"""
FINSIGHT user accounts.
Stores users in a small SQLite file. Passwords are never saved as text: each one is
mixed with a random "salt" and scrambled (PBKDF2) before it is stored.

On Hugging Face Spaces the normal disk is wiped whenever the Space restarts. If you
turn on "Persistent storage" for the Space, this file automatically uses /data instead,
so accounts survive restarts.
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

_default_dir = Path("/data") if Path("/data").is_dir() else Path(__file__).parent
DB_PATH = Path(os.environ.get("FINSIGHT_DB", _default_dir / "finsight_users.db"))

PBKDF2_ROUNDS = 200_000
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,30}$")
MIN_PASSWORD = 8
MAX_TRIES, LOCK_SECONDS = 5, 300  # 5 wrong passwords -> locked for 5 minutes

_fails = {}          # username -> (wrong tries so far, locked until timestamp)
_fails_lock = threading.Lock()


@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                pw_salt       TEXT NOT NULL,
                pw_hash       TEXT NOT NULL,
                display_name  TEXT,
                email         TEXT,
                income        REAL DEFAULT 0,
                savings_goal  REAL DEFAULT 0,
                budget_json   TEXT DEFAULT '{}',
                created_at    TEXT
            );
            CREATE TABLE IF NOT EXISTS analyses (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at   TEXT,
                n_purchases  INTEGER,
                total_spent  REAL,
                money_in     REAL,
                top_group    TEXT
            );
        """)


def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)


def _password_ok(row, password):
    if not row:
        _hash(password or "", b"\x00" * 16)  # keep the timing similar for unknown usernames
        return False
    return hmac.compare_digest(_hash(password or "", bytes.fromhex(row["pw_salt"])).hex(), row["pw_hash"])


def _public(row):
    """The safe version of a user (no password data)."""
    return {
        "id": row["id"],
        "username": row["username"],
        "display_name": row["display_name"] or row["username"],
        "email": row["email"] or "",
        "income": float(row["income"] or 0),
        "savings_goal": float(row["savings_goal"] or 0),
        "budgets": json.loads(row["budget_json"] or "{}"),
        "created_at": row["created_at"],
    }


def _check_new_password(password):
    if len(password or "") < MIN_PASSWORD:
        return f"Your password needs at least {MIN_PASSWORD} characters."
    return None


def create_user(username, password, display_name=""):
    """Returns (user, error_message). Exactly one of them is None."""
    username = (username or "").strip().lower()
    if not USERNAME_RE.match(username):
        return None, "Usernames need 3 to 30 letters, numbers, dots, dashes or underscores."
    err = _check_new_password(password)
    if err:
        return None, err
    salt = secrets.token_bytes(16)
    name = (display_name or "").strip()[:60] or username
    try:
        with _conn() as con:
            cur = con.execute(
                "INSERT INTO users (username, pw_salt, pw_hash, display_name, created_at) VALUES (?,?,?,?,?)",
                (username, salt.hex(), _hash(password, salt).hex(), name, time.strftime("%Y-%m-%d")))
            row = con.execute("SELECT * FROM users WHERE id = ?", (cur.lastrowid,)).fetchone()
            return _public(row), None
    except sqlite3.IntegrityError:
        return None, "That username is already taken. Please choose another one."


def verify_user(username, password):
    """Returns (user, error_message). Exactly one of them is None."""
    username = (username or "").strip().lower()
    now = time.time()
    with _fails_lock:
        _tries, locked_until = _fails.get(username, (0, 0))
        if locked_until > now:
            return None, "Too many wrong tries. Please wait a few minutes and try again."

    with _conn() as con:
        row = con.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        ok = _password_ok(row, password)
        user = _public(row) if (row and ok) else None

    with _fails_lock:
        if ok:
            _fails.pop(username, None)
        else:
            tries = _fails.get(username, (0, 0))[0] + 1
            _fails[username] = (0, now + LOCK_SECONDS) if tries >= MAX_TRIES else (tries, 0)
    if ok:
        return user, None
    return None, "Wrong username or password."


def get_user(user_id):
    with _conn() as con:
        row = con.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return _public(row) if row else None


def update_profile(user_id, display_name, email, income):
    """Returns (updated_user, error_message)."""
    name = (display_name or "").strip()[:60]
    email = (email or "").strip()[:120]
    try:
        income = float(income or 0)
    except (TypeError, ValueError):
        return None, "Your income must be a number."
    if not name:
        return None, "Please type a name to show in the app."
    if email and not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return None, "That email address does not look right."
    if income < 0:
        return None, "Your income cannot be negative."
    with _conn() as con:
        con.execute("UPDATE users SET display_name = ?, email = ?, income = ? WHERE id = ?",
                    (name, email, income, user_id))
        row = con.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return _public(row), None


def change_password(user_id, old_password, new_password):
    """Returns an error message, or None when it worked."""
    err = _check_new_password(new_password)
    if err:
        return err
    with _conn() as con:
        row = con.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not _password_ok(row, old_password):
            return "Your current password is not correct."
        salt = secrets.token_bytes(16)
        con.execute("UPDATE users SET pw_salt = ?, pw_hash = ? WHERE id = ?",
                    (salt.hex(), _hash(new_password, salt).hex(), user_id))
    return None


def save_budget(user_id, budgets, savings_goal):
    clean = {str(k): max(float(v or 0), 0.0) for k, v in budgets.items()}
    with _conn() as con:
        con.execute("UPDATE users SET budget_json = ?, savings_goal = ? WHERE id = ?",
                    (json.dumps(clean), max(float(savings_goal or 0), 0.0), user_id))


def add_analysis(user_id, total_spent, money_in, top_group, n_purchases):
    with _conn() as con:
        con.execute(
            "INSERT INTO analyses (user_id, created_at, n_purchases, total_spent, money_in, top_group) "
            "VALUES (?,?,?,?,?,?)",
            (user_id, time.strftime("%Y-%m-%d %H:%M"), int(n_purchases), float(total_spent),
             float(money_in), str(top_group)))


def list_analyses(user_id, limit=10):
    with _conn() as con:
        rows = con.execute("SELECT * FROM analyses WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                           (user_id, limit)).fetchall()
        return [dict(r) for r in rows]


def count_analyses(user_id):
    with _conn() as con:
        return con.execute("SELECT COUNT(*) FROM analyses WHERE user_id = ?", (user_id,)).fetchone()[0]


def delete_user(user_id, password):
    """Deletes the account and its history. Returns True when it worked."""
    with _conn() as con:
        row = con.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not _password_ok(row, password):
            return False
        con.execute("DELETE FROM analyses WHERE user_id = ?", (user_id,))
        con.execute("DELETE FROM users WHERE id = ?", (user_id,))
    return True


init_db()
