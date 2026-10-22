"""
Look inside the FINSIGHT accounts database.

    python inspect_users.py

Read-only: it prints what is stored and changes nothing. Passwords cannot be
shown because they are not stored - only a PBKDF2 hash of them is.
"""
import json
import sqlite3
from pathlib import Path

import users  # reuse the app's own path logic

DB = users.DB_PATH


def main():
    if not Path(DB).exists():
        print(f"No database at {DB}\n"
              "Create an account in the app first, then run this again.")
        return

    print(f"Database: {DB}")
    print(f"Size    : {Path(DB).stat().st_size:,} bytes\n")

    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row

    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    print("Tables:", ", ".join(tables), "\n")

    print("=" * 70)
    print("USERS")
    print("=" * 70)
    rows = con.execute("SELECT * FROM users ORDER BY id").fetchall()
    if not rows:
        print("(no accounts yet)")
    for r in rows:
        budgets = json.loads(r["budget_json"] or "{}")
        active = {k: v for k, v in budgets.items() if v}
        print(f"""
  id            {r['id']}
  username      {r['username']}
  display_name  {r['display_name']}
  email         {r['email'] or '(not set)'}
  income        R{r['income']:,.2f}
  savings_goal  R{r['savings_goal']:,.2f}
  created_at    {r['created_at']}
  budgets       {active or '(none saved)'}
  pw_salt       {r['pw_salt'][:16]}...  (random per user)
  pw_hash       {r['pw_hash'][:16]}...  (PBKDF2-SHA256, 200k rounds)""")

    print("\n" + "=" * 70)
    print("ANALYSES (last 20)")
    print("=" * 70)
    rows = con.execute("""
        SELECT a.*, u.username
        FROM analyses a JOIN users u ON u.id = a.user_id
        ORDER BY a.id DESC LIMIT 20
    """).fetchall()
    if not rows:
        print("(no statements analysed yet)")
    else:
        print(f"{'when':<17} {'user':<12} {'txns':>5} {'spent':>12} "
              f"{'money in':>12}  top group")
        for r in rows:
            print(f"{r['created_at']:<17} {r['username']:<12} "
                  f"{r['n_purchases']:>5} {r['total_spent']:>12,.2f} "
                  f"{r['money_in']:>12,.2f}  {r['top_group']}")

    con.close()
    print("\nNote: passwords are never stored, so they cannot be recovered - "
          "only reset from inside the app.")


if __name__ == "__main__":
    main()
