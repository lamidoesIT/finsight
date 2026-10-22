"""
FINSIGHT - Bank statement analyser (Gradio app for Hugging Face Spaces)
Flow:  PDF upload -> pdf_extraction.extract_transactions -> feature engineering
       -> trained XGBoost model -> category predictions -> insights + graphs
       -> (optional) budget set by the user -> budget vs reality + savings.
Users log in first (accounts live in users.py) and each user has a profile,
a saved budget plan and a short history of their analyses.
"""
import os
import hmac
import base64
import html
import re
import tempfile
import textwrap
from pathlib import Path

import inspect

import pandas as pd
import gradio as gr
import plotly.graph_objects as go

# ------------------------------------------------------------------
# Workaround for an upstream Gradio bug.
#
# On launch, Gradio decides whether our theme is custom by comparing it with
# every built-in theme. Font lists mix GoogleFont objects and plain strings,
# and GoogleFont.__eq__ assumes the other side is also a GoogleFont:
#
#     return self.name == other.name and ...
#     AttributeError: 'str' object has no attribute 'name'
#
# So the comparison explodes whenever a GoogleFont lines up with a string
# (the built-in "glass" theme triggers it). Making the comparison type-safe
# fixes it for any combination of fonts and any Gradio version.
# ------------------------------------------------------------------
try:
    from gradio.themes.utils.fonts import GoogleFont as _GoogleFont

    if not getattr(_GoogleFont, "_finsight_eq_patched", False):
        _orig_font_eq = _GoogleFont.__eq__

        def _safe_font_eq(self, other):
            if not isinstance(other, _GoogleFont):
                return NotImplemented
            return _orig_font_eq(self, other)

        _GoogleFont.__eq__ = _safe_font_eq
        _GoogleFont.__hash__ = lambda self: hash(self.name)
        _GoogleFont._finsight_eq_patched = True
except Exception:  # a Gradio version without this class needs no patch
    pass
from plotly.subplots import make_subplots

from pdf_extraction import extract_transactions, ExtractionError
import users


HERE = Path(__file__).parent

# ------------------------------------------------------------------
# 1 + 2. Model and feature engineering now live in finsight_core.py so the
#        Gradio app and the batch ETL pipeline share one implementation.
# ------------------------------------------------------------------
from finsight_core import (  # noqa: E402
    CATEGORIES, HIGH_CONFIDENCE, LOW_CONFIDENCE,
    build_features, predict_categories,
)


# ------------------------------------------------------------------
# 3. Plain-language wording for each spending group
# ------------------------------------------------------------------
ICONS = {
    "cart": '<circle cx="8" cy="21" r="1"/><circle cx="19" cy="21" r="1"/><path d="M2.05 2.05h2l2.66 12.42a2 2 0 0 0 2 1.58h9.78a2 2 0 0 0 1.95-1.57l1.65-7.43H5.12"/>',
    "utensils": '<path d="M3 2v7c0 1.1.9 2 2 2h4a2 2 0 0 0 2-2V2"/><path d="M7 2v20"/><path d="M21 15V2a5 5 0 0 0-5 5v6c0 1.1.9 2 2 2h3Zm0 0v7"/>',
    "bus": '<path d="M8 6v6"/><path d="M15 6v6"/><path d="M2 12h19.6"/><path d="M18 18h3s.5-1.7.8-2.8c.1-.4.2-.8.2-1.2 0-.4-.1-.8-.2-1.2l-1.4-5C20.1 6.8 19.1 6 18 6H4a2 2 0 0 0-2 2v10h3"/><circle cx="7" cy="18" r="2"/><path d="M9 18h5"/><circle cx="16" cy="18" r="2"/>',
    "bank": '<line x1="3" x2="21" y1="22" y2="22"/><line x1="6" x2="6" y1="18" y2="11"/><line x1="10" x2="10" y1="18" y2="11"/><line x1="14" x2="14" y1="18" y2="11"/><line x1="18" x2="18" y1="18" y2="11"/><polygon points="12 2 20 7 4 7"/>',
    "bag": '<path d="M6 2 3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4Z"/><path d="M3 6h18"/><path d="M16 10a4 4 0 0 1-8 0"/>',
    "film": '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M7 3v18"/><path d="M3 7.5h4"/><path d="M3 12h18"/><path d="M3 16.5h4"/><path d="M17 3v18"/><path d="M17 7.5h4"/><path d="M17 16.5h4"/>',
    "zap": '<path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/>',
    "send": '<path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/>',
    "package": '<path d="m7.5 4.27 9 5.15"/><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/>',
    "pin": '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>',
    "trophy": '<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/><path d="M4 22h16"/><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/><path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/><path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"/>',
    "repeat": '<path d="m17 2 4 4-4 4"/><path d="M3 11v-1a4 4 0 0 1 4-4h14"/><path d="m7 22-4-4 4-4"/><path d="M21 13v1a4 4 0 0 1-4 4H3"/>',
    "wallet": '<rect width="20" height="12" x="2" y="6" rx="2"/><circle cx="12" cy="12" r="2"/><path d="M6 12h.01M18 12h.01"/>',
    "coins": '<circle cx="8" cy="8" r="6"/><path d="M18.09 10.37A6 6 0 1 1 10.34 18"/><path d="M7 6h1v4"/><path d="m16.71 13.88.7.71-2.82 2.82"/>',
    "calendar": '<rect width="18" height="18" x="3" y="4" rx="2"/><path d="M16 2v4"/><path d="M8 2v4"/><path d="M3 10h18"/>',
    "calendar-days": '<rect width="18" height="18" x="3" y="4" rx="2"/><path d="M16 2v4"/><path d="M8 2v4"/><path d="M3 10h18"/><path d="M8 14h.01M12 14h.01M16 14h.01M8 18h.01M12 18h.01M16 18h.01"/>',
    "help": '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
}


def svg_icon(name, size=22, color="currentColor"):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
            f'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
            f'style="flex:none;vertical-align:middle">{ICONS.get(name, ICONS["pin"])}</svg>')


def dot(color):
    return (f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
            f'background:{color};margin-right:8px;vertical-align:baseline"></span>')


GROUP_INFO = {
    "groceries": ("cart", "#22C55E", "Supermarkets and food shops"),
    "food": ("utensils", "#F97316", "Restaurants, takeaways and food delivery"),
    "transport": ("bus", "#3B82F6", "Fuel, taxis, e-hailing and public transport"),
    "banking": ("bank", "#6366F1", "Bank charges and banking services"),
    "shopping": ("bag", "#EC4899", "Clothes, online shops and things you bought for yourself"),
    "entertainment": ("film", "#A855F7", "Streaming, cinema, games and fun"),
    "utilities": ("zap", "#EAB308", "Electricity, airtime, data and monthly bills"),
    "money sent": ("send", "#14B8A6", "Money you sent or transferred to other people"),
    "other": ("package", "#94A3B8", "Payments that did not fit any other group"),
}
FALLBACK_COLORS = ["#0EA5E9", "#F43F5E", "#84CC16", "#8B5CF6", "#F59E0B", "#06B6D4"]


def group_info(name):
    key = str(name).strip().lower()
    if key in GROUP_INFO:
        return GROUP_INFO[key]
    return ("pin", FALLBACK_COLORS[sum(map(ord, key)) % len(FALLBACK_COLORS)], "")


def color_for(name):
    return group_info(name)[1]


def icon_for(name):
    return group_info(name)[0]


def rand(value, decimals=2):
    return f"R{value:,.{decimals}f}"


def per_100(share):
    r = share * 100
    return "less than R1" if r < 1 else f"R{r:.0f}"


def sureness_label(conf):
    if conf >= HIGH_CONFIDENCE:
        return "Very sure"
    if conf >= LOW_CONFIDENCE:
        return "Fairly sure"
    return "Not sure"


def friendly_insights(df):
    """Plain-language observations about the statement. Returns [(icon, text)]."""
    data = df.dropna(subset=["amount_zar", "predicted_category"]).copy()
    total = data["amount_zar"].sum()
    by_group = (data.groupby("predicted_category")["amount_zar"].agg(["sum", "count"])
                .sort_values("sum", ascending=False))
    items = []

    top, top_sum = by_group.index[0], by_group["sum"].iloc[0]
    items.append(("trophy", f"Most of your money went to **{top}**: {rand(top_sum)}. "
                        f"That is about {per_100(top_sum / total)} of every R100 you spent."))

    often = by_group["count"].idxmax()
    items.append(("repeat", f"You paid for **{often}** more often than anything else: "
                        f"**{int(by_group.loc[often, 'count'])} times**."))

    big = data.loc[data["amount_zar"].idxmax()]
    items.append(("wallet", f"Your biggest single purchase was **{rand(big['amount_zar'])}** "
                        f"({big['description_sa'][:40]}) on {big['txn_date']:%d %b}."))

    small = data[data["amount_zar"] < 100]
    if len(small) >= 3:
        items.append(("coins", f"**{len(small)} small purchases** under R100 added up to "
                            f"**{rand(small['amount_zar'].sum())}**. Small amounts add up faster than you think."))

    daily = data.groupby(data["txn_date"].dt.normalize())["amount_zar"].sum()
    if len(daily) > 1:
        busiest = daily.idxmax()
        items.append(("calendar", f"Your busiest spending day was **{busiest:%d %b}**, "
                            f"when you spent **{rand(daily.max())}**."))
        days = (data["txn_date"].max() - data["txn_date"].min()).days + 1
        if days >= 7:
            items.append(("calendar-days", f"Over these {days} days you spent about "
                                f"**{rand(total / days, 0)} per day** on average."))

    unsure = int((data["confidence"] < LOW_CONFIDENCE).sum())
    if unsure:
        items.append(("help", f"For **{unsure} purchase(s)** the app was not very sure which group they belong to. "
                            "You can check them on the 'How sure are we?' tab."))
    return items


# ------------------------------------------------------------------
# 4. Look and feel (colours taken from the FINSIGHT logo)
# ------------------------------------------------------------------
# Two sets of colours, on purpose:
#
#   *_HEX  real colour values, for things that cannot read CSS variables:
#          the Gradio theme object and the Plotly charts.
#   bare   CSS variable references, used inside every HTML string we build.
#          Because they are variables, the whole interface re-colours itself
#          when the <body> gains the "dark" class - no HTML is regenerated.
NAVY_HEX, TEAL_HEX, TEAL_DARK_HEX = "#0F2A43", "#5EC8D8", "#148C9E"
TEAL_50_HEX, TEAL_100_HEX = "#F1FAFC", "#DCF1F6"
INK_HEX, MUTED_HEX, GRID_HEX, BORDER_HEX = "#16263A", "#64748B", "#EDF2F6", "#E2EBF0"
SURFACE_HEX, CANVAS_HEX = "#FFFFFF", "#F6F9FB"

# Dark palette: deep blue-greys rather than pure black, so the teal keeps its
# punch and large surfaces do not glare.
NAVY_DARK_HEX, INK_DARK_HEX, MUTED_DARK_HEX = "#E8F4F8", "#E6EDF3", "#94A6B8"
BORDER_DARK_HEX, GRID_DARK_HEX = "#24384D", "#1E3044"
SURFACE_DARK_HEX, CANVAS_DARK_HEX = "#111F2E", "#0B1623"

NAVY = "var(--fs-navy)"
TEAL = "var(--fs-teal)"
TEAL_DARK = "var(--fs-teal-dark)"
TEAL_50 = "var(--fs-teal-50)"
TEAL_100 = "var(--fs-teal-100)"
INK = "var(--fs-ink)"
MUTED = "var(--fs-muted)"
GRID = "var(--fs-grid)"
BORDER = "var(--fs-border)"
SURFACE = "var(--fs-surface)"
CANVAS = "var(--fs-canvas)"

SHADOW_SM = "var(--fs-shadow-sm)"
SHADOW_MD = "var(--fs-shadow-md)"
SHADOW_LG = "var(--fs-shadow-lg)"
RADIUS = "18px"

# Shown in the footer so the app is clearly attributable when embedded in a portfolio.
AUTHOR_NAME = os.environ.get("FINSIGHT_AUTHOR", "Lami Langa")
AUTHOR_GITHUB = os.environ.get("FINSIGHT_GITHUB", "https://github.com/LamidoesIT")
AUTHOR_LINKEDIN = os.environ.get("FINSIGHT_LINKEDIN", "https://www.linkedin.com/in/lami-langa-a063272b5")


def _load_logo():
    path = HERE / "logo.png"
    if path.exists():
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
    return None


LOGO_SRC = _load_logo()
FAVICON = HERE / "favicon.png"

BRAND = gr.themes.Color(
    c50="#F1FAFC", c100="#DCF1F6", c200="#BBE6EF", c300="#8FD8E6", c400="#5EC8D8",
    c500="#2FAFC4", c600="#148C9E", c700="#116F7F", c800="#135866", c900="#0F2A43",
    c950="#0A1C2E", name="finsight",
)

THEME = gr.themes.Base(
    primary_hue=BRAND,
    secondary_hue=BRAND,
    neutral_hue=gr.themes.colors.slate,
    # Keep exactly one GoogleFont first, then plain string fallbacks: some Gradio
    # versions compare this list element-wise against a built-in theme and crash
    # if a GoogleFont lines up with a plain string.
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "Inter", "ui-sans-serif",
          "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill=CANVAS_HEX, body_background_fill_dark=CANVAS_DARK_HEX,
    body_text_color=INK_HEX, body_text_color_dark=INK_DARK_HEX,
    block_background_fill=SURFACE_HEX, block_background_fill_dark=SURFACE_DARK_HEX,
    block_border_color=BORDER_HEX, block_border_color_dark=BORDER_DARK_HEX,
    block_shadow="0 4px 14px rgba(27, 60, 91, 0.07)",
    block_shadow_dark="0 4px 14px rgba(27, 60, 91, 0.07)",
    block_title_text_color=NAVY_HEX, block_title_text_color_dark=NAVY_DARK_HEX,
    block_label_text_color=MUTED_HEX, block_label_text_color_dark=MUTED_DARK_HEX,
    button_primary_background_fill="linear-gradient(135deg, #1B3C5B, #2A5F8A)",
    button_primary_background_fill_dark="linear-gradient(135deg, #1B3C5B, #2A5F8A)",
    button_primary_background_fill_hover="linear-gradient(135deg, #12293F, #1B3C5B)",
    button_primary_background_fill_hover_dark="linear-gradient(135deg, #12293F, #1B3C5B)",
    button_primary_text_color="white", button_primary_text_color_dark="white",
    table_even_background_fill=SURFACE_HEX, table_even_background_fill_dark=SURFACE_DARK_HEX,
    table_odd_background_fill="#F5FBFC", table_odd_background_fill_dark="#16293B",
    table_border_color=BORDER_HEX, table_border_color_dark=BORDER_DARK_HEX,
    table_text_color=INK_HEX, table_text_color_dark=INK_DARK_HEX,
)

CSS = """
/* ---------- design tokens: light ---------- */
:root {
  --fs-navy: #0F2A43; --fs-teal: #5EC8D8; --fs-teal-dark: #148C9E;
  --fs-teal-50: #F1FAFC; --fs-teal-100: #DCF1F6;
  --fs-ink: #16263A; --fs-muted: #64748B;
  --fs-border: #E2EBF0; --fs-grid: #EDF2F6;
  --fs-canvas: #F6F9FB; --fs-surface: #FFFFFF;
  --fs-track: #EDF3F6;
  --fs-glow-1: rgba(228,245,249,.9); --fs-glow-2: rgba(234,240,251,.9);
  --fs-shadow-sm: 0 1px 2px rgba(15,42,67,.06);
  --fs-shadow-md: 0 6px 20px -8px rgba(15,42,67,.16), 0 1px 3px rgba(15,42,67,.05);
  --fs-shadow-lg: 0 18px 40px -16px rgba(15,42,67,.22), 0 2px 6px rgba(15,42,67,.05);
  --fs-nav-active: linear-gradient(135deg, #0F2A43, #1D4E6E);
  --fs-input-bg: #FCFDFE;
  --fs-warn-bg: #FFFBEB; --fs-warn-border: #FDE68A; --fs-warn-text: #92400E;
  --fs-ok-bg: #ECFDF5; --fs-ok-border: #BBF7D0; --fs-ok-text: #166534;
  --fs-err-bg: #FEF2F2; --fs-err-border: #FECACA; --fs-err-text: #991B1B;
}

/* ---------- design tokens: dark ----------
   Gradio puts the class "dark" on <body>, so one class flips every surface,
   border and text colour across the whole interface. */
body.dark {
  --fs-navy: #E8F4F8; --fs-teal: #5EC8D8; --fs-teal-dark: #6FD6E6;
  --fs-teal-50: rgba(94,200,216,.10); --fs-teal-100: rgba(94,200,216,.26);
  --fs-ink: #E6EDF3; --fs-muted: #94A6B8;
  --fs-border: #24384D; --fs-grid: #1E3044;
  --fs-canvas: #0B1623; --fs-surface: #111F2E;
  --fs-track: #1C2E42;
  --fs-glow-1: rgba(20,140,158,.20); --fs-glow-2: rgba(45,80,140,.18);
  --fs-shadow-sm: 0 1px 2px rgba(0,0,0,.40);
  --fs-shadow-md: 0 6px 20px -8px rgba(0,0,0,.60), 0 1px 3px rgba(0,0,0,.40);
  --fs-shadow-lg: 0 18px 40px -16px rgba(0,0,0,.68), 0 2px 6px rgba(0,0,0,.45);
  --fs-nav-active: linear-gradient(135deg, #148C9E, #1D6F85);
  --fs-input-bg: #0E1B29;
  --fs-warn-bg: rgba(234,179,8,.12); --fs-warn-border: rgba(234,179,8,.38); --fs-warn-text: #FBCF5B;
  --fs-ok-bg: rgba(34,197,94,.12); --fs-ok-border: rgba(34,197,94,.36); --fs-ok-text: #6EE7A0;
  --fs-err-bg: rgba(239,68,68,.12); --fs-err-border: rgba(239,68,68,.36); --fs-err-text: #FCA5A5;
}

/* soft ambient background instead of a flat fill */
body, .gradio-container {
  background:
    radial-gradient(900px 420px at 12% -5%, var(--fs-glow-1) 0%, transparent 60%),
    radial-gradient(760px 420px at 88% 0%, var(--fs-glow-2) 0%, transparent 58%),
    var(--fs-canvas) !important;
  transition: background-color .2s ease, color .2s ease;
}
.gradio-container { max-width: 1180px !important; margin: 0 auto !important;
                    padding-top: 18px !important; }
* { -webkit-font-smoothing: antialiased; }

/* ---------- blocks: flatter, lighter, more breathing room ---------- */
.block, .form, .panel {
  border-radius: 18px !important;
  border-color: var(--fs-border) !important;
}
.gap, .gradio-container .main > .wrap { gap: 14px !important; }
label > span, .block-title { letter-spacing: -.005em !important; }

/* ---------- buttons ---------- */
button.primary, button.lg.primary {
  letter-spacing: -.01em !important;
  box-shadow: 0 8px 18px -8px rgba(15,42,67,.45) !important;
  transition: transform .12s ease, box-shadow .12s ease, filter .12s ease !important;
}
button.primary:hover { transform: translateY(-1px); filter: saturate(1.05); }
button.primary:active { transform: translateY(0); }
button.secondary {
  background: var(--fs-surface) !important; border: 1px solid var(--fs-border) !important;
  color: var(--fs-navy) !important; font-weight: 620 !important;
  box-shadow: var(--fs-shadow-sm) !important;
}
button.secondary:hover { background: var(--fs-teal-50) !important;
                         border-color: var(--fs-teal-100) !important; }

/* ---------- inputs ---------- */
input[type="text"], input[type="password"], input[type="number"], textarea {
  border-radius: 12px !important; border: 1px solid var(--fs-border) !important;
  background: var(--fs-input-bg) !important; transition: border-color .12s, box-shadow .12s !important;
}
input:focus, textarea:focus {
  border-color: var(--fs-teal) !important;
  box-shadow: 0 0 0 4px rgba(94,200,216,.18) !important; outline: none !important;
}

/* ---------- tables ---------- */
.gradio-container table thead th {
  background: var(--fs-teal-50) !important; color: var(--fs-navy) !important;
  font-weight: 680 !important; font-size: .86rem !important;
  text-transform: uppercase !important; letter-spacing: .04em !important;
  border-bottom: 1px solid var(--fs-border) !important;
}
.gradio-container table td { font-size: .93rem !important; }
.gradio-container table tbody tr:hover td { background: var(--fs-teal-50) !important; }
footer { display: none !important; }
.html-container { padding: 0 !important; }
.modebar-container { display: none !important; }
/* keep the page inside the screen on phones; tab row scrolls sideways */
.gradio-container, .gradio-container .main, .gradio-container .wrap, main.contain, .column, .tabs, .tabitem
{ min-width: 0 !important; max-width: 100% !important; }
.tab-wrapper { width: 100% !important; max-width: 100% !important; overflow-x: auto !important;
               scrollbar-width: none; }
.tab-wrapper::-webkit-scrollbar { display: none; }
.tab-container:not(.visually-hidden) { width: max-content !important; max-width: none !important;
                                       overflow: visible !important; }
.overflow-menu { border: none !important; background: transparent !important;
                 border-radius: 999px !important; color: var(--fs-muted) !important; }
.overflow-menu:hover { background: var(--fs-teal-50) !important; }
button[role="tab"] { white-space: nowrap !important; flex: none !important; font-weight: 650 !important; font-size: 0.97rem !important; color: #64748B !important; }
button[role="tab"][aria-selected="true"] { color: #1B3C5B !important; }
/* ---------- upload zone ---------- */
#upload-card {
  border: 1.5px dashed var(--fs-teal-100) !important;
  background: var(--fs-teal-50) !important;
  border-radius: 18px !important; box-shadow: none !important;
  transition: border-color .15s, background .15s !important;
}
#upload-card:hover { border-color: var(--fs-teal-dark) !important;
                     background: var(--fs-teal-100) !important; }

#run-btn, #compare-btn {
  font-size: 1.02rem !important; font-weight: 680 !important; border-radius: 14px !important;
  min-height: 52px !important; letter-spacing: -.01em !important;
}
#login-card { max-width: 560px !important; width: 100% !important; margin: 10px auto 26px !important; }
/* the login/signup fields sit in nested white blocks: flatten them so the
   rounded corners don't stack into a double border */
#login-card .form { border: 0 !important; background: transparent !important;
                    box-shadow: none !important; gap: 12px !important; }
#login-card .block { border: 0 !important; background: transparent !important;
                     box-shadow: none !important; padding: 0 !important; }
#login-card .tabitem { background: var(--fs-surface) !important; border: 1px solid var(--fs-border) !important;
                       border-radius: 18px !important; padding: 20px !important;
                       box-shadow: var(--fs-shadow-md) !important; margin-top: 14px !important; }
#logout-btn { border-radius: 12px !important; height: 100% !important; min-height: 46px; }

/* ---------- pill navigation ---------- */
.tab-wrapper {
  background: color-mix(in srgb, var(--fs-surface) 86%, transparent) !important;
  backdrop-filter: blur(8px);
  border: 1px solid var(--fs-border) !important; border-radius: 999px !important;
  padding: 5px !important; box-shadow: var(--fs-shadow-md) !important;
  position: sticky !important; top: 8px !important; z-index: 30 !important;
}
.tab-wrapper::after, .tab-container::after, button[role="tab"]::after { display: none !important; }
button[role="tab"] {
  border: none !important; border-radius: 999px !important; padding: 9px 17px !important;
  background: transparent !important; font-weight: 640 !important; font-size: .94rem !important;
  color: var(--fs-muted) !important; letter-spacing: -.01em !important;
  transition: background .15s, color .15s, box-shadow .15s !important;
}
button[role="tab"]:hover { background: var(--fs-teal-50) !important; color: var(--fs-navy) !important; }
button[role="tab"][aria-selected="true"] {
  background: var(--fs-nav-active) !important; color: #FFFFFF !important;
  box-shadow: 0 6px 14px -6px rgba(15,42,67,.55) !important;
}
.tabitem { padding-top: 18px !important; }
#login-card .tab-wrapper { width: 100% !important; position: static !important; }

/* ---------- charts in dark mode ----------
   Plotly figures are built on the server in light colours and are not
   regenerated when the theme flips. They render as inline SVG, so the text
   and gridlines can simply be restyled with CSS. The data colours are left
   alone on purpose: the category palette reads well on both backgrounds. */
body.dark .js-plotly-plot .gtitle,
body.dark .js-plotly-plot .gtitle tspan { fill: var(--fs-navy) !important; }
body.dark .js-plotly-plot .xtick text,
body.dark .js-plotly-plot .ytick text,
body.dark .js-plotly-plot .legendtext,
body.dark .js-plotly-plot .annotation-text,
body.dark .js-plotly-plot .annotation-text tspan,
body.dark .js-plotly-plot g.infolayer text { fill: var(--fs-ink) !important; }
body.dark .js-plotly-plot .xtitle, body.dark .js-plotly-plot .ytitle { fill: var(--fs-muted) !important; }
body.dark .js-plotly-plot .gridlayer path,
body.dark .js-plotly-plot .zerolinelayer path { stroke: var(--fs-grid) !important; }
/* slice/bar outlines are white in light mode; match the dark surface instead */
body.dark .js-plotly-plot .pielayer path.surface { stroke: var(--fs-surface) !important; }

/* The logo artwork is dark navy, which disappears on a dark background.
   Inverting lightness and rotating the hue back keeps the brand teal
   recognisable instead of flipping it to orange. */
body.dark img[alt^="FINSIGHT"] { filter: invert(1) hue-rotate(180deg) saturate(1.15); }

/* ---------- theme toggle ---------- */
#theme-btn {
  border-radius: 12px !important; min-height: 46px !important;
  height: 100% !important; font-size: 1.05rem !important; padding: 0 !important;
}
#theme-btn:hover { background: var(--fs-teal-50) !important; }

/* ---------- misc ---------- */
button.lg, button.primary { border-radius: 12px !important; }
.fs-card { transition: transform .14s ease, box-shadow .14s ease; }
.fs-card:hover { transform: translateY(-2px); box-shadow: var(--fs-shadow-lg); }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
/* ---------- phones ----------
   The tab strip scrolls sideways on narrow screens, which hid the
   "Create account" tab completely. Let the tabs wrap onto two rows
   instead: everything stays visible with nothing to discover. */
@media (max-width: 760px) {
  .gradio-container { padding-left: 10px !important; padding-right: 10px !important; }
  button[role="tab"] { padding: 8px 13px !important; font-size: .88rem !important; }

  .tab-wrapper { overflow-x: visible !important; border-radius: 18px !important; }

  /* top bar: identity on its own line, the two buttons side by side below it.
     Three across does not fit a phone - the button labels spill out. */
  #top-bar { flex-wrap: wrap !important; gap: 8px !important; }
  #top-bar > * { min-width: 0 !important; }
  #top-bar > *:first-child { flex: 1 1 100% !important; }
  #top-bar > *:not(:first-child) { flex: 1 1 calc(50% - 8px) !important; }
  #theme-btn, #logout-btn {
    white-space: nowrap !important; font-size: .9rem !important;
    min-height: 44px !important; padding: 0 12px !important; width: 100% !important;
  }
  .tab-container:not(.visually-hidden) {
    width: 100% !important; display: flex !important; flex-wrap: wrap !important;
    justify-content: center !important; gap: 4px !important;
  }
  button[role="tab"] { flex: 0 1 auto !important; }
}

/* The login card only ever has two tabs: give them half the width each
   at any screen size, so they always read as a pair of real buttons. */
#login-card .tab-container:not(.visually-hidden) {
  width: 100% !important; display: flex !important;
}
#login-card button[role="tab"] { flex: 1 1 0 !important; text-align: center !important;
                                 justify-content: center !important; }
"""

DARK_LABEL, LIGHT_LABEL = "☾  Dark", "☀  Light"

# Applied once when the page loads: use the saved choice, and fall back to
# whatever the operating system is set to. Wrapped in try/catch because
# localStorage throws in private windows and inside some embeds (the app is
# designed to be iframed into a portfolio site).
RESTORE_THEME_JS = f"""
() => {{
  let pref = null;
  try {{ pref = localStorage.getItem('finsight-theme'); }} catch (e) {{}}
  if (!pref && window.matchMedia
      && window.matchMedia('(prefers-color-scheme: dark)').matches) {{
    pref = 'dark';
  }}
  const dark = pref === 'dark';
  document.body.classList.toggle('dark', dark);
  const label = dark ? '{LIGHT_LABEL}' : '{DARK_LABEL}';
  return [label, label];
}}
"""

# Flips the theme and remembers it. No server round-trip: one class on <body>
# re-points every CSS variable, so the whole interface recolours instantly.
TOGGLE_THEME_JS = f"""
() => {{
  const dark = document.body.classList.toggle('dark');
  try {{ localStorage.setItem('finsight-theme', dark ? 'dark' : 'light'); }} catch (e) {{}}
  const label = dark ? '{LIGHT_LABEL}' : '{DARK_LABEL}';
  return [label, label];
}}
"""


def card_html(label, value, sub, accent=TEAL):
    """A stat tile: thin accent rail on the left, label, big number, one line of context."""
    return (
        f'<div class="fs-card" style="position:relative;background:{SURFACE};border:1px solid {BORDER};'
        f'border-radius:{RADIUS};padding:17px 20px 16px 22px;box-shadow:{SHADOW_MD};overflow:hidden">'
        f'<span style="position:absolute;left:0;top:0;bottom:0;width:4px;background:{accent}"></span>'
        f'<div style="font-size:.74rem;font-weight:680;letter-spacing:.07em;text-transform:uppercase;'
        f'color:{MUTED}">{label}</div>'
        f'<div style="font-size:1.72rem;font-weight:760;color:{NAVY};margin:7px 0 3px;line-height:1.2;'
        f'letter-spacing:-.02em;font-variant-numeric:tabular-nums">{value}</div>'
        f'<div style="font-size:.85rem;color:{MUTED};line-height:1.45">{sub}</div></div>'
    )


def card_grid(cards):
    return ('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));'
            f'gap:14px;margin:4px 0 18px">{"".join(cards)}</div>')


def bold(text):
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(text))


def hint_html(text):
    return (f'<div style="background:{SURFACE};border:1px dashed {BORDER};border-radius:{RADIUS};'
            f'padding:42px 24px;text-align:center;color:{MUTED};line-height:1.6;font-size:.95rem">'
            f'<div style="margin-bottom:10px;opacity:.6">{svg_icon("zap", 26, TEAL_DARK)}</div>{text}</div>')


def section_title(text, sub=""):
    sub_html = (f'<div style="color:{MUTED};font-size:.92rem;margin-top:3px;line-height:1.5">{sub}</div>'
                if sub else "")
    return (f'<div style="margin:22px 2px 14px;padding-left:12px;border-left:3px solid {TEAL}">'
            f'<div style="font-weight:730;font-size:1.14rem;color:{NAVY};letter-spacing:-.015em">{text}</div>'
            f'{sub_html}</div>')


HINT_MONEY = hint_html("Press the dark button at the top to see where your money went.")
HINT_DAY = hint_html("Press the dark button at the top to see your spending day by day.")
HINT_TX = hint_html("Press the dark button at the top to see every purchase and its group.")
HINT_CONF = hint_html("Press the dark button at the top to see how sure the app was.")
HINT_CMP = hint_html("First upload a statement and press the dark button at the top. "
                     "Then fill in your plan on the 'My budget' tab and press the button there.")

_logo_block = (
    f'<img src="{LOGO_SRC}" alt="FINSIGHT: clarity in spending" style="display:block;margin:0 auto;height:150px;max-width:100%">'
    if LOGO_SRC else
    f'<div style="font-size:2.2rem;font-weight:800;letter-spacing:.06em;color:{NAVY}">FIN'
    f'<span style="color:{TEAL}">SIGHT</span></div>'
)

HERO_HTML = f"""
<div style="position:relative;background:{SURFACE};border:1px solid {BORDER};border-radius:24px;
            padding:30px 28px 24px;text-align:center;box-shadow:{SHADOW_LG};overflow:hidden">
  <div style="position:absolute;inset:0;background:
       radial-gradient(420px 180px at 50% -20%, #DCF1F6 0%, rgba(220,241,246,0) 70%);
       pointer-events:none"></div>
  <div style="position:relative">
    <div style="display:inline-flex;align-items:center;gap:7px;background:{TEAL_50};
                border:1px solid {TEAL_100};color:{TEAL_DARK};border-radius:999px;
                padding:5px 13px;font-size:.74rem;font-weight:700;letter-spacing:.08em;
                text-transform:uppercase;margin-bottom:16px">
      {svg_icon("zap", 13, TEAL_DARK)} Machine-learning demo
    </div>
    {_logo_block}
    <div style="color:#334155;font-size:1.08rem;margin-top:12px;line-height:1.6;max-width:430px;
                margin-left:auto;margin-right:auto">
      Upload a bank statement and see, in plain words, exactly where your money went.
    </div>
    <div style="display:inline-flex;align-items:center;gap:8px;color:{MUTED};font-size:.8rem;
                margin-top:14px;background:var(--fs-warn-bg);border:1px solid var(--fs-warn-border);color:var(--fs-warn-text);
                border-radius:10px;padding:7px 13px;line-height:1.4;text-align:left">
      {svg_icon("alert", 15, "var(--fs-warn-text)")}
      <span>Demo only &mdash; please use the sample statements, not real personal ones.</span>
    </div>
  </div>
</div>
"""

EMPTY_STATE_HTML = f"""
<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:22px;padding:34px 24px 30px;
            text-align:center;box-shadow:{SHADOW_MD}">
  <div style="font-size:1.42rem;font-weight:770;color:{NAVY};letter-spacing:-.02em">
    Let's see where your money went</div>
  <div style="color:{MUTED};margin:7px 0 26px;font-size:.98rem">Three steps, about a minute</div>
  <div style="display:flex;justify-content:center;gap:14px;flex-wrap:wrap;text-align:left">
    {"".join(
        f'<div class="fs-card" style="background:{SURFACE};border:1px solid {BORDER};border-radius:18px;'
        f'padding:20px;min-width:215px;max-width:265px;flex:1;box-shadow:{SHADOW_SM}">'
        f'<div style="display:flex;align-items:center;gap:11px;margin-bottom:12px">'
        f'<div style="width:42px;height:42px;border-radius:13px;background:{TEAL_50};'
        f'border:1px solid {TEAL_100};display:flex;align-items:center;justify-content:center;flex:none">'
        f'{svg_icon(icon, 22, TEAL_DARK)}</div>'
        f'<div style="color:{TEAL_DARK};font-weight:800;font-size:.72rem;letter-spacing:.1em">STEP {n}</div></div>'
        f'<div style="color:{NAVY};font-weight:720;margin:0 0 5px;font-size:1.02rem">{t}</div>'
        f'<div style="color:{MUTED};font-size:.88rem;line-height:1.55">{d}</div></div>'
        for n, icon, t, d in [
            ("1", "bank", "Upload your statement", "A PDF from your banking app or website."),
            ("2", "zap", "Press analyse", "We read it and sort every payment for you."),
            ("3", "trophy", "Explore your results", "Then set a budget and see how you did."),
        ])}
  </div>
</div>
"""

CONFIDENCE_EXPLAINER_HTML = f"""
<div style="background:{TEAL_50};border:1px solid {TEAL_100};border-radius:16px;padding:18px 22px;
            color:{INK};line-height:1.65;margin-top:6px">
  <div style="font-weight:750;color:{NAVY};font-size:1.05rem;margin-bottom:6px">How does the app decide?</div>
  The app reads the name of the shop or payment and compares it with many payments it has seen before.
  Then it puts your purchase into the spending group it thinks fits best.<br><br>
  <strong>How sure</strong> tells you how strongly the app believes it picked the right group.
  {dot("#22C55E")}<strong>Very sure</strong> means it is almost certain. {dot("#EAB308")}<strong>Fairly sure</strong>
  means it is probably right. {dot("#EF4444")}<strong>Not sure</strong> means it was close to picking a different group, so please check that purchase yourself.
  The last column shows which group it nearly picked instead.<br><br>
  Even when the app is very sure, it can still be wrong. Treat it as a helpful first guess.
</div>
"""

GLOSSARY_HTML = f"""
<div style="color:{INK};line-height:1.65">
  <p><strong>Bank statement:</strong> a list of everything that went in and out of your bank account over a
  period of time, usually a month.</p>
  <p><strong>Purchase (or payment):</strong> one payment, like buying lunch or paying a bill.
  Banks call these "transactions".</p>
  <p><strong>Spending group:</strong> a bucket we put each purchase into, like Groceries or Transport.
  The app chooses the group by reading the shop or payment name.</p>
  <p><strong>"Out of every R100":</strong> a way to picture your spending. If you had spent exactly R100,
  this shows how many rand would have gone to each group.</p>
  <p><strong>How sure:</strong> how strongly the app believes it chose the right group for a purchase.</p>
  <p><strong>Money that came in:</strong> money going into your account, like your salary.
  We do not put these into spending groups.</p>
  <p><strong>Budget:</strong> a plan for how much you want to spend on each group. You type it in on the
  'My budget' tab and we compare it with what you really spent.</p>
</div>
"""

BUDGET_INTRO_HTML = f"""
<div style="background:{TEAL_50};border:1px solid {TEAL_100};border-radius:16px;padding:18px 22px;
            color:{INK};line-height:1.65;margin-bottom:12px">
  <div style="font-weight:750;color:{NAVY};font-size:1.05rem;margin-bottom:6px">Plan your spending</div>
  Type how much you <strong>want</strong> to spend on each group, for the same period as your statement.
  If you do not have a plan for a group, leave it at 0.
  You can also say how much you would like to save. Then press the button at the bottom and we will
  show you how your plan compares with what you really spent.
</div>
"""


# ------------------------------------------------------------------
# 5. Charts (interactive, colourful, with a plain-English caption)
# ------------------------------------------------------------------
def chart_title(title, caption):
    wrapped = "<br>".join(textwrap.wrap(caption, width=46))  # wraps on phones instead of being cut off
    return (f"<b>{title}</b><br><span style='font-size:13px;color:{MUTED_HEX}'>{wrapped}</span>")


def style_fig(fig, title, caption, height=460):
    fig.update_layout(
        title=dict(text=chart_title(title, caption), x=0.02, xanchor="left", font=dict(size=19, color=NAVY_HEX)),
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, Inter, system-ui, sans-serif", size=14, color="#334155"),
        margin=dict(l=24, r=34, t=116, b=44),
        hoverlabel=dict(bgcolor="white", bordercolor=TEAL_HEX, font=dict(color=NAVY_HEX, size=14)),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_HEX, zeroline=False, linecolor=GRID_HEX, ticks="")
    fig.update_yaxes(showgrid=True, gridcolor=GRID_HEX, zeroline=False, linecolor=GRID_HEX, ticks="")
    return fig


def fig_money_donut(df):
    totals = df.groupby("predicted_category")["amount_zar"].sum().sort_values(ascending=False)
    fig = go.Figure(go.Pie(
        labels=list(totals.index), values=totals.values,
        hole=0.58, sort=False, direction="clockwise",
        marker=dict(colors=[color_for(c) for c in totals.index], line=dict(color="white", width=3)),
        texttemplate="%{percent:.0%}", textposition="inside", insidetextorientation="horizontal",
        textfont=dict(color="white", size=15),
        hovertemplate="<b>%{label}</b><br>R%{value:,.2f}<br>%{percent} of everything you spent<extra></extra>",
    ))
    style_fig(fig, "Out of every R100 you spent",
              "Each colour is a spending group. A bigger slice means more of your money went there.", height=560)
    fig.update_layout(
        margin=dict(l=24, r=24, t=116, b=24),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.04, yanchor="top", font=dict(size=14)),
        uniformtext=dict(minsize=12, mode="hide"),
        annotations=[dict(text=f"<b style='font-size:30px;color:{NAVY_HEX}'>{rand(totals.sum(), 0)}</b><br>"
                              f"<span style='color:{MUTED_HEX}'>spent in total</span>",
                          x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)],
    )
    return fig


def fig_daily_by_group(df):
    work = df.assign(day=df["txn_date"].dt.normalize())
    pivot = work.pivot_table(index="day", columns="predicted_category", values="amount_zar",
                             aggfunc="sum", fill_value=0)
    fig = go.Figure()
    for c in pivot.sum().sort_values(ascending=False).index:
        fig.add_bar(x=pivot.index, y=pivot[c], name=str(c), marker_color=color_for(c),
                    hovertemplate="%{x|%d %b}<br>" + c + ": R%{y:,.2f}<extra></extra>")
    style_fig(fig, "What you spent each day",
              "Taller bars mean you spent more that day. The colours show what you spent it on.", height=500)
    fig.update_layout(barmode="stack", bargap=0.2,
                      legend=dict(orientation="h", x=0, y=-0.18, font=dict(size=13)))
    fig.update_xaxes(showgrid=False, tickformat="%d %b")
    fig.update_yaxes(title_text="Rand spent", tickprefix="R", tickformat=",")
    return fig


def fig_sureness_by_group(df):
    avg = df.groupby("predicted_category")["confidence"].mean().sort_values()
    fig = go.Figure(go.Bar(
        x=avg.values * 100, y=list(avg.index), orientation="h",
        marker=dict(color=[color_for(c) for c in avg.index]),
        text=[f"{v:.0%}" for v in avg.values], textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{x:.0f}% sure on average<extra></extra>",
    ))
    style_fig(fig, "How sure we are about each group",
              "A longer bar means the app was more certain it put your purchases in the right group.",
              height=max(400, 54 * len(avg) + 170))
    fig.update_xaxes(range=[0, 112], ticksuffix="%")
    fig.update_yaxes(showgrid=False, automargin=True)
    fig.update_layout(bargap=0.35)
    return fig


def fig_budget_vs_spent(budgets, spent):
    cats = [c for c in CATEGORIES if budgets[c] > 0 or spent[c] > 0]
    fig = go.Figure()
    fig.add_bar(x=cats, y=[budgets[c] for c in cats], name="What you planned", marker_color=TEAL_HEX,
                hovertemplate="%{x}<br>Planned: R%{y:,.2f}<extra></extra>")
    fig.add_bar(x=cats, y=[spent[c] for c in cats], name="What you spent", marker_color=NAVY_HEX,
                hovertemplate="%{x}<br>Spent: R%{y:,.2f}<extra></extra>")
    style_fig(fig, "Planned vs spent",
              "When the dark bar is taller than the light one, you spent more than you planned.", height=480)
    fig.update_layout(barmode="group", bargap=0.25,
                      legend=dict(orientation="h", x=0, y=-0.2, font=dict(size=13)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(tickprefix="R", tickformat=",")
    return fig


# ------------------------------------------------------------------
# 6. Page content built from the results
# ------------------------------------------------------------------
def stat_cards(result, money_in_total):
    totals = result.groupby("predicted_category")["amount_zar"].sum()
    total = totals.sum()
    top = totals.idxmax()
    cards = [
        card_html("You spent in total", rand(total), f"on {len(result)} purchases", TEAL),
        card_html("Your biggest spending group",
                  f'<span style="display:inline-flex;align-items:center;gap:9px;font-size:1.34rem;'
                  f'white-space:nowrap">'
                  f'{svg_icon(icon_for(top), 23, color_for(top))}{html.escape(str(top))}</span>',
                  f"about {per_100(totals.max() / total)} of every R100", color_for(top)),
        card_html("A typical purchase", rand(result["amount_zar"].median()),
                  "Half of your purchases were smaller than this", "#38BDF8"),
        card_html("How sure is the app?", f"{result['confidence'].mean():.0%}",
                  "How certain it was about its answers, on average", NAVY),
    ]
    if money_in_total:
        cards.append(card_html("Money that came in", rand(money_in_total),
                               "Not counted in the spending groups", "#22C55E"))
    return card_grid(cards)


def short_version(result, note_extra, warning):
    totals = result.groupby("predicted_category")["amount_zar"].sum()
    top, total = totals.idxmax(), totals.sum()
    big = result.loc[result["amount_zar"].idxmax()]
    text = (f"This statement shows **{len(result)} purchases** adding up to **{rand(total)}**. "
            f"The biggest share, **{totals.max() / total:.0%}** ({rand(totals.max())}), went to "
            f"**{top}**. Your biggest single purchase was **{rand(big['amount_zar'])}**.")
    out = (f'<div style="background:{SURFACE};border:1px solid {BORDER};border-left:6px solid {TEAL};'
           f'border-radius:16px;padding:18px 22px;box-shadow:var(--fs-shadow-md);margin-bottom:12px">'
           f'<div style="font-weight:750;color:{NAVY};font-size:1.05rem;margin-bottom:4px">The short version</div>'
           f'<div style="color:{INK};line-height:1.65;font-size:1.02rem">{bold(text)}</div>'
           f'<div style="color:{MUTED};font-size:.86rem;margin-top:8px">{bold(note_extra)}</div></div>')
    if warning:
        out += (f'<div style="background:var(--fs-warn-bg);border:1px solid var(--fs-warn-border);color:var(--fs-warn-text);border-radius:12px;'
                f'padding:10px 14px;margin-bottom:12px;font-size:.9rem;display:flex;gap:10px;align-items:center">'
                f'{svg_icon("alert", 20, "var(--fs-warn-text)")}<span>{html.escape(warning)}</span></div>')
    return out


def insight_list(items):
    rows = "".join(
        f'<div style="display:flex;gap:14px;align-items:center;padding:13px 16px;border-radius:14px;'
        f'background:{SURFACE};border:1px solid {BORDER};margin-bottom:9px;box-shadow:var(--fs-shadow-sm)">'
        f'<div style="width:40px;height:40px;border-radius:12px;background:{TEAL_50};display:flex;'
        f'align-items:center;justify-content:center;flex:none">{svg_icon(icon, 22, TEAL_DARK)}</div>'
        f'<div style="color:{INK};line-height:1.55">{bold(text)}</div></div>'
        for icon, text in items
    )
    return section_title("What we noticed") + rows


def group_breakdown(result):
    totals = result.groupby("predicted_category")["amount_zar"].sum().sort_values(ascending=False)
    grand = totals.sum()
    rows = []
    for name, amount in totals.items():
        icon, colour, desc = group_info(name)
        share = amount / grand
        rows.append(
            f'<div class="fs-card" style="background:{SURFACE};border:1px solid {BORDER};border-radius:16px;padding:15px 18px;'
            f'margin-bottom:10px;box-shadow:var(--fs-shadow-sm)">'
            f'<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">'
            f'<div style="width:46px;height:46px;border-radius:13px;background:{colour}26;display:flex;'
            f'align-items:center;justify-content:center;flex:none">{svg_icon(icon, 24, colour)}</div>'
            f'<div style="flex:1;min-width:180px"><div style="font-weight:750;color:{NAVY};font-size:1.02rem">'
            f'{html.escape(str(name))}</div>'
            f'<div style="color:{MUTED};font-size:.86rem">{html.escape(desc)}</div></div>'
            f'<div style="text-align:right"><div style="font-weight:780;color:{NAVY};font-size:1.1rem">'
            f'{rand(amount)}</div>'
            f'<div style="color:{MUTED};font-size:.86rem">{per_100(share)} of every R100</div></div></div>'
            f'<div style="height:9px;background:var(--fs-track);border-radius:999px;margin-top:11px;overflow:hidden">'
            f'<div style="width:{max(share * 100, 1.5):.1f}%;height:100%;background:{colour};'
            f'border-radius:999px"></div></div></div>'
        )
    return section_title("Your spending groups", "What each group means, and how much of your money went there") \
        + "".join(rows)


def top_purchases_html(result):
    top = result.nlargest(5, "amount_zar")
    biggest = top["amount_zar"].max()
    rows = []
    for _, r in top.iterrows():
        icon, colour, _desc = group_info(r["predicted_category"])
        rows.append(
            f'<div class="fs-card" style="background:{SURFACE};border:1px solid {BORDER};border-radius:16px;padding:15px 18px;'
            f'margin-bottom:10px;box-shadow:var(--fs-shadow-sm)">'
            f'<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">'
            f'<div style="width:46px;height:46px;border-radius:13px;background:{colour}26;display:flex;'
            f'align-items:center;justify-content:center;flex:none">{svg_icon(icon, 24, colour)}</div>'
            f'<div style="flex:1;min-width:170px"><div style="font-weight:750;color:{NAVY};'
            f'overflow-wrap:anywhere">{html.escape(str(r["description_sa"]))}</div>'
            f'<div style="color:{MUTED};font-size:.86rem">{r["txn_date"]:%d %b} · '
            f'{html.escape(str(r["predicted_category"]))}</div></div>'
            f'<div style="font-weight:780;color:{NAVY};font-size:1.1rem">{rand(r["amount_zar"])}</div></div>'
            f'<div style="height:9px;background:var(--fs-track);border-radius:999px;margin-top:11px;overflow:hidden">'
            f'<div style="width:{max(r["amount_zar"] / biggest * 100, 4):.1f}%;height:100%;background:{colour};'
            f'border-radius:999px"></div></div></div>'
        )
    return section_title("Your 5 biggest purchases",
                         "These single payments took the biggest bites out of your money.") + "".join(rows)


def confidence_view(result):
    conf = result["confidence"]
    n = len(result)
    high = int((conf >= HIGH_CONFIDENCE).sum())
    low = int((conf < LOW_CONFIDENCE).sum())
    medium = n - high - low
    cards = card_grid([
        card_html(dot("#22C55E") + "Very sure", str(high), f"{high / n:.0%} of your purchases", "#22C55E"),
        card_html(dot("#EAB308") + "Fairly sure", str(medium), f"{medium / n:.0%} of your purchases", "#EAB308"),
        card_html(dot("#EF4444") + "Not sure", str(low), f"{low / n:.0%} of your purchases: please check these", "#EF4444"),
        card_html("Overall", f"{conf.mean():.0%}", "How sure the app was on average", NAVY),
    ])
    least = result.sort_values("confidence").head(10)
    review = pd.DataFrame({
        "Date": least["txn_date"].dt.strftime("%d %b %Y"),
        "What you paid for": least["description_sa"],
        "Our best guess": least["predicted_category"],
        "How sure": [f"{sureness_label(v)} ({v:.0%})" for v in least["confidence"]],
        "It could also be": [f"{c} ({p:.0%})" for c, p in
                             zip(least["second_guess"], least["second_guess_prob"])],
    })
    return cards, fig_sureness_by_group(result), review


def budget_rows_html(budgets, spent):
    """One progress-bar row per group: green if within budget, red if over."""
    rows = []
    for c in CATEGORIES:
        b, s = budgets[c], spent[c]
        if b <= 0 and s <= 0:
            continue
        icon, colour, _ = group_info(c)
        if b > 0:
            diff = s - b
            over = diff > 0
            bar_col = "#EF4444" if over else "#22C55E"
            width = min(s / b * 100, 100)
            status = f"{rand(abs(diff))} over budget" if over else f"{rand(abs(diff))} left"
            status_col = "#B91C1C" if over else "#15803D"
        else:
            bar_col, width, status, status_col = "#94A3B8", 100, "No budget set", MUTED
        rows.append(
            f'<div class="fs-card" style="background:{SURFACE};border:1px solid {BORDER};border-radius:16px;padding:15px 18px;'
            f'margin-bottom:10px;box-shadow:var(--fs-shadow-sm)">'
            f'<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">'
            f'<div style="width:46px;height:46px;border-radius:13px;background:{colour}26;display:flex;'
            f'align-items:center;justify-content:center;flex:none">{svg_icon(icon, 24, colour)}</div>'
            f'<div style="flex:1;min-width:170px"><div style="font-weight:750;color:{NAVY}">{html.escape(c)}</div>'
            f'<div style="color:{MUTED};font-size:.86rem">Planned {rand(b)} · Spent {rand(s)}</div></div>'
            f'<div style="font-weight:750;color:{status_col}">{status}</div></div>'
            f'<div style="height:9px;background:var(--fs-track);border-radius:999px;margin-top:11px;overflow:hidden">'
            f'<div style="width:{width:.1f}%;height:100%;background:{bar_col};border-radius:999px"></div>'
            f'</div></div>'
        )
    return section_title("Group by group",
                         "Green means you stayed within your plan. Red means you went over.") + "".join(rows)


# ------------------------------------------------------------------
# 7. The functions Gradio calls when the user clicks the buttons
# ------------------------------------------------------------------
def budget_insights(result, user):
    """If the user has a saved plan, say how this statement compares (shown on the Summary tab)."""
    budgets = (user or {}).get("budgets") or {}
    if sum(budgets.values()) <= 0:
        return []
    spent = result.groupby("predicted_category")["amount_zar"].sum()
    over = [(c, spent.get(c, 0) - b) for c, b in budgets.items() if b > 0 and spent.get(c, 0) > b]
    if not over:
        return [("trophy", "You stayed within your saved plan in every group. Nicely done!")]
    c, amount = max(over, key=lambda x: x[1])
    return [("alert", f"Compared with your saved plan, you went over in **{len(over)} group(s)**. "
                      f"The biggest was **{c}**, by **{rand(amount)}**. "
                      "Open the 'Compare' tab for the full picture.")]


def analyse(pdf_path, user=None, progress=gr.Progress()):
    if not pdf_path:
        raise gr.Error("Please upload your bank statement (a PDF) first.")

    progress(0.1, desc="Reading your PDF")
    try:
        extracted = extract_transactions(pdf_path)
    except ExtractionError as e:
        raise gr.Error(str(e))
    except Exception as e:
        raise gr.Error(f"Sorry, I could not read this PDF: {e}")

    extracted["amount"] = pd.to_numeric(extracted["amount"], errors="coerce")
    extracted = extracted.dropna(subset=["amount"])
    money_in = extracted[extracted["amount"] > 0]
    spending = extracted[extracted["amount"] < 0]
    if spending.empty:
        raise gr.Error("I found payments in this PDF, but none of them were money going out.")

    progress(0.4, desc="Sorting your purchases into groups")
    feats = build_features(spending)
    feats = feats.dropna(subset=["txn_date"])
    result = predict_categories(feats)

    unchecked = int((~extracted["balance_ok"]).sum()) if "balance_ok" in extracted else 0
    note = (f"We read **{len(extracted)} payments** from {extracted['page'].max()} page(s) of your statement."
            + (f" {len(money_in)} of them were money coming in, so they are not counted below."
               if len(money_in) else ""))
    warning = (f"{unchecked} amount(s) could not be double-checked against your running balance, "
               "so please check them yourself.") if unchecked else None

    table = pd.DataFrame({
        "Date": result["txn_date"].dt.strftime("%d %b %Y"),
        "What you paid for": result["description_sa"],
        "Amount (R)": result["amount_zar"],
        "Spending group": result["predicted_category"],
        "How sure": [f"{sureness_label(v)} ({v:.0%})" for v in result["confidence"]],
    })
    csv_path = Path(tempfile.mkdtemp()) / "my_spending.csv"
    table.to_csv(csv_path, index=False)

    progress(0.8, desc="Building your charts")
    conf_cards, sure_chart, review = confidence_view(result)
    show_plot = lambda fig: gr.Plot(value=fig, visible=True)

    money_in_total = float(money_in["amount"].sum()) if len(money_in) else 0.0

    # What the budget tab needs later: spending per group + money that came in
    stmt_state = {
        "spent": {str(k): float(v) for k, v in
                  result.groupby("predicted_category")["amount_zar"].sum().items()},
        "money_in": money_in_total,
    }

    if user:  # keep a short history on the user's profile
        try:
            top_group = result.groupby("predicted_category")["amount_zar"].sum().idxmax()
            users.add_analysis(user["id"], float(result["amount_zar"].sum()), money_in_total,
                               top_group, len(result))
        except Exception:
            pass

    gr.Info("Your results are ready. Have a look through the tabs.")

    return (
        stat_cards(result, money_in_total),
        short_version(result, note, warning),
        insight_list(friendly_insights(result) + budget_insights(result, user)),
        gr.HTML(visible=False), show_plot(fig_money_donut(result)), group_breakdown(result),
        gr.HTML(visible=False), show_plot(fig_daily_by_group(result)), top_purchases_html(result),
        gr.HTML(visible=False), gr.Dataframe(value=table, visible=True), gr.File(value=str(csv_path), visible=True),
        conf_cards, show_plot(sure_chart), gr.Dataframe(value=review, visible=True),
        stmt_state,
    )


def compare_budget(state, user, savings_goal, *budget_values):
    if not state:
        raise gr.Error("Please upload and analyse a statement first, using the button at the top.")
    budgets = {c: float(v or 0) for c, v in zip(CATEGORIES, budget_values)}
    if sum(budgets.values()) <= 0:
        raise gr.Error("Please type in a budget for at least one group first.")

    spent = {c: float(state["spent"].get(c, 0)) for c in CATEGORIES}
    money_in, goal = float(state["money_in"]), float(savings_goal or 0)
    income_from_profile = False
    if money_in <= 0 and user and user.get("income", 0) > 0:
        money_in, income_from_profile = float(user["income"]), True
    total_budget, total_spent = sum(budgets.values()), sum(spent.values())
    overspend = sum(max(spent[c] - budgets[c], 0) for c in CATEGORIES if budgets[c] > 0)
    diff = total_spent - total_budget

    cards = [
        card_html("You planned to spend", rand(total_budget), "Added up across your groups", TEAL),
        card_html("You actually spent", rand(total_spent), "On this statement", NAVY),
        card_html("Over budget by" if diff > 0 else "Under budget by", rand(abs(diff)),
                  "You spent more than you planned" if diff > 0 else "Nice, you stayed within your plan",
                  "#EF4444" if diff > 0 else "#22C55E"),
    ]

    # Savings: real savings if we know the money that came in, otherwise what is left of the budget
    if money_in > 0:
        saved = money_in - total_spent
        if saved >= 0:
            if goal and saved >= goal:
                sub, accent = "Goal reached!", "#22C55E"
            elif goal:
                sub, accent = f"{rand(goal - saved)} short of your {rand(goal)} goal", "#EAB308"
            else:
                sub = ("Your profile income minus money you spent" if income_from_profile
                       else "Money that came in minus money you spent")
                accent = "#22C55E"
            cards.append(card_html("You saved", rand(saved), sub, accent))
        else:
            cards.append(card_html("You spent more than came in", rand(abs(saved)),
                                   "More money went out than came in", "#EF4444"))
    else:
        left = total_budget - total_spent
        cards.append(card_html("Left from your budget", rand(max(left, 0)),
                               "No money coming in was found, so this is your budget minus your spending",
                               "#22C55E"))
    cards.append(card_html("What overspending cost you", rand(overspend),
                           "This is what you could have saved by sticking to your plan", "#F97316"))

    notes = []
    for c in sorted(CATEGORIES, key=lambda c: spent[c] - budgets[c], reverse=True):
        if budgets[c] > 0 and spent[c] > budgets[c]:
            pct = (spent[c] - budgets[c]) / budgets[c]
            notes.append(("alert", f"You spent **{rand(spent[c] - budgets[c])} more** than planned on "
                                   f"**{c}** ({pct:.0%} over)."))
    unbudgeted = sum(spent[c] for c in CATEGORIES if budgets[c] <= 0)
    if unbudgeted > 0:
        notes.append(("help", f"**{rand(unbudgeted)}** went to groups you did not set a budget for."))
    if not any(n[0] == "alert" for n in notes):
        notes.insert(0, ("trophy", "You stayed within budget in every group you planned for."))

    return (
        card_grid(cards),
        gr.Plot(value=fig_budget_vs_spent(budgets, spent), visible=True),
        insight_list(notes) + budget_rows_html(budgets, spent),
        gr.Tabs(selected="compare"),
    )


# ------------------------------------------------------------------
# 8. Accounts: login, sign-up, profile, saved budget
# ------------------------------------------------------------------
def msg_html(text, ok=True):
    bg, fg, bd = (("var(--fs-ok-bg)", "var(--fs-ok-text)", "var(--fs-ok-border)") if ok
                  else ("var(--fs-err-bg)", "var(--fs-err-text)", "var(--fs-err-border)"))
    return (f'<div style="background:{bg};border:1px solid {bd};color:{fg};border-radius:12px;'
            f'padding:10px 14px;margin-top:8px;font-size:.92rem">{html.escape(text)}</div>')


def initials(name):
    parts = [p for p in re.split(r"\s+", (name or "").strip()) if p]
    return "".join(p[0] for p in parts[:2]).upper() or "?"


def avatar_html(name, size=40):
    return (f'<div style="width:{size}px;height:{size}px;border-radius:50%;'
            f'background:linear-gradient(135deg,{NAVY},{TEAL_DARK});color:#fff;display:flex;'
            f'align-items:center;justify-content:center;font-weight:750;font-size:{size * 0.4:.0f}px;'
            f'flex:none">{html.escape(initials(name))}</div>')


def welcome_html(user):
    """The top bar of the app: logo on the left, who is logged in on the right."""
    logo = (f'<img src="{LOGO_SRC}" alt="FINSIGHT" style="height:40px">' if LOGO_SRC else
            f'<span style="font-weight:800;letter-spacing:.06em;color:{NAVY};font-size:1.2rem">FIN'
            f'<span style="color:{TEAL}">SIGHT</span></span>')
    return (f'<div style="display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;'
            f'background:color-mix(in srgb, {SURFACE} 92%, transparent);backdrop-filter:blur(8px);'
            f'border:1px solid {BORDER};'
            f'border-radius:16px;padding:10px 18px;box-shadow:var(--fs-shadow-md)">{logo}'
            f'<div style="display:flex;align-items:center;gap:10px">{avatar_html(user["display_name"])}'
            f'<div style="line-height:1.25"><div style="font-weight:750;color:{NAVY}">'
            f'{html.escape(user["display_name"])}</div>'
            f'<div style="color:{MUTED};font-size:.8rem">@{html.escape(user["username"])}</div></div></div></div>')


def profile_info_html(user):
    n = users.count_analyses(user["id"])
    header = (f'<div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;background:{SURFACE};'
              f'border:1px solid {BORDER};border-radius:18px;padding:18px 22px;margin-bottom:12px;'
              f'box-shadow:var(--fs-shadow-md)">{avatar_html(user["display_name"], 64)}'
              f'<div><div style="font-weight:780;font-size:1.25rem;color:{NAVY}">'
              f'{html.escape(user["display_name"])}</div>'
              f'<div style="color:{MUTED}">@{html.escape(user["username"])} · member since '
              f'{html.escape(str(user["created_at"]))}</div></div></div>')
    return header + card_grid([
        card_html("Statements analysed", str(n), "Counted since you joined", "#22C55E"),
        card_html("Monthly income", rand(user["income"], 0) if user["income"] else "Not set",
                  "Used for your savings figure when a statement shows no income", TEAL),
        card_html("Saved plan", "Yes" if sum(user["budgets"].values()) > 0 else "Not yet",
                  "Set it on the Budget tab", NAVY),
    ])


def history_df(user_id):
    rows = users.list_analyses(user_id)
    return pd.DataFrame({
        "Date": [r["created_at"] for r in rows],
        "Purchases": [r["n_purchases"] for r in rows],
        "Spent (R)": [round(r["total_spent"], 2) for r in rows],
        "Money in (R)": [round(r["money_in"], 2) for r in rows],
        "Biggest group": [r["top_group"] for r in rows],
    })


def _session_values(user):
    """Values for session_outputs when someone logs in (same order as that list)."""
    b = user["budgets"]
    return (user, gr.Column(visible=False), gr.Column(visible=True), welcome_html(user),
            user["display_name"], user["email"], user["income"], profile_info_html(user),
            history_df(user["id"]),
            *[float(b.get(c, 0)) for c in CATEGORIES], user["savings_goal"])


def _logout_values():
    return (None, gr.Column(visible=True), gr.Column(visible=False), "",
            "", "", 0, "", history_df(-1),
            *[0] * len(CATEGORIES), 0)


def reset_results():
    """Puts every results tab back to its empty starting state (same order as reset_outputs)."""
    return (None, gr.Tabs(selected="summary"),
            EMPTY_STATE_HTML, "", "",
            gr.HTML(value=HINT_MONEY, visible=True), gr.Plot(value=None, visible=False), "",
            gr.HTML(value=HINT_DAY, visible=True), gr.Plot(value=None, visible=False), "",
            gr.HTML(value=HINT_TX, visible=True), gr.Dataframe(value=None, visible=False),
            gr.File(value=None, visible=False),
            HINT_CONF, gr.Plot(value=None, visible=False), gr.Dataframe(value=None, visible=False),
            None,
            HINT_CMP, gr.Plot(value=None, visible=False), "")


def _skip_session():
    return (gr.skip(),) * len(session_outputs)


def _skip_reset():
    return (gr.skip(),) * len(reset_outputs)


def do_login(username, password):
    user, err = users.verify_user(username, password)
    if err:
        return (*_skip_session(), msg_html(err, ok=False))
    return (*_session_values(user), "")


def do_signup(display_name, username, password, password2):
    if password != password2:
        return (*_skip_session(), msg_html("The two passwords do not match.", ok=False))
    user, err = users.create_user(username, password, display_name)
    if err:
        return (*_skip_session(), msg_html(err, ok=False))
    return (*_session_values(user), "")


def do_logout():
    return (*_logout_values(), *reset_results())


def save_profile(user, name, email, income):
    if not user:
        raise gr.Error("Please log in first.")
    updated, err = users.update_profile(user["id"], name, email, income)
    if err:
        return gr.skip(), gr.skip(), gr.skip(), msg_html(err, ok=False)
    return updated, welcome_html(updated), profile_info_html(updated), msg_html("Your profile was saved.")


def do_change_password(user, old, new, new2):
    if not user:
        raise gr.Error("Please log in first.")
    if new != new2:
        return msg_html("The two new passwords do not match.", ok=False), gr.skip(), gr.skip(), gr.skip()
    err = users.change_password(user["id"], old, new)
    if err:
        return msg_html(err, ok=False), gr.skip(), gr.skip(), gr.skip()
    return msg_html("Your password was changed."), "", "", ""


def do_delete(user, password):
    if not user:
        raise gr.Error("Please log in first.")
    if not users.delete_user(user["id"], password):
        return (*_skip_session(), *_skip_reset(), msg_html("That password is not correct.", ok=False), gr.skip())
    return (*_logout_values(), *reset_results(), "", msg_html("Your account and history were deleted."))


def save_my_budget(user, goal, *budget_values):
    if not user:
        raise gr.Error("Please log in first.")
    budgets = {c: float(v or 0) for c, v in zip(CATEGORIES, budget_values)}
    users.save_budget(user["id"], budgets, float(goal or 0))
    gr.Info("Your plan was saved.")
    return msg_html("Saved. We will fill in this plan for you every time you log in."), users.get_user(user["id"])


def suggest_plan(state):
    """A gentle starting point: last statement's spending minus 10%, rounded to the nearest R10."""
    if not state:
        raise gr.Error("Please analyse a statement first, then we can suggest a plan.")
    gr.Info("We took 10% off what you spent in each group. Change any number you like.")
    return [round(state["spent"].get(c, 0) * 0.9 / 10) * 10 for c in CATEGORIES]


def refresh_history(user):
    return history_df(user["id"] if user else -1)


# ------------------------------------------------------------------
# 9. The interface
# ------------------------------------------------------------------
# Gradio 6 takes theme/css in launch(); Gradio 5 takes them in Blocks().
_STYLE_IN_LAUNCH = "css" in inspect.signature(gr.Blocks.launch).parameters
_blocks_kwargs = {"title": "FINSIGHT: Clarity in spending"}
if not _STYLE_IN_LAUNCH:
    _blocks_kwargs.update(theme=THEME, css=CSS)

FEATURE_CHIPS_HTML = (
    '<div style="display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin:14px 0 8px">'
    + "".join(
        f'<span style="display:inline-flex;align-items:center;gap:7px;background:{SURFACE};'
        f'border:1px solid {BORDER};color:{NAVY};border-radius:999px;padding:7px 14px;'
        f'font-size:.84rem;font-weight:620;box-shadow:var(--fs-shadow-sm)">'
        f'{svg_icon(icon, 14, TEAL_DARK)}{t}</span>'
        for icon, t in [("zap", "Plain-English insights"), ("wallet", "Your own budget"),
                        ("coins", "Track your savings")])
    + "</div>"
)

FOOTER_HTML = f"""
<div style="margin:30px 2px 8px;padding-top:18px;border-top:1px solid {BORDER};
            display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;
            color:{MUTED};font-size:.84rem">
  <div>
    <strong style="color:{NAVY}">FINSIGHT</strong> &mdash; bank statement categorisation with XGBoost.
    Built as a data-science portfolio project by {html.escape(AUTHOR_NAME)}.
  </div>
  <div style="display:flex;gap:14px">
    <a href="{AUTHOR_GITHUB}" target="_blank" rel="noopener"
       style="color:{TEAL_DARK};text-decoration:none;font-weight:650">GitHub</a>
    <a href="{AUTHOR_LINKEDIN}" target="_blank" rel="noopener"
       style="color:{TEAL_DARK};text-decoration:none;font-weight:650">LinkedIn</a>
  </div>
</div>
"""

with gr.Blocks(**_blocks_kwargs) as demo:
    user_state = gr.State(None)   # who is logged in (None = nobody)
    stmt_state = gr.State(None)   # spending totals of the statement that was analysed

    # ---------------- Welcome / login screen ----------------
    with gr.Column(visible=True, elem_id="login-card") as login_view:
        gr.HTML(HERO_HTML, padding=False)
        gr.HTML(FEATURE_CHIPS_HTML, padding=False)
        with gr.Tabs():
            with gr.Tab("Log in"):
                login_user = gr.Textbox(label="Username", max_lines=1)
                login_pass = gr.Textbox(label="Password", type="password", max_lines=1)
                login_btn = gr.Button("Log in", variant="primary")
                login_msg = gr.HTML(padding=False)
                gr.HTML(f'<div style="color:{MUTED};font-size:.88rem;text-align:center">'
                        'New here? Open the <strong>Create account</strong> tab. It takes 30 seconds.</div>',
                        padding=False)
            with gr.Tab("Create account"):
                su_name = gr.Textbox(label="Your name", max_lines=1)
                su_user = gr.Textbox(label="Choose a username", max_lines=1,
                                     info="3 to 30 letters, numbers, dots, dashes or underscores")
                su_pass = gr.Textbox(label="Choose a password", type="password", max_lines=1,
                                     info=f"At least {users.MIN_PASSWORD} characters")
                su_pass2 = gr.Textbox(label="Type the password again", type="password", max_lines=1)
                su_btn = gr.Button("Create my account", variant="primary")
                signup_msg = gr.HTML(padding=False)
        gr.HTML(f'<div style="color:{MUTED};font-size:.8rem;text-align:center;margin-top:10px;line-height:1.5">'
                'This is a demo. Please do not reuse a password from your bank or email.</div>', padding=False)
        with gr.Row():
            gr.HTML("", padding=False)
            theme_btn_login = gr.Button(DARK_LABEL, elem_id="theme-btn", scale=0, min_width=130)
            gr.HTML("", padding=False)
        gr.HTML(FOOTER_HTML, padding=False)

    # ---------------- The app (only visible after logging in) ----------------
    with gr.Column(visible=False) as app_view:
        with gr.Row(equal_height=True, elem_id="top-bar"):
            with gr.Column(scale=6):
                welcome_out = gr.HTML(padding=False)
            with gr.Column(scale=1, min_width=112):
                theme_btn = gr.Button(DARK_LABEL, elem_id="theme-btn")
            with gr.Column(scale=1, min_width=112):
                logout_btn = gr.Button("Log out", elem_id="logout-btn")

        with gr.Row(equal_height=True):
            pdf_in = gr.File(label="1. Upload your bank statement (PDF)", file_types=[".pdf"], type="filepath",
                             height=130, scale=4, elem_id="upload-card")
            run_btn = gr.Button("2. Show me where my money went", variant="primary", scale=1, min_width=210,
                                elem_id="run-btn")

        with gr.Tabs() as tabs:
            with gr.Tab("Home", id="summary"):
                cards_out = gr.HTML(EMPTY_STATE_HTML, padding=False)
                short_out = gr.HTML(padding=False)
                insights_out = gr.HTML(padding=False)
                with gr.Accordion("New to this? Words explained", open=False):
                    gr.HTML(GLOSSARY_HTML, padding=False)

            with gr.Tab("Where it went"):
                money_hint = gr.HTML(HINT_MONEY, padding=False)
                donut_out = gr.Plot(show_label=False, visible=False)
                breakdown_out = gr.HTML(padding=False)

            with gr.Tab("Day by day"):
                day_hint = gr.HTML(HINT_DAY, padding=False)
                daily_out = gr.Plot(show_label=False, visible=False)
                top5_out = gr.HTML(padding=False)

            with gr.Tab("Purchases"):
                tx_hint = gr.HTML(HINT_TX, padding=False)
                table_out = gr.Dataframe(label="Every purchase, and the group we put it in", wrap=True,
                                         interactive=False, visible=False)
                csv_out = gr.File(label="Download your results (CSV file)", visible=False)

            with gr.Tab("How sure?"):
                conf_cards_out = gr.HTML(HINT_CONF, padding=False)
                sure_out = gr.Plot(show_label=False, visible=False)
                review_out = gr.Dataframe(label="The 10 purchases the app was least sure about", wrap=True,
                                          interactive=False, visible=False)
                gr.HTML(CONFIDENCE_EXPLAINER_HTML, padding=False)

            with gr.Tab("Budget", id="budget"):
                gr.HTML(BUDGET_INTRO_HTML, padding=False)
                budget_inputs = []
                for i in range(0, len(CATEGORIES), 3):
                    with gr.Row():
                        for c in CATEGORIES[i:i + 3]:
                            budget_inputs.append(
                                gr.Number(label=f"{c.title()} (R)", value=0, minimum=0, precision=2))
                savings_in = gr.Number(label="How much would you like to save? (R)", value=0, minimum=0,
                                       precision=2)
                with gr.Row():
                    suggest_btn = gr.Button("Not sure where to start? Suggest a plan", scale=2)
                    save_budget_btn = gr.Button("Save as my plan", scale=1, min_width=150)
                compare_btn = gr.Button("3. Compare my plan with what I spent", variant="primary",
                                        elem_id="compare-btn")
                budget_msg = gr.HTML(padding=False)

            with gr.Tab("Compare", id="compare"):
                cmp_cards = gr.HTML(HINT_CMP, padding=False)
                cmp_plot = gr.Plot(show_label=False, visible=False)
                cmp_detail = gr.HTML(padding=False)

            with gr.Tab("Profile", id="profile"):
                pf_info = gr.HTML(padding=False)
                gr.HTML(section_title("About you", "This is what we use to personalise the app."), padding=False)
                pf_name = gr.Textbox(label="Your name (shown in the app)", max_lines=1)
                pf_email = gr.Textbox(label="Email (optional)", max_lines=1)
                pf_income = gr.Number(label="Your monthly take-home income (R) (optional)", value=0, minimum=0,
                                      precision=2,
                                      info="We only use this for the savings figure when your statement "
                                           "does not show any money coming in.")
                pf_save_btn = gr.Button("Save my profile", variant="primary")
                pf_msg = gr.HTML(padding=False)

                gr.HTML(section_title("Your past analyses", "The last 10 statements you looked at."),
                        padding=False)
                history_out = gr.Dataframe(value=history_df(-1), interactive=False, wrap=True)

                with gr.Accordion("Change my password", open=False):
                    pw_old = gr.Textbox(label="Current password", type="password", max_lines=1)
                    pw_new = gr.Textbox(label="New password", type="password", max_lines=1)
                    pw_new2 = gr.Textbox(label="Type the new password again", type="password", max_lines=1)
                    pw_btn = gr.Button("Change my password")
                    pw_msg = gr.HTML(padding=False)

                with gr.Accordion("Delete my account", open=False):
                    gr.HTML(f'<div style="color:{MUTED};font-size:.9rem;margin-bottom:6px">This removes your '
                            'profile, saved plan and history for good. Type your password to confirm.</div>',
                            padding=False)
                    del_pass = gr.Textbox(label="Your password", type="password", max_lines=1)
                    del_btn = gr.Button("Delete my account for good", variant="stop")
                    del_msg = gr.HTML(padding=False)

        example_files = sorted(str(p) for p in (HERE / "examples").glob("*.pdf"))
        if example_files:
            gr.Examples(examples=[[p] for p in example_files], inputs=pdf_in,
                        label="No statement handy? Try a sample one")

        gr.HTML(FOOTER_HTML, padding=False)

    # ---------------- Wiring ----------------
    outputs = [cards_out, short_out, insights_out,
               money_hint, donut_out, breakdown_out,
               day_hint, daily_out, top5_out,
               tx_hint, table_out, csv_out,
               conf_cards_out, sure_out, review_out,
               stmt_state]
    compare_outputs = [cmp_cards, cmp_plot, cmp_detail]

    # everything that changes when someone logs in or out
    session_outputs = [user_state, login_view, app_view, welcome_out,
                       pf_name, pf_email, pf_income, pf_info, history_out,
                       *budget_inputs, savings_in]
    # everything that must be wiped so the next person never sees old results
    reset_outputs = [pdf_in, tabs, *outputs, *compare_outputs]

    login_btn.click(do_login, [login_user, login_pass], [*session_outputs, login_msg]) \
        .then(lambda: ("", "", ""), None, [login_pass, su_pass, su_pass2])
    login_pass.submit(do_login, [login_user, login_pass], [*session_outputs, login_msg]) \
        .then(lambda: ("", "", ""), None, [login_pass, su_pass, su_pass2])
    su_btn.click(do_signup, [su_name, su_user, su_pass, su_pass2], [*session_outputs, signup_msg]) \
        .then(lambda: ("", "", ""), None, [login_pass, su_pass, su_pass2])
    logout_btn.click(do_logout, None, [*session_outputs, *reset_outputs])

    run_btn.click(analyse, [pdf_in, user_state], outputs) \
        .then(refresh_history, user_state, history_out)
    compare_btn.click(compare_budget, [stmt_state, user_state, savings_in, *budget_inputs],
                      [*compare_outputs, tabs])
    suggest_btn.click(suggest_plan, stmt_state, budget_inputs)
    save_budget_btn.click(save_my_budget, [user_state, savings_in, *budget_inputs], [budget_msg, user_state])

    pf_save_btn.click(save_profile, [user_state, pf_name, pf_email, pf_income],
                      [user_state, welcome_out, pf_info, pf_msg])
    pw_btn.click(do_change_password, [user_state, pw_old, pw_new, pw_new2],
                 [pw_msg, pw_old, pw_new, pw_new2])
    del_btn.click(do_delete, [user_state, del_pass], [*session_outputs, *reset_outputs, del_msg, login_msg])

    theme_targets = [theme_btn, theme_btn_login]
    theme_btn.click(None, None, theme_targets, js=TOGGLE_THEME_JS)
    theme_btn_login.click(None, None, theme_targets, js=TOGGLE_THEME_JS)

    demo.load(None, None, theme_targets, js=RESTORE_THEME_JS)

if __name__ == "__main__":
    _launch_kwargs = {"theme": THEME, "css": CSS} if _STYLE_IN_LAUNCH else {}
    if FAVICON.exists():
        _launch_kwargs["favicon_path"] = str(FAVICON)

    demo.launch(share=True, **_launch_kwargs)