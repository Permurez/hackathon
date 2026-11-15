"""style.py - the app's look for non-technical viewers: the slides' dark theme with the vessel background picture,
cards for numbers, dark charts, a plain-language intro block and certainty badges (measurement / vs group / hint)."""
from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

import vascular as V

BG = Path(__file__).resolve().parent / "static" / "bg.jpg"

CSS = """
.stApp { background: linear-gradient(90deg, rgba(7, 12, 32, 0.55) 0%, rgba(7, 12, 32, 0.30) 60%, rgba(7, 12, 32, 0.15) 100%),
                     __BG__; background-size: cover; background-position: center; background-attachment: fixed; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: rgba(9, 16, 40, 0.78); backdrop-filter: blur(6px);
                            border-right: 1px solid rgba(201, 211, 227, 0.12); }
.stMarkdown p, .stMarkdown li, [data-testid="stCaptionContainer"], h1, h2, h3 { text-shadow: 0 1px 4px rgba(0, 0, 0, 0.75); }
[data-testid="stImage"] img { background: rgba(9, 16, 40, 0.80); padding: 10px; border-radius: 14px;
                              backdrop-filter: blur(4px); }
[data-testid="stTabs"], [data-testid="stDataFrame"] { background: rgba(9, 16, 40, 0.55); border-radius: 14px; padding: 6px 10px; }
[data-testid="stAppDeployButton"], #MainMenu, footer { display: none !important; }
h1 { font-weight: 700 !important; letter-spacing: -0.02em; }
[data-testid="stMetric"] { background: rgba(12, 21, 48, 0.82); border: 1px solid rgba(201, 211, 227, 0.16);
                           border-radius: 14px; padding: 14px 18px; backdrop-filter: blur(4px); }
[data-testid="stMetricValue"] { font-size: 1.75rem; }
[data-testid="stImage"] img { border-radius: 12px; }
[data-testid="stExpander"] details { background: rgba(16, 27, 58, 0.6); border-radius: 12px; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { background: rgba(16, 27, 58, 0.6); border-radius: 10px 10px 0 0; padding: 6px 16px; }
.hero { padding: 22px 26px; border-radius: 18px; margin: 4px 0 18px 0; backdrop-filter: blur(6px);
        background: linear-gradient(135deg, rgba(60, 20, 50, 0.78), rgba(12, 24, 60, 0.78));
        border: 1px solid rgba(201, 211, 227, 0.16); }
.hero .lead { margin: 0; color: #E8ECF3; font-size: 1.12rem; line-height: 1.5; }
.steps { display: flex; gap: 12px; margin-top: 14px; flex-wrap: wrap; }
.step { flex: 1; min-width: 190px; background: rgba(11, 21, 48, 0.72); border-radius: 12px; padding: 10px 14px;
        border: 1px solid rgba(201, 211, 227, 0.12); color: #C9D3E3; font-size: 0.95rem; }
.step b { color: #FF6B86; margin-right: 6px; }
.badge { display: inline-block; padding: 1px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
         margin-right: 6px; }
.badge.pomiar { background: rgba(110, 168, 255, 0.20); color: #A9CBFF; }
.badge.grupa { background: rgba(178, 140, 255, 0.20); color: #D3BEFF; }
.badge.wskazowka { background: rgba(255, 107, 134, 0.20); color: #FFA6B8; }
.legend { color: #B6C2D6; font-size: 0.9rem; margin: -6px 0 10px 0; }
"""

BADGES = {"pomiar": "measured", "grupa": "vs group", "wskazowka": "hint"}


@lru_cache(maxsize=1)
def _bg() -> str:
    if not BG.exists():
        return "none"
    return f'url("data:image/jpeg;base64,{base64.b64encode(BG.read_bytes()).decode()}")'


def apply() -> None:
    """Theme the page and the charts; call once per run, before any page draws."""
    V.INK, V.MUTED, V.GREY, V.BLUE = "#F3F5FA", "#9AA8C0", "#8D9AB3", "#6EA8FF"
    plt.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none", "savefig.transparent": True,
        "text.color": "#E8ECF3", "axes.labelcolor": "#C9D3E3", "axes.titlecolor": "#F3F5FA", "axes.edgecolor": "#5A6A8A",
        "xtick.color": "#9AA8C0", "ytick.color": "#9AA8C0", "legend.facecolor": "#13204A", "legend.edgecolor": "#5A6A8A",
        "legend.labelcolor": "#E8ECF3", "boxplot.boxprops.color": "#C9D3E3", "boxplot.whiskerprops.color": "#C9D3E3",
        "boxplot.capprops.color": "#C9D3E3"})
    st.markdown(f"<style>{CSS.replace('__BG__', _bg())}</style>", unsafe_allow_html=True)


def hero(lead: str, steps: list[str]) -> None:
    """Plain-language block at the top of a page: what it is, then the steps."""
    items = "".join(f'<div class="step"><b>{i}</b>{s}</div>' for i, s in enumerate(steps, 1))
    st.markdown(f'<div class="hero"><p class="lead">{lead}</p><div class="steps">{items}</div></div>',
                unsafe_allow_html=True)


def badge(kind: str) -> str:
    return f'<span class="badge {kind}">{BADGES[kind]}</span>'


def legend() -> None:
    st.markdown(f'<p class="legend">How to read: {badge("pomiar")} measured directly · {badge("grupa")} compared '
                f'with healthy people aged 18–42 · {badge("wskazowka")} a likelihood, not a certainty</p>',
                unsafe_allow_html=True)
