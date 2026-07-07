"""
Shared chart configuration, dark layout, and responsive column utilities.
"""

import streamlit as st
from typing import Optional, Dict, Any

DARK_BG = "#0E1117"
DARK_CARD = "#1C2833"
DARK_GRID = "#2C3E50"

CHART_COLORS = {
    "blue": "#3498DB",
    "green": "#27AE60",
    "red": "#E74C3C",
    "orange": "#F39C12",
    "purple": "#8E44AD",
    "teal": "#1ABC9C",
}

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "modeBarButtonsToAdd": ["drawline", "eraseshape"],
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "displaylogo": False,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "kenya_economic_pulse",
        "height": 600,
        "width": 900,
    },
}


def dark_layout(
    title: str = "",
    height: int = 400,
    xtitle: str = "",
    ytitle: str = "",
    use_legend: bool = True,
    margin: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    layout = {
        "title": (
            {"text": title, "font": {"color": "white", "size": 16}}
            if title else None
        ),
        "plot_bgcolor": DARK_BG,
        "paper_bgcolor": DARK_BG,
        "font": {"color": "white"},
        "height": height,
        "hovermode": "x unified",
        "xaxis": {"gridcolor": DARK_GRID, "title": xtitle or ""},
        "yaxis": {"gridcolor": DARK_GRID, "title": ytitle or ""},
        "margin": margin or {"l": 40, "r": 20, "t": 50, "b": 40},
    }
    if use_legend:
        layout["legend"] = {
            "bgcolor": DARK_CARD,
            "font": {"color": "white"},
        }
    return {k: v for k, v in layout.items() if v is not None}


def responsive_columns(count: int, min_cols: int = 2) -> list:
    cols = min(count, 6)
    if cols < min_cols:
        cols = min_cols
    return st.columns(cols)


def kpi_card(col, label: str, value: str, color: str, subtitle: str = ""):
    col.markdown(
        f"""
    <div style='background:{DARK_CARD}; padding:1rem; border-radius:10px;
                border-top:3px solid {color}; text-align:center; min-height:90px;'>
        <div style='color:#AAB7B8; font-size:.75rem; margin-bottom:.3rem'>{label}</div>
        <div style='color:{color}; font-size:1.4rem; font-weight:bold'>{value}</div>
        {f"<div style='color:#AAB7B8; font-size:.7rem'>{subtitle}</div>" if subtitle else ""}
    </div>
    """,
        unsafe_allow_html=True,
    )
