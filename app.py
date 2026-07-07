"""
Kenya Economic Pulse — Main Streamlit Application
10-page interactive data science dashboard for Kenya economic analysis.
Author: Stephen Muema | Data Scientist & ML Engineer
Portfolio: https://muemastephenportfolio.netlify.app/
GitHub:    https://github.com/Kaks753/KE_economic_pulse
Version:   2.3.0  |  April 2026
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from utils.chart_config import PLOTLY_CONFIG, responsive_columns, kpi_card, dark_layout

# ── Page config (MUST be first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="Kenya Economic Pulse | Stephen Muema",
    page_icon="\U0001f1f0\U0001f1ea",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":    "https://github.com/Kaks753/KE_economic_pulse",
        "Report a bug": "mailto:stephenmuema@proton.me",
        "About":       "Kenya Economic Pulse v2.3 — Built by Stephen Muema, Data Scientist",
    }
)

# ── Theme toggle ─────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

theme = st.session_state["theme"]
is_dark = theme == "dark"

BG_COLOR = "#0E1117" if is_dark else "#F5F5F5"
CARD_BG = "#1C2833" if is_dark else "#FFFFFF"
TEXT_COLOR = "#ECF0F1" if is_dark else "#1C2833"
MUTED_TEXT = "#AAB7B8" if is_dark else "#566573"
GRID_COLOR = "#2C3E50" if is_dark else "#D5D8DC"
SIDEBAR_BG = "linear-gradient(180deg, #0D1B2A 0%, #1B2838 100%)" if is_dark else "linear-gradient(180deg, #EBF5FB 0%, #D6EAF8 100%)"
BORDER_COLOR = "#2C3E50" if is_dark else "#D5D8DC"

# ── Global CSS styles ────────────────────────────────────────────────
st.markdown(f"""
<style>
    :root {{
        --bg: {BG_COLOR};
        --card: {CARD_BG};
        --text: {TEXT_COLOR};
        --muted: {MUTED_TEXT};
        --grid: {GRID_COLOR};
        --sidebar-bg: {SIDEBAR_BG};
        --border: {BORDER_COLOR};
    }}
    .stApp {{ background-color: var(--bg); }}

    [data-testid="stSidebar"] {{
        background: var(--sidebar-bg);
        border-right: 1px solid var(--border);
    }}
    [data-testid="stSidebar"] * {{ color: {"#ECF0F1" if is_dark else "#1C2833"} !important; }}

    [data-testid="stMetric"] {{
        background: var(--card);
        border-radius: 10px;
        padding: 12px 16px;
        border: 1px solid var(--border);
    }}
    [data-testid="stMetricLabel"] {{ color: var(--muted) !important; font-size: .85rem; }}
    [data-testid="stMetricValue"] {{ color: var(--text) !important; }}

    .streamlit-expanderHeader {{ color: #AED6F1 !important; }}

    .stTabs [data-baseweb="tab-list"]  {{ background: var(--card); border-radius: 10px; padding: 4px; }}
    .stTabs [data-baseweb="tab"]       {{ color: var(--muted) !important; border-radius: 8px; padding: 8px 16px; }}
    .stTabs [aria-selected="true"]     {{ background: #2980B9 !important; color: white !important; }}

    .stDownloadButton button {{
        background: #2980B9 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
    }}

    ::-webkit-scrollbar        {{ width: 6px; }}
    ::-webkit-scrollbar-track  {{ background: var(--bg); }}
    ::-webkit-scrollbar-thumb  {{ background: #2C3E50; border-radius: 3px; }}

    [data-testid="stSidebarNav"] {{ background: transparent; }}
    [data-testid="stSidebarNav"]::before {{
        content: "Kenya Economic Pulse";
        display: block;
        font-size: 1.5rem;
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }}

    @media (max-width: 768px) {{
        [data-testid="column"] {{ min-width: 100% !important; }}
    }}
</style>
""", unsafe_allow_html=True)

# ── Sidebar theme toggle ────────────────────────────────────────────
with st.sidebar:
    st.markdown("### \U0001f3a8 Appearance")
    light_mode = st.toggle("Light Mode", value=not is_dark, key="theme_toggle")
    new_theme = "light" if light_mode else "dark"
    if new_theme != st.session_state["theme"]:
        st.session_state["theme"] = new_theme
        st.rerun()

# ── Page imports (after set_page_config) ────────────────────────────
from utils.data_fetcher import get_all_data
from pages import (
    economic_indicators,
    county_inequality,
    mobile_money,
    youth_unemployment,
    forecaster,
    county_comparison,
    policy_simulator,
    anomaly_detection,
    nlq_engine,
    developer,
)

# ── Data loading with caching ────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    return get_all_data()


# ── Overview page render function ────────────────────────────────────
def render_overview():
    data = load_data()
    macro = data["macro"]
    mm = data["mobile_money"]
    yu = data["youth_unemp"]
    county = data["county"]

    # Hero banner — compact
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1B2631 0%, #154360 50%, #0E6655 100%);
                padding: 1.2rem 1.8rem; border-radius: 14px; margin-bottom: 1.2rem;
                display:flex; align-items:center; gap:1rem; flex-wrap:wrap;'>
        <div style='font-size:2.4rem; line-height:1'>🇰🇪</div>
        <div style='flex:1; min-width:200px;'>
            <h1 style='color:white; font-size:1.6rem; margin:0; font-weight:700;'>Kenya Economic Pulse</h1>
            <p style='color:#AED6F1; font-size:.85rem; margin:.2rem 0 .5rem;'>
                Real-time economic intelligence — World Bank · KNBS · CBK · Machine Learning
            </p>
            <div style='display:flex; gap:.5rem; flex-wrap:wrap;'>
                <span style='background:rgba(255,255,255,.12); color:white; padding:.25rem .75rem;
                             border-radius:20px; font-size:.75rem;'>📊 10 Dashboards</span>
                <span style='background:rgba(255,255,255,.12); color:white; padding:.25rem .75rem;
                             border-radius:20px; font-size:.75rem;'>🤖 7 ML Models</span>
                <span style='background:rgba(255,255,255,.12); color:white; padding:.25rem .75rem;
                             border-radius:20px; font-size:.75rem;'>🗺️ 47 Counties</span>
                <span style='background:rgba(255,255,255,.12); color:white; padding:.25rem .75rem;
                             border-radius:20px; font-size:.75rem;'>💬 NLQ Engine</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI strip — responsive (wraps to 3 cols on mobile)
    st.markdown("### 🌍 Kenya at a Glance")
    last_mm = mm.iloc[-1]
    last_yu = yu.iloc[-1]
    gdp_val = macro["GDP Growth (%)"].dropna().iloc[-1] if "GDP Growth (%)" in macro.columns else 4.8
    inf_val = macro["Inflation Rate (%)"].dropna().iloc[-1] if "Inflation Rate (%)" in macro.columns else 7.8

    kpi_data = [
        ("🇰🇪 Population",   "56.4M",                                   "#3498DB"),
        ("📈 GDP Growth",     f"{gdp_val:.1f}%",                         "#27AE60"),
        ("💸 Inflation",      f"{inf_val:.1f}%",                         "#E74C3C"),
        ("📱 M-Pesa Users",   f"{last_mm['MPesa_Users_M']:.0f}M",        "#8E44AD"),
        ("🎓 Youth Unemp.",   f"{last_yu['Youth_Unemployment_Pct']:.1f}%","#F39C12"),
        ("🏚️ Poverty Rate",  f"{last_mm['Poverty_Rate_National']:.1f}%", "#E67E22"),
    ]
    kpi_cols = responsive_columns(len(kpi_data))
    for i, (lbl, val, color) in enumerate(kpi_data):
        kpi_card(kpi_cols[i], lbl, val, color)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick charts: 2-column layout
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 📈 GDP Growth vs Poverty (2007–2023)")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if not macro.empty and "GDP Growth (%)" in macro.columns:
            mac_trim = macro[macro["Year"] >= 2007].copy()
            gdp_vals = mac_trim["GDP Growth (%)"].fillna(0)
            fig.add_trace(go.Bar(
                x=mac_trim["Year"], y=gdp_vals,
                name="GDP Growth (%)",
                marker_color=["#27AE60" if v >= 0 else "#E74C3C" for v in gdp_vals],
                opacity=0.75
            ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=mm["Year"], y=mm["Poverty_Rate_National"],
            name="Poverty Rate (%)", mode="lines+markers",
            line=dict(color="#E74C3C", width=2.5), marker=dict(size=6)
        ), secondary_y=True)
        fig.update_layout(
            **dark_layout(height=340, margin={"l": 10, "r": 10, "t": 30, "b": 20})
        )
        fig.update_layout(
            legend=dict(bgcolor="#1C2833", font=dict(color="white"), x=0, y=1.1, orientation="h"),
        )
        fig.update_yaxes(gridcolor="#2C3E50", secondary_y=False, title_text="GDP Growth (%)")
        fig.update_yaxes(gridcolor="#2C3E50", secondary_y=True, title_text="Poverty (%)")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with col_right:
        st.markdown("#### 📱 Mobile Financial Inclusion Rise")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=mm["Year"], y=mm["Financial_Inclusion_Pct"],
            fill="tozeroy",
            fillcolor="rgba(52,152,219,0.2)",
            line=dict(color="#3498DB", width=2.5),
            name="Financial Inclusion (%)"
        ))
        fig2.add_trace(go.Scatter(
            x=mm["Year"], y=mm["MPesa_Users_M"] * 2,
            line=dict(color="#27AE60", width=2, dash="dot"),
            name="M-Pesa Users (×2 scaled)"
        ))
        fig2.update_layout(
            **dark_layout(height=340, margin={"l": 10, "r": 10, "t": 30, "b": 20}, use_legend=False),
            legend=dict(bgcolor="#1C2833", font=dict(color="white"), x=0, y=1.1, orientation="h"),
        )
        st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)

    # County snapshot
    st.markdown("#### 🗺️ County Poverty Snapshot — Top 10 vs Bottom 10")
    top_poor = county.nlargest(10, "Poverty_Rate")[["County", "Poverty_Rate", "Region"]]
    top_rich = county.nsmallest(10, "Poverty_Rate")[["County", "Poverty_Rate", "Region"]]
    snap_df = pd.concat([
        top_poor.assign(Type="Highest Poverty"),
        top_rich.assign(Type="Lowest Poverty")
    ])
    fig_snap = px.bar(
        snap_df, x="County", y="Poverty_Rate", color="Type",
        color_discrete_map={"Highest Poverty": "#E74C3C", "Lowest Poverty": "#27AE60"},
        barmode="group",
        labels={"Poverty_Rate": "Poverty Rate (%)"},
    )
    fig_snap.update_layout(
        **dark_layout(height=360, margin={"l": 10, "r": 10, "t": 20, "b": 10}, use_legend=False),
        xaxis=dict(gridcolor="#2C3E50", tickangle=-30),
        legend=dict(bgcolor="#1C2833", font=dict(color="white")),
    )
    st.plotly_chart(fig_snap, use_container_width=True, config=PLOTLY_CONFIG)

    # Youth unemployment trend
    st.markdown("#### 🎓 Youth Unemployment Trend (2005–2023)")
    fig_yu = go.Figure()
    fig_yu.add_trace(go.Scatter(
        x=yu["Year"], y=yu["Youth_Unemployment_Pct"],
        fill="tozeroy",
        fillcolor="rgba(142,68,173,0.2)",
        line=dict(color="#8E44AD", width=2.5),
        mode="lines+markers",
        name="Youth Unemployment (%)",
        marker=dict(size=7)
    ))
    fig_yu.add_hline(
        y=13.6, line_dash="dash", line_color="#27AE60",
        annotation_text="🌍 Global avg: 13.6%",
        annotation_font_color="#27AE60"
    )
    fig_yu.update_layout(
        **dark_layout(height=300, ytitle="Youth Unemployment (%)", margin={"l": 10, "r": 10, "t": 20, "b": 20}, use_legend=False),
        legend=dict(bgcolor="#1C2833", font=dict(color="white")),
    )
    st.plotly_chart(fig_yu, use_container_width=True, config=PLOTLY_CONFIG)

    # Report download
    st.markdown("---")
    st.markdown("### 📥 Download Executive Summary Report")
    rpt_col1, rpt_col2, rpt_col3 = st.columns([2, 2, 3])
    with rpt_col1:
        try:
            from utils.report_generator import generate_executive_summary, REPORTLAB_OK
            if REPORTLAB_OK:
                with st.spinner("Generating PDF…"):
                    pdf_bytes = generate_executive_summary(data)
                if pdf_bytes:
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"Kenya_Economic_Pulse_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        help="Full executive summary with KPIs, findings & policy recommendations"
                    )
            else:
                st.info("Install `reportlab` for PDF reports")
        except Exception as e:
            st.warning(f"PDF unavailable: {e}")
    with rpt_col2:
        try:
            from utils.report_generator import generate_html_summary
            html_report = generate_html_summary(data)
            st.download_button(
                label="🌐 Download HTML Report",
                data=html_report.encode(),
                file_name=f"Kenya_Economic_Pulse_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html",
                help="Executive summary as a styled HTML file (open in any browser)"
            )
        except Exception as e:
            st.warning(f"HTML report error: {e}")
    with rpt_col3:
        st.markdown("""
        <div style='background:#1C2833; padding:.7rem 1rem; border-radius:8px;
                    border-left:3px solid #3498DB; font-size:.82rem; color:#AAB7B8;'>
            <b style='color:white'>Report includes:</b> KPI dashboard · Key findings ·
            County snapshot (top/bottom poverty) · 6 policy recommendations · Data sources
        </div>
        """, unsafe_allow_html=True)

    # Stakeholder insights
    st.markdown("---")
    st.markdown("""
    <div style='background:#1C2833; padding:1.2rem 1.5rem; border-radius:12px;
                border:1px solid #2C3E50; margin-bottom:1rem;'>
        <h3 style='color:#3498DB; margin-top:0; font-size:1.1rem;'>💡 Stakeholder Insights & Dashboard Guide</h3>
        <div style='display:grid; grid-template-columns:1fr 1fr; gap:.8rem; margin-top:.8rem;'>
            <div style='background:#0E1117; padding:.7rem 1rem; border-radius:8px; border-left:3px solid #3498DB;'>
                <b style='color:#3498DB; font-size:.85rem;'>🏛️ Policy Makers & Government</b>
                <p style='color:#AAB7B8; font-size:.8rem; margin:.3rem 0 0; line-height:1.6;'>
                    Use <b>Policy Simulator</b> to model interventions (mobile expansion, education spend, FDI).
                    <b>County Inequality Map</b> identifies which counties need emergency support.
                    <b>Anomaly Detection</b> flags economic shocks for early response.
                </p>
            </div>
            <div style='background:#0E1117; padding:.7rem 1rem; border-radius:8px; border-left:3px solid #27AE60;'>
                <b style='color:#27AE60; font-size:.85rem;'>📈 Investors & Financial Analysts</b>
                <p style='color:#AAB7B8; font-size:.8rem; margin:.3rem 0 0; line-height:1.6;'>
                    <b>Economic Indicators</b> page shows GDP, inflation, debt trends with 5-year forecasts.
                    <b>Economic Forecaster</b> provides ARIMA + Holt-Winters projections with confidence intervals.
                    <b>Safaricom M-PESA Impact</b> quantifies digital finance market opportunity.
                </p>
            </div>
            <div style='background:#0E1117; padding:.7rem 1rem; border-radius:8px; border-left:3px solid #F39C12;'>
                <b style='color:#F39C12; font-size:.85rem;'>🎓 Researchers & Academics</b>
                <p style='color:#AAB7B8; font-size:.8rem; margin:.3rem 0 0; line-height:1.6;'>
                    <b>County Comparison</b> enables side-by-side multi-county research across 6 socioeconomic dimensions.
                    <b>NL Query Engine</b> provides instant data answers. All CSV datasets are freely downloadable
                    from each page.
                </p>
            </div>
            <div style='background:#0E1117; padding:.7rem 1rem; border-radius:8px; border-left:3px solid #E74C3C;'>
                <b style='color:#E74C3C; font-size:.85rem;'>📰 Journalists & Civil Society</b>
                <p style='color:#AAB7B8; font-size:.8rem; margin:.3rem 0 0; line-height:1.6;'>
                    <b>Youth Unemployment</b> page shows the crisis modelling.
                    <b>County Inequality Map</b> exposes the NE Kenya development gap.
                    Download the <b>Executive Summary PDF</b> above for briefing documents.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # About project box
    with st.expander("🔬 About This Project & Tech Stack"):
        st.markdown("""
        **Kenya Economic Pulse** is an end-to-end data science portfolio combining open-source datasets
        with machine learning to answer Kenya's most pressing economic questions.

        | Page | What it does |
        |------|-------------|
        | 📊 Economic Indicators | 20+ macro indicators · Holt-Winters + ARIMA forecasts |
        | 🗺️ County Inequality Map | 47 counties · KMeans clustering · Folium choropleth |
        | 💳 M-PESA Impact Predictor | GBM regression R²=0.904 · Feature importance |
        | 🎓 Youth Unemployment | ML forecast · Scenario simulator |
        | 🔮 Economic Forecaster | Holt-Winters + ARIMA(2,1,2) · Confidence intervals |
        | ⚖️ County Comparison | Up to 5 counties · Radar + bar charts |
        | 🏛️ Policy Simulator | 5-lever what-if · Waterfall chart |
        | 🔍 Anomaly Detection | Isolation Forest · Known economic shocks |
        | 💬 NL Query Engine | Plain-English questions · 16 topic patterns |
        | 👨‍💻 Developer / About | Full methodology + data sources |

        **Stack:** Python · Streamlit · Scikit-learn · Plotly · Folium · Statsmodels ·
        World Bank API · KNBS · CBK · ReportLab
        """)


# ── Define wrapper functions to ensure unique URL pathnames ─────────
# Each wrapper has a unique function name to prevent duplicate pathname errors

def page_overview():
    render_overview()

def page_economic_indicators():
    data = load_data()
    economic_indicators.render(data)

def page_county_inequality():
    data = load_data()
    county_inequality.render(data)

def page_mpesa_predictor():
    data = load_data()
    mobile_money.render(data)

def page_youth_unemployment():
    data = load_data()
    youth_unemployment.render(data)

def page_economic_forecaster():
    data = load_data()
    forecaster.render(data)

def page_county_comparison():
    data = load_data()
    county_comparison.render(data)

def page_policy_simulator():
    data = load_data()
    policy_simulator.render(data)

def page_anomaly_detection():
    data = load_data()
    anomaly_detection.render(data)

def page_nlq_engine():
    data = load_data()
    nlq_engine.render(data)

def page_developer():
    data = load_data()
    developer.render(data)


# ── Define pages for st.navigation with unique function references ───
# Each st.Page gets a unique function (the wrapper functions above)

# Core Analysis pages
overview_page = st.Page(
    page_overview,
    title="Overview",
    icon="🏠"
)

economic_page = st.Page(
    page_economic_indicators,
    title="Economic Indicators",
    icon="📊"
)

county_page = st.Page(
    page_county_inequality,
    title="County Inequality Map",
    icon="🗺️"
)

mpesa_page = st.Page(
    page_mpesa_predictor,
    title="M-PESA Impact Predictor",
    icon="💳"
)

youth_page = st.Page(
    page_youth_unemployment,
    title="Youth Unemployment",
    icon="🎓"
)

# Advanced Tools pages
forecaster_page = st.Page(
    page_economic_forecaster,
    title="Economic Forecaster",
    icon="🔮"
)

comparison_page = st.Page(
    page_county_comparison,
    title="County Comparison",
    icon="⚖️"
)

policy_page = st.Page(
    page_policy_simulator,
    title="Policy Simulator",
    icon="🏛️"
)

anomaly_page = st.Page(
    page_anomaly_detection,
    title="Anomaly Detection",
    icon="🔍"
)

nlq_page = st.Page(
    page_nlq_engine,
    title="NL Query Engine",
    icon="💬"
)

developer_page = st.Page(
    page_developer,
    title="Developer / About",
    icon="👨‍💻"
)


# ── Organize pages into sections ─────────────────────────────────────
# The dictionary creates collapsible sections in the sidebar

pages = {
    "📊 Core Analysis": [
        overview_page,
        economic_page,
        county_page,
        mpesa_page,
        youth_page,
    ],
    "🔮 Advanced Tools": [
        forecaster_page,
        comparison_page,
        policy_page,
        anomaly_page,
        nlq_page,
        developer_page,
    ],
}

# ── Initialize navigation ───────────────────────────────────────────
# st.navigation creates the sidebar automatically
# It COMPLETELY IGNORES any files in your pages/ folder

pg = st.navigation(pages, position="sidebar")
pg.run()