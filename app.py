"""
app.py
Data Detective - Professional Enterprise Data Observability & Root-Cause Intelligence Platform.
Engineered with intelligent multi-tier caching, interactive diagnostic labs, and automated remediation.
"""

import time
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from profiler import run_full_profile, _get_categorical_columns
from anomaly import detect_outliers_iqr, detect_anomalies_isolation_forest, get_combined_anomaly_indices
from segmentation import compute_lift_by_category, generate_key_findings
from drift import detect_drift, split_by_date, calculate_psi
from health_score import compute_health_score

# ==============================================================================
# 1. PAGE CONFIGURATION & METADATA
# ==============================================================================
st.set_page_config(
    page_title="Data Detective | Data Observability & Root-Cause Engine",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# 2. DESIGN SYSTEM & MODERN UI CSS INJECTION
# ==============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg-base: #0B0F19;
    --bg-surface: #111827;
    --bg-elevated: #1E293B;
    --bg-glass: rgba(17, 24, 39, 0.75);
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-strong: rgba(99, 102, 241, 0.35);
    --text-main: #F9FAFB;
    --text-muted: #94A3B8;
    --accent-indigo: #6366F1;
    --accent-cyan: #38BDF8;
    --accent-emerald: #10B981;
    --accent-amber: #F59E0B;
    --accent-crimson: #EF4444;
}

/* Global typography */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-main);
}

/* Clean background canvas */
.stApp {
    background: radial-gradient(circle at 10% 0%, #171c2f 0%, #0B0F19 60%, #07090e 100%);
}

/* Streamlit Header / Toolbar clean-up */
header[data-testid="stHeader"] {
    background: transparent;
}
div[data-testid="stToolbar"] {
    display: none;
}

/* Glassmorphic Container Cards */
.dd-card {
    background: var(--bg-glass);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1.35rem 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.35), 0 8px 10px -6px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.dd-card:hover {
    border-color: rgba(99, 102, 241, 0.25);
}

/* Metric Scorecard Cards */
.dd-metric-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(17, 24, 39, 0.85) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1.15rem 1.25rem;
    position: relative;
    overflow: hidden;
}
.dd-metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--accent-indigo), var(--accent-cyan));
}
.dd-metric-label {
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
    color: var(--text-muted);
    margin-bottom: 0.35rem;
}
.dd-metric-val {
    font-size: 1.85rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.02em;
    line-height: 1.1;
    font-feature-settings: "tnum";
}
.dd-metric-sub {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 0.35rem;
    display: flex;
    align-items: center;
    gap: 0.35rem;
}

/* Status Pill Badges */
.dd-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.22rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.dd-pill-cyan {
    background: rgba(56, 189, 248, 0.12);
    color: #38BDF8;
    border: 1px solid rgba(56, 189, 248, 0.3);
}
.dd-pill-indigo {
    background: rgba(99, 102, 241, 0.15);
    color: #818CF8;
    border: 1px solid rgba(99, 102, 241, 0.35);
}
.dd-pill-emerald {
    background: rgba(16, 185, 129, 0.12);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.dd-pill-amber {
    background: rgba(245, 158, 11, 0.12);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}
.dd-pill-crimson {
    background: rgba(239, 68, 68, 0.12);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

/* Key Finding Banner */
.dd-finding-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(30, 41, 59, 0.6) 100%);
    border-left: 4px solid var(--accent-indigo);
    border-top: 1px solid var(--border-subtle);
    border-right: 1px solid var(--border-subtle);
    border-bottom: 1px solid var(--border-subtle);
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}
.dd-finding-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 0.25rem;
}
.dd-finding-desc {
    font-size: 0.88rem;
    color: #CBD5E1;
    line-height: 1.45;
}

/* Micro Progress Bar */
.dd-progress-wrap {
    width: 100%;
    height: 6px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 4px;
    margin-top: 0.5rem;
    overflow: hidden;
}
.dd-progress-bar {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Monospace code styling */
code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Customizing Streamlit Tabs */
div[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-size: 0.92rem;
    font-weight: 600;
    color: #94A3B8;
    padding: 0.65rem 1.15rem;
    border-radius: 8px 8px 0 0;
    background-color: transparent;
    transition: all 0.2s ease;
}
div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
    color: #FFFFFF;
    background-color: rgba(255, 255, 255, 0.03);
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #FFFFFF !important;
    border-bottom: 2px solid var(--accent-indigo) !important;
    background-color: rgba(99, 102, 241, 0.08) !important;
}

/* Streamlit dataframes modern styling */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    overflow: hidden;
}

/* Sleek scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0B0F19;
}
::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #475569;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==============================================================================
# 3. PLOTLY THEME CONFIGURATION
# ==============================================================================
def apply_plotly_dark_theme(fig: go.Figure, height: int = 380, title: str = None) -> go.Figure:
    """Applies a consistent enterprise dark-mode theme to any Plotly chart."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17, 24, 39, 0.4)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#CBD5E1", size=12),
        title=dict(
            text=title or "",
            font=dict(size=14, color="#FFFFFF", family="Plus Jakarta Sans, sans-serif", weight=700),
            x=0.01,
            y=0.96
        ) if title else None,
        height=height,
        margin=dict(l=40, r=30, t=50 if title else 25, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#94A3B8")
        ),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            linecolor="rgba(255, 255, 255, 0.1)",
            zerolinecolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(color="#94A3B8", size=11),
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            linecolor="rgba(255, 255, 255, 0.1)",
            zerolinecolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(color="#94A3B8", size=11),
        ),
        hoverlabel=dict(
            bgcolor="#1E293B",
            font_size=12,
            font_family="Plus Jakarta Sans, sans-serif",
            bordercolor="rgba(99, 102, 241, 0.5)"
        ),
    )
    return fig


# ==============================================================================
# 4. HIGH-PERFORMANCE CACHING LAYER (@st.cache_data)
# ==============================================================================
@st.cache_data(show_spinner=False)
def cached_load_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Fast in-memory cached CSV ingestion."""
    return pd.read_csv(io.BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def cached_run_profile(df: pd.DataFrame) -> dict:
    """Caches core profiling: missing counts, duplicates, type detection, stats."""
    return run_full_profile(df)

@st.cache_data(show_spinner=False)
def cached_health_score(df: pd.DataFrame, profile_report: dict) -> dict:
    """Caches weighted composite health score calculations."""
    return compute_health_score(df, profile_report)

@st.cache_data(show_spinner=False)
def cached_anomaly_detection(df: pd.DataFrame, contamination: float = 0.05) -> tuple:
    """Caches Isolation Forest and IQR anomaly detection indices."""
    iqr_dict = detect_outliers_iqr(df)
    iso_dict = detect_anomalies_isolation_forest(df, contamination=contamination)
    combined = get_combined_anomaly_indices(df, contamination=contamination)
    return iqr_dict, iso_dict, combined

@st.cache_data(show_spinner=False)
def cached_lift_and_findings(df: pd.DataFrame, anomaly_indices: list) -> tuple:
    """Caches categorical lift calculations and chi-square hypothesis testing."""
    lift_results = compute_lift_by_category(df, anomaly_indices)
    findings = generate_key_findings(lift_results)
    return lift_results, findings

@st.cache_data(show_spinner=False)
def cached_drift_analysis(df: pd.DataFrame, date_col: str, split_date_str: str = None) -> dict:
    """Caches Kolmogorov-Smirnov and PSI distribution shift testing."""
    try:
        baseline, current, split_point = split_by_date(df, date_col, split_point=split_date_str)
        drift_results = detect_drift(baseline, current)
        return {
            "baseline": baseline,
            "current": current,
            "split_point": str(split_point),
            "results": drift_results,
            "error": None
        }
    except Exception as e:
        return {"error": str(e)}


# ==============================================================================
# 5. SIDEBAR: DATA INGESTION & WORKBENCH TELEMETRY
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.5rem 0 1rem 0;">
        <div style="background: linear-gradient(135deg, #6366F1, #38BDF8); width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; box-shadow: 0 0 15px rgba(99, 102, 241, 0.5);">
            🔎
        </div>
        <div>
            <div style="font-weight: 800; font-size: 1.15rem; color: #FFFFFF; letter-spacing: -0.02em;">Data Detective</div>
            <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 500;">ENTERPRISE OBSERVABILITY</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📥 Ingestion Source")
    uploaded_file = st.file_uploader(
        "Upload CSV Datafile",
        type=["csv"],
        help="Upload standard structured tabular CSV datasets."
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("⚡ Load Demo", use_container_width=True, help="Load synthetic e-commerce transactions dataset with injected quality flaws"):
            st.session_state["use_sample"] = True
            st.session_state["uploaded_custom"] = False
    with col_btn2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state["use_sample"] = False
            st.session_state["uploaded_custom"] = False
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Engine Parameters")
    contamination_slider = st.slider(
        "Anomaly Sensitivity (Isolation Forest)",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
        help="Expected proportion of outliers in the data (contamination rate)."
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.76rem; color: #64748B; line-height: 1.5;">
        <strong style="color: #94A3B8;">Root-Cause Philosophy:</strong><br>
        Traditional profilers report <em>what</em> is broken. Data Detective automatically answers <em>where</em> anomalies concentrate and <em>why</em> using Chi-Square lift segmentation.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 6. DATASET LOADER & INITIALIZATION
# ==============================================================================
df = None
dataset_name = ""

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    df = cached_load_dataframe(file_bytes, uploaded_file.name)
    dataset_name = uploaded_file.name
elif st.session_state.get("use_sample", True):
    # Default to sample dataset so user lands immediately on a live, interactive experience
    with open("sample_data.csv", "rb") as f:
        file_bytes = f.read()
    df = cached_load_dataframe(file_bytes, "sample_data.csv")
    dataset_name = "sample_data.csv (Demo Transactions Dataset)"

if df is None:
    st.info("Please upload a CSV file or click 'Load Demo' in the sidebar to begin investigation.")
    st.stop()


# ==============================================================================
# 7. CORE INTELLIGENCE PIPELINE (CACHED)
# ==============================================================================
start_t = time.perf_counter()
profile_report = cached_run_profile(df)
health = cached_health_score(df, profile_report)
iqr_anomalies, iso_anomalies, anomaly_indices = cached_anomaly_detection(df, contamination=contamination_slider)
lift_results, findings = cached_lift_and_findings(df, anomaly_indices)
elapsed_ms = round((time.perf_counter() - start_t) * 1000, 1)

# Health Grade assignment
health_val = health["health_score"]
if health_val >= 90:
    grade, grade_color, grade_desc = "A+", "#10B981", "Optimal Data Integrity"
elif health_val >= 80:
    grade, grade_color, grade_desc = "A-", "#34D399", "High Quality (Minor Schema Flaws)"
elif health_val >= 70:
    grade, grade_color, grade_desc = "B", "#FBBF24", "Moderate Degradation Detected"
elif health_val >= 55:
    grade, grade_color, grade_desc = "C", "#F97316", "Significant Quality Risk"
else:
    grade, grade_color, grade_desc = "F", "#EF4444", "Critical Anomaly & Integrity Failure"


# ==============================================================================
# 8. TOP TELEMETRY & COMMAND BANNER
# ==============================================================================
mem_kb = round(df.memory_usage(deep=True).sum() / 1024, 1)
mem_str = f"{mem_kb} KB" if mem_kb < 1024 else f"{round(mem_kb/1024, 2)} MB"

st.markdown(f"""
<div class="dd-card" style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 1rem; padding: 1rem 1.4rem; border-left: 4px solid #6366F1;">
    <div style="display: flex; align-items: center; gap: 0.85rem;">
        <span class="dd-pill dd-pill-indigo">● ACTIVE SESSION</span>
        <div style="font-weight: 700; font-size: 1.05rem; color: #FFFFFF;">{dataset_name}</div>
    </div>
    <div style="display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap;">
        <span class="dd-pill dd-pill-cyan">📊 {len(df):,} Rows</span>
        <span class="dd-pill dd-pill-cyan">📋 {len(df.columns)} Columns</span>
        <span class="dd-pill dd-pill-indigo">💾 {mem_str}</span>
        <span class="dd-pill dd-pill-amber">⚡ Processed in {elapsed_ms}ms</span>
        <span class="dd-pill" style="background: rgba(99, 102, 241, 0.2); color: {grade_color}; border: 1px solid {grade_color}; font-weight: 800;">
            GRADE {grade} ({health_val}/100)
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 9. 7-TAB ANALYTICAL WORKBENCH
# ==============================================================================
tab_overview, tab_rootcause, tab_anomalies, tab_diagnostics, tab_drift, tab_correlations, tab_remediation = st.tabs([
    "📊 Executive Health",
    "🔎 Root-Cause & Lift",
    "🚨 Anomaly Lab",
    "🧪 Schema & Diagnostics",
    "📈 Chrono & Drift",
    "📐 Correlation Matrix",
    "🛠️ Remediation Studio",
])


# ------------------------------------------------------------------------------
# TAB 1: EXECUTIVE HEALTH & INSIGHTS HUB
# ------------------------------------------------------------------------------
with tab_overview:
    col_left, col_right = st.columns([1.1, 2.2])

    with col_left:
        st.markdown(f"""
        <div class="dd-card" style="text-align: center; padding: 1.75rem 1.25rem;">
            <div class="dd-metric-label">Composite Health Score</div>
            <div style="position: relative; display: inline-flex; align-items: center; justify-content: center; margin: 1rem 0;">
                <div style="font-size: 3.5rem; font-weight: 900; color: {grade_color}; letter-spacing: -0.03em; line-height: 1;">
                    {health_val}
                </div>
                <div style="font-size: 1.1rem; color: var(--text-muted); font-weight: 600; margin-left: 0.2rem;">/100</div>
            </div>
            <div style="display: flex; justify-content: center; margin-bottom: 0.85rem;">
                <span class="dd-pill" style="background: rgba(255,255,255,0.06); color: {grade_color}; border: 1px solid {grade_color}; font-size: 0.85rem; padding: 0.3rem 0.85rem;">
                    GRADE {grade} • {grade_desc}
                </span>
            </div>
            <div style="font-size: 0.8rem; color: #94A3B8; line-height: 1.45; text-align: left; background: rgba(0,0,0,0.25); padding: 0.75rem; border-radius: 8px; border: 1px solid var(--border-subtle);">
                Weighted index evaluated across Completeness (30%), Validity (30%), Uniqueness (20%), and Categorical Consistency (20%).
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        sub = health["sub_scores"]
        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)

        def render_sub_card(col, label, val, sub_text, color_accent):
            bar_color = color_accent
            col.markdown(f"""
            <div class="dd-metric-card" style="margin-bottom: 0.85rem;">
                <div class="dd-metric-label">{label}</div>
                <div class="dd-metric-val">{val}%</div>
                <div class="dd-progress-wrap">
                    <div class="dd-progress-bar" style="width: {val}%; background: {bar_color};"></div>
                </div>
                <div class="dd-metric-sub">{sub_text}</div>
            </div>
            """, unsafe_allow_html=True)

        render_sub_card(c1, "Completeness", sub["completeness"], "Inverse of null & missing value density", "#38BDF8")
        render_sub_card(c2, "Uniqueness", sub["uniqueness"], f"{profile_report['duplicates']['duplicate_rows']} duplicate rows detected", "#10B981")
        render_sub_card(c3, "Type Validity", sub["validity"], "Data conforms to inferred semantic types", "#818CF8")
        render_sub_card(c4, "Consistency", sub["consistency"], "Free of near-duplicate category labels", "#F59E0B")

    # STANDOUT HERO FEATURE: KEY ROOT-CAUSE FINDINGS
    st.markdown("### ⚡ Root-Cause Intelligence Summary")
    if findings:
        for f in findings:
            p_str = f"p = {f['p_value']:.5f}" if f.get("p_value") is not None else "p < 0.001"
            st.markdown(f"""
            <div class="dd-finding-card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.35rem;">
                    <div class="dd-finding-title">🎯 Segment Anomaly Over-Representation: {f['column']}</div>
                    <span class="dd-pill dd-pill-indigo">Chi-Square Confirmed ({p_str})</span>
                </div>
                <div class="dd-finding-desc">{f['text']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="dd-card" style="padding: 1rem 1.25rem;">
            <div style="color: #94A3B8; font-size: 0.9rem;">
                No statistically significant categorical clustering detected among anomalies. Anomalies appear randomly distributed across all segments.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # DATASET PREVIEW & SCHEMA INSPECTION
    with st.expander("🔍 Interactive Data Catalog & Raw Preview (First 25 Records)", expanded=False):
        st.dataframe(df.head(25), use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 2: ROOT-CAUSE & LIFT INVESTIGATOR (THE HERO FEATURE)
# ------------------------------------------------------------------------------
with tab_rootcause:
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <div style="font-size: 1.2rem; font-weight: 700; color: #FFFFFF;">Root-Cause Lift & Segment Attribution</div>
        <div style="font-size: 0.85rem; color: #94A3B8;">
            Calculates <strong>Lift</strong> for each categorical segment: <code>(Share of Anomalies in Segment) / (Baseline Share in Entire Dataset)</code>. 
            A lift &gt; 1.0 indicates that quality failures or outliers are disproportionately concentrated in that specific cohort.
        </div>
    </div>
    """, unsafe_allow_html=True)

    cat_cols = _get_categorical_columns(df)
    if not cat_cols or not lift_results:
        st.info("No categorical columns or anomalous rows available for lift segmentation.")
    else:
        selected_cat = st.selectbox(
            "Select Categorical Column to Investigate",
            options=list(lift_results.keys()),
            help="Examine anomaly concentration across distinct segment values."
        )

        res = lift_results[selected_cat]
        cat_data = res["categories"]
        cat_df = pd.DataFrame(cat_data)

        # Lift & Chi-Square Header Cards
        stat_sig = res["statistically_significant"]
        p_val = res["chi_square_p_value"]
        p_badge = f'<span class="dd-pill dd-pill-emerald">✓ Statistically Significant (p = {p_val})</span>' if stat_sig else f'<span class="dd-pill dd-pill-amber">⚠ Not Statistically Significant (p = {p_val})</span>'

        st.markdown(f"""
        <div class="dd-card" style="display: flex; align-items: center; justify-content: space-between; padding: 0.9rem 1.25rem;">
            <div>
                <strong style="color: #FFFFFF; font-size: 0.95rem;">Chi-Square Contingency Test:</strong>
                <span style="color: #94A3B8; font-size: 0.88rem; margin-left: 0.5rem;">Evaluates whether anomaly frequency differs from expected random baseline distribution.</span>
            </div>
            <div>{p_badge}</div>
        </div>
        """, unsafe_allow_html=True)

        col_chart, col_table = st.columns([1.6, 1.2])

        with col_chart:
            # Dual-bar comparison chart
            fig_lift = go.Figure()
            fig_lift.add_trace(go.Bar(
                name="Overall Baseline %",
                x=cat_df["category"],
                y=cat_df["overall_pct"],
                marker_color="rgba(148, 163, 184, 0.4)",
                marker_line=dict(width=1, color="rgba(148, 163, 184, 0.6)"),
                hovertemplate="<b>%{x}</b><br>Baseline Share: %{y:.1f}%<extra></extra>"
            ))
            fig_lift.add_trace(go.Bar(
                name="Anomaly Share %",
                x=cat_df["category"],
                y=cat_df["anomaly_pct"],
                marker_color="#6366F1",
                marker_line=dict(width=1, color="#818CF8"),
                hovertemplate="<b>%{x}</b><br>Anomaly Share: %{y:.1f}%<extra></extra>"
            ))

            fig_lift.update_layout(
                barmode="group",
                title=dict(text=f"Anomaly Concentration vs. Baseline: {selected_cat}"),
                xaxis_title="Category Segment",
                yaxis_title="Share (%)",
            )
            apply_plotly_dark_theme(fig_lift, height=360)
            st.plotly_chart(fig_lift, use_container_width=True)

        with col_table:
            # Lift Multiplier Bar Chart
            cat_df_sorted = cat_df.sort_values("lift", ascending=True)
            colors = ["#EF4444" if l > 1.5 else "#F59E0B" if l > 1.0 else "#10B981" for l in cat_df_sorted["lift"]]

            fig_bar = go.Figure(go.Bar(
                x=cat_df_sorted["lift"],
                y=cat_df_sorted["category"],
                orientation="h",
                marker_color=colors,
                text=[f"{l:.2f}x" for l in cat_df_sorted["lift"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Lift Multiplier: %{x:.2f}x<extra></extra>"
            ))
            fig_bar.add_vline(x=1.0, line_dash="dash", line_color="rgba(255,255,255,0.4)", annotation_text="Baseline (1.0x)")
            fig_bar.update_layout(
                title=dict(text=f"Lift Multiplier by Segment"),
                xaxis_title="Lift (Anomaly % / Baseline %)",
                yaxis_title="",
            )
            apply_plotly_dark_theme(fig_bar, height=360)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Tabular Segment Breakdown
        st.markdown("#### 📋 Detailed Lift Breakdown Table")
        display_lift_df = cat_df.copy()
        display_lift_df.columns = ["Segment Value", "Baseline Share (%)", "Anomaly Share (%)", "Lift Ratio"]
        st.dataframe(display_lift_df, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 3: ANOMALY & OUTLIER LAB
# ------------------------------------------------------------------------------
with tab_anomalies:
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <div style="font-size: 1.2rem; font-weight: 700; color: #FFFFFF;">Multivariate & Univariate Anomaly Lab</div>
        <div style="font-size: 0.85rem; color: #94A3B8;">
            Flags anomalies using <strong>Isolation Forest</strong> (multivariate dimension tree partitioning) and <strong>IQR</strong> (interquartile distribution boundaries).
        </div>
    </div>
    """, unsafe_allow_html=True)

    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.markdown(f"""
    <div class="dd-metric-card">
        <div class="dd-metric-label">Combined Anomalies Flagged</div>
        <div class="dd-metric-val" style="color: #EF4444;">{len(anomaly_indices):,}</div>
        <div class="dd-metric-sub">{round(len(anomaly_indices)/len(df)*100, 2)}% of total records</div>
    </div>
    """, unsafe_allow_html=True)

    c_m2.markdown(f"""
    <div class="dd-metric-card">
        <div class="dd-metric-label">Isolation Forest Detections</div>
        <div class="dd-metric-val">{iso_anomalies.get('anomaly_count', 0):,}</div>
        <div class="dd-metric-sub">Contamination rate: {int(contamination_slider*100)}%</div>
    </div>
    """, unsafe_allow_html=True)

    iqr_total_outliers = sum(v["outlier_count"] for v in iqr_anomalies.values())
    c_m3.markdown(f"""
    <div class="dd-metric-card">
        <div class="dd-metric-label">Univariate IQR Outlier Flags</div>
        <div class="dd-metric-val">{iqr_total_outliers:,}</div>
        <div class="dd-metric-sub">Across {len(iqr_anomalies)} numeric columns</div>
    </div>
    """, unsafe_allow_html=True)

    # 2D/3D Scatter Exploration
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) >= 2:
        st.markdown("#### 🔭 Interactive Anomaly Projection Scatter Plot")
        sc_col1, sc_col2 = st.columns(2)
        with sc_col1:
            x_axis = st.selectbox("X-Axis Feature", options=num_cols, index=0)
        with sc_col2:
            y_axis = st.selectbox("Y-Axis Feature", options=num_cols, index=min(1, len(num_cols)-1))

        plot_df = df.copy()
        plot_df["Status"] = "Normal Record"
        plot_df.loc[plot_df.index.isin(anomaly_indices), "Status"] = "Anomalous Record"

        fig_scatter = px.scatter(
            plot_df,
            x=x_axis,
            y=y_axis,
            color="Status",
            color_discrete_map={"Normal Record": "rgba(148, 163, 184, 0.45)", "Anomalous Record": "#EF4444"},
            hover_data=[c for c in df.columns if c not in [x_axis, y_axis]][:4],
            title=f"Anomaly Spatial Distribution: {x_axis} vs. {y_axis}"
        )
        apply_plotly_dark_theme(fig_scatter, height=420)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Anomaly Rows Inspector
    st.markdown("#### 🚨 Flagged Anomaly Records")
    if anomaly_indices:
        anomaly_df = df.loc[df.index.isin(anomaly_indices)]
        st.dataframe(anomaly_df, use_container_width=True)

        csv_buffer = io.StringIO()
        anomaly_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Flagged Anomalies CSV",
            data=csv_buffer.getvalue(),
            file_name="flagged_anomalies.csv",
            mime="text/csv",
        )
    else:
        st.success("Zero anomalous records detected under current parameter thresholds.")


# ------------------------------------------------------------------------------
# TAB 4: SCHEMA, MISSING VALUES & QUALITY DIAGNOSTICS
# ------------------------------------------------------------------------------
with tab_diagnostics:
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <div style="font-size: 1.2rem; font-weight: 700; color: #FFFFFF;">Data Quality & Schema Diagnostics</div>
        <div style="font-size: 0.85rem; color: #94A3B8;">
            Deep-dive audit into missingness patterns, schema/type coercion mismatches, exact duplicates, and fuzzy string cluster variants.
        </div>
    </div>
    """, unsafe_allow_html=True)

    diag_tab1, diag_tab2, diag_tab3, diag_tab4 = st.tabs([
        "⚠️ Missing Values",
        "🔀 Data Type Mismatches",
        "🔤 Inconsistent Categories",
        "👥 Duplicate Rows"
    ])

    with diag_tab1:
        missing_dict = profile_report["missing_values"]
        missing_rows = [
            {"Column": col, "Missing Count": v["missing_count"], "Missing (%)": v["missing_pct"]}
            for col, v in missing_dict.items()
        ]
        missing_df = pd.DataFrame(missing_rows).sort_values("Missing (%)", ascending=False)
        missing_non_zero = missing_df[missing_df["Missing (%)"] > 0]

        if not missing_non_zero.empty:
            fig_miss = px.bar(
                missing_non_zero,
                x="Column",
                y="Missing (%)",
                color="Missing (%)",
                color_continuous_scale=["#38BDF8", "#F59E0B", "#EF4444"],
                text="Missing (%)",
                title="Missing Value Density by Column (%)"
            )
            fig_miss.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            apply_plotly_dark_theme(fig_miss, height=320)
            st.plotly_chart(fig_miss, use_container_width=True)
            st.dataframe(missing_non_zero, use_container_width=True)
        else:
            st.markdown("""
            <div class="dd-card" style="border-left: 4px solid #10B981; padding: 1rem 1.25rem;">
                <span class="dd-pill dd-pill-emerald">✓ 100% Complete</span>
                <span style="margin-left: 0.5rem; color: #FFFFFF; font-size: 0.9rem;">No missing or null entries found in any column.</span>
            </div>
            """, unsafe_allow_html=True)

    with diag_tab2:
        type_issues = {k: v for k, v in profile_report["data_types"].items() if v["type_mismatch"]}
        if type_issues:
            for col, info in type_issues.items():
                st.markdown(f"""
                <div class="dd-finding-card" style="border-left-color: #F59E0B; margin-bottom: 0.75rem;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div class="dd-finding-title">Column <code>{col}</code> Type Coercion Opportunity</div>
                        <span class="dd-pill dd-pill-amber">Type Mismatch</span>
                    </div>
                    <div class="dd-finding-desc">
                        Stored in memory as <code>{info['declared_dtype']}</code>, but 90%+ of values conform to <strong>{info['inferred_type']}</strong>.
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="dd-card" style="border-left: 4px solid #10B981; padding: 1rem 1.25rem;">
                <span class="dd-pill dd-pill-emerald">✓ Types Match Inferred Schema</span>
                <span style="margin-left: 0.5rem; color: #FFFFFF; font-size: 0.9rem;">All column data types align accurately with their actual semantic payloads.</span>
            </div>
            """, unsafe_allow_html=True)

    with diag_tab3:
        inconsistent = profile_report["inconsistent_categories"]
        if inconsistent:
            for col, clusters in inconsistent.items():
                for group in clusters:
                    st.markdown(f"""
                    <div class="dd-finding-card" style="border-left-color: #818CF8;">
                        <div class="dd-finding-title">Category Inconsistency in <code>{col}</code></div>
                        <div class="dd-finding-desc">
                            The following variant labels appear to represent the same entity: <strong>{', '.join([f'"{g}"' for g in group])}</strong>.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="dd-card" style="border-left: 4px solid #10B981; padding: 1rem 1.25rem;">
                <span class="dd-pill dd-pill-emerald">✓ Clean Categorical Encodings</span>
                <span style="margin-left: 0.5rem; color: #FFFFFF; font-size: 0.9rem;">No fuzzy duplicate strings or case-mismatched categorical variants detected.</span>
            </div>
            """, unsafe_allow_html=True)

    with diag_tab4:
        dup = profile_report["duplicates"]
        if dup["duplicate_rows"] > 0:
            st.markdown(f"""
            <div class="dd-finding-card" style="border-left-color: #EF4444;">
                <div class="dd-finding-title">Duplicate Records Flagged</div>
                <div class="dd-finding-desc">
                    Found <strong>{dup['duplicate_rows']}</strong> exact duplicate rows ({dup['duplicate_pct']}% of total dataset).
                </div>
            </div>
            """, unsafe_allow_html=True)
            if dup["duplicate_row_indices"]:
                st.dataframe(df.loc[dup["duplicate_row_indices"]], use_container_width=True)
        else:
            st.markdown("""
            <div class="dd-card" style="border-left: 4px solid #10B981; padding: 1rem 1.25rem;">
                <span class="dd-pill dd-pill-emerald">✓ 100% Unique Records</span>
                <span style="margin-left: 0.5rem; color: #FFFFFF; font-size: 0.9rem;">Zero exact duplicate rows found in dataset.</span>
            </div>
            """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TAB 5: CHRONO & DISTRIBUTION DRIFT
# ------------------------------------------------------------------------------
with tab_drift:
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <div style="font-size: 1.2rem; font-weight: 700; color: #FFFFFF;">Distribution Drift & Chrono Inspector</div>
        <div style="font-size: 0.85rem; color: #94A3B8;">
            Compares temporal distributions across two chronological partitions using <strong>Kolmogorov-Smirnov (KS)</strong> two-sample tests and <strong>Population Stability Index (PSI)</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    date_candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower() or "year" in c.lower()]
    if not date_candidates:
        date_candidates = df.columns.tolist()

    date_col = st.selectbox(
        "Select Temporal Partition Column",
        options=date_candidates,
        help="Date/timestamp column used to partition baseline vs. current observation periods."
    )

    if date_col:
        drift_data = cached_drift_analysis(df, date_col)
        if drift_data.get("error"):
            st.warning(f"Could not calculate temporal drift on '{date_col}': {drift_data['error']}")
        else:
            drift_res = drift_data["results"]
            split_p = drift_data["split_point"]
            baseline_df = drift_data["baseline"]
            current_df = drift_data["current"]

            st.markdown(f"""
            <div class="dd-card" style="padding: 0.9rem 1.25rem; display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <strong style="color: #FFFFFF;">Temporal Split Point:</strong> 
                    <code style="color: #38BDF8; margin-left: 0.4rem;">{split_p}</code>
                </div>
                <div style="display: flex; gap: 0.65rem;">
                    <span class="dd-pill dd-pill-cyan">Baseline: {len(baseline_df):,} Rows</span>
                    <span class="dd-pill dd-pill-indigo">Current: {len(current_df):,} Rows</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            drift_table_rows = []
            for col_name, stats in drift_res.items():
                psi_str = f"{stats['psi']:.4f}" if stats['psi'] is not None else "N/A"
                drift_table_rows.append({
                    "Feature": col_name,
                    "Baseline Mean": stats["baseline_mean"],
                    "Current Mean": stats["current_mean"],
                    "KS-Statistic": stats["ks_statistic"],
                    "KS p-value": stats["ks_p_value"],
                    "PSI Score": psi_str,
                    "Drift Status": "🚨 DRIFT DETECTED" if stats["drift_detected"] else "✓ STABLE"
                })

            drift_summary_df = pd.DataFrame(drift_table_rows)
            st.dataframe(drift_summary_df, use_container_width=True)

            # Interactive Distribution Curve Overlay
            drift_features = list(drift_res.keys())
            if drift_features:
                sel_feature = st.selectbox("Inspect Distribution Shift for Feature", options=drift_features)

                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=baseline_df[sel_feature].dropna(),
                    histnorm='probability density',
                    name="Baseline Distribution",
                    marker_color="rgba(56, 189, 248, 0.5)",
                    opacity=0.6
                ))
                fig_dist.add_trace(go.Histogram(
                    x=current_df[sel_feature].dropna(),
                    histnorm='probability density',
                    name="Current Period Distribution",
                    marker_color="rgba(99, 102, 241, 0.6)",
                    opacity=0.6
                ))
                fig_dist.update_layout(
                    barmode="overlay",
                    title=dict(text=f"Overlaid Probability Density: {sel_feature} (Baseline vs. Current)"),
                    xaxis_title=sel_feature,
                    yaxis_title="Probability Density",
                )
                apply_plotly_dark_theme(fig_dist, height=360)
                st.plotly_chart(fig_dist, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 6: CORRELATION MATRIX & COLLINEARITY
# ------------------------------------------------------------------------------
with tab_correlations:
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <div style="font-size: 1.2rem; font-weight: 700; color: #FFFFFF;">Correlation Matrix & Collinearity Radar</div>
        <div style="font-size: 0.85rem; color: #94A3B8;">
            Pearson pairwise correlation coefficients across numeric features with automated high-collinearity warnings (|r| &ge; 0.70).
        </div>
    </div>
    """, unsafe_allow_html=True)

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] >= 2:
        corr_matrix = numeric_df.corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Feature Pairwise Correlation Heatmap"
        )
        apply_plotly_dark_theme(fig_corr, height=440)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Automated high collinearity pairs detector
        pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                r = corr_matrix.iloc[i, j]
                if abs(r) >= 0.70:
                    pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], round(r, 3)))

        if pairs:
            st.markdown("#### ⚡ Collinear Feature Pairs")
            for p1, p2, r_val in pairs:
                st.markdown(f"""
                <div class="dd-finding-card" style="border-left-color: #38BDF8;">
                    <strong>{p1}</strong> & <strong>{p2}</strong> exhibit strong linear dependency (r = {r_val}).
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("At least 2 numeric columns are required to construct a correlation heatmap.")


# ------------------------------------------------------------------------------
# TAB 7: REMEDIATION STUDIO & REPORT EXPORT
# ------------------------------------------------------------------------------
with tab_remediation:
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <div style="font-size: 1.2rem; font-weight: 700; color: #FFFFFF;">Remediation Studio & Audit Export</div>
        <div style="font-size: 0.85rem; color: #94A3B8;">
            Automated production-ready data cleaning script generator and executive audit report export.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_rem1, col_rem2 = st.columns([1.2, 1])

    with col_rem1:
        st.markdown("#### 🐍 Auto-Generated Python Remediation Script")
        # Build cleaning script addressing actual findings
        script_lines = [
            "# Auto-generated Data Cleaning Pipeline by Data Detective",
            "import pandas as pd",
            "import numpy as np",
            "",
            "def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:",
            "    df_clean = df.copy()",
            "",
        ]

        # 1. Deduplication
        if profile_report["duplicates"]["duplicate_rows"] > 0:
            script_lines.append("    # 1. Drop duplicate rows")
            script_lines.append("    df_clean = df_clean.drop_duplicates()")
            script_lines.append("")

        # 2. Type conversions
        type_issues = {k: v for k, v in profile_report["data_types"].items() if v["type_mismatch"]}
        if type_issues:
            script_lines.append("    # 2. Correct inferred data types")
            for col, v in type_issues.items():
                if "numeric" in v["inferred_type"]:
                    script_lines.append(f"    df_clean['{col}'] = pd.to_numeric(df_clean['{col}'], errors='coerce')")
                elif "datetime" in v["inferred_type"]:
                    script_lines.append(f"    df_clean['{col}'] = pd.to_datetime(df_clean['{col}'], errors='coerce')")
            script_lines.append("")

        # 3. Categorical normalization
        inconsistent = profile_report["inconsistent_categories"]
        if inconsistent:
            script_lines.append("    # 3. Standardize inconsistent categorical string labels")
            for col, clusters in inconsistent.items():
                for group in clusters:
                    canonical = group[0].strip().title()
                    for alias in group:
                        if alias != canonical:
                            script_lines.append(f"    df_clean['{col}'] = df_clean['{col}'].replace('{alias}', '{canonical}')")
            script_lines.append("")

        # 4. Missing value strategy
        missing_dict = profile_report["missing_values"]
        has_missing = any(v["missing_pct"] > 0 for v in missing_dict.values())
        if has_missing:
            script_lines.append("    # 4. Impute or flag missing values")
            for col, v in missing_dict.items():
                if v["missing_pct"] > 0:
                    if col in df.select_dtypes(include=[np.number]).columns:
                        script_lines.append(f"    df_clean['{col}'] = df_clean['{col}'].fillna(df_clean['{col}'].median())")
                    else:
                        script_lines.append(f"    df_clean['{col}'] = df_clean['{col}'].fillna('UNKNOWN')")
            script_lines.append("")

        script_lines.append("    return df_clean")
        clean_code = "\n".join(script_lines)

        st.code(clean_code, language="python")
        st.download_button(
            label="📥 Download clean_pipeline.py",
            data=clean_code,
            file_name="clean_pipeline.py",
            mime="text/x-python",
            use_container_width=True
        )

    with col_rem2:
        st.markdown("#### 📄 Executive Audit Report (Markdown)")
        audit_md = f"""# Data Detective Observability Report
**Dataset**: {dataset_name}  
**Audit Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Health Score**: {health_val}/100 (Grade {grade} - {grade_desc})  

## 1. Quality Pillar Sub-Scores
- **Completeness**: {sub['completeness']}%
- **Uniqueness**: {sub['uniqueness']}%
- **Validity**: {sub['validity']}%
- **Consistency**: {sub['consistency']}%

## 2. Key Root-Cause Findings
"""
        if findings:
            for f in findings:
                audit_md += f"- **{f['column']}**: {f['text']} (p-value: {f.get('p_value', '<0.001')})\n"
        else:
            audit_md += "- No statistically significant segment anomaly concentrations.\n"

        audit_md += f"""
## 3. Detected Anomalies
- Total combined flagged anomalies: {len(anomaly_indices)} rows ({round(len(anomaly_indices)/len(df)*100, 2)}%)
- Isolation Forest anomalies: {iso_anomalies.get('anomaly_count', 0)}
- Exact duplicate rows: {profile_report['duplicates']['duplicate_rows']}

*Generated by Data Detective Observability Platform*
"""
        st.text_area("Audit Summary Preview", audit_md, height=260)
        st.download_button(
            label="📥 Download Full Audit Report (.md)",
            data=audit_md,
            file_name=f"data_detective_audit_{int(time.time())}.md",
            mime="text/markdown",
            use_container_width=True
        )

# ==============================================================================
# 10. CLEAN WORKSPACE FOOTER
# ==============================================================================
st.markdown("---")
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; font-size: 0.76rem; color: #64748B; padding: 0.5rem 0;">
    <div>Data Detective Observability Engine v2.0 • High-Performance Caching Active</div>
    <div>Statistical Confidence: Chi-Square (p &lt; 0.05) • KS Two-Sample Drift Detection</div>
</div>
""", unsafe_allow_html=True)
