import streamlit as st

def apply_theme():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset & Base ───────────────────────────── */
:root {
    --bg-base:        #070A13;
    --bg-surface:     #0E1220;
    --bg-card:        rgba(255,255,255,0.035);
    --bg-card-hover:  rgba(255,255,255,0.065);
    --border:         rgba(255,255,255,0.08);
    --border-hover:   rgba(255,255,255,0.16);
    --txt:            #EAEDF2;
    --txt-sub:        #8B92A5;
    --accent:         #6C63FF;
    --accent2:        #3B82F6;
    --green:          #22C55E;
    --amber:          #F59E0B;
    --red:            #EF4444;
    --r:              14px;
    --shadow:         0 4px 24px rgba(0,0,0,0.5);
}
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    color: var(--txt);
}

/* ── App background ─────────────────────────── */
.stApp {
    background: var(--bg-base);
    background-image:
        radial-gradient(ellipse 60% 40% at 10% 20%, rgba(108,99,255,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 50% 50% at 90% 80%, rgba(59,130,246,0.06) 0%, transparent 60%);
}

/* ── Sidebar ────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] section { padding: 0 1rem 1rem; }

/* ── Main content padding ───────────────────── */
.block-container {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1400px !important;
}

/* ── Hide default top padding ──────────────── */
.stApp > header { display: none !important; }

/* ── Typography ─────────────────────────────── */
h1,h2,h3,h4 { color: var(--txt) !important; letter-spacing: -0.02em; }
h1 { font-size: 2rem    !important; font-weight: 700 !important; margin-bottom: 0.25rem !important; margin-top: 0 !important; }
h2 { font-size: 1.3rem  !important; font-weight: 600 !important; margin-top: 2rem !important; margin-bottom: 0.5rem !important; }
h3 { font-size: 1.05rem !important; font-weight: 600 !important; }
p  { color: var(--txt-sub); line-height: 1.6; }

/* ── Glass Card ─────────────────────────────── */
.gc {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 20px 22px;
    box-shadow: var(--shadow);
    transition: border-color 0.25s, background 0.25s, transform 0.25s;
    margin-bottom: 0;
}
.gc:hover {
    border-color: var(--border-hover);
    background: var(--bg-card-hover);
    transform: translateY(-1px);
}

/* ── KPI card ───────────────────────────────── */
.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--txt-sub);
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 1.85rem;
    font-weight: 700;
    line-height: 1.1;
    color: var(--txt);
}
.kpi-sub {
    font-size: 0.78rem;
    color: var(--txt-sub);
    margin-top: 5px;
}
.kpi-badge-green { color: var(--green); font-weight: 600; }
.kpi-badge-red   { color: var(--red);   font-weight: 600; }

/* ── Divider ────────────────────────────────── */
.div-line {
    height: 1px;
    background: var(--border);
    margin: 1.6rem 0;
}

/* ── Section label ──────────────────────────── */
.section-tag {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 6px;
}

/* ── Metrics native ─────────────────────────── */
[data-testid="stMetricValue"] { font-size: 1.75rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { font-size: 0.78rem !important; font-weight: 600 !important; color: var(--txt-sub) !important; }
[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 18px 20px !important;
    transition: border-color .25s, transform .25s;
}
[data-testid="metric-container"]:hover {
    border-color: var(--border-hover);
    transform: translateY(-1px);
}

/* ── Streamlit tabs ─────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 4px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    padding: 10px 18px !important;
    font-size: 0.88rem;
    font-weight: 500;
    color: var(--txt-sub) !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: var(--txt) !important;
    border-bottom: 2px solid var(--accent) !important;
    font-weight: 600 !important;
}

/* ── Selectbox / inputs ─────────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--txt) !important;
}
.stSlider [data-testid="stSlider"] { color: var(--accent); }

/* ── Button ─────────────────────────────────── */
.stButton > button {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    color: var(--txt);
    font-weight: 500;
    font-size: 0.88rem;
    padding: 8px 18px;
    transition: all .2s;
}
.stButton > button:hover {
    background: var(--bg-card-hover);
    border-color: var(--accent);
    color: var(--accent);
}

/* ── Expander ───────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--txt) !important;
}

/* ── DataFrames ─────────────────────────────── */
.stDataFrame { border: 1px solid var(--border) !important; border-radius: var(--r) !important; }
[data-testid="stDataFrameResizable"] > div { background: var(--bg-card) !important; }

/* ── Sidebar radio ──────────────────────────── */
[data-testid="stRadio"] label {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    padding: 10px 12px !important;
    border-radius: 10px !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: var(--txt-sub) !important;
    transition: all .2s !important;
    cursor: pointer !important;
    margin: 2px 0 !important;
}
[data-testid="stRadio"] label:hover {
    background: var(--bg-card-hover) !important;
    color: var(--txt) !important;
}
[data-testid="stRadio"] [aria-checked="true"] ~ label,
[data-testid="stRadio"] input:checked + label {
    background: rgba(108,99,255,0.12) !important;
    color: var(--txt) !important;
    border-left: 2px solid var(--accent) !important;
}

/* ── Sidebar radio bullet hide ──────────────── */
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { margin: 0; }

/* ── Alert/info ─────────────────────────────── */
[data-testid="stAlert"] {
    background: rgba(108,99,255,0.08) !important;
    border: 1px solid rgba(108,99,255,0.2) !important;
    border-radius: var(--r) !important;
    color: var(--txt) !important;
}

/* ── Warning ────────────────────────────────── */
[data-testid="stNotification"] {
    border-radius: var(--r) !important;
}

/* ── Recommendation progress bar ────────────── */
.rec-bar-track {
    background: rgba(255,255,255,0.07);
    border-radius: 6px;
    height: 6px;
    width: 100%;
    overflow: hidden;
    margin: 8px 0;
}
.rec-bar-fill {
    height: 6px;
    border-radius: 6px;
}

/* ── Status dot ─────────────────────────────── */
.dot-green {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 6px var(--green);
}
</style>
""", unsafe_allow_html=True)


def page_header(tag: str, title: str, subtitle: str):
    """Renders a consistent premium page header."""
    st.markdown(f'<span class="section-tag">{tag}</span>', unsafe_allow_html=True)
    st.markdown(f"# {title}")
    st.markdown(f'<p style="margin-top:-10px;font-size:1.05rem;">{subtitle}</p>', unsafe_allow_html=True)
    st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)


def section_header(title: str):
    st.markdown(f'<h2 style="margin-top:1.5rem;">{title}</h2>', unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = ""):
    """Renders a premium KPI card using Streamlit metric-styled HTML."""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
<div class="gc" style="text-align:left;">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  {sub_html}
</div>""", unsafe_allow_html=True)


def divider():
    st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
