import streamlit as st

STEAM_CSS = """
<style>
/* ── 전체 배경 ── */
.stApp {
    background: #1b2838 !important;
    font-family: "Motiva Sans", Arial, sans-serif;
}
.main {
    background: #1b2838 !important;
}
header { visibility: hidden; }
footer { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 100% !important; }

/* ── 사이드바 ── */
section[data-testid="stSidebar"] {
    background: #16202d !important;
    border-right: 1px solid #2a3f5a !important;
}
div[data-testid="stSidebarContent"] { background: #16202d !important; }
[data-testid="stSidebarNav"] { background: #16202d !important; }
[data-testid="stSidebarNav"] a {
    color: #c6d4df !important;
    border-radius: 4px !important;
    padding: 8px 12px !important;
    font-size: 14px !important;
    text-decoration: none !important;
}
[data-testid="stSidebarNav"] a:hover {
    background: #2a3f5a !important;
    color: #fff !important;
}
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: #2a475e !important;
    color: #66c0f4 !important;
    border-left: 3px solid #66c0f4 !important;
}

/* ── 히어로 배너 ── */
.steam-hero {
    background: linear-gradient(135deg, #0e1922 0%, #1a3a5c 50%, #0d2035 100%);
    border-bottom: 2px solid #66c0f4;
    text-align: center;
    padding: 52px 40px;
    margin-bottom: 0;
}
.steam-hero-badge {
    display: inline-block;
    background: rgba(102,192,244,0.15);
    border: 1px solid #66c0f4;
    color: #66c0f4;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 2px;
    margin-bottom: 18px;
}
.steam-hero-title {
    font-size: 48px;
    font-weight: 900;
    color: #ffffff;
    letter-spacing: -1px;
    text-shadow: 0 2px 24px rgba(0,0,0,0.8);
    margin-bottom: 12px;
    line-height: 1.1;
}
.steam-hero-subtitle {
    font-size: 17px;
    color: #8cbdd8;
    font-weight: 400;
}

/* ── KPI 카드 ── */
[data-testid="metric-container"] {
    background: #16202d !important;
    border: 1px solid #2a3f5a !important;
    border-radius: 4px !important;
    padding: 18px 20px !important;
}
[data-testid="metric-container"] label {
    color: #8cbdd8 !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 26px !important;
    font-weight: 700 !important;
}

/* ── 섹션 헤더 ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 30px 0 14px 0;
    border-bottom: 1px solid #2a3f5a;
    padding-bottom: 8px;
}
.section-header-text {
    font-size: 16px;
    font-weight: 700;
    color: #c6d4df;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.section-accent {
    width: 3px; height: 18px;
    background: #66c0f4;
    border-radius: 2px;
    display: inline-block;
    flex-shrink: 0;
}

/* ── 데이터프레임 ── */
[data-testid="stDataFrame"] {
    background: #16202d !important;
    border: 1px solid #2a3f5a !important;
    border-radius: 4px !important;
}

/* ── selectbox ── */
[data-baseweb="select"] > div {
    background: #16202d !important;
    border: 1px solid #4d7899 !important;
    color: #c6d4df !important;
}
[data-baseweb="select"] span { color: #c6d4df !important; }
[data-baseweb="popover"] ul { background: #16202d !important; border: 1px solid #2a3f5a !important; }
[data-baseweb="popover"] li { color: #c6d4df !important; }
[data-baseweb="popover"] li:hover { background: #2a475e !important; }

/* ── 버튼 ── */
.stButton button {
    background: linear-gradient(to right, #4d7899, #2a6496) !important;
    color: white !important;
    border: none !important;
    border-radius: 3px !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    padding: 10px 24px !important;
}
.stButton button:hover { filter: brightness(1.2); }

/* ── 탭 ── */
[data-testid="stTabs"] button {
    color: #8cbdd8 !important;
    font-size: 13px;
    text-transform: uppercase;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #66c0f4 !important;
    border-bottom: 2px solid #66c0f4 !important;
}

/* ── 기타 ── */
hr { border-color: #2a3f5a !important; }
[data-testid="stAlert"] {
    background: #16202d !important;
    border: 1px solid #2a3f5a !important;
    border-radius: 4px !important;
}
h1, h2, h3 { color: #c6d4df !important; }
p, label, span { color: #c6d4df; }

/* ── 네비게이션 바 ── */
.steam-nav-bar {
    background: linear-gradient(to bottom, #15232e, #1b2838);
    padding: 0 30px;
    border-bottom: 2px solid #000e1a;
    display: flex;
    align-items: center;
    position: sticky;
    top: 0;
    z-index: 999;
    box-shadow: 0 2px 8px rgba(0,0,0,0.5);
}
.steam-nav-logo {
    font-size: 24px;
    font-weight: 900;
    color: #66c0f4 !important;
    padding: 14px 24px 14px 0;
    border-right: 1px solid #2a3f5a;
    margin-right: 12px;
    letter-spacing: 3px;
    font-style: italic;
    text-decoration: none !important;
    display: inline-block;
}

.steam-nav-logo:hover {
    color: white !important;
}
.steam-nav-link {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 19px 16px 17px 16px;
    display: inline-block;
    text-decoration: none !important;
    border-bottom: 2px solid transparent;
    transition: color 0.15s, border-color 0.15s;
    margin-bottom: -2px;
}
.steam-nav-link:hover {
    color: #ffffff !important;
    text-decoration: none !important;
}
.nav-active {
    color: #ffffff !important;
    border-bottom: 2px solid #66c0f4 !important;
}
.nav-inactive {
    color: #8f98a0 !important;
    border-bottom: 2px solid transparent !important;
}

/* 링크 밑줄 전체 제거 */
a { text-decoration: none !important; }
</style>
"""


def steam_theme():
    st.markdown(STEAM_CSS, unsafe_allow_html=True)


def steam_nav(active="홈"):
    pages = [
        ("홈", "Home.py"),
        ("장르 분석", "pages/1_장르분석.py"),
        ("가격 분석", "pages/2_가격분석.py"),
        ("흥행 예측", "pages/3_예측모델.py"),
    ]

    links = ""

    # STEAM 로고
    links += (
        '<a href="Home.py" target="_self" '
        'class="steam-nav-logo">STEAM</a>'
    )

    for label, href in pages:
        is_active = (
            active in label
            or label in active
        )

        css = (
            "nav-active"
            if is_active
            else "nav-inactive"
        )

        links += (
            f'<a href="{href}" '
            f'target="_self" '
            f'class="steam-nav-link {css}">'
            f'{label}'
            f'</a>'
        )

    html = f"""
    <div class="steam-nav-bar">
        {links}
    </div>
    """

    st.markdown(
        html,
        unsafe_allow_html=True
    )

def steam_hero(title, subtitle, badge="STEAM ANALYTICS"):
    st.markdown(
        '<div class="steam-hero">'
        f'<div class="steam-hero-badge">{badge}</div>'
        f'<div class="steam-hero-title">{title}</div>'
        f'<div class="steam-hero-subtitle">{subtitle}</div>'
        '</div>',
        unsafe_allow_html=True
    )


def section_header(text):
    st.markdown(
        '<div class="section-header">'
        '<span class="section-accent"></span>'
        f'<span class="section-header-text">{text}</span>'
        '</div>',
        unsafe_allow_html=True
    )