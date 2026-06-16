import base64
from pathlib import Path

import streamlit as st
from utils.preprocessing import load_data


st.set_page_config(
    page_title="Steam Analytics",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

HOME_URL = "/Home"

df = load_data()

if "price" not in df.columns:
    df["price"] = 0
if "is_free" not in df.columns:
    df["is_free"] = 0
if "release_year" not in df.columns:
    df["release_year"] = 0
if "name" not in df.columns:
    df["name"] = "Unknown Game"

genre_cols = [c for c in df.columns if c.startswith("genre_")]

params = st.query_params
view = params.get("view", "all")
if isinstance(view, list):
    view = view[0]

filtered_df = df.copy()

if view == "free":
    filtered_df = df[df["is_free"] == 1].copy()
elif view == "paid":
    filtered_df = df[df["is_free"] == 0].copy()
elif view == "new":
    max_year = int(df["release_year"].max())
    filtered_df = df[df["release_year"] == max_year].copy()

top_genre = "Unknown"
top_count = 0

if genre_cols:
    genre_sum = df[genre_cols].sum().sort_values(ascending=False)
    top_col = genre_sum.index[0]
    top_genre = top_col.replace("genre_", "").replace("_", " ").title()
    top_count = int(max(genre_sum.iloc[0], 0))

current_count = len(filtered_df)
current_free = int(filtered_df["is_free"].sum()) if len(filtered_df) > 0 else 0
current_paid = current_count - current_free

sample_games = filtered_df["name"].dropna().astype(str).head(5).tolist()
sample_text = " · ".join(sample_games) if sample_games else "대표 게임 데이터가 없습니다."


def image_to_base64(path: str) -> str:
    image_path = Path(path)
    if not image_path.exists():
        return ""
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


banner_base64 = image_to_base64("assets/banner.png")

if banner_base64:
    hero_bg = f"""
    background-image:
        linear-gradient(90deg, rgba(3,8,18,0.92), rgba(3,8,18,0.48), rgba(3,8,18,0.10)),
        url("data:image/png;base64,{banner_base64}");
    """
else:
    hero_bg = """
    background-image:
        linear-gradient(90deg, rgba(3,8,18,0.95), rgba(3,8,18,0.45)),
        radial-gradient(circle at 75% 30%, rgba(255,110,40,0.48), transparent 26%),
        radial-gradient(circle at 20% 80%, rgba(102,192,244,0.22), transparent 25%),
        linear-gradient(135deg, #07111f, #143b66);
    """


featured_games = [
    {"title": "Limbus Company", "review": "매우 긍정적", "rank": "내 지역 순위: 10", "price": "무료", "image": "assets/featured_1.jpg"},
    {"title": "붉은사막", "review": "매우 긍정적", "rank": "내 지역 순위: 13", "price": "₩ 79,800", "image": "assets/featured_2.jpg"},
    {"title": "GrandChase", "review": "복합적", "rank": "내 지역 순위: 20", "price": "무료", "image": "assets/featured_3.jpg"},
    {"title": "FINAL FANTASY VII REMAKE", "review": "매우 긍정적", "rank": "내 지역 순위: 23", "price": "₩ 17,430", "image": "assets/featured_4.jpg"},
    {"title": "클레르 옵스퀴르: 33 원정대", "review": "압도적으로 긍정적", "rank": "내 지역 순위: 30", "price": "₩ 43,840", "image": "assets/featured_5.jpg"},
]

sale_games = [
    {"image": "assets/sale_1.png", "tag": "MIDWEEK DEAL", "discount": "-25%", "price": "₩ 20,250"},
    {"image": "assets/sale_2.png", "tag": "MIDWEEK DEAL", "discount": "-60%", "price": "₩ 8,600"},
    {"image": "assets/sale_3.png", "tag": "TODAY'S DEAL", "discount": "-50%", "price": "₩ 12,000"},
    {"image": "assets/sale_4.png", "tag": "TODAY'S DEAL", "discount": "-20%", "price": "₩ 43,840"},
    {"image": "assets/sale_5.png", "tag": "SPECIAL", "discount": "-65%", "price": "₩ 17,430"},
    {"image": "assets/sale_6.png", "tag": "SPECIAL", "discount": "-30%", "price": "₩ 18,900"},
    {"image": "assets/sale_7.png", "tag": "EVENT", "discount": "-50%", "price": "무료"},
    {"image": "assets/sale_8.png", "tag": "EVENT", "discount": "-80%", "price": "₩ 10,600"},
]

if "featured_idx" not in st.session_state:
    st.session_state.featured_idx = 0

idx = st.session_state.featured_idx % len(featured_games)
featured = featured_games[idx]

css = """
<style>
* { box-sizing: border-box; }

.stApp {
    background:
        radial-gradient(circle at 20% 0%, rgba(42, 71, 94, 0.45), transparent 28%),
        radial-gradient(circle at 80% 10%, rgba(26, 159, 255, 0.18), transparent 24%),
        linear-gradient(180deg, #0e141b 0%, #1b2838 42%, #0b1220 100%) !important;
    color: #c7d5e0;
}

header, footer { visibility: hidden; }

[data-testid="stSidebar"] { display: none !important; }

.block-container {
    padding: 0rem !important;
    max-width: 100% !important;
}

.steam-top {
    background: #171a21;
    height: 72px;
    display: flex;
    align-items: center;
    padding: 0 16%;
    gap: 42px;
}

.logo {
    font-size: 30px;
    font-weight: 900;
    color: #dcdedf !important;
    letter-spacing: 4px;
    font-family: Arial Black, Impact, sans-serif;
    text-decoration: none !important;
}

.logo:hover { color: #66c0f4 !important; }

.menu {
    display: flex;
    align-items: center;
    gap: 26px;
}

.menu a {
    color: #dcdedf !important;
    font-weight: 900;
    font-size: 15px;
    letter-spacing: 0.4px;
    font-family: Arial Black, Impact, sans-serif;
    text-decoration: none !important;
}

.menu a:hover { color: #1a9fff !important; }

.menu .active {
    color: #1a9fff !important;
    border-bottom: 2px solid #1a9fff;
    padding-bottom: 5px;
}

.store-nav {
    background: linear-gradient(90deg, #386fa8, #1b4f8f, #0e2f5a);
    height: 48px;
    width: 86%;
    margin: 0 auto;
    display: flex;
    align-items: center;
    padding: 0 18px;
    gap: 22px;
    box-shadow: 0 0 25px rgba(0,0,0,0.45);
}

.store-nav a {
    color: white !important;
    font-size: 13px;
    font-weight: 900;
    text-decoration: none !important;
    font-family: Arial Black, Impact, sans-serif;
}

.store-nav a:hover,
.store-nav .selected {
    color: #66c0f4 !important;
    border-bottom: 2px solid #66c0f4;
}

.hero {
    width: 86%;
    margin: 0 auto;
    min-height: 570px;
    __HERO_BG__
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    border: 1px solid rgba(102,192,244,0.35);
    padding: 64px;
    position: relative;
    overflow: hidden;
}

.hero::after {
    content: "";
    position: absolute;
    inset: 0;
    background:
        linear-gradient(180deg, rgba(0,0,0,0.08), rgba(0,0,0,0.35)),
        radial-gradient(circle at 20% 50%, rgba(0,0,0,0.45), transparent 35%);
    pointer-events: none;
}

.hero-inner {
    position: relative;
    z-index: 2;
    max-width: 760px;
}

.badge {
    display: inline-block;
    background: rgba(0, 153, 255, 0.12);
    border: 1px solid #1a9fff;
    color: #1a9fff;
    padding: 8px 16px;
    font-size: 12px;
    font-weight: 900;
    letter-spacing: 2px;
    border-radius: 3px;
    margin-bottom: 24px;
    font-family: Arial Black, Impact, sans-serif;
}

.hero-title {
    color: white;
    font-size: 68px;
    font-weight: 1000;
    line-height: 1.08;
    margin-bottom: 22px;
    font-family: Arial Black, Impact, sans-serif;
    text-shadow:
        0 0 10px rgba(0,0,0,1),
        0 0 24px rgba(0,0,0,0.95),
        0 0 48px rgba(0,0,0,0.85);
}

.hero-desc {
    color: #e6f3ff;
    font-size: 20px;
    line-height: 1.7;
    max-width: 760px;
    text-shadow: 0 2px 12px rgba(0,0,0,0.95);
}

.hero-sample {
    color: #8cbdd8;
    margin-top: 20px;
    font-size: 14px;
    text-shadow: 0 2px 10px rgba(0,0,0,0.9);
}

.hero-buttons {
    display: flex;
    gap: 14px;
    margin-top: 32px;
}

.hero-buttons a { text-decoration: none !important; }

.hero-btn {
    padding: 14px 24px;
    border-radius: 4px;
    color: white;
    font-weight: 900;
    font-size: 14px;
    font-family: Arial Black, Impact, sans-serif;
    display: inline-block;
}

.hero-btn.green { background: linear-gradient(90deg, #75b022, #588a1b); }
.hero-btn.blue { background: linear-gradient(90deg, #1a9fff, #0066cc); }

.content {
    width: 86%;
    margin: 34px auto 80px auto;
}

.section-title-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 26px 0 14px 0;
}

.section-title {
    color: white;
    font-size: 24px;
    font-weight: 900;
    font-family: Arial Black, Impact, sans-serif;
}

.see-more {
    background: #dcdedf;
    color: #111 !important;
    padding: 7px 18px;
    border-radius: 3px;
    font-size: 13px;
    font-weight: 900;
    text-decoration: none !important;
}

.featured-image-big-wrap {
    width: 100%;
    background: #16202d;
    border: 1px solid #2a3f5a;
    box-shadow: 0 18px 45px rgba(0,0,0,0.5);
    overflow: hidden;
}

.featured-image-big-wrap img {
    height: 560px !important;
    object-fit: cover !important;
}

.dots {
    text-align: center;
    margin-top: 11px;
}

.dot {
    display: inline-block;
    width: 16px;
    height: 7px;
    background: #4d5b66;
    border-radius: 6px;
    margin: 0 3px;
}

.active-dot { background: white; }

.carousel-control-space {
    margin-top: 8px;
    margin-bottom: 34px;
}

div[data-testid="stButton"] button {
    background: #1b4f8f !important;
    color: white !important;
    border: 1px solid #66c0f4 !important;
    border-radius: 3px !important;
    font-weight: 900 !important;
    min-width: 42px !important;
    height: 32px !important;
}

.sale-grid {
    background: linear-gradient(180deg, #1f3b57, #152434);
    border: 1px solid #2a3f5a;
    padding: 18px;
}

.sale-card {
    background: #101923;
    border: 1px solid #2a3f5a;
    overflow: hidden;
    min-height: 230px;
}

.sale-img-wrap img {
    height: 170px !important;
    object-fit: cover !important;
}

.sale-info {
    background: #0e1824;
    display: flex;
    justify-content: flex-end;
    align-items: center;
}

.sale-tag {
    color: #c7d5e0;
    font-size: 12px;
    font-weight: 900;
    padding: 8px 10px;
    margin-right: auto;
}

.discount {
    background: #a4d007;
    color: black;
    font-weight: 900;
    padding: 8px 10px;
}

.sale-price {
    color: white;
    font-weight: 900;
    padding: 8px 10px;
}

.kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-top: 28px;
}

.kpi {
    background: linear-gradient(180deg, #16202d, #101923);
    border: 1px solid rgba(102,192,244,0.2);
    padding: 20px;
}

.kpi-label {
    color: #8f98a0;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.kpi-value {
    color: white;
    font-size: 26px;
    font-weight: 900;
    margin-top: 6px;
    font-family: Arial Black, Impact, sans-serif;
}

.card-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
}

.card {
    background: linear-gradient(180deg, #16202d, #101923);
    border: 1px solid rgba(102,192,244,0.2);
    padding: 24px;
    min-height: 160px;
    transition: 0.15s;
}

.card:hover {
    transform: translateY(-4px);
    border-color: #66c0f4;
    box-shadow: 0 0 28px rgba(102,192,244,0.18);
}

.card h3 {
    color: #66c0f4;
    margin-top: 0;
    font-family: Arial Black, Impact, sans-serif;
}

.card p {
    color: #c7d5e0;
    line-height: 1.6;
}

@media (max-width: 1100px) {
    .kpi-row,
    .card-grid {
        grid-template-columns: 1fr;
    }

    .hero-title { font-size: 46px; }

    .steam-top { padding: 0 4%; }

    .menu { gap: 12px; }
}
</style>
"""

css = css.replace("__HERO_BG__", hero_bg)
st.markdown(css, unsafe_allow_html=True)

st.markdown(
    f"""
<div class="steam-top">
    <a href="{HOME_URL}" target="_self" class="logo">● STEAM</a>
    <div class="menu">
        <a class="active" href="{HOME_URL}" target="_self">STORE</a>
        <a href="/장르분석" target="_self">GENRE</a>
        <a href="/예측모델" target="_self">PREDICTION</a>
        <a href="/숨은명작" target="_self">HIDDEN GEM</a>
        <a href="/게임검색" target="_self">SEARCH</a>
    </div>
</div>

<div class="store-nav">
    <a class="{'selected' if view == 'all' else ''}" href="{HOME_URL}?view=all" target="_self">전체 게임</a>
    <a class="{'selected' if view == 'free' else ''}" href="{HOME_URL}?view=free" target="_self">무료 게임</a>
    <a class="{'selected' if view == 'paid' else ''}" href="{HOME_URL}?view=paid" target="_self">유료 게임</a>
    <a class="{'selected' if view == 'genre' else ''}" href="{HOME_URL}?view=genre" target="_self">인기 장르</a>
    <a class="{'selected' if view == 'new' else ''}" href="{HOME_URL}?view=new" target="_self">신작 게임</a>
</div>

<div class="hero">
    <div class="hero-inner">
        <div class="badge">STEAM ANALYTICS</div>
        <div class="hero-title">스팀의<br>게임데이터 분석</div>
        <div class="hero-desc">Steam 게임 시장 데이터를 기반으로 장르 · 흥행 가능성 · 숨은 명작 흐름을 분석합니다.</div>
        <div class="hero-sample">대표 데이터: {sample_text}</div>
        <div class="hero-buttons">
            <a href="/장르분석" target="_self"><div class="hero-btn green">장르 분석 보기</div></a>
            <a href="/숨은명작" target="_self"><div class="hero-btn blue">숨은 명작 탐지</div></a>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True
)

st.markdown('<div class="content">', unsafe_allow_html=True)

st.markdown(
    """
<div class="section-title-row">
    <div class="section-title">Featured & Recommended</div>
    <a class="see-more" href="/숨은명작" target="_self">더 보기</a>
</div>
""",
    unsafe_allow_html=True
)

st.markdown('<div class="featured-image-big-wrap">', unsafe_allow_html=True)

if Path(featured["image"]).exists():
    st.image(featured["image"], use_container_width=True)
else:
    st.warning(f"이미지를 찾지 못했습니다: {featured['image']}")

st.markdown("</div>", unsafe_allow_html=True)

dots = ""
for i in range(len(featured_games)):
    dot_class = "dot active-dot" if i == idx else "dot"
    dots += f'<span class="{dot_class}"></span>'

st.markdown(f'<div class="dots">{dots}</div>', unsafe_allow_html=True)

st.markdown('<div class="carousel-control-space">', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns([4, 0.35, 0.9, 0.35, 4])

with c2:
    if st.button("◀", key="featured_prev"):
        st.session_state.featured_idx = (idx - 1) % len(featured_games)
        st.rerun()

with c4:
    if st.button("▶", key="featured_next"):
        st.session_state.featured_idx = (idx + 1) % len(featured_games)
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
<div class="section-title-row">
    <div class="section-title">Discounts & Events</div>
    <a class="see-more" href="/게임검색" target="_self">더 보기</a>
</div>
""",
    unsafe_allow_html=True
)

st.markdown('<div class="sale-grid">', unsafe_allow_html=True)

sale_cols = st.columns(4)
for i, game in enumerate(sale_games):
    with sale_cols[i % 4]:
        st.markdown('<div class="sale-card">', unsafe_allow_html=True)
        st.markdown('<div class="sale-img-wrap">', unsafe_allow_html=True)
        if Path(game["image"]).exists():
            st.image(game["image"], use_container_width=True)
        else:
            st.warning(f"이미지 없음: {game['image']}")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
<div class="sale-info">
    <span class="sale-tag">{game["tag"]}</span>
    <span class="discount">{game["discount"]}</span>
    <span class="sale-price">{game["price"]}</span>
</div>
""",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    f"""
<div class="section-title-row">
    <div class="section-title">데이터 요약</div>
</div>

<div class="kpi-row">
    <div class="kpi">
        <div class="kpi-label">Selected Games</div>
        <div class="kpi-value">{current_count:,}</div>
    </div>
    <div class="kpi">
        <div class="kpi-label">Top Genre</div>
        <div class="kpi-value">{top_genre}</div>
    </div>
    <div class="kpi">
        <div class="kpi-label">Free Games</div>
        <div class="kpi-value">{current_free:,}</div>
    </div>
    <div class="kpi">
        <div class="kpi-label">Paid Games</div>
        <div class="kpi-value">{current_paid:,}</div>
    </div>
</div>

<div class="section-title-row">
    <div class="section-title">Special Sections</div>
</div>

<div class="card-grid">
    <a href="/장르분석" target="_self" style="text-decoration:none;">
        <div class="card">
            <h3>📈 Genre Analysis</h3>
            <p>연도별 장르 변화와 인기 장르 순위를 확인합니다.</p>
        </div>
    </a>
    <a href="/예측모델" target="_self" style="text-decoration:none;">
        <div class="card">
            <h3>🏆 Success Prediction</h3>
            <p>게임의 흥행 가능성을 예측합니다.</p>
        </div>
    </a>
    <a href="/숨은명작" target="_self" style="text-decoration:none;">
        <div class="card">
            <h3>💎 Hidden Gem</h3>
            <p>숨은 명작 게임을 탐지하고 추천합니다.</p>
        </div>
    </a>
    <a href="/게임검색" target="_self" style="text-decoration:none;">
        <div class="card">
            <h3>🔍 Smart Search</h3>
            <p>자연어 기반으로 원하는 게임을 검색합니다.</p>
        </div>
    </a>
</div>
""",
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)