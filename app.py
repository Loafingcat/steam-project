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

genre_cols = [c for c in df.columns if c.startswith("genre_")]

if "price" not in df.columns:
    df["price"] = 0
if "is_free" not in df.columns:
    df["is_free"] = 0
if "release_year" not in df.columns:
    df["release_year"] = 0
if "name" not in df.columns:
    df["name"] = "Unknown Game"

params = st.query_params
view = params.get("view", "all")
if isinstance(view, list):
    view = view[0]

filtered_df = df.copy()

top_genre = "Unknown"
top_count = 0

if genre_cols:
    genre_sum = df[genre_cols].sum().sort_values(ascending=False)
    top_col = genre_sum.index[0]
    top_genre = top_col.replace("genre_", "").replace("_", " ").title()
    top_count = int(max(genre_sum.iloc[0], 0))

view_title = "스팀의<br>게임데이터 분석"
view_subtitle = "Steam 게임 시장 데이터를 기반으로 장르 · 가격 · 흥행 가능성을 분석합니다."
view_badge = "STEAM ANALYTICS"
feature_title = top_genre
feature_desc = f"현재 데이터에서 가장 많이 등장한 장르는 {top_genre}입니다. 총 {top_count:,}개의 게임이 이 장르에 포함되어 있습니다."

if view == "free":
    filtered_df = df[df["is_free"] == 1].copy()
    view_title = "무료 게임<br>데이터 분석"
    view_subtitle = "무료 게임만 모아 시장 흐름과 장르 분포를 확인합니다."
    view_badge = "FREE TO PLAY"
    feature_title = "Free Games"
    feature_desc = f"전체 데이터 중 무료 게임은 총 {len(filtered_df):,}개입니다."

elif view == "paid":
    filtered_df = df[df["is_free"] == 0].copy()
    view_title = "유료 게임<br>데이터 분석"
    view_subtitle = "유료 게임만 모아 가격과 장르 흐름을 확인합니다."
    view_badge = "PAID GAMES"
    feature_title = "Paid Games"
    feature_desc = f"전체 데이터 중 유료 게임은 총 {len(filtered_df):,}개입니다."

elif view == "genre":
    view_title = "인기 장르<br>트렌드 분석"
    view_subtitle = "데이터에서 가장 많이 등장한 장르와 장르별 흐름을 확인합니다."
    view_badge = "POPULAR GENRES"
    feature_title = top_genre
    feature_desc = f"가장 많이 등장한 장르는 {top_genre}이며, 총 {top_count:,}개 게임이 포함되어 있습니다."

elif view == "new":
    max_year = int(df["release_year"].max())
    filtered_df = df[df["release_year"] == max_year].copy()
    view_title = "신작 게임<br>데이터 분석"
    view_subtitle = f"{max_year}년에 출시된 게임만 모아 최신 흐름을 확인합니다."
    view_badge = "NEW RELEASES"
    feature_title = f"{max_year} New Releases"
    feature_desc = f"{max_year}년에 출시된 게임은 총 {len(filtered_df):,}개입니다."

current_count = len(filtered_df)
current_avg_price = (
    filtered_df[filtered_df["price"] > 0]["price"].mean()
    if len(filtered_df[filtered_df["price"] > 0]) > 0
    else 0
)
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
    hero_bg = f'''
    background-image:
        linear-gradient(90deg, rgba(3,8,18,0.92), rgba(3,8,18,0.48), rgba(3,8,18,0.10)),
        url("data:image/png;base64,{banner_base64}");
    '''
else:
    hero_bg = '''
    background-image:
        linear-gradient(90deg, rgba(3,8,18,0.95), rgba(3,8,18,0.45)),
        radial-gradient(circle at 75% 30%, rgba(255,110,40,0.48), transparent 26%),
        radial-gradient(circle at 20% 80%, rgba(102,192,244,0.22), transparent 25%),
        linear-gradient(135deg, #07111f, #143b66);
    '''

css = """
<style>
.stApp {
    background:
        radial-gradient(circle at 20% 0%, rgba(42, 71, 94, 0.45), transparent 28%),
        radial-gradient(circle at 80% 10%, rgba(26, 159, 255, 0.18), transparent 24%),
        linear-gradient(180deg, #0e141b 0%, #1b2838 42%, #0b1220 100%) !important;
    color: #c7d5e0;
}

header, footer {
    visibility: hidden;
}

.block-container {
    padding-top: 0rem !important;
    padding-left: 0rem !important;
    padding-right: 0rem !important;
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

.logo:hover {
    color: #66c0f4 !important;
}

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

.menu a:hover {
    color: #1a9fff !important;
}

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

.store-nav a:hover {
    color: #66c0f4 !important;
}

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
    box-sizing: border-box;
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

.hero-buttons a {
    text-decoration: none !important;
}

.hero-btn {
    padding: 14px 24px;
    border-radius: 4px;
    color: white;
    font-weight: 900;
    font-size: 14px;
    font-family: Arial Black, Impact, sans-serif;
    display: inline-block;
}

.hero-btn.green {
    background: linear-gradient(90deg, #75b022, #588a1b);
}

.hero-btn.blue {
    background: linear-gradient(90deg, #1a9fff, #0066cc);
}

.content {
    width: 76%;
    margin: 34px auto 80px auto;
}

.section-title {
    color: white;
    font-size: 23px;
    font-weight: 900;
    margin: 26px 0 16px 0;
    font-family: Arial Black, Impact, sans-serif;
}

.feature-grid {
    display: grid;
    grid-template-columns: 1.45fr 1fr;
    gap: 18px;
}

.feature-main {
    min-height: 350px;
    background:
        linear-gradient(90deg, rgba(0,0,0,0.82), rgba(0,0,0,0.18)),
        radial-gradient(circle at 70% 30%, rgba(102,192,244,0.20), transparent 30%),
        linear-gradient(135deg, #0d141f, #213a55);
    border: 1px solid rgba(102,192,244,0.18);
    box-shadow: 0 18px 45px rgba(0,0,0,0.45);
    padding: 34px;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
}

.feature-main h2 {
    color: white;
    font-size: 42px;
    margin: 0 0 12px 0;
    font-family: Arial Black, Impact, sans-serif;
}

.feature-main p {
    color: #c7d5e0;
    font-size: 17px;
}

.kpi-box {
    background: linear-gradient(180deg, #1f3b57, #152434);
    border: 1px solid rgba(102,192,244,0.22);
    padding: 22px;
}

.kpi-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
}

.kpi {
    background: rgba(0,0,0,0.25);
    border: 1px solid rgba(102,192,244,0.18);
    padding: 18px;
}

.kpi-label {
    color: #8f98a0;
    font-size: 12px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.kpi-value {
    color: white;
    font-size: 27px;
    font-weight: 900;
    margin-top: 6px;
    font-family: Arial Black, Impact, sans-serif;
}

.card-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
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
        <a href="/가격분석" target="_self">PRICE</a>
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
        <div class="badge">{view_badge}</div>
        <div class="hero-title">{view_title}</div>
        <div class="hero-desc">{view_subtitle}</div>
        <div class="hero-sample">대표 데이터: {sample_text}</div>
        <div class="hero-buttons">
            <a href="/장르분석" target="_self"><div class="hero-btn green">대시보드 탐색</div></a>
            <a href="/가격분석" target="_self"><div class="hero-btn blue">가격 트렌드 보기</div></a>
        </div>
    </div>
</div>

<div class="content">
    <div class="section-title">Featured & Recommended</div>

    <div class="feature-grid">
        <div class="feature-main">
            <h2>{feature_title}</h2>
            <p>{feature_desc}</p>
        </div>

        <div class="kpi-box">
            <div class="kpi-grid">
                <div class="kpi">
                    <div class="kpi-label">Selected Games</div>
                    <div class="kpi-value">{current_count:,}</div>
                </div>
                <div class="kpi">
                    <div class="kpi-label">Avg Price</div>
                    <div class="kpi-value">${current_avg_price:.2f}</div>
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
        </div>
    </div>

    <div class="section-title">Special Sections</div>

    <div class="card-grid">
        <a href="/장르분석" target="_self" style="text-decoration:none;">
            <div class="card">
                <h3>📈 Genre Analysis</h3>
                <p>연도별 장르 변화와 인기 장르 순위를 확인합니다.</p>
            </div>
        </a>
        <a href="/가격분석" target="_self" style="text-decoration:none;">
            <div class="card">
                <h3>💰 Price Analysis</h3>
                <p>무료/유료 게임, 가격대, 장르별 평균 가격을 비교합니다.</p>
            </div>
        </a>
        <a href="/예측모델" target="_self" style="text-decoration:none;">
            <div class="card">
                <h3>🏆 Success Prediction</h3>
                <p>팀원이 만든 실시간 예측 기능과 연결될 예정입니다.</p>
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
                <p>자연어로 게임을 검색합니다. ML vs DL 비교.</p>
            </div>
        </a>
    </div>
</div>
""",
    unsafe_allow_html=True
)

st.markdown('<div class="content">', unsafe_allow_html=True)

search = st.text_input(
    "게임 검색",
    placeholder="🔍 Search games...",
    label_visibility="collapsed"
)

if search:
    result = df[df["name"].astype(str).str.contains(search, case=False, na=False)].copy()

    st.subheader("검색 결과")

    if len(result) > 0:
        show_cols = [c for c in ["name", "release_year", "price", "is_free"] if c in result.columns]
        st.dataframe(result[show_cols].head(20), use_container_width=True)
    else:
        st.warning("검색 결과가 없습니다.")

st.markdown("</div>", unsafe_allow_html=True)