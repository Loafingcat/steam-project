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
    view_subtitle = "가장 많이 등장한 장르와 장르별 흐름을 확인합니다."
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
    if len(filtered_df[filtered_df["price"] > 0]) > 0 else 0
)
current_free = int(filtered_df["is_free"].sum()) if len(filtered_df) > 0 else 0
current_paid = current_count - current_free

sample_games = filtered_df["name"].dropna().astype(str).head(5).tolist()
sample_text = " · ".join(sample_games) if sample_games else "대표 게임 데이터가 없습니다."


def image_to_base64(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return base64.b64encode(p.read_bytes()).decode("utf-8")


banner_b64 = image_to_base64("assets/banner.png")
if banner_b64:
    hero_bg = (
        f'background-image: linear-gradient(90deg, rgba(3,8,18,0.93),'
        f' rgba(3,8,18,0.52), rgba(3,8,18,0.10)),'
        f' url("data:image/png;base64,{banner_b64}");'
    )
else:
    hero_bg = (
        "background-image: linear-gradient(90deg, rgba(3,8,18,0.95),"
        " rgba(3,8,18,0.45)),"
        " radial-gradient(circle at 75% 30%, rgba(255,110,40,0.48), transparent 26%),"
        " linear-gradient(135deg, #07111f, #143b66);"
    )

# ── 필터 active 클래스 ──
def fa(v):
    return "filter-active" if view == v else ""

css = """
<style>
* { box-sizing: border-box; }
.stApp {
    background:
        radial-gradient(circle at 18% 0%, rgba(42,71,94,0.45), transparent 28%),
        radial-gradient(circle at 82% 8%, rgba(26,159,255,0.18), transparent 25%),
        linear-gradient(180deg, #0e141b 0%, #1b2838 42%, #0b1220 100%) !important;
    color: #c7d5e0;
}
header, footer { visibility: hidden; }
.block-container {
    padding-top: 0rem !important;
    padding-left: 0rem !important;
    padding-right: 0rem !important;
    max-width: 100% !important;
}

/* ── 상단 검정 바 ── */
.top-bar {
    background: #171a21;
    height: 62px;
    display: flex;
    align-items: center;
    padding: 0 7%;
    gap: 0;
}
.top-logo {
    font-size: 26px;
    font-weight: 900;
    color: #c6d4df;
    letter-spacing: 4px;
    font-family: Arial Black, Impact, sans-serif;
    padding-right: 28px;
    border-right: 1px solid #2a3f5a;
    margin-right: 4px;
    flex-shrink: 0;
}
.top-links {
    display: flex;
    align-items: center;
    height: 100%;
    flex: 1;
}
.top-link {
    color: #c6d4df;
    font-size: 13px;
    font-weight: 700;
    font-family: Arial, sans-serif;
    padding: 0 16px;
    height: 100%;
    display: flex;
    align-items: center;
    text-decoration: none !important;
    border-bottom: 3px solid transparent;
    transition: color 0.12s, border-color 0.12s;
    letter-spacing: 0.3px;
}
.top-link:hover { color: #fff; border-bottom-color: #4a90d9; }
.top-link.tl-active { color: #fff; border-bottom-color: #4a90d9; }
.top-search-wrap {
    margin-left: auto;
    display: flex;
    align-items: center;
    background: #316282;
    border: 1px solid #4f94bc;
    height: 34px;
    width: 260px;
    flex-shrink: 0;
}
.top-search-wrap input {
    background: transparent;
    border: none;
    outline: none;
    color: #c6d4df;
    font-size: 13px;
    padding: 0 10px;
    width: 100%;
    height: 100%;
}
.top-search-icon {
    color: #8cbdd8;
    padding: 0 10px;
    font-size: 14px;
    flex-shrink: 0;
}

/* ── 파란 필터 바 ── */
.filter-bar {
    background: linear-gradient(90deg, #2b5278, #1b4069, #142e4e);
    height: 44px;
    display: flex;
    align-items: center;
    padding: 0 7%;
    gap: 0;
    border-bottom: 1px solid #0d1f30;
}
.filter-link {
    color: #b8d2e8;
    font-size: 13px;
    font-weight: 700;
    font-family: Arial, sans-serif;
    padding: 0 18px;
    height: 100%;
    display: flex;
    align-items: center;
    text-decoration: none !important;
    border-bottom: 3px solid transparent;
    transition: color 0.12s, border-color 0.12s;
    letter-spacing: 0.3px;
}
.filter-link:hover { color: #fff; }
.filter-active { color: #fff !important; border-bottom-color: #66c0f4 !important; }

/* ── 히어로 ── */
.hero {
    width: 100%;
    min-height: 490px;
    __HERO_BG__
    background-size: cover;
    background-position: center 40%;
    background-repeat: no-repeat;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "";
    position: absolute;
    inset: 0;
    background:
        linear-gradient(90deg, rgba(6,12,22,0.88) 0%, rgba(6,12,22,0.42) 55%, rgba(6,12,22,0.08) 100%),
        linear-gradient(180deg, rgba(0,0,0,0.05) 60%, rgba(0,0,0,0.30) 100%);
    pointer-events: none;
}
.hero-inner {
    position: relative;
    z-index: 2;
    padding: 60px 8%;
    max-width: 700px;
}
.badge {
    display: inline-block;
    background: rgba(0,153,255,0.12);
    border: 1px solid #1a9fff;
    color: #1a9fff;
    padding: 6px 14px;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: 2px;
    border-radius: 2px;
    margin-bottom: 20px;
    font-family: Arial Black, Impact, sans-serif;
}
.hero-title {
    color: white;
    font-size: 62px;
    font-weight: 1000;
    line-height: 1.08;
    margin-bottom: 18px;
    font-family: Arial Black, Impact, sans-serif;
    text-shadow: 0 0 8px rgba(0,0,0,1), 0 0 22px rgba(0,0,0,0.95);
}
.hero-desc {
    color: #daeeff;
    font-size: 18px;
    line-height: 1.65;
    text-shadow: 0 2px 10px rgba(0,0,0,0.95);
    max-width: 560px;
}
.hero-sample {
    color: #7ab0cc;
    margin-top: 16px;
    font-size: 13px;
    text-shadow: 0 2px 8px rgba(0,0,0,0.9);
}
.hero-buttons {
    display: flex;
    gap: 12px;
    margin-top: 28px;
}
.hero-btn {
    padding: 12px 22px;
    border-radius: 3px;
    color: white;
    font-weight: 900;
    font-size: 13px;
    font-family: Arial Black, Impact, sans-serif;
    display: inline-block;
    cursor: pointer;
}
.hero-btn.green { background: linear-gradient(90deg, #75b022, #4e7a12); }
.hero-btn.blue  { background: linear-gradient(90deg, #1a9fff, #0060c0); }

/* ── 콘텐츠 영역 ── */
.content {
    width: 86%;
    margin: 32px auto 80px auto;
}
.section-title {
    color: #c6d4df;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin: 28px 0 14px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #2a3f5a;
}

/* ── Featured ── */
.featured-wrap {
    display: grid;
    grid-template-columns: 1fr 340px;
    gap: 0;
    background: #16202d;
    border: 1px solid #2a3f5a;
}
.featured-main {
    min-height: 340px;
    background:
        linear-gradient(90deg, rgba(0,0,0,0.70) 0%, rgba(0,0,0,0.10) 100%),
        radial-gradient(circle at 60% 35%, rgba(102,192,244,0.28), transparent 40%),
        linear-gradient(135deg, #182c43, #070d15);
    padding: 32px;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
}
.featured-main h2 {
    color: white;
    font-size: 40px;
    margin: 0 0 10px 0;
    font-family: Arial Black, Impact, sans-serif;
}
.featured-main p { color: #c7d5e0; font-size: 16px; line-height: 1.6; }
.featured-side {
    background: #1e3248;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.side-label {
    color: #66c0f4;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 4px;
}
.side-thumbs {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
    flex: 1;
}
.side-thumb {
    background:
        radial-gradient(circle at 55% 40%, rgba(102,192,244,0.22), transparent 40%),
        linear-gradient(135deg, #1e3652, #0a1118);
    border: 1px solid #2a3f5a;
    border-radius: 2px;
    min-height: 72px;
}
.side-tag {
    color: #a4d007;
    font-size: 15px;
    font-weight: 900;
    margin-top: 8px;
}

/* ── KPI ── */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-top: 12px;
}
.kpi-card {
    background: #16202d;
    border: 1px solid #2a3f5a;
    padding: 16px 18px;
}
.kpi-label {
    color: #8f98a0;
    font-size: 11px;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.kpi-value {
    color: white;
    font-size: 26px;
    font-weight: 900;
    margin-top: 4px;
    font-family: Arial Black, Impact, sans-serif;
}

/* ── Discounts ── */
.discount-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
}
.discount-card {
    background: linear-gradient(135deg, #1e2d3d, #0e1924);
    border: 1px solid #2a3f5a;
    padding: 18px;
    min-height: 180px;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    position: relative;
    overflow: hidden;
    transition: border-color 0.15s, transform 0.15s;
}
.discount-card:hover {
    border-color: #66c0f4;
    transform: translateY(-3px);
}
.deal-badge {
    position: absolute;
    top: 0; left: 0;
    background: #9b2f7f;
    color: white;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: 1px;
}
.discount-title { color: white; font-size: 18px; font-weight: 900; }
.discount-sub { color: #a4d007; font-size: 13px; font-weight: 700; margin-top: 6px; }

/* ── Nav Cards ── */
.nav-cards {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
}
.nav-card {
    background: linear-gradient(180deg, #16202d, #101923);
    border: 1px solid #2a3f5a;
    padding: 22px;
    min-height: 140px;
    transition: border-color 0.15s, transform 0.15s;
    cursor: pointer;
}
.nav-card:hover {
    border-color: #66c0f4;
    transform: translateY(-3px);
    box-shadow: 0 0 24px rgba(102,192,244,0.16);
}
.nav-card h3 {
    color: #66c0f4;
    margin: 0 0 8px 0;
    font-family: Arial Black, Impact, sans-serif;
    font-size: 16px;
}
.nav-card p { color: #8f98a0; line-height: 1.5; font-size: 14px; margin: 0; }

/* ── Streamlit 버튼 숨기기 (기능용만) ── */
div[data-testid="stButton"] > button {
    display: none !important;
}

/* ── 검색 결과 ── */
.search-result-wrap { margin-top: 24px; }
</style>
"""
css = css.replace("__HERO_BG__", hero_bg)
st.markdown(css, unsafe_allow_html=True)

# ── 상단 검정 메뉴바 (HTML) ──
top_bar_html = f"""
<div class="top-bar">
  <div class="top-logo">● STEAM</div>
  <div class="top-links">
    <a class="top-link tl-active" href="?view=all" target="_self">STORE</a>
    <a class="top-link" href="?_page=genre" target="_self">GENRE</a>
    <a class="top-link" href="?_page=price" target="_self">PRICE</a>
    <a class="top-link" href="?_page=predict" target="_self">PREDICTION</a>
  </div>
  <div class="top-search-wrap">
    <span class="top-search-icon">🔍</span>
    <input type="text" id="top-search-input" placeholder="Search the store" />
  </div>
</div>
"""
st.markdown(top_bar_html, unsafe_allow_html=True)

# ── 상단 메뉴 실제 버튼 (숨겨진 Streamlit 버튼) ──
_c = st.columns([1, 1, 1, 1, 8])
with _c[0]:
    if st.button("__genre__", key="nav_genre"):
        st.switch_page("pages/1_장르분석.py")
with _c[1]:
    if st.button("__price__", key="nav_price"):
        st.switch_page("pages/2_가격분석.py")
with _c[2]:
    if st.button("__predict__", key="nav_predict"):
        st.switch_page("pages/3_예측모델.py")

# ── 파란 필터 바 ──
filter_bar_html = f"""
<div class="filter-bar">
  <a class="filter-link {fa('all')}"    href="?view=all"   target="_self">전체 게임</a>
  <a class="filter-link {fa('free')}"   href="?view=free"  target="_self">무료 게임</a>
  <a class="filter-link {fa('paid')}"   href="?view=paid"  target="_self">유료 게임</a>
  <a class="filter-link {fa('genre')}"  href="?view=genre" target="_self">인기 장르</a>
  <a class="filter-link {fa('new')}"    href="?view=new"   target="_self">신작 게임</a>
</div>
"""
st.markdown(filter_bar_html, unsafe_allow_html=True)

# ── 히어로 배너 ──
hero_html = f"""
<div class="hero">
  <div class="hero-inner">
    <div class="badge">{view_badge}</div>
    <div class="hero-title">{view_title}</div>
    <div class="hero-desc">{view_subtitle}</div>
    <div class="hero-sample">대표 데이터: {sample_text}</div>
    <div class="hero-buttons">
      <div class="hero-btn green" onclick="location.href='?view=genre'">대시보드 탐색</div>
      <div class="hero-btn blue"  onclick="location.href='?view=paid'">가격 트렌드 보기</div>
    </div>
  </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

# ── 콘텐츠 ──
content_html = f"""
<div class="content">
  <div class="section-title">Featured &amp; Recommended</div>
  <div class="featured-wrap">
    <div class="featured-main">
      <h2>{feature_title}</h2>
      <p>{feature_desc}</p>
    </div>
    <div class="featured-side">
      <div class="side-label">Related Categories</div>
      <div class="side-thumbs">
        <div class="side-thumb"></div>
        <div class="side-thumb"></div>
        <div class="side-thumb"></div>
        <div class="side-thumb"></div>
      </div>
      <div class="side-tag">↗ Top Category</div>
    </div>
  </div>

  <div class="kpi-row">
    <div class="kpi-card">
      <div class="kpi-label">Selected Games</div>
      <div class="kpi-value">{current_count:,}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Avg Price</div>
      <div class="kpi-value">${current_avg_price:.2f}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Free Games</div>
      <div class="kpi-value">{current_free:,}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Paid Games</div>
      <div class="kpi-value">{current_paid:,}</div>
    </div>
  </div>

  <div class="section-title">Discounts &amp; Events</div>
  <div class="discount-row">
    <div class="discount-card">
      <div class="deal-badge">DATA DEAL</div>
      <div class="discount-title">무료 게임 인사이트</div>
      <div class="discount-sub">Free to Play 분석</div>
    </div>
    <div class="discount-card">
      <div class="deal-badge">TREND</div>
      <div class="discount-title">인기 장르 흐름</div>
      <div class="discount-sub">Top Genre View</div>
    </div>
    <div class="discount-card">
      <div class="deal-badge">PRICE</div>
      <div class="discount-title">가격대 분석</div>
      <div class="discount-sub">Average Price Insight</div>
    </div>
    <div class="discount-card">
      <div class="deal-badge">PREDICT</div>
      <div class="discount-title">흥행 예측</div>
      <div class="discount-sub">Coming Soon</div>
    </div>
  </div>

  <div class="section-title">Browse by Category</div>
  <div class="nav-cards">
    <div class="nav-card" onclick="location.href='?_nav=genre'">
      <h3>📈 Genre Analysis</h3>
      <p>연도별 장르 변화와 인기 장르 순위를 확인합니다.</p>
    </div>
    <div class="nav-card" onclick="location.href='?_nav=price'">
      <h3>💰 Price Analysis</h3>
      <p>무료/유료 게임, 가격대, 장르별 평균 가격을 비교합니다.</p>
    </div>
    <div class="nav-card" onclick="location.href='?_nav=predict'">
      <h3>🏆 Success Prediction</h3>
      <p>팀원이 만든 실시간 예측 기능과 연결될 예정입니다.</p>
    </div>
  </div>
</div>
"""
st.markdown(content_html, unsafe_allow_html=True)

# ── 페이지 이동 감지 ──
_page = st.query_params.get("_page", "")
_nav  = st.query_params.get("_nav", "")
if _page == "genre" or _nav == "genre":
    st.switch_page("pages/1_장르분석.py")
elif _page == "price" or _nav == "price":
    st.switch_page("pages/2_가격분석.py")
elif _page == "predict" or _nav == "predict":
    st.switch_page("pages/3_예측모델.py")

# ── 실제 검색창 (Streamlit) ──
st.markdown('<div style="width:86%;margin:0 auto;">', unsafe_allow_html=True)
search = st.text_input(
    "게임 검색",
    placeholder="🔍 게임 이름으로 검색...",
    label_visibility="collapsed",
    key="game_search"
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