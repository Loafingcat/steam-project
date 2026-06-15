import streamlit as st
import pandas as pd
import plotly.express as px
from utils.preprocessing import loaddata
from utils.preprocessing import load_data

df = load_data()
st.set_page_config(
    page_title="Steam Analytics",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

def clean_year(x):
    if pd.isna(x):
        return pd.NA
    try:
        x = int(float(x))
    except:
        return pd.NA
    if x < 0:
        return 2014 + abs(x) - 1
    if x < 100:
        return 2000 + x
    return x

df = loaddata().copy()

if "releaseyear" not in df.columns:
    if "year" in df.columns:
        df["releaseyear"] = df["year"]
    elif "release_year" in df.columns:
        df["releaseyear"] = df["release_year"]

if "releaseyear" in df.columns:
    df["releaseyear_fixed"] = df["releaseyear"].apply(clean_year)
    df = df[df["releaseyear_fixed"].notna()].copy()
    df["releaseyear_fixed"] = df["releaseyear_fixed"].astype(int)
else:
    df["releaseyear_fixed"] = pd.NA

if "price" in df.columns:
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
if "isfree" in df.columns:
    df["isfree"] = pd.to_numeric(df["isfree"], errors="coerce").fillna(0)
if "requiredage" in df.columns:
    df["requiredage"] = pd.to_numeric(df["requiredage"], errors="coerce").fillna(0).astype(int)
elif "required_age" in df.columns:
    df["required_age"] = pd.to_numeric(df["required_age"], errors="coerce").fillna(0).astype(int)

genre_cols = [c for c in df.columns if c.startswith("genre")]
selected_year = "전체"
if "releaseyear_fixed" in df.columns and df["releaseyear_fixed"].notna().any():
    years = sorted(df["releaseyear_fixed"].dropna().unique().tolist())
    selected_year = st.sidebar.selectbox("기준 연도", ["전체"] + years)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #111827 40%, #1b2838 100%);
        color: #e5e7eb;
    }
    [data-testid="stSidebar"] {
        background-color: #0b1220;
    }
    [data-testid="stSidebar"] * {
        color: #dfe7ef;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #66c0f4 !important;
    }
    p, span, label {
        color: #d6dde5 !important;
    }
    [data-testid="metric-container"] {
        background: #1e293b;
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 15px;
    }
    [data-testid="stDataFrame"] {
        border-radius: 15px;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="background:#171a21;padding:28px;border-radius:20px;border:1px solid #2a475e;margin-bottom:24px;">
        <h1 style="color:#66c0f4;margin-bottom:8px;">Steam Analytics Dashboard</h1>
        <p style="font-size:17px;color:#cfd8e3;margin:0;">
            연도, 가격, 연령등급, 장르 분포를 확인하는 대시보드입니다.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

total_games = len(df)
avg_price = df["price"].dropna().mean() if "price" in df.columns else 0
free_count = int(df["isfree"].fillna(0).sum()) if "isfree" in df.columns else 0

age_col = None
if "requiredage" in df.columns:
    age_col = "requiredage"
elif "required_age" in df.columns:
    age_col = "required_age"

unique_age_groups = int(df[age_col].nunique()) if age_col else 0

with c1:
    st.metric("총 게임 수", f"{total_games:,}")
with c2:
    st.metric("평균 가격", f"${avg_price:.2f}")
with c3:
    st.metric("무료 게임 수", f"{free_count:,}")
with c4:
    st.metric("연령등급 수", f"{unique_age_groups:,}")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("전체 장르 출시 수 순위")
    if genre_cols:
        base_df = df
        if selected_year != "전체":
            base_df = df[df["releaseyear_fixed"] == selected_year].copy()

        genre_sum = base_df[genre_cols].sum().sort_values(ascending=False).head(10)
        genre_df = genre_sum.reset_index()
        genre_df.columns = ["장르", "게임 수"]

        if len(genre_df) > 0:
            fig1 = px.bar(
                genre_df,
                x="게임 수",
                y="장르",
                orientation="h",
                title="전체" if selected_year == "전체" else f"{selected_year}년 장르 순위"
            )
            fig1.update_layout(
                template="plotly_dark",
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                font=dict(color="white")
            )
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.warning("표시할 장르 데이터가 없습니다.")
    else:
        st.info("genre 컬럼을 찾지 못했습니다.")

with right:
    st.subheader("연도별 장르 추이")
    if genre_cols and "releaseyear_fixed" in df.columns:
        line_df = df.dropna(subset=["releaseyear_fixed"]).copy()
        line_df["releaseyear_fixed"] = line_df["releaseyear_fixed"].astype(int)

        selected_genre = st.selectbox("장르 선택", genre_cols)

        trend = line_df.groupby("releaseyear_fixed")[genre_cols].sum().reset_index()
        trend["releaseyear_fixed"] = trend["releaseyear_fixed"].astype(str) + "년"

        fig2 = px.line(
            trend,
            x="releaseyear_fixed",
            y=selected_genre,
            markers=True,
            title=f"{selected_genre} 추이"
        )
        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font=dict(color="white")
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("releaseyear 또는 genre 컬럼이 없습니다.")

st.divider()

bottom_left, bottom_right = st.columns(2)

with bottom_left:
    st.subheader("가격 분포")
    if "price" in df.columns:
        fig3 = px.histogram(df, x="price", nbins=50, title="가격 분포")
        fig3.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font=dict(color="white")
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("price 컬럼이 없습니다.")

with bottom_right:
    st.subheader("연령등급별 평균 가격")
    if age_col and "price" in df.columns:
        age_price = (
            df.dropna(subset=[age_col, "price"])
              .groupby(age_col, as_index=False)["price"]
              .mean()
              .sort_values(age_col)
        )
        age_price[age_col] = age_price[age_col].astype(int).astype(str) + "세"

        fig4 = px.bar(
            age_price,
            x=age_col,
            y="price",
            title="연령등급별 평균 가격"
        )
        fig4.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font=dict(color="white")
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("requiredage / required_age 또는 price 컬럼이 없습니다.")

st.divider()
st.subheader("데이터 미리보기")
st.dataframe(df.head(20), use_container_width=True)