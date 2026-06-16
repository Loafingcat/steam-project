import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils.preprocessing import load_data
from style import steam_theme, steam_nav, steam_hero, section_header

st.set_page_config(page_title="장르 분석 | Steam Analytics", page_icon="🎮", layout="wide")
steam_theme()
steam_nav(active="장르")
steam_hero(
    title="Genre Analysis",
    subtitle="연도별 게임 장르 트렌드 · 인기 장르 순위",
    badge="GENRE TRENDS"
)

df = load_data()
genre_cols  = [c for c in df.columns if c.startswith("genre_")]
label_map   = {c: c.replace("genre_", "").replace("_", " ").title() for c in genre_cols}

year_min = int(df["release_year"].min())
year_max = int(df["release_year"].max())
# 연도 목록 (정수 변환 보장)
year_list = sorted([int(y) for y in df["release_year"].dropna().unique()])

st.write("")
tab1, tab2 = st.tabs(["📈  연도별 장르 변화", "🏆  전체 장르 순위"])

# ────────────────────────────────────────────
with tab1:
    section_header("연도별 장르 변화 트렌드")
    col_ctrl, col_graph = st.columns([1, 3])

    with col_ctrl:
        st.markdown('<p style="color:#8cbdd8;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">장르 선택</p>', unsafe_allow_html=True)
        selected_col = st.selectbox(
            "장르", genre_cols,
            format_func=lambda x: label_map[x],
            label_visibility="collapsed"
        )
        st.markdown('<p style="color:#8cbdd8;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin:12px 0 4px 0;">연도 범위</p>', unsafe_allow_html=True)
        year_range = st.slider(
            "연도 범위", year_min, year_max, (year_min, year_max),
            step=1, label_visibility="collapsed"
        )

    with col_graph:
        genre_year = df.groupby("release_year")[genre_cols].sum().reset_index()
        genre_year["release_year"] = genre_year["release_year"].astype(int)
        filtered = genre_year[
            (genre_year["release_year"] >= year_range[0]) &
            (genre_year["release_year"] <= year_range[1])
        ]
        fig = px.area(
            filtered, x="release_year", y=selected_col,
            markers=True,
            title=f"{label_map[selected_col]} — 연도별 출시 수",
            color_discrete_sequence=["#66c0f4"]
        )
        fig.update_traces(
            line=dict(width=2.5),
            marker=dict(size=7, color="#66c0f4"),
            fillcolor="rgba(102,192,244,0.15)"
        )
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#16202d", plot_bgcolor="#16202d",
            font_color="#c6d4df", title_font_size=15,
            xaxis=dict(gridcolor="#2a3f5a", title="출시 연도", dtick=1),
            yaxis=dict(gridcolor="#2a3f5a", title="게임 수"),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    section_header("연도별 전체 장르 비교")
    genre_year_all = df.groupby("release_year")[genre_cols].sum().reset_index()
    genre_year_all["release_year"] = genre_year_all["release_year"].astype(int)
    fig2 = px.line(
        genre_year_all, x="release_year", y=genre_cols,
        title="전체 장르 동시 비교",
        labels={"value": "게임 수", "release_year": "출시 연도", "variable": "장르"}
    )
    fig2.for_each_trace(lambda t: t.update(name=label_map.get(t.name, t.name)))
    fig2.update_layout(
        template="plotly_dark", paper_bgcolor="#16202d", plot_bgcolor="#16202d",
        font_color="#c6d4df",
        xaxis=dict(gridcolor="#2a3f5a", dtick=1),
        yaxis=dict(gridcolor="#2a3f5a"),
        legend=dict(bgcolor="#16202d", bordercolor="#2a3f5a"),
        margin=dict(l=10, r=10, t=50, b=10), height=420
    )
    st.plotly_chart(fig2, use_container_width=True)

# ────────────────────────────────────────────
with tab2:
    section_header("전체 장르 출시 수 순위")

    col_ctrl2, _ = st.columns([1, 3])
    with col_ctrl2:
        st.markdown('<p style="color:#8cbdd8;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">기준 연도</p>', unsafe_allow_html=True)
        # 정수 변환된 연도 목록 사용 → 소수점 없음
        year_options = ["전체"] + [str(y) for y in sorted(year_list, reverse=True)]
        selected_year = st.selectbox(
            "연도", year_options, label_visibility="collapsed"
        )

    if selected_year == "전체":
        target = df
        title_str = "전체 기간"
    else:
        target = df[df["release_year"].astype(int) == int(selected_year)]
        title_str = f"{selected_year}년"

    top = target[genre_cols].sum().sort_values(ascending=True)
    top.index = [label_map[c] for c in top.index]

    n = len(top)
    colors = [f"rgba(102,192,244,{0.35 + 0.65*(i/max(n-1,1)):.2f})" for i in range(n)]

    fig3 = go.Figure(go.Bar(
        x=top.values, y=top.index,
        orientation="h",
        marker_color=colors,
        text=[f"{int(v):,}" for v in top.values],
        textposition="outside",
        textfont=dict(color="#c6d4df", size=11)
    ))
    fig3.update_layout(
        template="plotly_dark", paper_bgcolor="#16202d", plot_bgcolor="#16202d",
        font_color="#c6d4df",
        title=f"{title_str} 장르별 출시 수",
        xaxis=dict(gridcolor="#2a3f5a", title="게임 수"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=90, t=40, b=10), height=520
    )
    st.plotly_chart(fig3, use_container_width=True)