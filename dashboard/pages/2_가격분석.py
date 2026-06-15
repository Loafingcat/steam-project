import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils.preprocessing import load_data
from style import steam_theme, steam_nav, steam_hero, section_header

st.set_page_config(page_title="가격 분석 | Steam Analytics", page_icon="💰", layout="wide")
steam_theme()
steam_nav(active="가격")
steam_hero(
    title="Price Analysis",
    subtitle="게임 가격 · 장르 · 연령 등급 · IP 유형 세부 분석",
    badge="PRICE TRENDS"
)

df = load_data()

# ── 파생 컬럼 준비 ──
genre_cols = [c for c in df.columns if c.startswith("genre_")]
label_map  = {c: c.replace("genre_", "").replace("_", " ").title() for c in genre_cols}

# IP 유형: developers_game_count 기반으로 구분
# 1개 게임만 만든 개발사 → 신규 IP, 그 외 → 기존 IP
if "developers_game_count" in df.columns:
    df["ip_type"] = df["developers_game_count"].apply(
        lambda x: "신규 IP" if x <= 1 else "기존 IP"
    )
else:
    df["ip_type"] = "알 수 없음"

# 연령 등급: required_age 기반
def age_label(age):
    if age == 0:   return "전체 이용가"
    elif age <= 12: return "12세 이상"
    elif age <= 15: return "15세 이상"
    elif age <= 18: return "18세 이상 (성인)"
    else:           return "기타"

if "required_age" in df.columns:
    df["age_group"] = df["required_age"].apply(age_label)
else:
    df["age_group"] = "알 수 없음"

st.write("")

# ── KPI ──
free_count   = int(df["is_free"].sum())
paid_count   = len(df) - free_count
avg_price    = df[df["price"] > 0]["price"].mean()
median_price = df[df["price"] > 0]["price"].median()

c1, c2, c3, c4 = st.columns(4)
c1.metric("평균 가격 (유료)", f"${avg_price:.2f}")
c2.metric("중앙값 가격",      f"${median_price:.2f}")
c3.metric("무료 게임",        f"{free_count:,}")
c4.metric("유료 게임",        f"{paid_count:,}")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊  가격 분포",
    "🎮  장르 × 가격",
    "🔞  연령 등급 분석",
    "🏷️  IP 유형 분석",
])

PLOT_STYLE = dict(
    template="plotly_dark",
    paper_bgcolor="#16202d",
    plot_bgcolor="#16202d",
    font_color="#c6d4df",
)

# ── 탭 1: 가격 분포 ──────────────────────────
with tab1:
    section_header("가격 분포 세부 설정")

    col_ctrl, col_graph = st.columns([1, 3])
    with col_ctrl:
        st.markdown('<p style="color:#8cbdd8;font-size:12px;letter-spacing:1px;margin-bottom:4px;">최대 가격 ($)</p>', unsafe_allow_html=True)
        price_max = st.slider("최대 가격", 1, 100, 60, step=1, label_visibility="collapsed")

        st.markdown('<p style="color:#8cbdd8;font-size:12px;letter-spacing:1px;margin:10px 0 4px 0;">최소 가격 ($)</p>', unsafe_allow_html=True)
        price_min = st.slider("최소 가격", 0, 50, 0, step=1, label_visibility="collapsed")

        st.markdown('<p style="color:#8cbdd8;font-size:12px;letter-spacing:1px;margin:10px 0 4px 0;">구간 수</p>', unsafe_allow_html=True)
        bin_count = st.slider("구간 수", 10, 100, 40, label_visibility="collapsed")

        show_free = st.checkbox("무료 게임 포함", value=False)

    with col_graph:
        mask = (df["price"] >= price_min) & (df["price"] <= price_max)
        if not show_free:
            mask = mask & (df["price"] > 0)
        filtered = df[mask]

        fig = px.histogram(
            filtered, x="price", nbins=bin_count,
            title=f"가격 분포  (${price_min} ~ ${price_max},  {len(filtered):,}개 게임)",
            color_discrete_sequence=["#66c0f4"]
        )
        fig.update_layout(
            **PLOT_STYLE,
            xaxis=dict(gridcolor="#2a3f5a", title="가격 ($)"),
            yaxis=dict(gridcolor="#2a3f5a", title="게임 수"),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # 가격 구간별 통계
    section_header("가격 구간별 게임 수")
    bins   = [0, 5, 10, 20, 30, 50, 100]
    labels = ["$0~5", "$5~10", "$10~20", "$20~30", "$30~50", "$50+"]
    paid_df = df[df["price"] > 0].copy()
    paid_df["price_range"] = pd.cut(paid_df["price"], bins=bins, labels=labels, right=False)
    range_count = paid_df["price_range"].value_counts().reindex(labels).fillna(0)

    fig_range = go.Figure(go.Bar(
        x=range_count.index, y=range_count.values,
        marker_color=["#66c0f4","#5aaddf","#4d9bbf","#3f8aa0","#2a7080","#1a5a6a"],
        text=[f"{int(v):,}" for v in range_count.values],
        textposition="outside", textfont=dict(color="#c6d4df")
    ))
    fig_range.update_layout(
        **PLOT_STYLE,
        xaxis=dict(gridcolor="#2a3f5a", title="가격 구간"),
        yaxis=dict(gridcolor="#2a3f5a", title="게임 수"),
        margin=dict(l=10, r=10, t=30, b=10), height=320
    )
    st.plotly_chart(fig_range, use_container_width=True)

# ── 탭 2: 장르 × 가격 ────────────────────────
with tab2:
    section_header("장르별 평균 / 중앙값 가격")

    col_ctrl2, col_graph2 = st.columns([1, 3])
    with col_ctrl2:
        st.markdown('<p style="color:#8cbdd8;font-size:12px;letter-spacing:1px;margin-bottom:4px;">통계 방식</p>', unsafe_allow_html=True)
        stat_mode = st.radio(
            "통계", ["평균 가격", "중앙값 가격"],
            label_visibility="collapsed"
        )
        st.markdown('<p style="color:#8cbdd8;font-size:12px;letter-spacing:1px;margin:10px 0 4px 0;">무료 게임 제외</p>', unsafe_allow_html=True)
        exclude_free = st.checkbox("무료 제외", value=True, label_visibility="collapsed")

    with col_graph2:
        base = df[df["price"] > 0] if exclude_free else df
        rows = []
        for col in genre_cols:
            subset = base[base[col] == 1]["price"]
            if len(subset) < 10:
                continue
            rows.append({
                "장르": label_map[col],
                "평균 가격": round(subset.mean(), 2),
                "중앙값 가격": round(subset.median(), 2),
                "게임 수": len(subset)
            })
        genre_price_df = pd.DataFrame(rows).sort_values(stat_mode, ascending=True)

        fig2 = go.Figure(go.Bar(
            x=genre_price_df[stat_mode],
            y=genre_price_df["장르"],
            orientation="h",
            marker_color="#66c0f4",
            text=[f"${v}" for v in genre_price_df[stat_mode]],
            textposition="outside", textfont=dict(color="#c6d4df")
        ))
        fig2.update_layout(
            **PLOT_STYLE,
            title=f"장르별 {stat_mode}",
            xaxis=dict(gridcolor="#2a3f5a", title=f"{stat_mode} ($)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=80, t=40, b=10), height=480
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── 탭 3: 연령 등급 분석 ─────────────────────
with tab3:
    section_header("연령 등급별 분석")

    age_order = ["전체 이용가", "12세 이상", "15세 이상", "18세 이상 (성인)", "기타"]
    age_count = df["age_group"].value_counts().reindex(age_order).fillna(0)

    col_pie, col_bar = st.columns(2)

    with col_pie:
        fig_pie = px.pie(
            values=age_count.values,
            names=age_count.index,
            title="연령 등급 비율",
            color_discrete_sequence=["#66c0f4","#4d9bbf","#2a7080","#1a5a6a","#2a475e"],
            hole=0.4
        )
        fig_pie.update_traces(textfont_color="white")
        fig_pie.update_layout(
            **PLOT_STYLE,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(bgcolor="#16202d")
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        # 연령 등급별 평균 가격
        age_price = (
            df[df["price"] > 0]
            .groupby("age_group")["price"]
            .mean()
            .reindex(age_order)
            .dropna()
        )
        fig_age = go.Figure(go.Bar(
            x=age_price.index, y=age_price.values,
            marker_color=["#66c0f4","#4d9bbf","#2a7080","#1a5a6a","#2a475e"],
            text=[f"${v:.2f}" for v in age_price.values],
            textposition="outside", textfont=dict(color="#c6d4df")
        ))
        fig_age.update_layout(
            **PLOT_STYLE,
            title="연령 등급별 평균 가격",
            xaxis=dict(gridcolor="#2a3f5a"),
            yaxis=dict(gridcolor="#2a3f5a", title="평균 가격 ($)"),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_age, use_container_width=True)

    # 연령 등급 × 장르 히트맵
    section_header("연령 등급 × 장르 분포")
    heat_data = []
    for col in genre_cols:
        for age in age_order:
            cnt = len(df[(df[col] == 1) & (df["age_group"] == age)])
            heat_data.append({"장르": label_map[col], "연령 등급": age, "게임 수": cnt})
    heat_df = pd.DataFrame(heat_data).pivot(index="장르", columns="연령 등급", values="게임 수").fillna(0)
    heat_df = heat_df.reindex(columns=[c for c in age_order if c in heat_df.columns])

    fig_heat = px.imshow(
        heat_df,
        color_continuous_scale=[[0,"#16202d"],[0.5,"#2a475e"],[1,"#66c0f4"]],
        title="연령 등급 × 장르 히트맵",
        text_auto=True,
        aspect="auto"
    )
    fig_heat.update_layout(
        **PLOT_STYLE,
        margin=dict(l=10, r=10, t=50, b=10), height=480
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ── 탭 4: IP 유형 분석 ───────────────────────
with tab4:
    section_header("신규 IP vs 기존 IP 분석")

    ip_count = df["ip_type"].value_counts()

    col_l, col_r = st.columns(2)

    with col_l:
        fig_ip_pie = px.pie(
            values=ip_count.values, names=ip_count.index,
            title="IP 유형 비율",
            color_discrete_sequence=["#66c0f4", "#2a475e"],
            hole=0.4
        )
        fig_ip_pie.update_traces(textfont_color="white")
        fig_ip_pie.update_layout(
            **PLOT_STYLE,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(bgcolor="#16202d")
        )
        st.plotly_chart(fig_ip_pie, use_container_width=True)

    with col_r:
        # IP 유형별 평균 가격
        ip_price = df[df["price"] > 0].groupby("ip_type")["price"].mean()
        fig_ip_price = go.Figure(go.Bar(
            x=ip_price.index, y=ip_price.values,
            marker_color=["#66c0f4", "#2a475e"],
            text=[f"${v:.2f}" for v in ip_price.values],
            textposition="outside", textfont=dict(color="#c6d4df")
        ))
        fig_ip_price.update_layout(
            **PLOT_STYLE,
            title="IP 유형별 평균 가격",
            xaxis=dict(gridcolor="#2a3f5a"),
            yaxis=dict(gridcolor="#2a3f5a", title="평균 가격 ($)"),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_ip_price, use_container_width=True)

    # IP × 장르 비교
    section_header("IP 유형 × 장르 비교")
    col_ctrl3, col_graph3 = st.columns([1, 3])
    with col_ctrl3:
        st.markdown('<p style="color:#8cbdd8;font-size:12px;letter-spacing:1px;margin-bottom:4px;">IP 유형 선택</p>', unsafe_allow_html=True)
        ip_filter = st.selectbox(
            "IP", ["전체", "신규 IP", "기존 IP"], label_visibility="collapsed"
        )
    with col_graph3:
        ip_df = df if ip_filter == "전체" else df[df["ip_type"] == ip_filter]
        ip_genre = ip_df[genre_cols].sum().sort_values(ascending=True)
        ip_genre.index = [label_map[c] for c in ip_genre.index]

        fig_ip_genre = go.Figure(go.Bar(
            x=ip_genre.values, y=ip_genre.index,
            orientation="h",
            marker_color="#66c0f4",
            text=[f"{int(v):,}" for v in ip_genre.values],
            textposition="outside", textfont=dict(color="#c6d4df")
        ))
        fig_ip_genre.update_layout(
            **PLOT_STYLE,
            title=f"{ip_filter} — 장르별 게임 수",
            xaxis=dict(gridcolor="#2a3f5a", title="게임 수"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=90, t=40, b=10), height=480
        )
        st.plotly_chart(fig_ip_genre, use_container_width=True)