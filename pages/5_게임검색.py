import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
from search_engine import search_df, route_method, set_dl_model

st.set_page_config(page_title="게임 검색", page_icon="🔍", layout="wide")


@st.cache_resource
def preload_dl():
    try:
        from sentence_transformers import SentenceTransformer
        with open("data/processed/embed_model_name.txt") as f:
            name = f.read().strip()
        return SentenceTransformer(name)
    except Exception:
        return None


_m = preload_dl()
if _m:
    set_dl_model(_m)
    
@st.cache_data
def has_dl():
    return os.path.exists("data/processed/embeddings.npy")


@st.cache_data
def load_comparison():
    try:
        return pd.read_json("data/processed/comparison_search.json")
    except Exception:
        return None
    
    
tab1, tab2 = st.tabs(["🔍 게임 검색", "📊 ML vs DL 성능"])

# ══════════════ 탭 1: 검색 ══════════════
with tab1:
    st.title("🔍 게임 검색")
    st.caption("원하는 분위기를 자유롭게 입력하세요. "
               "짧은 키워드는 단어매칭(ML), 자연어·한국어는 의미검색(DL)으로 자동 처리됩니다.")

    query = st.text_input("어떤 게임을 찾으세요?",
                          placeholder="예: 잔잔한 힐링 게임 / roguelike / 친구랑 할 협동 호러")

    c1, c2, c3 = st.columns(3)
    with c1:
        quality = st.select_slider("결과 우선순위",
                                   ["관련성 우선", "균형", "인기·평점 우선"],
                                   value="균형")
    with c2:
        max_price = st.slider("최대 가격 ($)", 0, 60, 60)
    with c3:
        min_rev = st.select_slider("최소 리뷰 수", [0, 30, 100, 500, 1000], value=30)

    pop_weight = {"관련성 우선": 0.0, "균형": 0.35, "인기·평점 우선": 0.6}[quality]

    if query:
        method, why = route_method(query)
        if method == "dl" and not has_dl():
            st.warning("DL 임베딩 인덱스가 없어 ML로 검색합니다. "
                       "`uv run build_search_index.py`로 DL을 활성화하세요.")
            method = "ml"
            why = "DL 인덱스 없음 → ML 대체"

        badge = "🧠 DL 의미검색" if method == "dl" else "🔧 ML 단어매칭"
        st.markdown(f"**{badge}** 로 검색했습니다 · {why}")

        res = search_df(query, method=method, top_k=12,
                        min_reviews=min_rev,
                        max_price=max_price if max_price < 60 else None,
                        popularity_weight=pop_weight)

        if len(res) == 0:
            st.info("조건에 맞는 결과가 없습니다. 필터를 완화해 보세요.")
        else:
            # 카드형 2열 출력
            cols = st.columns(2)
            for i, (_, r) in enumerate(res.iterrows()):
                with cols[i % 2]:
                    tags = " · ".join(r["tag_list"][:5]) if len(r["tag_list"]) else ""
                    price = "무료" if r["price"] == 0 else f"${r['price']:.2f}"
                    pos = r["positive_ratio"] * 100
                    pos_color = "#66c0f4" if pos >= 80 else ("#a0a0a0" if pos >= 60 else "#c0685a")
                    st.markdown(
                        f"""<div style="background:#1b2838;border:1px solid #2a475e;
                        border-radius:6px;padding:14px 16px;margin-bottom:10px;">
                        <div style="font-size:16px;font-weight:700;color:#fff;">{r['name']}</div>
                        <div style="font-size:12px;color:#8f98a0;margin:4px 0;">{tags}</div>
                        <div style="font-size:13px;color:#c7d5e0;">
                        {price} · <span style="color:{pos_color};">긍정 {pos:.0f}%</span>
                        · 리뷰 {int(r['total_reviews']):,} · {int(r['release_year'])}</div>
                        </div>""",
                        unsafe_allow_html=True)
    else:
        st.markdown("##### 예시 검색어")
        st.write(" · ".join(f"`{e}`" for e in
                 ["잔잔한 힐링 게임", "roguelike", "친구랑 할 협동 호러",
                  "감성적인 스토리 게임", "open world"]))

# ══════════════ 탭 2: ML vs DL ══════════════
with tab2:
    st.title("📊 ML vs DL 성능 비교")
    df = load_comparison()
    if df is None:
        st.warning("평가 결과 없음 — `uv run eval_search.py`를 먼저 실행하세요.")
    else:
        # 쿼리 유형별 평균
        pivot = (df.pivot_table(index="qtype", columns="method",
                                values="precision", aggfunc="mean")
                 .reindex(["word", "natural", "korean"]).reset_index())
        label_map = {"word": "영어 키워드", "natural": "자연어 문장", "korean": "한국어"}
        pivot["유형"] = pivot["qtype"].map(label_map)

        melted = pivot.melt(id_vars="유형", value_vars=[c for c in ["ML", "DL"] if c in pivot],
                            var_name="방법", value_name="Precision@20")
        fig = px.bar(melted, x="유형", y="Precision@20", color="방법",
                     barmode="group", color_discrete_map={"ML": "#3498db", "DL": "#e74c3c"},
                     title="쿼리 유형별 Precision@20")
        st.plotly_chart(fig, width="stretch")

        st.markdown("""
        **읽는 법**
        - **영어 키워드**: 단어가 그대로 겹치니 ML(단어매칭)이 강하거나 대등
        - **자연어 문장**: 의미를 풀어야 하니 DL(의미검색)이 우세
        - **한국어**: ML은 영어 설명문과 매칭 불가 → DL만 작동

        → **데이터·쿼리 성격이 모델 선택을 결정한다.** 그래서 실제 검색에서는
        쿼리를 보고 ML/DL을 자동 선택하는 하이브리드 방식을 적용했다.
        """)

        with st.expander("쿼리별 상세 결과"):
            st.dataframe(df[["method", "qtype", "query", "precision", "recall"]],
                         width="stretch", hide_index=True)