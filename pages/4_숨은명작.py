"""숨은 명작 추천 — 장르/플랫폼 필터 → 명작 목록"""
import os
import sys
import streamlit as st
import pandas as pd
import mlflow

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

st.set_page_config(page_title="숨은 명작", page_icon="💎", layout="wide")
st.title("💎 숨은 명작 추천")

mlflow.set_tracking_uri("sqlite:///mlflow.db")


@st.cache_data
def load_catalog():
    return pd.read_parquet("data/processed/games_catalog.parquet")


@st.cache_data
def load_X_all():
    return pd.read_parquet("data/processed/X_all_hidden_gem.parquet")


@st.cache_resource
def load_model():
    return mlflow.sklearn.load_model("models:/steam_hidden_gem_predictor@champion")


catalog = load_catalog()
X_all = load_X_all()
model = load_model()
catalog["gem_score"] = model.predict_proba(X_all)[:, 1]

# ── 필터 ──
c1, c2, c3 = st.columns(3)
with c1:
    sel_genres = st.multiselect(
        "장르", ["Indie", "Action", "Adventure", "RPG", "Strategy",
                "Casual", "Simulation", "Puzzle", "Horror"], ["Indie"])
with c2:
    sel_plat = st.multiselect("플랫폼", ["windows", "mac", "linux"], ["windows"])
with c3:
    max_price = st.slider("최대 가격 ($)", 0, 60, 20)

mask = catalog["price"] <= max_price
for p in sel_plat:
    if p in catalog.columns:
        mask &= catalog[p] == 1
if sel_genres:
    mask &= catalog["genres"].apply(lambda s: any(g in str(s) for g in sel_genres))
filtered = catalog[mask]

# ── 탭 ──
tab1, tab2, tab3 = st.tabs(["✅ 확정 숨은 명작", "🔮 잠재 명작 (모델 발굴)", "🚨 리뷰 이상 패턴"])

with tab1:
    st.caption("Wilson score 하한 85%+ · owners 10만 미만 · 리뷰 30+ · 이상 패턴 제외")
    gems = (filtered[filtered["target_hidden_gem"] == 1]
            .sort_values("gem_score", ascending=False).head(20))
    for _, r in gems.iterrows():
        price = "무료" if r["price"] == 0 else f"${r['price']:.2f}"
        st.markdown(
            f"**{r['name']}** · {price} · "
            f"긍정 {r['positive_ratio']*100:.0f}% · "
            f"리뷰 {int(r['total_reviews_calc']):,} · "
            f"모델점수 {r['gem_score']:.2f}")
        st.divider()

with tab2:
    st.caption("리뷰 부족으로 라벨 미부여, 모델이 명작 패턴으로 판단한 게임")
    latent = (filtered[(filtered["target_hidden_gem"] == 0)
                       & (filtered["total_reviews_calc"] < 30)]
              .sort_values("gem_score", ascending=False).head(20))
    for _, r in latent.iterrows():
        price = "무료" if r["price"] == 0 else f"${r['price']:.2f}"
        st.markdown(
            f"**{r['name']}** · {price} · "
            f"리뷰 {int(r['total_reviews_calc']):,} · "
            f"모델점수 {r['gem_score']:.2f}")
        st.divider()

with tab3:
    st.caption("Isolation Forest 탐지 — 리뷰 폭격, owners 오류, 데이터 중복 후보")
    sus = (catalog[catalog["is_suspicious"] == 1]
           .sort_values("anomaly_score", ascending=False).head(20))
    for _, r in sus.iterrows():
        st.markdown(
            f"**{r['name']}** · "
            f"긍정 {r['positive_ratio']*100:.0f}% · "
            f"리뷰 {int(r['total_reviews_calc']):,} · "
            f"이상점수 {r['anomaly_score']:.2f}")
        st.divider()