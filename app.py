"""
Steam Games — MLOps 대시보드 + 모델 서빙 (팀원 C 작업 베이스)
MLflow 레지스트리의 champion 모델을 로드해 실시간 예측 제공.

실행: streamlit run app.py
"""
import json
import sys

import joblib
import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, "src")

st.set_page_config(page_title="Steam ML vs DL | MLOps", page_icon="🎮", layout="wide")
mlflow.set_tracking_uri("sqlite:///mlflow.db")

PAGES = ["1. 파이프라인 현황", "2. 실험 비교", "3. 모델 레지스트리",
         "4. 실시간 예측 데모", "5. 숨은 명작 추천", "6. 장르 분석",
         "7. 한계점 & 개선"]
page = st.sidebar.radio("페이지", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("전처리 → 학습(MLflow) → champion 등록 → 서빙")


@st.cache_data
def load_meta():
    with open("data/processed/pipeline_meta.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_comparison(task):
    try:
        with open(f"data/processed/comparison_{task}.json", encoding="utf-8") as f:
            return pd.DataFrame(json.load(f))
    except FileNotFoundError:
        return None


@st.cache_resource
def get_champion(task):
    return mlflow.sklearn.load_model(f"models:/steam_{task}_predictor@champion")


@st.cache_resource
def get_scaler(task):
    return joblib.load(f"data/processed/scaler_{task}.joblib")


@st.cache_data
def get_feature_names(task):
    return pd.read_parquet(f"data/processed/X_test_{task}.parquet").columns.tolist()


# ═════════ 1. 파이프라인 현황 ═════════
if page == PAGES[0]:
    st.title("🔧 파이프라인 현황")
    meta = load_meta()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("모델링 데이터", f"{meta['rows_after_cleaning']:,}건")
    c2.metric("피처 수", meta["final_feature_count"])
    c3.metric("흥행 양성비", f"{meta['label_stats']['hit']['positive_ratio']*100:.1f}%")
    c4.metric("숨은명작 양성비", f"{meta['label_stats']['hidden_gem']['positive_ratio']*100:.1f}%")

    st.subheader("클리닝 단계별 제거 내역")
    st.dataframe(pd.DataFrame(list(meta["cleaning_steps"].items()),
                              columns=["단계", "건수"]),
                 use_container_width=True, hide_index=True)

    st.subheader("⚠️ Leakage 차단 컬럼 (출시 후 데이터)")
    st.code(", ".join(meta["leakage_columns_dropped"]))
    st.caption(f"라벨 생성에만 사용 후 물리 제거 | data_hash: {meta.get('data_hash')}")

# ═════════ 2. 실험 비교 ═════════
elif page == PAGES[1]:
    st.title("📊 실험 비교 — ML vs DL")
    task = st.selectbox("타스크", ["hidden_gem", "hit"],
                        format_func=lambda x: "숨은 명작 탐지" if x == "hidden_gem" else "흥행 예측")
    df = load_comparison(task)
    if df is None:
        st.warning("학습 미실행 — `python run_all.py` 먼저 실행하세요."); st.stop()

    metric = "pr_auc" if task == "hidden_gem" else "f1"
    st.caption(f"1차 비교 지표: **{metric.upper()}** "
               + ("(불균형 데이터 → PR-AUC가 ROC-AUC보다 엄격)" if task == "hidden_gem" else ""))
    num_cols = df.select_dtypes("number").columns
    st.dataframe(df.style.background_gradient(
        subset=[c for c in [metric, "f1"] if c in num_cols], cmap="Greens"),
        use_container_width=True, hide_index=True)

    fig = px.bar(df.sort_values(metric), x=metric, y="name", color="family",
                 orientation="h", color_discrete_map={"ML": "#3498db", "DL": "#e74c3c"},
                 title=f"모델별 {metric.upper()}")
    st.plotly_chart(fig, use_container_width=True)

    if task == "hidden_gem" and "f1@thr" in df.columns:
        st.subheader("임계값 튜닝 효과 (기본 0.5 → 최적)")
        m = df.melt(id_vars="name", value_vars=["f1", "f1@thr"],
                    var_name="유형", value_name="F1")
        fig = px.bar(m, x="name", y="F1", color="유형", barmode="group",
                     color_discrete_map={"f1": "#95a5a6", "f1@thr": "#27ae60"})
        st.plotly_chart(fig, use_container_width=True)
        st.info("MLP_raw는 기본 임계값 0.5에서 F1=0 (양성 예측 전무) → "
                "임계값 최적화만으로 0.37 회복. 불균형 데이터에서 임계값 튜닝의 중요성.")

# ═════════ 3. 레지스트리 ═════════
elif page == PAGES[2]:
    st.title("📦 모델 레지스트리")
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    for task, label in [("hidden_gem", "숨은 명작 탐지"), ("hit", "흥행 예측")]:
        name = f"steam_{task}_predictor"
        st.subheader(label)
        try:
            rows = []
            for v in client.search_model_versions(f"name='{name}'"):
                aliases = list(getattr(v, "aliases", []) or [])
                rows.append({"version": v.version,
                             "alias": ", ".join(aliases) or "-",
                             "등록일": pd.Timestamp(v.creation_timestamp, unit="ms")
                             .strftime("%m-%d %H:%M")})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(f"`mlflow.sklearn.load_model('models:/{name}@champion')`")
        except Exception:
            st.warning("등록된 모델 없음")

# ═════════ 4. 예측 데모 ═════════
elif page == PAGES[3]:
    st.title("🎮 실시간 예측 데모")
    task = st.radio("대상", ["hidden_gem", "hit"], horizontal=True,
                    format_func=lambda x: "숨은 명작" if x == "hidden_gem" else "흥행")
    try:
        model = get_champion(task)
        scaler = get_scaler(task)
        feats = get_feature_names(task)
    except Exception:
        st.warning("champion 없음 — run_all.py 먼저 실행"); st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        price = st.slider("가격 ($)", 0.0, 60.0, 9.99, 0.5)
        ach = st.slider("업적 수", 0, 100, 20)
        dlc = st.slider("DLC 수", 0, 10, 0)
    with c2:
        month = st.selectbox("출시 월", range(1, 13), index=5)
        year = st.slider("출시 연도", 2015, 2025, 2024)
        self_pub = st.checkbox("자체 퍼블리싱 (인디)", True)
    with c3:
        plats = st.multiselect("플랫폼", ["windows", "mac", "linux"], ["windows"])
        genres = st.multiselect(
            "장르", ["indie", "action", "casual", "adventure", "rpg",
                    "strategy", "simulation"], ["indie"])

    row = {c: 0.0 for c in feats}
    row.update({"price": price, "log_price": np.log1p(price),
                "is_free": int(price == 0), "n_achievements": ach, "n_dlc": dlc,
                "release_year": year, "release_month": month,
                "month_sin": np.sin(2 * np.pi * month / 12),
                "month_cos": np.cos(2 * np.pi * month / 12),
                "is_q4_release": int(month in (10, 11, 12)),
                "n_platforms": len(plats),
                "is_self_published": int(self_pub),
                "n_genre": len(genres)})
    for p in plats:
        if p in row: row[p] = 1
    for g in genres:
        k = f"genre_{g}"
        if k in row: row[k] = 1

    X = pd.DataFrame([row])[feats]
    Xs = pd.DataFrame(scaler.transform(X), columns=feats)
    prob = float(model.predict_proba(Xs)[0, 1])
    st.metric("숨은 명작 확률" if task == "hidden_gem" else "흥행 확률",
              f"{prob*100:.1f}%")
    st.progress(min(prob, 1.0))
    st.caption(f"champion: {type(model).__name__} | "
               f"registry: steam_{task}_predictor@champion")

# ═════════ 숨은 명작 추천 ═════════
elif page == PAGES[4]:
    st.title("💎 숨은 명작 추천")

    @st.cache_data
    def load_recommend_data():
        catalog = pd.read_parquet("data/processed/games_catalog.parquet")
        X_all = pd.read_parquet("data/processed/X_all_hidden_gem.parquet")
        return catalog, X_all

    catalog, X_all = load_recommend_data()
    model = get_champion("hidden_gem")

    # 모델 점수 계산 (전체 게임, 캐싱)
    @st.cache_data
    def score_all():
        return model.predict_proba(X_all)[:, 1]
    catalog = catalog.copy()
    catalog["gem_score"] = score_all()

    # ── 필터 UI ──
    c1, c2, c3 = st.columns(3)
    with c1:
        sel_genres = st.multiselect(
            "장르", ["Indie", "Action", "Adventure", "RPG", "Strategy",
                    "Casual", "Simulation", "Puzzle"], ["Indie"])
    with c2:
        sel_plat = st.multiselect("플랫폼", ["windows", "mac", "linux"], ["windows"])
    with c3:
        max_price = st.slider("최대 가격 ($)", 0, 60, 20)

    # ── 필터 적용 ──
    mask = (catalog["price"] <= max_price)
    for p in sel_plat:
        mask &= (catalog[p] == 1)
    if sel_genres:
        genre_mask = catalog["genres"].apply(
            lambda s: any(g in str(s) for g in sel_genres))
        mask &= genre_mask
    filtered = catalog[mask]

    # ── 2층 추천 ──
    tab1, tab2, tab3 = st.tabs(["✅ 확정 숨은 명작", "🔮 잠재 명작 (모델 발굴)",
                                 "🚨 리뷰 패턴 이상"])

    with tab1:
        st.caption("긍정률 85%+ · owners 10만 미만 · 리뷰 30+ (검증된 명작)")
        gems = (filtered[filtered["target_hidden_gem"] == 1]
                .sort_values("wilson_lb", ascending=False).head(20))
        st.dataframe(gems[["name", "price", "positive_ratio",
                           "total_reviews_calc", "release_year"]]
                     .rename(columns={"positive_ratio": "긍정률",
                                      "total_reviews_calc": "리뷰수"}),
                     use_container_width=True, hide_index=True)

    with tab2:
        st.caption("리뷰가 부족해 라벨을 못 받았지만 모델이 명작 패턴으로 판단한 게임")
        latent = (filtered[(filtered["target_hidden_gem"] == 0)
                           & (filtered["total_reviews_calc"] < 30)]
                  .sort_values("gem_score", ascending=False).head(20))
        st.dataframe(latent[["name", "price", "gem_score",
                             "total_reviews_calc", "release_year"]]
                     .rename(columns={"gem_score": "모델 점수",
                                      "total_reviews_calc": "리뷰수"}),
                     use_container_width=True, hide_index=True)
        
    with tab3:
        st.caption("Isolation Forest가 탐지한 통계적 이상 패턴 — "
                   "리뷰 폭격, 조작 의심, 데이터 오류 후보 (확정이 아닌 검토 대상)")
        sus = (catalog[catalog["is_suspicious"] == 1]
               .sort_values("anomaly_score", ascending=False).head(20))
        st.dataframe(sus[["name", "total_reviews_calc", "positive_ratio",
                          "anomaly_score", "price"]]
                     .rename(columns={"total_reviews_calc": "리뷰수",
                                      "positive_ratio": "긍정률",
                                      "anomaly_score": "이상점수"}),
                     use_container_width=True, hide_index=True)
        st.info("실제 탐지 사례: Call of Duty MW II/III (긍정률 22~32% — 리뷰 폭격), "
                "동일 게임 중복 등재 (데이터 오류). 이상 플래그 게임은 "
                "숨은 명작 라벨에서 자동 제외됩니다.")
        
# ═════════ 장르 선호도 분석 ═════════
elif page == PAGES[5]:
    st.title("📊 장르별 수요·공급 분석")

    @st.cache_data
    def genre_stats():
        catalog = pd.read_parquet("data/processed/games_catalog.parquet")
        import ast
        rows = []
        for _, r in catalog.iterrows():
            try:
                genres = ast.literal_eval(str(r["genres"]))
            except (ValueError, SyntaxError):
                continue
            for g in genres:
                rows.append({"genre": g, "owners": r["owners_lower"],
                             "reviews": r["total_reviews_calc"]})
        gdf = pd.DataFrame(rows)
        agg = gdf.groupby("genre").agg(
            게임수=("genre", "size"),
            총보유자=("owners", "sum"),
            총리뷰=("reviews", "sum")).reset_index()
        agg["게임당_평균보유자"] = agg["총보유자"] / agg["게임수"]
        return agg.nlargest(12, "게임수")

    agg = genre_stats()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(agg.sort_values("총보유자"),
                     x="총보유자", y="genre", orientation="h",
                     title="장르별 총 보유자 수 (유저 선호도 proxy)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(agg.sort_values("게임수"),
                     x="게임수", y="genre", orientation="h",
                     title="장르별 게임 수 (공급)")
        st.plotly_chart(fig, use_container_width=True)

    # 핵심 인사이트: 게임당 평균 보유자 = 경쟁 대비 수요
    fig = px.scatter(agg, x="게임수", y="게임당_평균보유자",
                     text="genre", size="총보유자", log_y=True,
                     title="공급 vs 게임당 수요 — 우상단=블루오션, 우하단=레드오션")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Indie는 게임 수가 압도적이지만 게임당 평균 보유자는 낮음 → "
            "전형적 레드오션. 발표에서 '숨은 명작이 Indie에 몰리는 이유'와 연결 가능.")

# ═════════ 5. 한계점 ═════════
else:
    st.title("⚠️ 한계점 & 🚀 향후 개선")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("한계점")
        st.markdown("""
- **리뷰 수 = 인기 proxy**: 실제 판매량 데이터 부재 (Boxleiter 추정에 의존)
- **n_dlc leakage 가능성**: DLC는 출시 후 증가 — 부분적 미래정보 포함
- **텍스트/이미지 부재**: 이 데이터셋엔 설명문이 없어 콘텐츠 신호 미활용
- **리뷰 10개 미만 66% 제외**: 진짜 '완전히 묻힌' 게임은 분석 불가
- **숨은 명작 라벨의 자의성**: 85%/30~500 기준은 합리적 추정일 뿐
""")
    with c2:
        st.subheader("향후 개선")
        st.markdown("""
- Steam API로 설명문 수집 → 텍스트 임베딩 추가 (DL 강화)
- SteamSpy owners 데이터 결합으로 라벨 정밀화
- Optuna로 숨은 명작 모델도 자동 튜닝
- GitHub Actions CI/CD + 드리프트 모니터링
- 비용 민감 임계값 (큐레이션 노출 비용 반영)
""")
