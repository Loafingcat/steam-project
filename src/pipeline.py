"""전처리 파이프라인 — artermiloff (games_march2025_cleaned.csv, 47컬럼)

이전 데이터셋과의 차이:
  - estimated_owners 있음 → owners 기반 라벨 복원
  - tags가 리스트가 아닌 딕셔너리 문자열: {'Survival': 14838, ...}
  - windows/mac/linux가 이미 bool 컬럼
  - pct_pos_total은 자체 계산값과 평균 10.7%p 차이 → 긍정률 직접 계산
  - short_description 있음 → DL 텍스트 실험(TF-IDF) 가능
"""
import ast, hashlib, json, logging, os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

log = logging.getLogger("pipeline")

# 468MB 파일 → 로딩 시점에 대용량 텍스트/URL 컬럼 제외 (메모리 절약 핵심)
DROP_ON_LOAD = ["detailed_description", "about_the_game", "reviews",
                "header_image", "website", "support_url", "support_email",
                "metacritic_url", "screenshots", "movies", "notes", "packages",
                "full_audio_languages"]


# ──────────── [1] LOAD ────────────
def load_raw(cfg):
    path = cfg["data"]["raw_path"]
    log.info(f"[1/5] 로딩: {path}")
    header = pd.read_csv(path, nrows=0)
    drop = [c for c in DROP_ON_LOAD if c in header.columns]
    usecols = [c for c in header.columns if c not in drop]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    for c in df.select_dtypes("int64").columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype(np.float32)
    log.info(f"  {len(df):,}행 × {len(df.columns)}열 (제외 {len(drop)}열)")
    return df


# ──────────── [2] CLEAN ────────────
def clean(df, cfg, meta):
    log.info("[2/5] 클리닝")
    c = cfg["cleaning"]
    steps, n0 = {}, len(df)

    n = len(df); df = df.drop_duplicates(subset=["appid"])
    steps["중복 제거"] = n - len(df)

    n = len(df)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce", format="mixed")
    df = df.dropna(subset=["release_date"])
    steps["출시일 파싱 실패 제거"] = n - len(df)

    # 데이터 수집 기준일에서 너무 최근 출시작 → owners/평가 미성숙
    n = len(df)
    cutoff = pd.Timestamp("2025-03-10") - pd.Timedelta(days=c["min_days_since_release"])
    df = df[df["release_date"] <= cutoff]
    steps[f"출시 {c['min_days_since_release']}일 미만 제거"] = n - len(df)

    # 리뷰 최소 기준 — positive/negative 직접 합산 (pct_pos_total 불신)
    n = len(df)
    df["total_reviews_calc"] = df["positive"].fillna(0) + df["negative"].fillna(0)
    df = df[df["total_reviews_calc"] >= c["min_total_reviews"]]
    steps[f"리뷰 {c['min_total_reviews']}개 미만 제거"] = n - len(df)

    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    steps["가격 클리핑"] = int((df["price"] > c["price_clip_max"]).sum())
    df["price"] = df["price"].clip(upper=c["price_clip_max"])

    df["required_age"] = pd.to_numeric(df["required_age"], errors="coerce").fillna(0).clip(0, 21)
    for col in ["dlc_count", "achievements"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    meta["cleaning_steps"] = steps
    meta["rows_after_cleaning"] = len(df)
    for k, v in steps.items():
        log.info(f"  - {k}: {v:,}건")
    log.info(f"  클리닝 후: {len(df):,}행 ({len(df)/n0*100:.1f}%)")
    return df.reset_index(drop=True)


# ──────────── [3] LABEL ────────────
def _owners_lower(s):
    """'100000 - 200000' → 100000.0"""
    try:
        return float(str(s).replace(",", "").split("-")[0].strip())
    except (ValueError, AttributeError):
        return np.nan


def make_labels(df, cfg, meta):
    log.info("[3/5] 타겟 라벨 (owners 구간 하한 기준)")
    lc = cfg["labels"]

    df["owners_lower"] = df["estimated_owners"].apply(_owners_lower)
    df = df.dropna(subset=["owners_lower"])

    # 긍정률 직접 계산 — pct_pos_total은 집계 기준이 달라 평균 10.7%p 차이
    df["positive_ratio"] = df["positive"] / df["total_reviews_calc"].replace(0, np.nan)
    df["positive_ratio"] = df["positive_ratio"].fillna(0)

    df["target_hit"] = (df["owners_lower"] >= lc["hit"]["owners_lower_threshold"]).astype(int)

    g = lc["hidden_gem"]
    df["target_hidden_gem"] = (
        (df["positive_ratio"] >= g["positive_ratio_min"])
        & (df["owners_lower"] < g["owners_upper_threshold"])
        & (df["total_reviews_calc"] >= g["min_total_reviews"])
    ).astype(int)

    meta["label_stats"] = {
        "hit": {"positive_ratio": round(float(df["target_hit"].mean()), 4)},
        "hidden_gem": {"positive_ratio": round(float(df["target_hidden_gem"].mean()), 4)},
    }
    log.info(f"  흥행: {df['target_hit'].mean()*100:.1f}% | "
             f"숨은명작: {df['target_hidden_gem'].mean()*100:.1f}%")
    return df.reset_index(drop=True)


# ──────────── [4] FEATURES ────────────
def _parse_list(s):
    """리스트 문자열 → list. tags는 딕셔너리 문자열이므로 keys 반환."""
    if pd.isna(s):
        return []
    try:
        v = ast.literal_eval(str(s))
        if isinstance(v, dict):
            return list(v.keys())
        return v if isinstance(v, list) else []
    except (ValueError, SyntaxError):
        return []


def _multihot(df, col, top_n, prefix):
    lists = df[col].apply(_parse_list)
    top = pd.Series([i for lst in lists for i in lst]).value_counts().head(top_n).index
    for item in top:
        safe = str(item).lower().replace(" ", "_").replace("-", "_").replace("/", "_")[:30]
        df[f"{prefix}_{safe}"] = lists.apply(lambda x: int(item in x))
    df[f"n_{prefix}"] = lists.apply(len)
    log.info(f"  - {col}: 상위 {len(top)}개 멀티핫 + n_{prefix}")
    return df


def engineer(df, cfg, meta):
    log.info("[4/5] 피처 엔지니어링")
    fc = cfg["features"]

    df["is_free"] = (df["price"] == 0).astype(int)
    df["log_price"] = np.log1p(df["price"])

    # 플랫폼은 이미 bool → int 변환만
    for p in ["windows", "mac", "linux"]:
        df[p] = df[p].astype(bool).astype(int)
    df["n_platforms"] = df[["windows", "mac", "linux"]].sum(axis=1)

    df = _multihot(df, "genres", fc["top_n_genres"], "genre")
    df = _multihot(df, "tags", fc["top_n_tags"], "tag")        # dict → keys
    df = _multihot(df, "categories", fc["top_n_categories"], "cat")
    df["n_languages"] = df["supported_languages"].apply(lambda s: len(_parse_list(s)))

    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month
    df["release_dow"] = df["release_date"].dt.dayofweek
    df["month_sin"] = np.sin(2 * np.pi * df["release_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["release_month"] / 12)
    df["is_q4_release"] = df["release_month"].isin([10, 11, 12]).astype(int)

    for col in ["developers", "publishers"]:
        first = df[col].apply(lambda s: (_parse_list(s) or ["unknown"])[0])
        df[f"{col}_game_count"] = first.map(first.value_counts())
    dev1 = df["developers"].apply(lambda s: (_parse_list(s) or ["?"])[0])
    pub1 = df["publishers"].apply(lambda s: (_parse_list(s) or ["?"])[0])
    df["is_self_published"] = (dev1 == pub1).astype(int)

    # ═══ ⚠️ LEAKAGE 물리 차단 ═══
    leak = [c for c in fc["leakage_columns"] if c in df.columns]
    drop_etc = [c for c in ["appid", "name", "developers", "publishers",
                            "categories", "genres", "tags",
                            "supported_languages", "release_date",
                            "owners_lower", "positive_ratio",
                            "total_reviews_calc"] if c in df.columns]
    df = df.drop(columns=leak + drop_etc)
    meta["leakage_columns_dropped"] = leak
    log.warning(f"  ⚠️ LEAKAGE 차단 {len(leak)}개: {leak}")

    n_feat = len([c for c in df.columns
                  if not c.startswith("target_") and c != "short_description"])
    meta["final_feature_count"] = n_feat
    log.info(f"  최종 피처 수: {n_feat}")
    return df


# ──────────── [5] SPLIT + SAVE ────────────
def split_scale_save(df, cfg, meta):
    log.info("[5/5] 분할 / 스케일링 / 저장")
    out = cfg["data"]["processed_dir"]
    os.makedirs(out, exist_ok=True)
    sc = cfg["split"]
    text_col = df["short_description"] if "short_description" in df.columns else None
    feats = [c for c in df.columns
             if not c.startswith("target_") and c != "short_description"]

    meta["data_hash"] = hashlib.sha256(
        pd.util.hash_pandas_object(df[feats], index=False).values.tobytes()
    ).hexdigest()[:12]

    for target in ["target_hit", "target_hidden_gem"]:
        tag = target.replace("target_", "")
        X, y = df[feats].astype(np.float32), df[target]
        idx_tr, idx_te = train_test_split(
            df.index, test_size=sc["test_size"],
            random_state=sc["random_state"], stratify=y)
        X_tr, X_te = X.loc[idx_tr], X.loc[idx_te]
        y_tr, y_te = y.loc[idx_tr], y.loc[idx_te]

        scaler = StandardScaler()
        X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=feats, index=idx_tr)
        X_te_s = pd.DataFrame(scaler.transform(X_te), columns=feats, index=idx_te)

        X_tr_s.to_parquet(f"{out}/X_train_{tag}.parquet")
        X_te_s.to_parquet(f"{out}/X_test_{tag}.parquet")
        y_tr.to_frame().to_parquet(f"{out}/y_train_{tag}.parquet")
        y_te.to_frame().to_parquet(f"{out}/y_test_{tag}.parquet")
        joblib.dump(scaler, f"{out}/scaler_{tag}.joblib")

        # DL 텍스트 실험용 TF-IDF (train에만 fit)
        if text_col is not None and cfg["data"].get("keep_text"):
            tc = cfg["text"]
            tfidf = TfidfVectorizer(max_features=tc["tfidf_max_features"],
                                    ngram_range=(1, tc["tfidf_ngram_max"]),
                                    stop_words="english")
            T_tr = tfidf.fit_transform(text_col.loc[idx_tr].fillna(""))
            T_te = tfidf.transform(text_col.loc[idx_te].fillna(""))
            joblib.dump(tfidf, f"{out}/tfidf_{tag}.joblib")
            joblib.dump(T_tr, f"{out}/T_train_{tag}.joblib")
            joblib.dump(T_te, f"{out}/T_test_{tag}.joblib")

        log.info(f"  [{tag}] train {len(X_tr):,} / test {len(X_te):,} "
                 f"(양성 {y_tr.mean()*100:.1f}%)")

    with open(f"{out}/pipeline_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"  meta 저장 | data_hash={meta['data_hash']}")


# ───── 팀원 인터페이스 (시그니처 불변 — B/C 코드 수정 불필요) ─────
def load_processed(task="hit", with_text=False, processed_dir="data/processed"):
    """task: 'hit' | 'hidden_gem'. with_text=True면 TF-IDF 행렬 추가 반환."""
    X_tr = pd.read_parquet(f"{processed_dir}/X_train_{task}.parquet")
    X_te = pd.read_parquet(f"{processed_dir}/X_test_{task}.parquet")
    y_tr = pd.read_parquet(f"{processed_dir}/y_train_{task}.parquet").iloc[:, 0]
    y_te = pd.read_parquet(f"{processed_dir}/y_test_{task}.parquet").iloc[:, 0]
    if with_text:
        T_tr = joblib.load(f"{processed_dir}/T_train_{task}.joblib")
        T_te = joblib.load(f"{processed_dir}/T_test_{task}.joblib")
        return X_tr, X_te, y_tr, y_te, T_tr, T_te
    return X_tr, X_te, y_tr, y_te