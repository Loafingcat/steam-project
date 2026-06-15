"""
[의미 검색] 인덱스 구축 — ML vs DL 비교용

게임 설명문(short_description)으로 두 가지 검색 인덱스를 만든다:
  - ML:  TF-IDF (단어 빈도 기반) — 단어가 겹쳐야 매칭
  - DL:  Sentence-Transformer 임베딩 (의미 기반) — 동의어/다국어 매칭

산출물(data/processed/):
  search_catalog.parquet     검색 대상 게임 메타 (이름, 태그, gem 정보)
  tfidf_vectorizer.joblib    ML 벡터라이저
  tfidf_matrix.joblib        ML 문서 행렬
  embeddings.npy             DL 임베딩 행렬
  embed_model_name.txt       사용한 임베딩 모델 이름

실행:
    uv run build_search_index.py
    uv run build_search_index.py --no-dl   # 임베딩 생략 (TF-IDF만)
"""
import argparse
import ast
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("search")

RAW = "data/raw/games_march2025_cleaned.csv"
OUT = "data/processed"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 한/영 교차 검색 지원


def parse_tags(s):
    """tags 딕셔너리 문자열 → 키 리스트"""
    if pd.isna(s):
        return []
    try:
        v = ast.literal_eval(str(s))
        return list(v.keys()) if isinstance(v, dict) else (v if isinstance(v, list) else [])
    except (ValueError, SyntaxError):
        return []


def parse_owners_lower(s):
    try:
        return float(str(s).replace(",", "").split("-")[0].strip())
    except (ValueError, AttributeError):
        return np.nan


def build_catalog():
    """검색 대상 게임 로드 + 클리닝 (전처리 파이프라인과 동일 기준)"""
    log.info("[1/3] 검색 카탈로그 구축")
    df = pd.read_csv(RAW, usecols=[
        "appid", "name", "short_description", "tags", "genres",
        "price", "positive", "negative", "estimated_owners",
        "windows", "mac", "linux", "release_date"])

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce", format="mixed")
    df = df[df["release_date"] <= pd.Timestamp("2025-03-10") - pd.Timedelta(days=180)]
    df["total_reviews"] = df["positive"].fillna(0) + df["negative"].fillna(0)
    df = df[df["total_reviews"] >= 10]
    df = df.dropna(subset=["short_description"])
    df = df[df["short_description"].astype(str).str.len() >= 20].reset_index(drop=True)

    df["tag_list"] = df["tags"].apply(parse_tags)
    df["genre_list"] = df["genres"].apply(parse_tags)
    df["owners_lower"] = df["estimated_owners"].apply(parse_owners_lower)
    df["positive_ratio"] = (df["positive"] / df["total_reviews"].replace(0, np.nan)).fillna(0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["release_year"] = df["release_date"].dt.year

    keep = ["appid", "name", "short_description", "tag_list", "genre_list",
            "price", "positive_ratio", "owners_lower", "total_reviews",
            "windows", "mac", "linux", "release_year"]
    catalog = df[keep].reset_index(drop=True)
    log.info(f"  검색 대상: {len(catalog):,}개 게임")
    return catalog


def build_tfidf(catalog):
    """[ML] TF-IDF 인덱스"""
    log.info("[2/3] ML 인덱스 (TF-IDF) 구축")
    texts = catalog["short_description"].astype(str).tolist()
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    mat = tfidf.fit_transform(texts)
    joblib.dump(tfidf, f"{OUT}/tfidf_vectorizer.joblib")
    joblib.dump(mat, f"{OUT}/tfidf_matrix.joblib")
    log.info(f"  TF-IDF 행렬: {mat.shape}")


def build_embeddings(catalog):
    """[DL] Sentence-Transformer 임베딩"""
    log.info("[3/3] DL 인덱스 (임베딩) 구축 — 최초 1회 모델 다운로드(~120MB)")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    texts = catalog["short_description"].astype(str).tolist()
    emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                       normalize_embeddings=True)
    np.save(f"{OUT}/embeddings.npy", emb.astype(np.float32))
    with open(f"{OUT}/embed_model_name.txt", "w") as f:
        f.write(EMBED_MODEL)
    log.info(f"  임베딩 행렬: {emb.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dl", action="store_true", help="임베딩 생략 (TF-IDF만)")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    catalog = build_catalog()
    catalog.to_parquet(f"{OUT}/search_catalog.parquet")

    build_tfidf(catalog)
    if not args.no_dl:
        build_embeddings(catalog)
    else:
        log.info("  --no-dl: DL 임베딩 생략")

    log.info("✅ 검색 인덱스 구축 완료")
    log.info("   다음: uv run eval_search.py  (ML vs DL 정량 비교)")


if __name__ == "__main__":
    main()
