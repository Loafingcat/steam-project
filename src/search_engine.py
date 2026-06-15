"""
[의미 검색] 검색 엔진 — app.py와 평가 스크립트가 공통으로 사용

두 가지 검색 방식을 같은 인터페이스로 제공:
  search(query, method="ml")  → TF-IDF 코사인 유사도
  search(query, method="dl")  → 임베딩 코사인 유사도

DL은 임베딩 모델을 처음 호출 시 1회 로드 (캐시).
"""
import ast
import functools

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

OUT = "data/processed"
_dl_model = None  # 임베딩 모델 캐시

def set_dl_model(model):
    """외부에서 캐시된 모델을 주입 (Streamlit 캐싱용)"""
    global _dl_model
    _dl_model = model


def route_method(query: str) -> tuple:
    """쿼리 성격을 보고 ML/DL 자동 선택"""
    has_korean = any('\uac00' <= c <= '\ud7a3' for c in query)
    word_count = len(query.split())
    if has_korean:
        return "dl", "한국어 쿼리 → 의미 임베딩"
    if word_count <= 3:
        return "ml", "짧은 키워드 → 단어매칭"
    return "dl", "자연어 문장 → 의미 임베딩"


def load_catalog():
    return pd.read_parquet(f"{OUT}/search_catalog.parquet")


@functools.lru_cache(maxsize=1)
def _load_tfidf():
    return (joblib.load(f"{OUT}/tfidf_vectorizer.joblib"),
            joblib.load(f"{OUT}/tfidf_matrix.joblib"))


@functools.lru_cache(maxsize=1)
def _load_embeddings():
    return np.load(f"{OUT}/embeddings.npy")


def _get_dl_model():
    global _dl_model
    if _dl_model is None:
        from sentence_transformers import SentenceTransformer
        with open(f"{OUT}/embed_model_name.txt") as f:
            name = f.read().strip()
        _dl_model = SentenceTransformer(name)
    return _dl_model


def search(query: str, method: str = "dl", top_k: int = 10):
    """쿼리로 게임 검색 → 유사도 순 인덱스 + 점수 반환

    method: 'ml'(TF-IDF) | 'dl'(임베딩)
    반환: (인덱스 배열, 유사도 배열)
    """
    if method == "ml":
        tfidf, mat = _load_tfidf()
        q = tfidf.transform([query])
        sim = cosine_similarity(q, mat).flatten()
    elif method == "dl":
        model = _get_dl_model()
        emb = _load_embeddings()
        q = model.encode([query], normalize_embeddings=True)
        sim = (emb @ q.T).flatten()
    else:
        raise ValueError(method)

    top = np.argsort(-sim)[:top_k]
    return top, sim[top]


def route_method(query: str) -> tuple:
    """쿼리 성격을 보고 ML/DL 자동 선택.
    - 한국어 포함 → DL (ML은 영어 설명문과 매칭 불가)
    - 3단어 이하 영어 키워드 → ML (단어매칭이 빠르고 정확)
    - 그 외 자연어 문장 → DL (의미매칭)
    반환: (method, 이유)
    """
    has_korean = any('\uac00' <= c <= '\ud7a3' for c in query)
    word_count = len(query.split())
    if has_korean:
        return "dl", "한국어 쿼리 → 의미 임베딩"
    if word_count <= 3:
        return "ml", "짧은 키워드 → 단어매칭"
    return "dl", "자연어 문장 → 의미 임베딩"


def search_df(query: str, method: str = "dl", top_k: int = 10,
              min_reviews: int = 0, max_price: float = None,
              platforms: list = None, popularity_weight: float = 0.0):
    """검색 + 필터 + 결과 DataFrame 반환 (app.py용)

    method: 'ml' | 'dl' | 'auto'(쿼리 보고 자동 선택)
    popularity_weight: 0~1. 관련성 점수에 인기/평점을 섞는 비율.
                       0이면 순수 관련성순, 0.3이면 검증된 게임이 위로.
    """
    if method == "auto":
        method, _ = route_method(query)

    catalog = load_catalog()
    # 넉넉히 뽑아서 필터 후 top_k
    idx, scores = search(query, method, top_k=top_k * 15)
    res = catalog.iloc[idx].copy()
    res["relevance"] = scores

    if min_reviews:
        res = res[res["total_reviews"] >= min_reviews]
    if max_price is not None:
        res = res[res["price"] <= max_price]
    if platforms:
        for p in platforms:
            res = res[res[p] == True]  # noqa: E712

    if popularity_weight > 0 and len(res) > 0:
        # 인기 점수 = 리뷰 수(로그 정규화) × 긍정률, 0~1 스케일
        import numpy as np
        log_rev = np.log1p(res["total_reviews"])
        pop = (log_rev / log_rev.max()) * res["positive_ratio"]
        # 관련성과 인기를 가중 결합
        rel_norm = res["relevance"] / (res["relevance"].max() + 1e-9)
        res["score"] = ((1 - popularity_weight) * rel_norm
                        + popularity_weight * pop)
    else:
        res["score"] = res["relevance"]

    return res.sort_values("score", ascending=False).head(top_k)
