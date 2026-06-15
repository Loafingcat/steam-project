# 자연어 게임 검색 — ML vs DL (의미 검색)

> "잔잔한 힐링 게임" 같은 자연어로 게임을 찾는 기능.
> 같은 검색을 ML(단어매칭)과 DL(의미매칭)으로 처리해 성능을 비교한다.

---

## 왜 이 기능인가

**스팀의 불편**: 태그 조합 필터만 있고 "분위기"로 검색이 안 됨.
"잔잔한 힐링겜"을 찾으려면 유저가 Relaxing+Casual+Singleplayer 태그를
일일이 알아서 조합해야 함.

**해결**: 게임 설명문(short_description)을 검색해서 자연어 의도와
의미가 맞는 게임을 추천. 유저는 분위기만 말하면 됨.

**ML vs DL 비교 가치**: 텍스트 데이터라 표현 방식에 따라 우열이 갈림.
정형 데이터(숨은 명작)에서는 ML이 이겼지만, 텍스트에서는 DL이 이긴다는
대조가 한 프로젝트 안에서 나옴.

---

## 두 가지 검색 방식

| 방식 | 원리 | 강점 / 약점 |
|---|---|---|
| **ML (TF-IDF)** | 단어 빈도 벡터 + 코사인 유사도 | 단어가 겹쳐야 매칭. "힐링"="relaxing" 모름. 빠름 |
| **DL (임베딩)** | Sentence-Transformer 의미 벡터 + 코사인 | 의미로 매칭. 동의어·다국어(한↔영) 가능. 모델 필요 |

핵심 차이 예시:
- 쿼리 "roguelike dungeon" → 설명문에 "roguelike" 단어가 없는 로그라이크
  게임을, **ML은 못 찾고(Precision 0%) DL은 찾음.**
- 쿼리 "잔잔한 힐링"(한국어) → **ML은 영어 설명문과 매칭 불가, DL은 가능**
  (다국어 임베딩 모델 사용 시).

---

## 실행

```bash
uv add sentence-transformers   # 최초 1회

# 인덱스 구축 (설명문 → TF-IDF + 임베딩)
uv run build_search_index.py          # 전체 (DL 포함, 최초 임베딩 5~10분)
uv run build_search_index.py --no-dl  # TF-IDF만 (빠름)

# ML vs DL 정량 평가 (MLflow 기록)
uv run eval_search.py

# Streamlit 검색 화면
uv run streamlit run app_search.py
```

---

## 평가 방법 (정답 없는 검색을 어떻게 비교하나)

검색은 정답 라벨이 없으므로 **게임의 실제 태그를 정답으로 활용**:

```
"relaxing peaceful game" 으로 검색
  → 상위 20개 결과 중 'Relaxing' 태그가 달린 게임 비율 = Precision@20
```

10개 쿼리(힐링/호러/로그라이크/퍼즐/오픈월드 등)에 대해
ML과 DL의 Precision@20, Recall@20을 측정해 MLflow에 기록.

검증된 ML 베이스라인 (TF-IDF, 46,467개 게임):
- 평균 Precision@20 ≈ 0.64
- 단어 그대로 겹치는 쿼리(horror, puzzle)는 높고(85~90%),
  단어가 안 겹치는 쿼리(roguelike)는 0% → DL이 보완할 지점

---

## 파일 구조

```
build_search_index.py     검색 인덱스 구축 (TF-IDF + 임베딩 생성)
eval_search.py            ML vs DL 정량 비교 (Precision/Recall@K)
app_search.py             Streamlit 검색 화면 (ML/DL 나란히 비교)
src/search_engine.py      검색 엔진 (search() 공통 인터페이스)
data/processed/
├── search_catalog.parquet    검색 대상 게임 메타
├── tfidf_vectorizer.joblib   ML 인덱스
├── tfidf_matrix.joblib       ML 문서 행렬
├── embeddings.npy            DL 임베딩 (--no-dl 시 없음)
├── embed_model_name.txt      임베딩 모델 이름
└── comparison_search.json    ML vs DL 평가 결과
```

---

## 임베딩 모델

`paraphrase-multilingual-MiniLM-L12-v2` (약 120MB)
- 다국어 지원 → 한국어 쿼리로 영어 설명문 검색 가능
- 최초 실행 시 HuggingFace에서 자동 다운로드 (인터넷 필요)
- 한 번 받으면 캐시되어 이후 오프라인 작동

### 더 나은 다국어 모델로 업그레이드는 가능

paraphrase-multilingual-MiniLM-L12-v2 / (현재)가볍고 빠름, 무난 / 120MB
paraphrase-multilingual-mpnet-base-v2 / 더 정확, 한국어도 더 강함 / 1.1GB
BAAI/bge-m3 / 최신, 다국어 검색 최강급 / 2.3GB

---

## 중요점

> "정형 데이터인 숨은 명작 탐지에서는 ML(부스팅)이 DL(MLP)을 이겼지만,
> 텍스트 데이터인 의미 검색에서는 DL(임베딩)이 ML(TF-IDF)을 이겼다.
> **데이터의 성격이 모델 선택을 결정한다**는 것을 한 프로젝트 안에서
> 두 사례로 실증했다."

이것이 ML vs DL 비교 프로젝트의 핵심 결론 — "무엇이 더 좋은가"가 아니라
"언제 무엇을 써야 하는가".
