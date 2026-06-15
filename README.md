# 🎮 Steam Games Analytics — ML vs DL 비교 프로젝트

> Steam 게임 데이터(89,618건)를 활용해 세 가지 문제를 ML과 DL로 풀고 비교한다.
> **숨은 명작 탐지**(정형), **흥행 예측**(정형), **자연어 검색**(텍스트).

---

## 핵심 결론

**데이터 성격과 문제 특성이 모델 선택을 결정한다.**

| 문제 | 데이터 | ML 최고 | DL 최고 | 승자 |
|---|---|---|---|---|
| 숨은 명작 탐지 | 정형 83피처, 불균형 1:9.3 | XGBoost PR-AUC 0.279 | MLP PR-AUC 0.222 | **ML** |
| 흥행 예측 | 정형 83피처, 다운샘플링 1:1 | LightGBM F1 0.58 | MLP F1 **0.62** | **DL** |
| 자연어 검색 | 텍스트 (설명문 4.6만) | TF-IDF P@20 0.64 | 임베딩 (한국어·자연어) | **DL** |

흥미로운 대조: 같은 정형 데이터인데 숨은 명작(극단 불균형)에선 ML이,
흥행 예측(균형 데이터)에선 DL이 이겼다. 불균형 처리 전략까지 결과를 바꾼다.

---

## 프로젝트 구성

### ① 전처리 파이프라인 (자동화)
- 89,618 → 46,572건 (180일 미만 제거 + 리뷰 최소 기준 + 클리핑)
- Wilson score로 소표본 긍정률 보정 + Isolation Forest로 리뷰 이상 932건 탐지
- 출시 후 데이터 17개 컬럼 leakage 물리 차단
- 모든 임계값은 config.yaml에서 관리, `uv run run_pipeline.py` 한 줄 실행
- MLflow 실험 추적 + 게이팅 기반 champion 자동 등록

### ② 숨은 명작 탐지 (정형 ML vs DL — 불균형 중심)
- **문제**: 양성 9.8% 극단 불균형에서 명작 패턴 찾기
- **ML 5종 + DL 2종** 불균형 처리 전략 비교 (가중치 vs SMOTE)
- **champion**: XGBoost+SMOTE (PR-AUC 0.279, 무작위 대비 2.85배)
- **발견**: SMOTE 확률 왜곡(F1 0.12), 임계값 튜닝으로 MLP F1 0.02→0.29 회복
- **결론**: 극단 불균형 + 정형 데이터에서 트리 기반 ML이 MLP보다 강건
- PR-AUC 0.10 → 랜덤 / 0.15 → 별로 / 0.20 → 쓸만함 / 0.30 → 꽤 좋음 / 0.40+ → 매우 좋음

### ③ 흥행 예측 (정형 ML vs DL — 다운샘플링 중심)
- **문제**: 흥행작 비율 약 18%, 다운샘플링으로 1:1 균형화 후 학습
- **진행 과정**:
  - Random Forest(baseline) → 불균형 문제 인식 → 다운샘플링 적용(Recall 0.28→0.78)
  - LightGBM 교체(F1 0.58) → PyTorch MLP 검증(과적합 발생→드롭아웃 조정→해결)
- **champion**: MLP (정확도 86%, F1 0.62) — DL이 ML을 근소하게 앞섬
- **발견**: 드롭아웃 0.3/0.2→0.4/0.3 조정만으로 과적합 방어 + 성능 안정화
- **결론**: 데이터 균형이 맞으면 DL도 정형 데이터에서 ML과 경쟁 가능

### ④ 자연어 게임 검색 (텍스트 ML vs DL)
- **문제**: "잔잔한 힐링 게임" 같은 자연어로 4.6만 게임 검색
- **ML**: TF-IDF 단어매칭 — 영어 키워드는 P@20 0.64로 강함
- **DL**: Sentence-Transformer 의미 임베딩 — 한국어·자연어에서 유일하게 작동
- **하이브리드 라우팅**: 짧은 키워드→ML, 자연어·한국어→DL 자동 선택
- **결론**: 의미 매칭이 필요한 텍스트 영역에서는 DL이 압도

### ⑤ Streamlit 대시보드
- 숨은 명작 추천 (확정 명작 / 잠재 명작 / 이상 패턴)
- 흥행 예측 모델 결과
- 자연어 검색 + ML vs DL 비교 차트
- 장르·가격 분석

---

## ML vs DL — 세 문제의 대조가 말하는 것

| | 숨은 명작 | 흥행 예측 | 자연어 검색 |
|---|---|---|---|
| 불균형 | 극단 (1:9.3) | 해소 (1:1) | 해당 없음 |
| 승자 | ML | DL | DL |
| 교훈 | 불균형이 심하면 ML이 안전 | 균형 맞추면 DL도 경쟁력 | 텍스트는 DL 영역 |

→ "무엇이 더 좋은가"에 대한 답은 없고, **"이 데이터·이 상황에서는 이것"**이 있을 뿐이다.

---

## 실행

```bash
uv run run_all.py              # 전처리 → 숨은명작 → 검색 인덱스 → 평가
uv run streamlit run Home.py   # 대시보드
```

## 기술 스택

Python 3.12 · uv · pandas · scikit-learn · XGBoost · LightGBM ·
PyTorch · imbalanced-learn · MLflow · sentence-transformers · Streamlit · Plotly

## 데이터

Kaggle [artermiloff/steam-games-dataset](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset)
(games_march2025_cleaned.csv, 89,618행 × 47열)
