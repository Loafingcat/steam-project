# Steam Games 전처리 파이프라인 — 설명서

> 데이터셋: Kaggle artermiloff/steam-games-dataset (`games_march2025_cleaned.csv`)
> 목적: ML vs DL 비교를 위한 흥행 예측 / 숨은 명작 탐지 데이터 생성

---

## 0. 환경세팅

uv init --python 3.12
uv add pandas numpy scikit-learn pyyaml pyarrow joblib

## 1. 한눈에 보기

| 항목 | 값 |
|---|---|
| 원본 | 89,618행 × 47열 (468MB) |
| 전처리 후 | 46,572행 × 83피처 |
| 타겟 | target_hit (흥행, 양성 15.6%) / target_hidden_gem (숨은 명작, 양성 19.7%) |
| 실행 | `uv run run_pipeline.py` (약 19초) |
| 산출물 | `data/processed/` — parquet 8개 + scaler 2개 + tfidf 3개 + meta JSON |

```
games_march2025_cleaned.csv (89,618행)
  │
  ├─[1] 로딩      대용량 13컬럼 제외 + 메모리 다운캐스팅
  ├─[2] 클리닝    미성숙작/저리뷰 제거 → 46,572행
  ├─[3] 라벨링    owners·긍정률로 타겟 2종 생성
  ├─[4] 피처      리스트 파싱·멀티핫·날짜 파생 → leakage 17컬럼 물리 제거
  └─[5] 분할/저장 stratified 8:2 → train에만 스케일링 → parquet
  │
  └─→ load_processed("hit" | "hidden_gem")  ← 팀원 인터페이스
```

---

## 2. 팀원 사용법 (모델 담당자는 이것만 알면 됨)

`src/pipeline.py`와 `data/processed/`를 같은 프로젝트에 두고:

```python
from pipeline import load_processed

# 흥행 예측
X_train, X_test, y_train, y_test = load_processed("hit")

# 숨은 명작 탐지
X_train, X_test, y_train, y_test = load_processed("hidden_gem")

# 텍스트(TF-IDF)까지 받기 — DL "정형 vs 정형+텍스트" 실험용
X_tr, X_te, y_tr, y_te, T_tr, T_te = load_processed("hit", with_text=True)
```

받는 데이터의 보장 사항:
- **스케일링 완료** (StandardScaler, train 기준)
- **leakage 차단 완료** — 출시 후 정보는 컬럼 자체가 없음. 아무 피처나 써도 안전
- **train/test 동일 분할** (random_state=42) — 팀원 간 비교 공정성 보장

---

## 3. 전처리 단계별 기준과 근거

### [1] 로딩
468MB 파일이라 메모리 관리가 첫 과제. `detailed_description`,
`screenshots`, `movies` 등 **대용량 텍스트·URL 13개 컬럼을 로딩 시점에 제외**하고,
숫자형은 float64→float32로 다운캐스팅했다.

### [2] 클리닝 — 4가지 제거 기준

| 기준 | 제거량 | 근거 |
|---|---|---|
| 출시 180일 미만 | 9,943건 | 라벨 신뢰성 (아래 상술) |
| 리뷰 10개 미만 | 33,103건 | 긍정률 표본 확보 |
| 출시일 파싱 실패 | 0건 | 날짜 포맷 정상 |
| 가격 $100 초과 클리핑 | 57건 | 이상값이 스케일링 왜곡 |

**▶ 왜 출시 180일 미만을 제거하나 (핵심 의사결정)**

두 라벨(owners, 긍정률) 모두 **출시 후 시간이 지나야 쌓이는 값**이다.
2025년 2월 출시작은 데이터 기준일(2025-03-10)까지 한 달뿐이라,
실제로는 흥행할 게임도 owners가 낮게 찍혀 "비흥행"으로 **잘못 라벨링**된다.
이건 모델이 극복할 노이즈가 아니라 **정답 자체가 틀린** 경우다.

180일(약 6개월)은 게임 초기 판매·리뷰가 안정화되는 통상 기간.
즉 이 데이터는 "출시 6개월 시점의 성과"를 기준으로 학습한다.
→ 한계: 실시간 신작 예측엔 부적합 (한계점 참조).
→ `config.yaml`의 `min_days_since_release`로 조정 가능.

**▶ 왜 리뷰 10개 미만을 제거하나**

숨은 명작 라벨이 긍정률에 의존하는데, 리뷰 3개의 "긍정률 100%"는
통계적으로 무의미하다. 최소 표본을 확보해 라벨을 신뢰 가능하게 만든다.
(숨은 명작은 추가로 30개 이상을 별도 요구)

### [3] 라벨링 — 타겟 2종

**target_hit (흥행 예측)** — `estimated_owners` 구간 하한 ≥ 100,000 → 양성 15.6%
- owners는 "100000 - 200000" 같은 **구간 문자열**이라 하한값을 파싱해 사용
- 10만은 인디 게임 기준 상업적 성공의 통상 경계선

**target_hidden_gem (숨은 명작)** — 다음 3조건 동시 만족 → 양성 19.7%
- 긍정률 ≥ 0.85 (품질 검증)
- owners < 100,000 (아직 묻혀 있음)
- 리뷰 ≥ 30 (긍정률 신뢰)
- hit 기준과 owners 임계값이 같아 **두 라벨은 상호배타**

**▶ 긍정률을 직접 계산한 이유 (데이터 함정)**

데이터에 `pct_pos_total` 컬럼이 있지만, `positive/(positive+negative)`로
직접 계산한 값과 **평균 10.7%p 차이**가 난다. pct_pos_total은 집계 기준이
다른 것으로 보여(전체 리뷰 vs 구매자 리뷰), 일관성을 위해 직접 계산했다.

### [4] 피처 엔지니어링 (83개 생성)

| 그룹 | 피처 | 설명 |
|---|---|---|
| 가격 | price, log_price, is_free | 분포가 오른쪽 꼬리 → 로그 변환. 무료는 별도 플래그 |
| 플랫폼 | windows, mac, linux, n_platforms | 이미 bool 컬럼 |
| 장르/태그/카테고리 | genre_*, tag_*, cat_* + 개수 | 상위 N개 멀티핫 (희소성 방지) |
| 날짜 | year, month, dow, month_sin/cos, is_q4 | 월은 순환 인코딩(12월↔1월 인접) |
| 개발사 | developers_game_count, is_self_published | 이름 대신 규모 통계. 자체퍼블=인디 신호 |

**▶ tags 파싱 주의** — tags는 리스트가 아니라
`{'Survival': 14838, 'Shooter': 12727, ...}` **딕셔너리 문자열**이다.
`ast.literal_eval` 후 dict면 keys만 추출해 멀티핫 처리.

**▶ Leakage 물리 차단 (이 파이프라인의 핵심)**

`positive`, `peak_ccu`, `estimated_owners`, playtime 4종 등
**출시 후에만 알 수 있는 17개 컬럼**은 라벨 생성에만 쓰고 step4에서
DataFrame에서 제거한다. 이후 단계와 팀원 데이터에는 **컬럼이 존재하지
않으므로** 실수로라도 피처에 넣을 수 없다. "주의"가 아니라 "불가능"하게 만든 설계.

### [5] 분할 / 스케일링 / 저장

- **stratify=y** — 불균형(15.6%/19.7%)에서 train/test 양성비를 동일 유지
- **scaler는 train에만 fit** — test 통계 누출(leakage)도 방지. test엔 transform만
- **TF-IDF도 train에만 fit** — short_description을 300차원으로 (DL 텍스트 실험용)
- **데이터 SHA-256 해시 기록** — "어떤 데이터로 학습했나" 추적 (MLOps 재현성)

---

## 4. 재현성

- 모든 임계값은 `configs/config.yaml`에 — 코드 수정 없이 실험 가능
  (예: 흥행 기준을 owners 5만으로 낮추면 양성비 변화 → 민감도 실험)
- 클리닝 단계별 제거 건수는 `pipeline_meta.json`에 자동 기록 —
  이 문서의 모든 수치는 거기서 나온 것
- data_hash로 데이터 버전 추적

---

## 5. 한계점 (발표용)

- **owners가 구간 추정치** — SteamSpy 추정 기반이라 정확한 판매량 아님.
  "0 - 0" 구간도 8,418건 존재
- **출시 6개월 시점으로 범위 한정** — 180일 컷의 대가. 실시간 신작 예측 부적합
- **저리뷰 게임 제외** — 진짜 '완전히 묻힌' 게임(리뷰<10)은 분석 대상 외.
  우리의 '숨은 명작'은 "최소한 발견은 된 게임 중의 명작"
- **라벨 기준의 자의성** — 10만/85%/30 기준은 합리적 추정. config로 민감도 실험해 보완

---

## 6. 파일 구조

```
steam-project/
├── configs/config.yaml      # 모든 임계값
├── src/pipeline.py          # 전처리 5단계 + load_processed()
├── run_pipeline.py          # 실행 진입점
├── data/
│   ├── raw/                 # games_march2025_cleaned.csv
│   └── processed/           # 산출물 (팀원에게 전달)
└── logs/pipeline.log        # 실행 로그
```
