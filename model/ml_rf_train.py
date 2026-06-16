import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

# 파이썬 실행 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.pipeline import load_processed

from sklearn.utils import resample

print("[ML] 스팀 흥행 예측 데이터 로딩 중(정형 데이터 전용)")

# 텍스트 피처를 제외한 83개의 핵심 정형 피처셋만 로드
X_train, X_test, y_train, y_test = load_processed("hit", with_text=False)

print(f"불균형 조정 전 학습 데이터 피처 개수: {X_train.shape[1]}개")
print(f"불균형 조정 전 레이블 분포: 실패(0)={sum(y_train==0)}개, 성공(1)={sum(y_train==1)}개")

# =========================================================================
# 🌟 [다운샘플링 데이터 레벨 개선 구간]
# 다수 클래스(실패)의 양을 소수 클래스(성공) 수준으로 낮춰 데이터 비율을 1:1로 맞춥니다.
# =========================================================================

# 1. 조작을 편하게 하기 위해 분리된 X_train과 y_train을 하나의 데이터프레임으로 결합
train_data = pd.concat([X_train, y_train], axis=1)

# 2. 레이블(y) 값을 기준으로 흥행 실패(0) 데이터와 흥행 성공(1) 데이터를 분리
df_failure = train_data[train_data[y_train.name] == 0]
df_success = train_data[train_data[y_train.name] == 1]

# 3. 실패(0) 데이터셋에서 성공(1) 데이터 개수와 똑같은 양만큼 무작위 비복원 추출
df_failure_downsampled = resample(
    df_failure, 
    replace=False, 
    n_samples=len(df_success), 
    random_state=42
)

# 4. 다운샘플링된 실패 데이터와 기존 성공 데이터를 다시 하나로 결합
df_train_balanced = pd.concat([df_failure_downsampled, df_success])

# 5. 모델 학습에 사용할 수 있도록 다시 피처(X)와 레이블(y) 세트로 쪼개기
X_train = df_train_balanced.drop(columns=[y_train.name])
y_train = df_train_balanced[y_train.name]

print(f"다운샘플링 적용 후 레이블 분포: 실패(0)={sum(y_train==0)}개, 성공(1)={sum(y_train==1)}개 (1:1 완료)")
# =========================================================================

print("랜덤 포레스트 모델 학습 시작...")

# 하이퍼파라미터 설정: 튜닝 단계(ml_rf_train_tuning.py)에서 도출된 
# 성능과 인프라 비용(학습 속도) 간의 최적의 가성비 수치를 반영하여 학습을 진행(ml_rf_train_tuning으로 값 도출)
# n_jobs=-1을 통해 멀티코어 병렬 연산을 수행함으로써 대규모 인프라 자원을 효율적으로 소모하도록 함.
rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1,
                                  class_weight="balanced")  # class_weight="balanced" <- 데이터 비율에 맞춰 자동으로 소수 클래스(1)에 가중치를 부여
rf_model.fit(X_train, y_train)

# 예측 및 성능 평가 프로세스
rf_pred = rf_model.predict(X_test)
print("\n=== ✨ Random Forest 결과 보고서 ===")
print(f"테스트 정확도(Accuracy): {accuracy_score(y_test, rf_pred):.4f}")
print("\n상세 분류 지표:")
# 과적합 여부와 데이터 불균형 상태를 진단하기 위해 불균형 성능 지표(Precision, Recall, F1-Score)를 출력합니다.
print(classification_report(y_test, rf_pred))

print("특성 중요도 그래프 생성 중...")

#특성 중요도 분석 및 시각화
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1] # 중요도가 높은 순으로 내림차순 정렬
feature_names = X_train.columns

# 시각화 가독성을 위해 영향력이 가장 높은 상위 15개의 핵심 피처만 매끄럽게 필터링합니다.
top_n = min(15, X_train.shape[1])

plt.figure(figsize=(10, 6))
plt.title("Steam Game Success - Top Feature Importances", fontsize=14)
plt.bar(range(top_n), importances[indices[:top_n]], align="center", color="skyblue")
plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.tight_layout()

plt.show()