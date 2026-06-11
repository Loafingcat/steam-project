import sys
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 파이썬 실행 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.pipeline import load_processed

print("[ML] 스팀 흥행 예측 데이터 로딩 중(정형 데이터 전용)")

# 텍스트 피처를 제외한 83개의 핵심 정형 피처셋만 로드
X_train, X_test, y_train, y_test = load_processed("hit", with_text=False)

print(f"데이터 로드 완료 피처 개수: {X_train.shape[1]}개")

print("랜덤 포레스트 모델 학습 시작...")

# 하이퍼파라미터 설정: 튜닝 단계(ml_rf_train_tuning.py)에서 도출된 
# 성능과 인프라 비용(학습 속도) 간의 최적의 가성비 수치를 반영하여 학습을 진행(ml_rf_train_tuning으로 값 도출)
# n_jobs=-1을 통해 멀티코어 병렬 연산을 수행함으로써 대규모 인프라 자원을 효율적으로 소모하도록 함.
rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
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