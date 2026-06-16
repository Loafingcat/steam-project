import sys
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
import matplotlib.pyplot as plt

# 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.pipeline import load_processed

print("[ML] LightGBM 기반 스팀 흥행 예측 데이터 로딩 중")
X_train, X_test, y_train, y_test = load_processed("hit", with_text=False)

# 다운샘플링 적용 (1:1 비율 유지)
train_data = pd.concat([X_train, y_train], axis=1)
df_failure = train_data[train_data[y_train.name] == 0]
df_success = train_data[train_data[y_train.name] == 1]
df_failure_downsampled = resample(df_failure, replace=False, n_samples=len(df_success), random_state=42)
df_train_balanced = pd.concat([df_failure_downsampled, df_success])

X_train = df_train_balanced.drop(columns=[y_train.name])
y_train = df_train_balanced[y_train.name]

print("LightGBM 모델 학습 시작...")

# LightGBM 설정
lgbm_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)

lgbm_model.fit(X_train, y_train)

# 결과 출력
lgbm_pred = lgbm_model.predict(X_test)
print("\n=== ✨ LightGBM 결과 보고서 ===")
print(f"테스트 정확도(Accuracy): {accuracy_score(y_test, lgbm_pred):.4f}")
print("\n상세 분류 지표:")
print(classification_report(y_test, lgbm_pred))

# 중요도 시각화
plt.figure(figsize=(10, 6))
lgb.plot_importance(lgbm_model, max_num_features=15, title="LightGBM Top Feature Importances")
plt.tight_layout()
plt.show()