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

# =========================================================================
# ⚙️ [추가 전처리] 출시 연도(release_year) 이상치 보정
# =========================================================================
if 'release_year' in X_train.columns:
    # 2000년 이후의 정상적인 연도 데이터 기준 중앙값 계산
    valid_years = X_train['release_year'][X_train['release_year'] > 2000]
    median_year = valid_years.median() if not valid_years.empty else 2018.0
    
    # Train 및 Test 셋의 2000년 미만 유실 데이터를 중앙값으로 대체
    X_train.loc[X_train['release_year'] < 2000, 'release_year'] = median_year
    X_test.loc[X_test['release_year'] < 2000, 'release_year'] = median_year

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
# plt.show() # 스트림릿 백엔드 실행 시 블로킹 방지를 위해 주석 처리 유지

# =========================================================================
# 🌟 [스트림릿 연동을 위한 파일 추출]
# =========================================================================
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))

# 현재 폴더(model) 내부에 바로 저장하도록 경로 변경
lgbm_path = os.path.join(current_dir, "model_lgbm.pkl")
joblib.dump(lgbm_model, lgbm_path)
print(f"\n✅ [LGBM 완료] 스트림릿용 모델 저장 성공: {lgbm_path}")

feature_path = os.path.join(current_dir, "feature_names.pkl")
joblib.dump(list(X_train.columns), feature_path)
print(f"✅ [피처 완료] 컬럼 순서 리스트 저장 성공: {feature_path}")

# ⚙️ [추가 생성] 웹 앱 입력값 왜곡 방지용 StandardScaler 학습 및 저장
from sklearn.preprocessing import StandardScaler

print("\n웹 앱 연동용 전처리 스케일러(Scaler) 학습 중...")
scaler = StandardScaler()
scaler.fit(X_train)  # 학습 데이터의 평균과 표준편차를 그대로 기록

scaler_path = os.path.join(current_dir, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"✅ [스케일러 완료] 스트림릿용 전처리 스케일러 저장 성공: {scaler_path}")