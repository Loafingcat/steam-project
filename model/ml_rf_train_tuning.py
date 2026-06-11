import sys
import os
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 파이썬 실행 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.pipeline import load_processed

def main():
    print("[TUNING] 5-Fold 교차 검증용 데이터 로딩 중...")
    
    # 텍스트 피처(TF-IDF 300차원)를 제외한 83개의 핵심 정형 피처만 로드.
    X_train, X_test, y_train, y_test = load_processed("hit", with_text=False)
    
    # 탐색할 랜덤 포레스트 하이퍼파라미터 그리드 정의 (5 x 5 = 총 25개 조합)
    # n_estimators: 생성할 결정 트리의 개수 (많을수록 모델이 안정적이나 메모리와 연산량 급증)
    # max_depth: 트리의 최대 깊이 (과적합을 방지하고 균형 잡힌 규칙을 찾기 위한 마지노선 설정)
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [8, 12, 16, 20, 24]
    }
    
    print("\n[TUNING] 5-Fold 교차 검증 가성비 분석 그리드 서치 시작...")
    
    # n_jobs=-1로 설정하여 로컬 가용 CPU 코어를 모두 활용해 병렬 연산을 수행
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # 5-Fold(cv=5)를 적용하여 데이터의 편향을 방지
    # 전체 데이터를 5개 조각으로 쪼개어 80%로 학습하고 20%로 평가하는 과정을 5번 반복
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=5, 
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1 # 학습 진행률 트리거 표시
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    total_tuning_time = time.time() - start_time
    
    # 25개 조합에 대한 교차 검증 결과를 데이터프레임으로 파싱하여 자원 효율성 분석
    cv_results = pd.DataFrame(grid_search.cv_results_)
    
    # [가성비 점수 산출 로직]
    # 모델의 예측력(mean_test_score)을 연산에 걸린 평균 시간(mean_fit_time)으로 나눕니다.
    # 분모에 1을 더해 학습 속도가 0초에 가까운 가벼운 모델의 분모가 0이 되는 예외 상황을 방지합니다.
    # 이 연산을 통해 성능 상승 곡선이 완만해지면서 시간만 무한정 잡아먹는 헤비한 조합의 패널티를 계산합니다.
    cv_results['efficiency_score'] = cv_results['mean_test_score'] / (cv_results['mean_fit_time'] + 1)
    
    # 전체 결과 중 순수 점수(F1-Score)가 1등인 인덱스 추출
    best_perf_row = cv_results.loc[cv_results['mean_test_score'].idxmax()]
    
    # 전체 결과 중 시간 대비 성능 가성비(efficiency_score)가 1등인 인덱스 추출
    best_eff_row = cv_results.loc[cv_results['efficiency_score'].idxmax()]
    
    print("\n" + "="*60)
    print("            📊 5-Fold 하이퍼파라미터 가성비 분석 완료")
    print("="*60)
    print(f"🥇 [단순 성능 최고 조합] (인프라 자원 소모 무시)")
    print(f"   👉 파라미터:  max_depth={best_perf_row['param_max_depth']}, n_estimators={best_perf_row['param_n_estimators']}")
    print(f"   👉 5-Fold 검증 점수: {best_perf_row['mean_test_score']:.4f} | 평균 학습 시간: {best_perf_row['mean_fit_time']:.2f}초")
    print("-"*60)
    print(f"⚡ [추천 가성비 최고 조합] (인프라 비용 및 서비스 응답 속도 반영)")
    print(f"   👉 파라미터:  max_depth={best_eff_row['param_max_depth']}, n_estimators={best_eff_row['param_n_estimators']}")
    print(f"   👉 5-Fold 검증 점수: {best_eff_row['mean_test_score']:.4f} | 평균 학습 시간: {best_eff_row['mean_fit_time']:.2f}초")
    print("="*60)
    print(f"[알림] 5-Fold 전체 탐색 완료 시간: {total_tuning_time:.2f}초")
    print("위의 ⚡[추천 가성비 최고 조합] 수치를 'ml_rf_train.py'에 기입하고 최종 학습을 진행하세요!\n")

if __name__ == "__main__":
    main()