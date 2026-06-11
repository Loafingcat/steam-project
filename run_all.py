"""
MLOps 파이프라인 전체 실행 (DAG)

    python run_all.py                  # 전처리 → 숨은명작 학습 → 흥행 학습
    python run_all.py --skip-hit       # 흥행(팀원 B 파트) 제외
    python run_all.py --quick          # 빠른 검증 모드
    python run_all.py --trials 20      # 흥행 Optuna trial 수

실행 후:
    mlflow ui --backend-store-uri sqlite:///mlflow.db   # localhost:5000
    streamlit run app.py                                # localhost:8501
"""
import argparse
import subprocess
import sys
import time


def run(cmd: list, name: str):
    print(f"\n{'━'*55}\n━━━ {name}\n{'━'*55}")
    if subprocess.run(cmd).returncode != 0:
        print(f"❌ {name} 실패 — 중단"); sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-hit", action="store_true",
                    help="흥행 예측(팀원 B 파트) 생략")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()

    t0 = time.time()
    py = sys.executable

    run([py, "run_pipeline.py"], "Stage 1: 전처리 파이프라인")

    gem_cmd = [py, "train_hidden_gem.py"] + (["--quick"] if args.quick else [])
    run(gem_cmd, "Stage 2a: 숨은 명작 탐지 (내 모델)")

    if not args.skip_hit:
        trials = 5 if args.quick else args.trials
        run([py, "train_hit_reference.py", "--trials", str(trials)],
            "Stage 2b: 흥행 예측 (팀원 B 참고 템플릿)")

    print(f"\n{'═'*55}")
    print(f"✅ 전체 완료 ({(time.time()-t0)/60:.1f}분)")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("  streamlit run app.py")
    print("═" * 55)


if __name__ == "__main__":
    main()
