"""
MLOps 파이프라인 전체 실행 (DAG)

    python run_all.py                  # 전처리 → 숨은명작 학습
    python run_all.py --quick          # 빠른 검증 모드
    python run_all.py --ml-only        # ML만 (CI/CD용)

실행 후:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    streamlit run app.py
"""
import argparse
import subprocess
import sys
import time


def run(cmd, name):
    print(f"\n{'━'*55}\n━━━ {name}\n{'━'*55}")
    if subprocess.run(cmd).returncode != 0:
        print(f"❌ {name} 실패 — 중단"); sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ml-only", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    py = sys.executable

    run([py, "run_pipeline.py"], "Stage 1: 전처리 파이프라인")

    gem_cmd = [py, "train_hidden_gem.py"]
    if args.quick:
        gem_cmd.append("--quick")
    if args.ml_only:
        gem_cmd.append("--ml-only")
    run(gem_cmd, "Stage 2: 숨은 명작 탐지")

    print(f"\n{'═'*55}")
    print(f"✅ 전체 완료 ({(time.time()-t0)/60:.1f}분)")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("  streamlit run app.py")
    print("═" * 55)


if __name__ == "__main__":
    main()