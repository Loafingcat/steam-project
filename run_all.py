"""
MLOps 파이프라인 전체 실행 (DAG)

    uv run run_all.py                  # 전처리 → 숨은명작 → 검색인덱스 → 검색평가
    uv run run_all.py --quick          # 빠른 검증
    uv run run_all.py --ml-only        # ML만 (DL 제외, CI/CD용)
    uv run run_all.py --no-dl          # 검색 임베딩 생략

실행 후:
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
    uv run streamlit run app.py
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
    ap.add_argument("--no-dl", action="store_true", help="검색 임베딩 생략")
    args = ap.parse_args()

    t0 = time.time()
    py = sys.executable

    # Stage 1: 전처리
    run([py, "run_pipeline.py"], "Stage 1: 전처리 파이프라인")

    # Stage 2: 숨은 명작 (정형 ML vs DL)
    gem = [py, "train_hidden_gem.py"]
    if args.quick:
        gem.append("--quick")
    if args.ml_only:
        gem.append("--ml-only")
    run(gem, "Stage 2: 숨은 명작 탐지 (정형 데이터)")

    # Stage 3: 검색 인덱스 (텍스트 ML vs DL)
    idx = [py, "build_search_index.py"]
    if args.no_dl:
        idx.append("--no-dl")
    run(idx, "Stage 3: 의미 검색 인덱스 구축 (텍스트 데이터)")

    # Stage 4: 검색 평가
    if not args.no_dl:
        run([py, "eval_search.py"], "Stage 4: 검색 ML vs DL 평가")

    print(f"\n{'═'*55}")
    print(f"✅ 전체 완료 ({(time.time()-t0)/60:.1f}분)")
    print("  uv run mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("  uv run streamlit run app.py")
    print("═" * 55)


if __name__ == "__main__":
    main()
