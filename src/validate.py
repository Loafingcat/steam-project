"""전처리 산출물 자동 검증 — 예상 범위 벗어나면 실패"""
import json
import sys

def validate(meta_path="data/processed/pipeline_meta.json"):
    with open(meta_path) as f:
        meta = json.load(f)

    checks = []

    # 행 수 검증 (±20% 이내)
    rows = meta["rows_after_cleaning"]
    checks.append(("행 수", 35000 <= rows <= 55000, f"{rows:,}"))

    # 양성비 검증
    for task in ["hit", "hidden_gem"]:
        ratio = meta["label_stats"][task]["positive_ratio"]
        checks.append((f"{task} 양성비", 0.05 <= ratio <= 0.40, f"{ratio:.3f}"))

    # 피처 수 검증
    n_feat = meta["final_feature_count"]
    checks.append(("피처 수", 50 <= n_feat <= 120, str(n_feat)))

    # leakage 차단 확인
    n_leak = len(meta["leakage_columns_dropped"])
    checks.append(("leakage 차단", n_leak >= 15, f"{n_leak}개"))

    failed = [(name, val) for name, ok, val in checks if not ok]
    for name, ok, val in checks:
        print(f"  {'✅' if ok else '❌'} {name}: {val}")

    if failed:
        print(f"\n❌ 검증 실패 {len(failed)}건 — 파이프라인 결과를 확인하세요")
        sys.exit(1)
    else:
        print("\n✅ 전체 검증 통과")

if __name__ == "__main__":
    validate()