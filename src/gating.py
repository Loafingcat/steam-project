"""
모델 게이팅 — 새 모델이 기존 champion보다 나을 때만 교체

사용:
    from gating import gate_and_register
    gate_and_register(
        task="hidden_gem",
        new_run_id="abc123",
        new_metrics={"pr_auc": 0.41, "f1": 0.46},
        primary_metric="pr_auc",
        min_threshold=0.25,    # PR-AUC 무작위(0.197)보다 확실히 높아야
    )
"""
import logging
import mlflow
from mlflow.tracking import MlflowClient

log = logging.getLogger("pipeline")

REGISTRY_NAMES = {
    "hit": "steam_hit_predictor",
    "hidden_gem": "steam_hidden_gem_predictor",
}


def gate_and_register(task, new_run_id, new_metrics, primary_metric="pr_auc",
                      min_threshold=None):
    """
    게이팅 규칙:
      1. min_threshold가 있으면 → 이것 미달 시 등록 거부
      2. 기존 champion이 있으면 → 기존보다 나을 때만 교체
      3. 기존 champion이 없으면 → 1번만 통과하면 등록
    반환: (등록 여부, 사유)
    """
    client = MlflowClient()
    reg_name = REGISTRY_NAMES[task]
    new_val = new_metrics[primary_metric]

    # 규칙 1: 최소 기준
    if min_threshold is not None and new_val < min_threshold:
        log.warning(f"  ❌ 게이팅 실패: {primary_metric}={new_val:.4f} < "
                    f"최소 기준 {min_threshold}")
        return False, f"{primary_metric} 최소 기준 미달"

    # 기존 champion 확인
    old_val = None
    try:
        old_mv = client.get_model_version_by_alias(reg_name, "champion")
        old_run = client.get_run(old_mv.run_id)
        old_val = old_run.data.metrics.get(primary_metric)
    except Exception:
        pass  # 기존 champion 없음 → 바로 등록

    # 규칙 2: 기존보다 나아야 교체
    if old_val is not None and new_val <= old_val:
        log.warning(f"  ⏸️ 게이팅 스킵: 새 {primary_metric}={new_val:.4f} ≤ "
                    f"기존 champion {old_val:.4f}")
        return False, f"기존 champion({old_val:.4f})보다 낮음"

    # 통과 → 등록
    mv = mlflow.register_model(f"runs:/{new_run_id}/model", reg_name)
    client.set_registered_model_alias(reg_name, "champion", mv.version)
    improvement = f" (+{new_val - old_val:.4f})" if old_val else " (첫 등록)"
    log.info(f"  ✅ champion 교체: {reg_name} v{mv.version} "
             f"{primary_metric}={new_val:.4f}{improvement}")
    return True, f"등록 완료 v{mv.version}"