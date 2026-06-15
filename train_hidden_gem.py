"""
[내 모델 파트] 숨은 명작 탐지 (hidden_gem)

핵심 스토리: "양성 18.5% 불균형 + 어떤 처리 전략이 최선인가"
  실험군:
    A. LogisticRegression + class_weight     (선형 baseline)
    B. RandomForest + class_weight           (배깅 앙상블)
    C. XGBoost + scale_pos_weight            (부스팅 + 가중치)
    D. XGBoost + SMOTE                       (부스팅 + 오버샘플링)
    E. LightGBM + is_unbalance               (부스팅 변형)
    F. MLP + class 가중 없음 (raw)            (DL baseline)
    G. MLP + SMOTE                           (DL + 오버샘플링)
  → ML vs DL 비교 + 가중치 vs 오버샘플링 비교가 동시에 나옴

평가: PR-AUC(Average Precision) 1차 — 불균형에서 ROC-AUC는 과대평가 경향
      F1 2차, champion에 대해 임계값 튜닝(F1 최대화) 수행

실행:
    python train_hidden_gem.py            # 전체 실험 (MLflow 기록)
    python train_hidden_gem.py --quick    # 빠른 검증용
"""
import argparse
import logging
import sys
import time

import joblib
import mlflow
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from src.gating import gate_and_register

sys.path.insert(0, "src")
from pipeline import load_processed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("hidden_gem")

EXPERIMENT = "steam-ml-vs-dl"
TRACKING_URI = "sqlite:///mlflow.db"


def build_experiments(spw: float, quick: bool):
    """(이름, 모델, SMOTE 사용 여부, family) 목록"""
    n_est = 100 if quick else 300
    mlp_iter = 100 if quick else 400
    return [
        ("LR_classweight", LogisticRegression(max_iter=1000, class_weight="balanced",
                                              random_state=42), False, "ML"),
        ("RF_classweight", RandomForestClassifier(n_estimators=n_est, max_depth=12,
                                                  class_weight="balanced",
                                                  random_state=42, n_jobs=-1), False, "ML"),
        ("XGB_posweight", XGBClassifier(n_estimators=n_est, max_depth=6,
                                        learning_rate=0.1, scale_pos_weight=spw,
                                        eval_metric="logloss", verbosity=0,
                                        random_state=42, n_jobs=-1), False, "ML"),
        ("XGB_SMOTE", XGBClassifier(n_estimators=n_est, max_depth=6,
                                    learning_rate=0.1, eval_metric="logloss",
                                    verbosity=0, random_state=42, n_jobs=-1), True, "ML"),
        ("LGBM_unbalance", LGBMClassifier(n_estimators=n_est, max_depth=6,
                                          learning_rate=0.1, is_unbalance=True,
                                          verbose=-1, random_state=42, n_jobs=-1), False, "ML"),
        ("MLP_raw", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=mlp_iter,
                                  early_stopping=True, random_state=42), False, "DL"),
        ("MLP_SMOTE", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=mlp_iter,
                                    early_stopping=True, random_state=42), True, "DL"),
    ]


def tune_threshold(y_true, y_prob):
    """PR 곡선에서 F1 최대 임계값 탐색"""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    i = int(np.argmax(f1s))
    return float(thr[i]), float(f1s[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ml-only", action="store_true",           # ← 이 줄 추가
                    help="ML 모델만 실행 (DL 제외 — CI/CD용)")  # ← 이 줄 추가
    args = ap.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    X_tr, X_te, y_tr, y_te = load_processed("hidden_gem")
    spw = float((y_tr == 0).sum() / (y_tr == 1).sum())
    log.info(f"데이터: train {len(X_tr):,} / test {len(X_te):,} | "
             f"양성 {y_tr.mean()*100:.1f}% (불균형 1:{spw:.1f})")

    results = []
    experiments = build_experiments(spw, args.quick)
    if args.ml_only:                                                          # ← 추가
        experiments = [(n,m,s,f) for n,m,s,f in experiments if f == "ML"]     # ← 추가
        log.info("  --ml-only: DL 모델 제외")                                 # ← 추가
    for name, model, use_smote, family in experiments:
        
        Xf, yf = (SMOTE(random_state=42).fit_resample(X_tr, y_tr)
                  if use_smote else (X_tr, y_tr))
        t0 = time.time()
        model.fit(Xf, yf)
        train_t = time.time() - t0

        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        m = {
            "pr_auc": average_precision_score(y_te, y_prob),
            "roc_auc": roc_auc_score(y_te, y_prob),
            "f1": f1_score(y_te, y_pred),
            "precision": precision_score(y_te, y_pred, zero_division=0),
            "recall": recall_score(y_te, y_pred, zero_division=0),
        }
        thr, f1_at_thr = tune_threshold(y_te, y_prob)

        with mlflow.start_run(run_name=f"{name}-hidden_gem") as run:
            mlflow.log_params({"model_name": name, "model_family": family,
                               "task": "hidden_gem", "imbalance_strategy":
                               "SMOTE" if use_smote else "weight/none",
                               "scale_pos_weight": round(spw, 2)})
            mlflow.log_metrics({**m, "train_time_s": train_t,
                                "best_threshold": thr, "f1_at_best_thr": f1_at_thr})
            mlflow.sklearn.log_model(model, name="model")
            run_id = run.info.run_id

        results.append({"name": name, "family": family, "run_id": run_id,
                        "smote": use_smote, **{k: round(v, 4) for k, v in m.items()},
                        "best_thr": round(thr, 3), "f1@thr": round(f1_at_thr, 4),
                        "train_s": round(train_t, 1)})
        log.info(f"  {name:16s} [{family}] PR-AUC={m['pr_auc']:.4f} "
                 f"F1={m['f1']:.4f} (thr튜닝후 {f1_at_thr:.4f}) {train_t:.0f}s")

    # ── champion: 게이팅 후 등록 ──
    df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    best = df.iloc[0]

    registered, reason = gate_and_register(
        task="hidden_gem",
        new_run_id=best["run_id"],
        new_metrics={"pr_auc": best["pr_auc"], "f1": best["f1"]},
        primary_metric="pr_auc",
        min_threshold=0.25,
    )

    df.to_json("data/processed/comparison_hidden_gem.json",
               orient="records", force_ascii=False, indent=2)
    log.info("─" * 60)
    log.info(df[["name", "family", "pr_auc", "f1", "f1@thr", "best_thr"]]
             .to_string(index=False))
    if registered:
        log.info(f"🏆 champion 교체: {best['name']} (PR-AUC={best['pr_auc']}) — {reason}")
    else:
        log.info(f"⏸️ champion 유지: {reason}")


if __name__ == "__main__":
    main()
