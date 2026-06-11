"""
[팀원 B 참고용 템플릿] 흥행 예측 (hit) — MLflow + Optuna 패턴 예시

⚠️ 이 파일은 "이렇게 하면 된다"는 참고 답안입니다.
   B가 직접 모델/탐색공간을 설계해서 자기 것으로 만드는 걸 권장합니다.
   규칙 딱 3가지만 지키면 어떤 모델이든 자유:
     1. 데이터는 load_processed("hit")로 받는다
     2. 결과는 mlflow에 기록한다 (params + metrics + model)
     3. 최고 모델을 steam_hit_predictor에 등록하고 @champion alias를 단다

실행:
    python train_hit_reference.py --trials 20
"""
import argparse
import logging
import sys
import time

import mlflow
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

sys.path.insert(0, "src")
from pipeline import load_processed

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("hit")

EXPERIMENT = "steam-ml-vs-dl"
TRACKING_URI = "sqlite:///mlflow.db"


def build_model(name, params, spw):
    if name == "LogisticRegression":
        return LogisticRegression(C=params["C"], max_iter=1000,
                                  class_weight="balanced", random_state=42)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=params["n_estimators"],
                                      max_depth=params["max_depth"],
                                      class_weight="balanced",
                                      random_state=42, n_jobs=-1)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=params["n_estimators"],
                             max_depth=params["max_depth"],
                             learning_rate=params["learning_rate"],
                             scale_pos_weight=spw, eval_metric="logloss",
                             verbosity=0, random_state=42, n_jobs=-1)
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=params["n_estimators"],
                              max_depth=params["max_depth"],
                              learning_rate=params["learning_rate"],
                              num_leaves=params["num_leaves"],
                              is_unbalance=True, verbose=-1,
                              random_state=42, n_jobs=-1)
    if name == "MLP":
        layers = tuple(params[f"units_l{i}"] for i in range(params["n_layers"]))
        return MLPClassifier(hidden_layer_sizes=layers, alpha=params["alpha"],
                             learning_rate_init=params["lr"], max_iter=300,
                             early_stopping=True, random_state=42)


def suggest(name, trial):
    if name == "LogisticRegression":
        return {"C": trial.suggest_float("C", 1e-3, 10, log=True)}
    if name == "RandomForest":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 400),
                "max_depth": trial.suggest_int("max_depth", 4, 16)}
    if name in ("XGBoost", "LightGBM"):
        p = {"n_estimators": trial.suggest_int("n_estimators", 100, 500),
             "max_depth": trial.suggest_int("max_depth", 3, 10),
             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True)}
        if name == "LightGBM":
            p["num_leaves"] = trial.suggest_int("num_leaves", 20, 150)
        return p
    if name == "MLP":
        n = trial.suggest_int("n_layers", 1, 3)
        p = {"n_layers": n,
             "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
             "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True)}
        for i in range(n):
            p[f"units_l{i}"] = trial.suggest_int(f"units_l{i}", 32, 256, step=32)
        return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    X_tr, X_te, y_tr, y_te = load_processed("hit")
    spw = float((y_tr == 0).sum() / (y_tr == 1).sum())
    log.info(f"hit: train {len(X_tr):,} (양성 {y_tr.mean()*100:.1f}%) | trials={args.trials}")

    results = []
    for name in ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "MLP"]:
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(
            lambda tr: cross_val_score(build_model(name, suggest(name, tr), spw),
                                       X_tr, y_tr, cv=3, scoring="f1").mean(),
            n_trials=args.trials, show_progress_bar=False)

        model = build_model(name, study.best_params, spw)
        t0 = time.time(); model.fit(X_tr, y_tr); train_t = time.time() - t0
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        m = {"f1": f1_score(y_te, y_pred),
             "roc_auc": roc_auc_score(y_te, y_prob),
             "pr_auc": average_precision_score(y_te, y_prob),
             "accuracy": accuracy_score(y_te, y_pred),
             "precision": precision_score(y_te, y_pred, zero_division=0),
             "recall": recall_score(y_te, y_pred, zero_division=0)}

        with mlflow.start_run(run_name=f"{name}-hit") as run:
            mlflow.log_params({**study.best_params, "model_name": name,
                               "model_family": "DL" if name == "MLP" else "ML",
                               "task": "hit", "n_trials": args.trials})
            mlflow.log_metrics({**m, "train_time_s": train_t,
                                "cv_best_f1": study.best_value})
            mlflow.sklearn.log_model(model, name="model")
            rid = run.info.run_id

        results.append({"name": name, "family": "DL" if name == "MLP" else "ML",
                        "run_id": rid, **{k: round(v, 4) for k, v in m.items()},
                        "train_s": round(train_t, 1)})
        log.info(f"  {name:18s} F1={m['f1']:.4f} AUC={m['roc_auc']:.4f}")

    df = pd.DataFrame(results).sort_values("f1", ascending=False)
    best = df.iloc[0]
    mv = mlflow.register_model(f"runs:/{best['run_id']}/model", "steam_hit_predictor")
    from mlflow.tracking import MlflowClient
    MlflowClient().set_registered_model_alias("steam_hit_predictor", "champion", mv.version)
    df.to_json("data/processed/comparison_hit.json",
               orient="records", force_ascii=False, indent=2)
    log.info(f"🏆 champion: {best['name']} (F1={best['f1']}) → steam_hit_predictor v{mv.version}")


if __name__ == "__main__":
    main()
