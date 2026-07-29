"""Train delay classifier with MLflow tracking (XGBoost or sklearn HGB fallback)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from flight_agent.config import get_settings, load_project_config
from flight_agent.features.build import CATEGORICAL, FEATURE_COLUMNS, TARGET, build_training_frame

try:
    from xgboost import XGBClassifier

    _HAS_XGBOOST = True
except Exception:  # noqa: BLE001
    _HAS_XGBOOST = False


def _make_estimator(model_cfg: dict[str, Any], *, scale_pos_weight: float = 1.0):
    xgb_params = model_cfg.get("xgboost", {})
    if _HAS_XGBOOST:
        return (
            "xgboost",
            XGBClassifier(
                n_estimators=xgb_params.get("n_estimators", 400),
                max_depth=xgb_params.get("max_depth", 7),
                learning_rate=xgb_params.get("learning_rate", 0.05),
                subsample=xgb_params.get("subsample", 0.85),
                colsample_bytree=xgb_params.get("colsample_bytree", 0.85),
                min_child_weight=xgb_params.get("min_child_weight", 5),
                reg_lambda=xgb_params.get("reg_lambda", 1.0),
                scale_pos_weight=scale_pos_weight,
                objective="binary:logistic",
                eval_metric="auc",
                random_state=model_cfg.get("random_state", 42),
                n_jobs=4,
            ),
            {**xgb_params, "scale_pos_weight": scale_pos_weight},
        )
    return (
        "hist_gradient_boosting",
        HistGradientBoostingClassifier(
            max_depth=xgb_params.get("max_depth", 7),
            learning_rate=xgb_params.get("learning_rate", 0.05),
            max_iter=xgb_params.get("n_estimators", 400),
            class_weight="balanced",
            random_state=model_cfg.get("random_state", 42),
        ),
        {"fallback": "HistGradientBoostingClassifier", **xgb_params},
    )


def _time_based_split(df, *, test_size: float, random_state: int):
    """Hold out the most recent fraction of flights by fl_date (no shuffle leak)."""
    if "fl_date" not in df.columns:
        from sklearn.model_selection import train_test_split

        X = df[FEATURE_COLUMNS]
        y = df[TARGET].astype(int)
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    ordered = df.sort_values("fl_date").reset_index(drop=True)
    cut = max(1, int(len(ordered) * (1.0 - test_size)))
    if cut >= len(ordered):
        cut = len(ordered) - 1
    train_df = ordered.iloc[:cut]
    test_df = ordered.iloc[cut:]
    return (
        train_df[FEATURE_COLUMNS],
        test_df[FEATURE_COLUMNS],
        train_df[TARGET].astype(int),
        test_df[TARGET].astype(int),
    )


def _best_f1_threshold(y_true, proba: np.ndarray) -> tuple[float, float]:
    """Pick probability threshold that maximizes F1 on the holdout set."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.15, 0.75, 61):
        preds = (proba >= t).astype(int)
        score = float(f1_score(y_true, preds, zero_division=0))
        if score > best_f1:
            best_f1, best_t = score, float(t)
    return best_t, best_f1


def train_model(
    sample_limit: int | None = None,
    publish_hf: bool = False,
    publish_r2: bool | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    cfg = load_project_config()
    model_cfg = cfg["model"]
    hub_pair_only = bool(model_cfg.get("hub_pair_only", False))
    tune_threshold = bool(model_cfg.get("tune_threshold", True))

    df = build_training_frame(sample_limit=sample_limit)
    if df.empty:
        raise RuntimeError(
            "Training frame is empty — push curated marts to R2 "
            "(`flight dbt build --from-r2` / `flight warehouse push`)."
        )

    y_all = df[TARGET].astype(int)
    n_pos = int(y_all.sum())
    n_neg = int(len(y_all) - n_pos)
    # Milder than full neg/pos — avoids over-predicting delays.
    raw_spw = (n_neg / n_pos) if n_pos else 1.0
    scale_pos_weight = float(min(max(raw_spw * 0.6, 1.0), 4.0))

    X_train, X_test, y_train, y_test = _time_based_split(
        df,
        test_size=float(model_cfg.get("test_size", 0.2)),
        random_state=int(model_cfg.get("random_state", 42)),
    )
    print(
        f"Split (time-based): train={len(X_train):,} test={len(X_test):,} "
        f"scale_pos_weight={scale_pos_weight:.3f} hub_pair_only={hub_pair_only}",
        flush=True,
    )

    numeric = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL]
    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
            ("num", "passthrough", numeric),
        ]
    )

    algo_name, clf, logged_params = _make_estimator(
        model_cfg, scale_pos_weight=scale_pos_weight
    )
    pipe = Pipeline([("pre", pre), ("model", clf)])

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("flight-delay")

    with mlflow.start_run(run_name=f"{algo_name}_arr_delay_15") as run:
        print(f"Fitting {algo_name}…", flush=True)
        fit_kwargs: dict[str, Any] = {}
        if algo_name == "xgboost":
            # Early stopping on a time-based validation slice of the train set.
            n_val = max(1000, int(len(X_train) * 0.1))
            X_fit, y_fit = X_train.iloc[:-n_val], y_train.iloc[:-n_val]
            X_val, y_val = X_train.iloc[-n_val:], y_train.iloc[-n_val:]
            pipe.named_steps["pre"].fit(X_fit)
            X_fit_t = pipe.named_steps["pre"].transform(X_fit)
            X_val_t = pipe.named_steps["pre"].transform(X_val)
            clf.set_params(early_stopping_rounds=40)
            clf.fit(
                X_fit_t,
                y_fit,
                eval_set=[(X_val_t, y_val)],
                verbose=False,
            )
            # Rebuild pipeline with fitted pre + model for joblib serving.
            pipe = Pipeline([("pre", pipe.named_steps["pre"]), ("model", clf)])
        else:
            pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        threshold = 0.5
        f1_at_default = float(f1_score(y_test, (proba >= 0.5).astype(int), zero_division=0))
        if tune_threshold:
            threshold, _ = _best_f1_threshold(y_test, proba)
        preds = (proba >= threshold).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "f1_at_0_5": f1_at_default,
            "roc_auc": float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else 0.0,
            "threshold": float(threshold),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_rows_sampled": int(len(df)),
            "positive_rate": float(y_all.mean()),
            "scale_pos_weight": float(scale_pos_weight),
            "hub_pair_only": hub_pair_only,
            "algorithm": algo_name,
            "split": "time_based",
        }
        if algo_name == "xgboost" and hasattr(clf, "best_iteration"):
            metrics["best_iteration"] = int(clf.best_iteration) if clf.best_iteration is not None else -1

        test_df = X_test.copy()
        test_df["y_true"] = y_test.values
        test_df["y_pred"] = preds
        by_origin = {}
        for origin, g in test_df.groupby("origin"):
            by_origin[str(origin)] = {
                "n": int(len(g)),
                "f1": float(f1_score(g["y_true"], g["y_pred"], zero_division=0)),
                "delay_rate": float(g["y_true"].mean()),
            }

        mlflow.log_params({f"model_{k}": v for k, v in logged_params.items()})
        mlflow.log_param("algorithm", algo_name)
        mlflow.log_param("split", "time_based")
        mlflow.log_param("hub_pair_only", hub_pair_only)
        mlflow.log_metrics(
            {
                k: v
                for k, v in metrics.items()
                if isinstance(v, (float, int)) and k not in ("algorithm", "split", "hub_pair_only")
            }
        )

        model_dir = Path(settings.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.joblib"
        metrics_path = model_dir / "metrics.json"
        meta_path = model_dir / "meta.json"
        joblib.dump(pipe, model_path)
        try:
            from flight_agent.serve.services import load_meta, load_model

            load_model.cache_clear()
            load_meta.cache_clear()
        except Exception:  # noqa: BLE001
            pass
        mlflow.log_artifact(str(model_path), artifact_path="model")
        metrics_path.write_text(json.dumps({"overall": metrics, "by_origin": by_origin}, indent=2))
        meta = {
            "feature_columns": FEATURE_COLUMNS,
            "categorical": CATEGORICAL,
            "target": TARGET,
            "algorithm": algo_name,
            "mlflow_run_id": run.info.run_id,
            "model_path": str(model_path),
            "n_rows_sampled": int(len(df)),
            "split": "time_based",
            "threshold": float(threshold),
            "hub_pair_only": hub_pair_only,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(meta_path))

        sample_path = model_dir / "sample_features.parquet"
        df.head(min(2000, len(df))).to_parquet(sample_path, index=False)

        if publish_hf:
            from flight_agent.train.publish import publish_to_hf

            publish_to_hf(model_path, sample_path, metrics)

        do_publish_r2 = settings.r2_configured if publish_r2 is None else publish_r2
        if do_publish_r2:
            from flight_agent.train.r2_model import push_model_to_r2

            push_model_to_r2(model_dir)

        print(
            f"Training done: roc_auc={metrics['roc_auc']:.3f} "
            f"f1={metrics['f1']:.3f} (thr={threshold:.3f}) "
            f"n_train={metrics['n_train']}",
            flush=True,
        )
        return metrics
