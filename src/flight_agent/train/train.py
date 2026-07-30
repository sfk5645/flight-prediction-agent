"""Train delay classifier + minutes regressor with MLflow tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from flight_agent.config import get_settings, load_project_config
from flight_agent.features.build import (
    CATEGORICAL,
    FEATURE_COLUMNS,
    REGRESSION_TARGET,
    TARGET,
    build_training_frame,
)

try:
    from xgboost import XGBClassifier, XGBRegressor

    _HAS_XGBOOST = True
except Exception:  # noqa: BLE001
    _HAS_XGBOOST = False


def _make_classifier(model_cfg: dict[str, Any], *, scale_pos_weight: float = 1.0):
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


def _make_regressor(model_cfg: dict[str, Any]):
    xgb_params = model_cfg.get("xgboost", {})
    if _HAS_XGBOOST:
        return (
            "xgboost",
            XGBRegressor(
                n_estimators=xgb_params.get("n_estimators", 400),
                max_depth=xgb_params.get("max_depth", 7),
                learning_rate=xgb_params.get("learning_rate", 0.05),
                subsample=xgb_params.get("subsample", 0.85),
                colsample_bytree=xgb_params.get("colsample_bytree", 0.85),
                min_child_weight=xgb_params.get("min_child_weight", 5),
                reg_lambda=xgb_params.get("reg_lambda", 1.0),
                objective="reg:squarederror",
                eval_metric="mae",
                random_state=model_cfg.get("random_state", 42),
                n_jobs=4,
            ),
        )
    return (
        "hist_gradient_boosting",
        HistGradientBoostingRegressor(
            max_depth=xgb_params.get("max_depth", 7),
            learning_rate=xgb_params.get("learning_rate", 0.05),
            max_iter=xgb_params.get("n_estimators", 400),
            random_state=model_cfg.get("random_state", 42),
        ),
    )


def _time_based_frames(df, *, test_size: float):
    """Hold out the most recent fraction of flights by fl_date (no shuffle leak)."""
    if "fl_date" not in df.columns:
        from sklearn.model_selection import train_test_split

        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    ordered = df.sort_values("fl_date").reset_index(drop=True)
    cut = max(1, int(len(ordered) * (1.0 - test_size)))
    if cut >= len(ordered):
        cut = len(ordered) - 1
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def _best_f1_threshold(y_true, proba: np.ndarray) -> tuple[float, float]:
    """Pick probability threshold that maximizes F1 on the holdout set."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.15, 0.75, 61):
        preds = (proba >= t).astype(int)
        score = float(f1_score(y_true, preds, zero_division=0))
        if score > best_f1:
            best_f1, best_t = score, float(t)
    return best_t, best_f1


def _preprocessor() -> ColumnTransformer:
    numeric = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL]
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
            ("num", "passthrough", numeric),
        ]
    )


def train_model(
    sample_limit: int | None = None,
    publish_hf: bool = False,
) -> dict[str, Any]:
    settings = get_settings()
    cfg = load_project_config()
    model_cfg = cfg["model"]
    hub_pair_only = bool(model_cfg.get("hub_pair_only", False))
    tune_threshold = bool(model_cfg.get("tune_threshold", True))
    train_regressor = bool(model_cfg.get("train_regressor", True))
    clip_lo = float(model_cfg.get("regression_clip_min", -30))
    clip_hi = float(model_cfg.get("regression_clip_max", 240))

    df = build_training_frame(sample_limit=sample_limit)
    if df.empty:
        raise RuntimeError(
            "Training frame is empty — push curated marts to R2 "
            "(`flight dbt build --from-r2` / `flight warehouse push`)."
        )
    if REGRESSION_TARGET not in df.columns:
        raise RuntimeError(
            f"Training frame missing {REGRESSION_TARGET}; rebuild curated marts."
        )

    y_all = df[TARGET].astype(int)
    n_pos = int(y_all.sum())
    n_neg = int(len(y_all) - n_pos)
    raw_spw = (n_neg / n_pos) if n_pos else 1.0
    scale_pos_weight = float(min(max(raw_spw * 0.6, 1.0), 4.0))

    train_df, test_df = _time_based_frames(
        df,
        test_size=float(model_cfg.get("test_size", 0.2)),
    )
    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)
    print(
        f"Split (time-based): train={len(X_train):,} test={len(X_test):,} "
        f"scale_pos_weight={scale_pos_weight:.3f} hub_pair_only={hub_pair_only}",
        flush=True,
    )

    algo_name, clf, logged_params = _make_classifier(
        model_cfg, scale_pos_weight=scale_pos_weight
    )
    pre = _preprocessor()
    pipe = Pipeline([("pre", pre), ("model", clf)])

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("flight-delay")

    with mlflow.start_run(run_name=f"{algo_name}_arr_delay_15") as run:
        print(f"Fitting classifier ({algo_name})…", flush=True)
        if algo_name == "xgboost":
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
            pipe = Pipeline([("pre", pipe.named_steps["pre"]), ("model", clf)])
        else:
            pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        threshold = 0.5
        f1_at_default = float(
            f1_score(y_test, (proba >= 0.5).astype(int), zero_division=0)
        )
        if tune_threshold:
            threshold, _ = _best_f1_threshold(y_test, proba)
        preds = (proba >= threshold).astype(int)

        metrics: dict[str, Any] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "f1_at_0_5": f1_at_default,
            "roc_auc": float(roc_auc_score(y_test, proba))
            if y_test.nunique() > 1
            else 0.0,
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
            metrics["best_iteration"] = (
                int(clf.best_iteration) if clf.best_iteration is not None else -1
            )

        hold = X_test.copy()
        hold["y_true"] = y_test.values
        hold["y_pred"] = preds
        by_origin = {}
        for origin, g in hold.groupby("origin"):
            by_origin[str(origin)] = {
                "n": int(len(g)),
                "f1": float(f1_score(g["y_true"], g["y_pred"], zero_division=0)),
                "delay_rate": float(g["y_true"].mean()),
            }

        regression_metrics: dict[str, Any] | None = None
        reg_path = Path(settings.model_dir) / "model_regressor.joblib"
        if train_regressor:
            print(f"Fitting regressor ({algo_name}) on clipped arr_delay…", flush=True)
            y_reg_train = (
                train_df[REGRESSION_TARGET]
                .astype(float)
                .clip(lower=clip_lo, upper=clip_hi)
            )
            y_reg_test = (
                test_df[REGRESSION_TARGET]
                .astype(float)
                .clip(lower=clip_lo, upper=clip_hi)
            )
            reg_algo, reg_est = _make_regressor(model_cfg)
            reg_pre = _preprocessor()
            reg_pipe = Pipeline([("pre", reg_pre), ("model", reg_est)])
            if reg_algo == "xgboost":
                n_val = max(1000, int(len(X_train) * 0.1))
                X_fit = X_train.iloc[:-n_val]
                y_fit = y_reg_train.iloc[:-n_val]
                X_val = X_train.iloc[-n_val:]
                y_val = y_reg_train.iloc[-n_val:]
                reg_pipe.named_steps["pre"].fit(X_fit)
                X_fit_t = reg_pipe.named_steps["pre"].transform(X_fit)
                X_val_t = reg_pipe.named_steps["pre"].transform(X_val)
                reg_est.set_params(early_stopping_rounds=40)
                reg_est.fit(
                    X_fit_t,
                    y_fit,
                    eval_set=[(X_val_t, y_val)],
                    verbose=False,
                )
                reg_pipe = Pipeline(
                    [("pre", reg_pipe.named_steps["pre"]), ("model", reg_est)]
                )
            else:
                reg_pipe.fit(X_train, y_reg_train)

            y_hat = np.asarray(reg_pipe.predict(X_test), dtype=float)
            y_hat = np.clip(y_hat, clip_lo, clip_hi)
            mae = float(mean_absolute_error(y_reg_test, y_hat))
            rmse = float(np.sqrt(mean_squared_error(y_reg_test, y_hat)))
            # Bias: mean predicted − mean actual
            bias = float(y_hat.mean() - y_reg_test.mean())
            regression_metrics = {
                "mae_minutes": mae,
                "rmse_minutes": rmse,
                "bias_minutes": bias,
                "clip_min": clip_lo,
                "clip_max": clip_hi,
                "target": REGRESSION_TARGET,
                "algorithm": reg_algo,
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "mean_actual_minutes": float(y_reg_test.mean()),
                "mean_predicted_minutes": float(y_hat.mean()),
            }
            if reg_algo == "xgboost" and hasattr(reg_est, "best_iteration"):
                regression_metrics["best_iteration"] = (
                    int(reg_est.best_iteration)
                    if reg_est.best_iteration is not None
                    else -1
                )
            Path(settings.model_dir).mkdir(parents=True, exist_ok=True)
            joblib.dump(reg_pipe, reg_path)
            mlflow.log_metrics(
                {
                    "reg_mae_minutes": mae,
                    "reg_rmse_minutes": rmse,
                    "reg_bias_minutes": bias,
                }
            )
            mlflow.log_artifact(str(reg_path), artifact_path="model")
            print(
                f"Regressor done: mae={mae:.2f} min rmse={rmse:.2f} min "
                f"(clip=[{clip_lo:g},{clip_hi:g}])",
                flush=True,
            )

        mlflow.log_params({f"model_{k}": v for k, v in logged_params.items()})
        mlflow.log_param("algorithm", algo_name)
        mlflow.log_param("split", "time_based")
        mlflow.log_param("hub_pair_only", hub_pair_only)
        mlflow.log_param("train_regressor", train_regressor)
        mlflow.log_metrics(
            {
                k: v
                for k, v in metrics.items()
                if isinstance(v, (float, int))
                and k not in ("algorithm", "split", "hub_pair_only")
            }
        )

        model_dir = Path(settings.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.joblib"
        metrics_path = model_dir / "metrics.json"
        meta_path = model_dir / "meta.json"
        joblib.dump(pipe, model_path)
        try:
            from flight_agent.serve.services import (
                load_meta,
                load_model,
                load_regressor,
            )

            load_model.cache_clear()
            load_meta.cache_clear()
            load_regressor.cache_clear()
        except Exception:  # noqa: BLE001
            pass
        mlflow.log_artifact(str(model_path), artifact_path="model")
        payload = {"overall": metrics, "by_origin": by_origin}
        if regression_metrics is not None:
            payload["regression"] = regression_metrics
            metrics["regression"] = regression_metrics
        metrics_path.write_text(json.dumps(payload, indent=2))
        meta = {
            "feature_columns": FEATURE_COLUMNS,
            "categorical": CATEGORICAL,
            "target": TARGET,
            "regression_target": REGRESSION_TARGET if train_regressor else None,
            "algorithm": algo_name,
            "mlflow_run_id": run.info.run_id,
            "model_path": str(model_path),
            "regressor_path": str(reg_path) if train_regressor else None,
            "n_rows_sampled": int(len(df)),
            "split": "time_based",
            "threshold": float(threshold),
            "hub_pair_only": hub_pair_only,
            "regression_clip_min": clip_lo if train_regressor else None,
            "regression_clip_max": clip_hi if train_regressor else None,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(meta_path))

        sample_path = model_dir / "sample_features.parquet"
        df.head(min(2000, len(df))).to_parquet(sample_path, index=False)

        if publish_hf:
            from flight_agent.train.publish import publish_to_hf

            publish_to_hf(model_path, sample_path, metrics)

        print(
            f"Training done: roc_auc={metrics['roc_auc']:.3f} "
            f"f1={metrics['f1']:.3f} (thr={threshold:.3f}) "
            f"n_train={metrics['n_train']}",
            flush=True,
        )
        return metrics
