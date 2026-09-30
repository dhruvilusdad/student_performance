"""Model training, tuning, held-out evaluation, and artifact persistence."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, cast

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.data_loader import PROJECT_ROOT
from src.evaluate import comparison_frame, regression_metrics
from src.preprocessing import make_preprocessor, normalize_features
from src.utils import json_safe, target_scale

RANDOM_STATE = 42
MODEL_DIR = PROJECT_ROOT / "models"


def _pipeline(features: pd.DataFrame, estimator: Any) -> Pipeline:
    return Pipeline(
        steps=[("preprocessor", make_preprocessor(features)), ("model", estimator)]
    )


def _model_candidates() -> dict[str, Any]:
    candidates: dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    try:
        if importlib.util.find_spec("xgboost") is not None:
            xgboost = __import__("xgboost")
            XGBRegressor = xgboost.XGBRegressor

            candidates["XGBoost"] = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
    except (ImportError, ValueError):
        pass
    return candidates


def train_and_evaluate(
    frame: pd.DataFrame,
    target_column: str,
    model_dir: str | Path = MODEL_DIR,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compare regressors on one holdout split and persist the selected pipeline."""
    if target_column not in frame.columns:
        raise ValueError("Select a target column that exists in the dataset.")
    clean = frame.drop_duplicates().copy()
    target = pd.to_numeric(clean[target_column], errors="coerce")
    valid_target = target.notna()
    clean = clean.loc[valid_target].copy()
    target = target.loc[valid_target].astype(float)
    if len(clean) < 10:
        raise ValueError("At least ten rows with numeric target values are required for reliable holdout and cross-validation metrics.")

    feature_columns = [column for column in clean.columns if column != target_column]
    feature_columns = [column for column in feature_columns if clean[column].notna().any()]
    if not feature_columns:
        raise ValueError("No usable feature columns remain after excluding the target.")
    features = normalize_features(clean[feature_columns])
    if len(features.columns) == 0:
        raise ValueError("At least one usable feature column is required.")

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_STATE
    )
    folds = min(5, max(2, len(X_train) // 2))
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    candidates = _model_candidates()
    fitted_candidates: dict[str, Pipeline] = {}
    rows: list[dict[str, object]] = []

    for name, estimator in candidates.items():
        pipeline = _pipeline(X_train, estimator)
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=folds,
            scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
            n_jobs=1,
            error_score="raise",
        )
        pipeline.fit(X_train, y_train)
        metrics = regression_metrics(y_test, pipeline.predict(X_test))
        rows.append(
            {
                "Model": name,
                **metrics,
                "CV RMSE": float(-cv_scores["test_rmse"].mean()),
                "CV R2": float(cv_scores["test_r2"].mean()),
            }
        )
        fitted_candidates[name] = pipeline

    tuning_spaces = {
        "Random Forest (tuned)": (
            RandomForestRegressor(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
        "Gradient Boosting (tuned)": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [75, 125, 200],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [2, 3],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
    }
    for name, (estimator, parameters) in tuning_spaces.items():
        search = RandomizedSearchCV(
            estimator=_pipeline(X_train, estimator),
            param_distributions=parameters,
            n_iter=4,
            scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
            cv=folds,
            random_state=RANDOM_STATE,
            n_jobs=1,
            refit="rmse",
        )
        search.fit(X_train, y_train)
        pipeline = cast(Pipeline, search.best_estimator_)
        best_index = int(search.best_index_)
        metrics = regression_metrics(y_test, pipeline.predict(X_test))
        rows.append(
            {
                "Model": name,
                **metrics,
                "CV RMSE": float(-search.cv_results_["mean_test_rmse"][best_index]),
                "CV R2": float(search.cv_results_["mean_test_r2"][best_index]),
            }
        )
        fitted_candidates[name] = pipeline

    results = comparison_frame(rows)
    best_name = str(results.iloc[0]["Model"])
    best_pipeline = fitted_candidates[best_name]
    best_pipeline.fit(features, target)
    scale = target_scale(target)
    metadata: dict[str, Any] = {
        "target_column": target_column,
        "feature_columns": feature_columns,
        "model_name": best_name,
        "selection_metric": "Lowest cross-validation RMSE on the training split",
        "target_scale": scale,
        "metrics": results.iloc[0][["MAE", "MSE", "RMSE", "R2"]].to_dict(),
        "training_rows": int(len(features)),
        "holdout_rows": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "category_logic": scale["strategy"],
    }
    artifact = {"pipeline": best_pipeline, "metadata": metadata}
    joblib.dump(artifact, model_dir / "best_model.pkl")
    results.to_csv(model_dir / "model_comparison.csv", index=False)
    with (model_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(json_safe(metadata), file, indent=2, ensure_ascii=False)
    return artifact, results
