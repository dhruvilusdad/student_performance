"""Artifact loading, prediction, and model-level explanations."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import normalize_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"


def load_artifact(path: str | Path = MODEL_PATH) -> dict[str, Any]:
    """Load the fitted pipeline and metadata package."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError("No trained model is available yet. Train a model first.")
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact or "metadata" not in artifact:
        raise ValueError("The model artifact is invalid or incompatible with this application.")
    return artifact


def predict_student(artifact: dict[str, Any], student: pd.DataFrame) -> float:
    """Predict one or more rows using the saved, complete pipeline."""
    feature_columns = artifact["metadata"]["feature_columns"]
    missing = [column for column in feature_columns if column not in student.columns]
    if missing:
        raise ValueError(f"Prediction input is missing required features: {', '.join(missing)}")
    prediction = artifact["pipeline"].predict(normalize_features(student[feature_columns]))
    return float(np.asarray(prediction).reshape(-1)[0])


def _source_feature(name: str, feature_columns: list[str]) -> str:
    ordered_features = sorted(feature_columns, key=len, reverse=True)
    return next(
        (feature for feature in ordered_features
         if name.startswith(f"numerical__{feature}")
         or name.startswith(f"categorical__{feature}_")
         or name == f"categorical__{feature}"),
        name,
    )


def feature_importance(artifact: dict[str, Any]) -> pd.DataFrame:
    """Return global model importance, aggregating one-hot values by source feature."""
    pipeline = artifact["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype=float)).reshape(-1)
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])
    try:
        transformed_names = preprocessor.get_feature_names_out()
    except (AttributeError, ValueError):
        return pd.DataFrame(columns=["Feature", "Importance"])
    if len(transformed_names) != len(values):
        return pd.DataFrame(columns=["Feature", "Importance"])

    feature_columns = artifact["metadata"]["feature_columns"]
    aggregated: dict[str, float] = {}
    for transformed_name, value in zip(transformed_names, values):
        source = _source_feature(str(transformed_name), feature_columns)
        aggregated[source] = aggregated.get(source, 0.0) + float(value)
    result = pd.DataFrame(
        [{"Feature": feature, "Importance": importance} for feature, importance in aggregated.items()]
    )
    return result.sort_values("Importance", ascending=False).reset_index(drop=True)


def shap_importance(artifact: dict[str, Any], sample: pd.DataFrame) -> pd.DataFrame | None:
    """Try SHAP global importance when installed; otherwise allow fallback."""
    try:
        shap = importlib.import_module("shap")

        pipeline = artifact["pipeline"]
        preprocessor = pipeline.named_steps["preprocessor"]
        estimator = pipeline.named_steps["model"]
        transformed = preprocessor.transform(sample[artifact["metadata"]["feature_columns"]])
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        explainer = shap.Explainer(estimator, transformed[: min(50, len(transformed))])
        explanation = explainer(transformed[: min(100, len(transformed))])
        values = np.asarray(explanation.values)
        if values.ndim == 3:
            values = values[..., 0]
        names = preprocessor.get_feature_names_out()
        importance = np.abs(values).mean(axis=0)
        feature_columns = artifact["metadata"]["feature_columns"]
        aggregated: dict[str, float] = {}
        for name, value in zip(names, importance):
            source = _source_feature(str(name), feature_columns)
            aggregated[source] = aggregated.get(source, 0.0) + float(value)
        return pd.DataFrame(
            [{"Feature": feature, "Mean absolute SHAP value": value}
             for feature, value in aggregated.items()]
        ).sort_values("Mean absolute SHAP value", ascending=False).reset_index(drop=True)
    except Exception:
        return None
