"""Leakage-safe preprocessing pipelines for mixed tabular features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def feature_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identify numeric and categorical columns from the supplied frame."""
    numeric = features.select_dtypes(include=["number"]).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    return numeric, categorical


def normalize_features(features: pd.DataFrame) -> pd.DataFrame:
    """Coerce mostly numeric predictors and normalize remaining categories."""
    normalized = features.copy()
    for column in normalized.columns:
        values = pd.to_numeric(normalized[column], errors="coerce")
        observed = int(normalized[column].notna().sum())
        numeric_observed = int(values.notna().sum())
        if pd.api.types.is_numeric_dtype(normalized[column]) or (
            observed > 0 and numeric_observed >= observed * 0.9
        ):
            normalized[column] = values.replace([np.inf, -np.inf], np.nan)
        else:
            category_values = normalized[column].astype("string")
            normalized[column] = category_values.astype(object).where(
                category_values.notna(), np.nan
            )
    return normalized


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build a train-fitted transformer for numerical and categorical columns."""
    numeric_columns, categorical_columns = feature_types(features)
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numerical", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    if not transformers:
        raise ValueError("At least one usable feature column is required.")
    return ColumnTransformer(transformers=transformers, remainder="drop")
