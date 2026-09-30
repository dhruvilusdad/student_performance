"""Small shared helpers for target handling and performance labels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def target_scale(values: pd.Series) -> dict[str, float | str]:
    """Describe the observed target range and the category strategy."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    minimum = float(clean.min())
    maximum = float(clean.max())
    fixed_score_bands = minimum >= 0 and maximum <= 100 and maximum >= 75
    if fixed_score_bands:
        strategy = "Fixed 0-100 score bands: Poor <40, Average <60, Good <75, Excellent otherwise."
    else:
        strategy = "Observed target range divided into four equal-width bands."
    return {
        "minimum": minimum,
        "maximum": maximum,
        "strategy": strategy,
        "fixed_score_bands": fixed_score_bands,
    }


def performance_category(prediction: float, scale: dict[str, Any]) -> str:
    """Map a prediction to a documented label without assuming its scale."""
    value = float(prediction)
    if scale["fixed_score_bands"]:
        if value < 40:
            return "Poor"
        if value < 60:
            return "Average"
        if value < 75:
            return "Good"
        return "Excellent"
    minimum = float(scale["minimum"])
    maximum = float(scale["maximum"])
    if maximum <= minimum:
        return "Average"
    fraction = (value - minimum) / (maximum - minimum)
    if fraction < 0.25:
        return "Poor"
    if fraction < 0.50:
        return "Average"
    if fraction < 0.75:
        return "Good"
    return "Excellent"


def json_safe(value: Any) -> Any:
    """Convert common NumPy and pandas scalar values to JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if pd.isna(value):
        return None
    return value
