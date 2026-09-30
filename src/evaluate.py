"""Regression metrics and model-comparison helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Calculate standard regression metrics on a held-out data split."""
    mse = float(mean_squared_error(actual, predicted))
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan"),
    }


def comparison_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Order candidates by training-fold CV RMSE, keeping holdout for reporting."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["CV RMSE", "CV R2", "RMSE"], ascending=[True, False, True], na_position="last"
    ).reset_index(drop=True)
