"""Dataset loading, inspection, and data-driven target discovery."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "student-performance.csv"
TARGET_ALIASES = {
    "final_score",
    "performance",
    "score",
    "g3",
    "final_grade",
    "grade",
}


class DatasetSummary(TypedDict):
    rows: int
    columns: int
    column_names: list[str]
    data_types: dict[str, str]
    missing_values: dict[str, int]
    duplicate_rows: int


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the real CSV and provide actionable errors for common file issues."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            "Please place the Kaggle dataset at data/student-performance.csv"
        )
    try:
        frame = pd.read_csv(csv_path)
    except EmptyDataError as exc:
        raise ValueError("The CSV file is empty or has no readable header.") from exc
    except ParserError as exc:
        raise ValueError("The CSV could not be parsed. Check its delimiter and quoting.") from exc
    if frame.empty:
        raise ValueError("The CSV contains no student records.")
    if frame.shape[1] == 0:
        raise ValueError("The CSV contains no columns.")
    frame.columns = [str(column).strip() for column in frame.columns]
    if any(not column for column in frame.columns):
        raise ValueError("Every CSV column must have a non-empty name.")
    if frame.columns.duplicated().any():
        raise ValueError("Column names must be unique after trimming whitespace.")
    return frame


def dataset_summary(frame: pd.DataFrame) -> DatasetSummary:
    """Return basic dataset health and schema information for the dashboard."""
    data_types = {str(column): str(dtype) for column, dtype in frame.dtypes.items()}
    missing_values = {str(column): int(value) for column, value in frame.isna().sum().items()}
    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "column_names": list(frame.columns),
        "data_types": data_types,
        "missing_values": missing_values,
        "duplicate_rows": int(frame.duplicated().sum()),
    }


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Return columns that contain numeric values or are fully numeric-like."""
    result: list[str] = []
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any() and values.notna().sum() >= frame[column].notna().sum() * 0.9:
            result.append(str(column))
    return result


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")


def target_candidates(frame: pd.DataFrame) -> list[str]:
    """Find numeric-compatible columns matching recognized target names."""
    usable_numeric = set(numeric_columns(frame))
    return [
        str(column)
        for column in frame.columns
        if column in usable_numeric and _normalize_name(str(column)) in TARGET_ALIASES
    ]


def detect_target_column(frame: pd.DataFrame) -> str | None:
    """Return a recognized target only when exactly one candidate exists."""
    candidates = target_candidates(frame)
    return candidates[0] if len(candidates) == 1 else None
