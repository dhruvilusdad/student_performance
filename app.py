"""Streamlit dashboard for student performance analytics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import (
    DEFAULT_DATA_PATH,
    detect_target_column,
    dataset_summary,
    load_dataset,
    numeric_columns,
    target_candidates,
)
from src.predict import feature_importance, load_artifact, predict_student, shap_importance
from src.train import MODEL_DIR, train_and_evaluate
from src.utils import performance_category

st.set_page_config(
    page_title="AI Student Performance Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f7f6; color: #172a25; }
    [data-testid="stMainBlockContainer"] { color: #172a25; }
    [data-testid="stMainBlockContainer"] h1,
    [data-testid="stMainBlockContainer"] h2,
    [data-testid="stMainBlockContainer"] h3,
    [data-testid="stMainBlockContainer"] h4,
    [data-testid="stMainBlockContainer"] h5,
    [data-testid="stMainBlockContainer"] p,
    [data-testid="stMainBlockContainer"] li,
    [data-testid="stMainBlockContainer"] label,
    [data-testid="stMainBlockContainer"] [data-testid="stWidgetLabel"],
    [data-testid="stMainBlockContainer"] [data-testid="stWidgetLabel"] * {
        color: #172a25 !important;
    }
    [data-testid="stMainBlockContainer"] [data-testid="stTextInput"] input,
    [data-testid="stMainBlockContainer"] [data-testid="stTextArea"] textarea,
    [data-testid="stMainBlockContainer"] [data-testid="stNumberInput"] input,
    [data-testid="stMainBlockContainer"] [data-testid="stSelectbox"] [role="combobox"] {
        color: #f8faf9 !important;
        background-color: #181a1b !important;
    }
    [data-testid="stSidebar"] { background: #172a25; }
    [data-testid="stSidebar"] * { color: #f3f6f2; }
    .hero { padding: 1.2rem 1.5rem; border-left: 5px solid #d8784b;
            background: #ffffff; margin: 0.3rem 0 1.2rem; }
    .hero h1 { color: #172a25; font-size: 2rem; margin: 0; }
    .hero p { color: #53635c; margin: .35rem 0 0; }
    div[data-testid="stMetric"] { background: #ffffff; padding: 1rem;
            border: 1px solid #dfe5e1; border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _read_dataset(path: str) -> pd.DataFrame:
    return load_dataset(path)


def _show_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def _active_artifact(target: str | None, frame: pd.DataFrame) -> dict[str, Any] | None:
    artifact = st.session_state.get("artifact")
    if artifact is None:
        try:
            artifact = load_artifact()
            st.session_state["artifact"] = artifact
        except (FileNotFoundError, ValueError, OSError):
            return None
    metadata = artifact.get("metadata", {})
    if target != metadata.get("target_column"):
        return None
    if not set(metadata.get("feature_columns", [])).issubset(frame.columns):
        return None
    return artifact


def _metric_value(value: Any) -> str:
    try:
        numeric = float(value)
        return f"{numeric:.3f}" if np.isfinite(numeric) else "Not available"
    except (TypeError, ValueError):
        return "Not available"


def _numeric_target_values(frame: pd.DataFrame, target: str) -> pd.Series:
    return pd.to_numeric(frame[target], errors="coerce").dropna()


def _prediction_form(frame: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    feature_columns = artifact["metadata"]["feature_columns"]
    numeric_features = set(numeric_columns(frame))
    input_values: dict[str, Any] = {}
    with st.form("student_prediction_form"):
        columns = st.columns(2)
        for index, feature in enumerate(feature_columns):
            series = frame[feature]
            container = columns[index % 2]
            if feature in numeric_features:
                values = pd.to_numeric(series, errors="coerce").dropna()
                if values.empty:
                    input_values[feature] = container.number_input(feature, value=0.0)
                    continue
                lower = float(values.min())
                upper = float(values.max())
                default = float(values.median())
                if lower == upper:
                    lower -= 1.0
                    upper += 1.0
                is_integer = pd.api.types.is_integer_dtype(series)
                input_values[feature] = container.number_input(
                    feature,
                    min_value=lower,
                    max_value=upper,
                    value=min(max(default, lower), upper),
                    step=1.0 if is_integer else max((upper - lower) / 100, 0.01),
                    format="%.0f" if is_integer else "%.3f",
                )
            else:
                categories = series.dropna().astype(str).unique().tolist()
                if 1 < len(categories) <= 50:
                    options = categories + ["Other value..."]
                    selected = container.selectbox(feature, options)
                    input_values[feature] = (
                        container.text_input(f"{feature} value")
                        if selected == "Other value..."
                        else selected
                    )
                else:
                    default = categories[0] if categories else ""
                    input_values[feature] = container.text_input(feature, value=default)
        submitted = st.form_submit_button("Predict Performance", type="primary")
    if submitted:
        st.session_state["prediction_input"] = input_values
    return pd.DataFrame([st.session_state.get("prediction_input", input_values)])


def main() -> None:
    st.sidebar.title("Student Analytics")
    try:
        frame = _read_dataset(str(DEFAULT_DATA_PATH))
    except FileNotFoundError:
        st.title("🎓 AI Student Performance Analytics")
        st.error("Please place the Kaggle dataset at data/student-performance.csv")
        st.caption(f"Expected file: {DEFAULT_DATA_PATH}")
        st.stop()
        return
    except (ValueError, OSError) as error:
        st.title("🎓 AI Student Performance Analytics")
        st.error(str(error))
        st.stop()
        return

    numeric_targets = numeric_columns(frame)
    if not numeric_targets:
        st.title("🎓 AI Student Performance Analytics")
        st.error("No numeric-compatible columns were found. A numeric regression target is required.")
        st.write("Available columns:", list(frame.columns))
        st.stop()
        return

    recognized = target_candidates(frame)
    detected = detect_target_column(frame)
    options = ["G3"]
    default_index = 0
    selected_target = st.sidebar.selectbox(
        "Performance target",
        options,
        index=default_index,
        disabled=True,
        help="A unique recognized score column is preselected. Otherwise choose the regression target explicitly.",
    )
    target = selected_target
    if len(recognized) > 1:
        st.sidebar.warning("Several recognized target columns exist. Choose the intended target explicitly.")
    elif not recognized:
        st.sidebar.caption("No recognized target name found; choose the numeric target column.")

    navigation = st.sidebar.radio(
        "Workspace",
        [
            "Dashboard",
            "Student Prediction",
            "Data Analytics",
            "Model Performance",
            "Feature Importance",
            "About Project",
        ],
    )
    if st.sidebar.button("Train / Retrain Models", type="primary", use_container_width=True):
        if target is None:
            st.sidebar.error("Choose a numeric target before training.")
        else:
            try:
                with st.spinner("Training and evaluating regression pipelines..."):
                    artifact, comparison = train_and_evaluate(frame, target)
                st.session_state["artifact"] = artifact
                st.session_state["comparison"] = comparison
                st.sidebar.success(f"Selected: {artifact['metadata']['model_name']}")
            except Exception as error:
                st.sidebar.error(str(error))

    summary = dataset_summary(frame)
    st.sidebar.caption(f"{summary['rows']:,} rows · {summary['columns']} columns")
    artifact = _active_artifact(target, frame)
    target_values = _numeric_target_values(frame, target) if target else pd.Series(dtype=float)

    if navigation == "Dashboard":
        _show_header("AI Student Performance Analytics", "A data-driven view of student outcomes")
        if target is None:
            st.info("Select a numeric performance target in the sidebar to activate target-based analytics.")
        metrics = st.columns(4)
        metrics[0].metric("Students", f"{summary['rows']:,}")
        metrics[1].metric("Features", str(max(summary["columns"] - (1 if target else 0), 0)))
        metrics[2].metric("Average performance", _metric_value(target_values.mean()) if target else "Choose target")
        best_r2 = artifact["metadata"].get("metrics", {}).get("R2") if artifact else None
        metrics[3].metric("Best model R²", _metric_value(best_r2) if best_r2 is not None else "Not trained")
        left, right = st.columns([1.4, 1])
        with left:
            st.subheader("Performance distribution")
            if target and not target_values.empty:
                distribution = pd.DataFrame({target: target_values})
                st.plotly_chart(
                    px.histogram(distribution, x=target, nbins=25, marginal="box",
                                 color_discrete_sequence=["#d8784b"]),
                    use_container_width=True,
                )
            else:
                st.info("Choose a target column to view the outcome distribution.")
        with right:
            st.subheader("Dataset profile")
            st.write(f"**Rows:** {summary['rows']:,}")
            st.write(f"**Columns:** {summary['columns']}")
            st.write(f"**Duplicate records:** {summary['duplicate_rows']:,}")
            missing_total = int(sum(summary["missing_values"].values()))
            st.write(f"**Missing cells:** {missing_total:,}")
            st.write(f"**Source:** `{Path(DEFAULT_DATA_PATH).relative_to(Path(__file__).parent)}`")
        if artifact:
            st.caption(f"Best evaluated pipeline: {artifact['metadata']['model_name']} · holdout RMSE {_metric_value(artifact['metadata']['metrics'].get('RMSE'))}")
        else:
            st.info("Train models from the sidebar to enable predictions and model diagnostics.")
        with st.expander("Dataset schema and quality"):
            schema = pd.DataFrame(
                {
                    "Column": summary["column_names"],
                    "Data type": [summary["data_types"][column] for column in summary["column_names"]],
                    "Missing values": [summary["missing_values"][column] for column in summary["column_names"]],
                }
            )
            st.dataframe(schema, use_container_width=True, hide_index=True)

    elif navigation == "Student Prediction":
        _show_header("Student Prediction", "Estimate performance from the selected dataset's feature schema")
        if artifact is None:
            st.warning("Train a model for the selected target before making predictions.")
        else:
            st.write(f"Using **{artifact['metadata']['model_name']}** for **{target}**.")
            result_frame = _prediction_form(frame, artifact)
            if st.session_state.get("prediction_input"):
                try:
                    prediction = predict_student(artifact, result_frame)
                    category = performance_category(prediction, artifact["metadata"]["target_scale"])
                    st.metric("Predicted score", f"{prediction:.2f}")
                    st.success(f"Performance category: **{category}**")
                    st.caption(f"Model used: {artifact['metadata']['model_name']}")
                    importance = feature_importance(artifact).head(3)
                    if not importance.empty:
                        names = ", ".join(importance["Feature"].astype(str).tolist())
                        st.info(f"The model's strongest overall signals include: {names}. These are global associations, not causal explanations for this individual prediction.")
                except Exception as error:
                    st.error(f"Prediction could not be generated: {error}")

    elif navigation == "Data Analytics":
        _show_header("Data Analytics", "Explore distributions, relationships, and dataset quality")
        feature_options = [column for column in frame.columns if column != target]
        if not feature_options:
            st.info("Choose the target column to reveal the remaining features.")
        else:
            selected_feature = st.selectbox("Feature", feature_options)
            series = frame[selected_feature]
            left, right = st.columns(2)
            with left:
                st.subheader("Feature distribution")
                if pd.api.types.is_numeric_dtype(series):
                    st.plotly_chart(px.histogram(frame, x=selected_feature, marginal="box",
                                                 color_discrete_sequence=["#387d71"]), use_container_width=True)
                else:
                    counts = series.astype("string").fillna("Missing").value_counts().head(25)
                    chart_data = counts.rename_axis(selected_feature).reset_index(name="Records")
                    st.plotly_chart(px.bar(chart_data, x=selected_feature, y="Records",
                                           color_discrete_sequence=["#387d71"]), use_container_width=True)
            with right:
                st.subheader("Feature and target")
                if target is None:
                    st.info("Select a numeric target to see relationships.")
                elif pd.api.types.is_numeric_dtype(series):
                    relationship = frame[[selected_feature, target]].copy()
                    relationship[target] = pd.to_numeric(relationship[target], errors="coerce")
                    st.plotly_chart(px.scatter(relationship, x=selected_feature, y=target,
                                               opacity=0.65,
                                               color_discrete_sequence=["#d8784b"]), use_container_width=True)
                elif series.nunique(dropna=True) <= 25:
                    relationship = frame[[selected_feature, target]].copy()
                    relationship[target] = pd.to_numeric(relationship[target], errors="coerce")
                    st.plotly_chart(px.box(relationship, x=selected_feature, y=target,
                                           color_discrete_sequence=["#d8784b"]), use_container_width=True)
                else:
                    st.info("This high-cardinality feature is not grouped into a target chart.")
        st.subheader("Numeric correlations")
        numeric_frame = frame.select_dtypes(include=["number"]).copy()
        if target and target not in numeric_frame and target in frame:
            numeric_frame[target] = pd.to_numeric(frame[target], errors="coerce")
        if numeric_frame.shape[1] >= 2:
            corr = numeric_frame.corr(numeric_only=True)
            st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                      zmin=-1, zmax=1, aspect="auto"), use_container_width=True)
        else:
            st.info("At least two numeric columns are needed for a correlation matrix.")
        missing = frame.isna().sum().sort_values(ascending=False)
        if missing.gt(0).any():
            st.subheader("Missing values by feature")
            missing_frame = missing[missing.gt(0)].rename_axis("Feature").reset_index(name="Missing values")
            st.plotly_chart(px.bar(missing_frame, x="Feature", y="Missing values",
                                   color_discrete_sequence=["#c45e49"]), use_container_width=True)

    elif navigation == "Model Performance":
        _show_header("Model Performance", "Holdout evaluation and training-fold cross-validation")
        if artifact is None:
            st.warning("Train a model for the selected target to see evaluation results.")
        else:
            comparison = st.session_state.get("comparison")
            if comparison is None:
                comparison_path = MODEL_DIR / "model_comparison.csv"
                comparison = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
            if comparison.empty:
                st.metric("Selected model", artifact["metadata"]["model_name"])
                st.json(artifact["metadata"].get("metrics", {}))
                st.info("Train models in this session to view the full comparison table.")
            else:
                st.dataframe(comparison, use_container_width=True, hide_index=True)
                st.caption("Selection uses lowest training-fold CV RMSE; holdout metrics are reserved for final evaluation.")
                chart = comparison.melt(id_vars="Model", value_vars=["MAE", "RMSE"],
                                        var_name="Metric", value_name="Error")
                st.plotly_chart(px.bar(chart, x="Model", y="Error", color="Metric", barmode="group",
                                       color_discrete_sequence=["#387d71", "#d8784b"]), use_container_width=True)
                st.plotly_chart(px.bar(comparison, x="Model", y="R2",
                                       color_discrete_sequence=["#5879a5"]), use_container_width=True)
                cv_frame = comparison[["Model", "CV RMSE", "CV R2"]].copy()
                st.subheader("Cross-validation summary")
                st.dataframe(cv_frame, use_container_width=True, hide_index=True)

    elif navigation == "Feature Importance":
        _show_header("Feature Importance", "Global signals learned by the selected regression model")
        if artifact is None:
            st.warning("Train a model for the selected target to see feature importance.")
        else:
            importance = feature_importance(artifact)
            if importance.empty:
                st.info("The selected estimator does not expose built-in feature importance.")
            else:
                st.plotly_chart(px.bar(importance.head(20).sort_values("Importance"),
                                       x="Importance", y="Feature", orientation="h",
                                       color_discrete_sequence=["#d8784b"]), use_container_width=True)
                st.dataframe(importance, use_container_width=True, hide_index=True)
            st.caption("Importance describes model-level predictive influence, not causation or a student's individual reason for a score.")
            if st.checkbox("Calculate SHAP global importance when available"):
                with st.spinner("Computing SHAP values for a bounded sample..."):
                    shap_result = shap_importance(artifact, frame)
                if shap_result is None or shap_result.empty:
                    st.info("SHAP is unavailable or incompatible for this model; built-in importance remains available.")
                else:
                    st.dataframe(shap_result, use_container_width=True, hide_index=True)

    else:
        _show_header("About Project", "A modular regression and analytics workflow for real student datasets")
        st.write(
            "This application inspects the supplied Kaggle CSV, builds a leakage-safe preprocessing and regression pipeline, "
            "compares candidate models on a held-out split, and saves the complete selected pipeline with its metadata."
        )
        st.markdown(
            "**Dataset path**  \n`data/student-performance.csv`  \n\n"
            "**Target handling**  \nA unique recognized numeric target is preselected. If target names are ambiguous or unrecognized, "
            "choose the intended numeric column in the sidebar. Training never silently selects among ambiguous targets.  \n\n"
            "**Performance categories**  \nFor nonnegative targets observed on a 0-100-like scale (maximum at least 75), fixed score bands are used. "
            "Other target ranges are divided into four equal-width bands using the observed training target minimum and maximum. "
            "The chosen logic is stored in the model metadata."
        )


if __name__ == "__main__":
    main()
