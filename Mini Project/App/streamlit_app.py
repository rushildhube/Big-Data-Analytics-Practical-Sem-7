from __future__ import annotations

import io
import os
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import threading
import queue
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import main as forecast_main

warnings.filterwarnings("ignore")

def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #0f766e 55%, #f59e0b 100%);
            color: white;
            padding: 1.25rem 1.4rem;
            border-radius: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.18);
        }
        .hero h1 {
            margin: 0;
            font-size: 2rem;
        }
        .hero p {
            margin: 0.4rem 0 0;
            opacity: 0.92;
        }
        .section-card {
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 0.9rem;
            padding: 1rem;
            background: white;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_dataset() -> pd.DataFrame:
    dataset_path = forecast_main.locate_dataset()
    return forecast_main.load_data(dataset_path)


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def load_text(path: str) -> str:
    if not os.path.exists(path):
        return ""
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def project_asset_paths() -> dict[str, Path]:
    outputs_dir = PROJECT_ROOT / "Outputs"
    viz_dir = PROJECT_ROOT / "Visualizations"
    report_dir = PROJECT_ROOT / "For Report"
    return {
        "outputs": outputs_dir,
        "visualizations": viz_dir,
        "report": report_dir,
    }


def build_eda_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    overview = pd.DataFrame(
        [
            {"Metric": "Rows", "Value": len(df)},
            {"Metric": "Columns", "Value": len(df.columns)},
            {"Metric": "Ride Types", "Value": df["rideable_type"].nunique() if "rideable_type" in df else 0},
            {"Metric": "Stations", "Value": df["start_station_name"].nunique() if "start_station_name" in df else 0},
            {"Metric": "Members %", "Value": round((df["member_casual"].eq("member").mean() * 100), 2) if "member_casual" in df else np.nan},
        ]
    )
    tables["overview"] = overview

    missing = (
        df.isna().sum().reset_index().rename(columns={"index": "Column", 0: "Missing_Count"})
    )
    missing.columns = ["Column", "Missing_Count"]
    missing["Missing_%"] = (missing["Missing_Count"] / len(df)) * 100
    tables["missing"] = missing.sort_values("Missing_Count", ascending=False)

    if "rideable_type" in df:
        tables["rideable_type"] = df["rideable_type"].value_counts().rename_axis("rideable_type").reset_index(name="count")

    if "member_casual" in df:
        tables["member_casual"] = df["member_casual"].value_counts().rename_axis("member_casual").reset_index(name="count")

    if "start_station_name" in df:
        tables["top_start_stations"] = df["start_station_name"].value_counts().head(10).rename_axis("station").reset_index(name="count")
    if "end_station_name" in df:
        tables["top_end_stations"] = df["end_station_name"].value_counts().head(10).rename_axis("station").reset_index(name="count")

    if {"started_at", "date"}.issubset(df.columns):
        ts = df.copy()
        ts["started_at"] = pd.to_datetime(ts["started_at"], errors="coerce")
        ts = ts.dropna(subset=["started_at"])
        ts["date"] = ts["started_at"].dt.floor("D")
        ts["hour"] = ts["started_at"].dt.hour
        ts["dayofweek"] = ts["started_at"].dt.day_name()
        tables["daily_counts"] = ts.groupby("date").size().rename("count").reset_index()
        tables["hourly_counts"] = ts.groupby("hour").size().rename("count").reset_index()
        tables["dayofweek_counts"] = ts.groupby("dayofweek").size().reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        ).rename("count").reset_index()

    if "ride_duration_min" in df:
        duration = df["ride_duration_min"].dropna()
        outlier_summary = pd.DataFrame(
            [
                {"Metric": "Mean", "Value": round(duration.mean(), 2)},
                {"Metric": "Median", "Value": round(duration.median(), 2)},
                {"Metric": "Std Dev", "Value": round(duration.std(), 2)},
                {"Metric": "Min", "Value": round(duration.min(), 2)},
                {"Metric": "Max", "Value": round(duration.max(), 2)},
                {"Metric": "Rides > 2h", "Value": int((duration > 120).sum())},
                {"Metric": "Rides < 1m", "Value": int((duration < 1).sum())},
            ]
        )
        tables["duration_summary"] = outlier_summary

    numeric_cols = [c for c in ["ride_duration_min", "hour", "dayofweek", "start_lat", "start_lng", "end_lat", "end_lng"] if c in df.columns]
    if len(numeric_cols) >= 2:
        tables["correlations"] = df[numeric_cols].corr(numeric_only=True).reset_index().rename(columns={"index": "feature"})

    return tables


def show_metric_row(df: pd.DataFrame) -> None:
    cols = st.columns(len(df))
    for col, (_, row) in zip(cols, df.iterrows()):
        value = row["Value"]
        if isinstance(value, float):
            value = f"{value:,.2f}"
        elif isinstance(value, (int, np.integer)):
            value = f"{value:,}"
        col.metric(str(row["Metric"]), value)


def render_visualization_grid(image_paths: list[Path]) -> None:
    if not image_paths:
        st.info("No visualization PNGs found yet.")
        return

    # lightweight keyword -> description mapping for common saved plots
    viz_descriptions = {
        "hourly": "Hourly Ride Distribution: shows total rides aggregated by hour of day — useful to identify peak travel hours.",
        "heatmap": "Hour vs Day-of-week Heatmap: highlights when rides are concentrated across week and day-hour combinations.",
        "forecast": "Forecast plot: model predictions vs. observed values on the holdout period (includes baseline comparisons).",
        "trend": "Daily Trend: aggregated daily ride counts showing short-term trends and spikes.",
        "member": "Member vs Casual Distribution: breakdown of rider types (members and casual users).",
        "dayofweek": "Rides by Day of Week: bar chart summarizing total rides per weekday.",
        "duration": "Ride Duration Distribution: histogram of ride durations (0-60 minutes) showing skew and common trip lengths.",
        "bike_type": "Bike Type Distribution: share of electric vs classic bikes in the dataset.",
        "member_vs_casual": "Hourly Distribution by User Type: shows how members and casual users ride at different hours.",
    }

    for idx in range(0, len(image_paths), 2):
        cols = st.columns(2)
        for col, image_path in zip(cols, image_paths[idx : idx + 2]):
            name = image_path.name.lower()
            caption = image_path.name
            col.image(str(image_path), caption=caption, use_container_width=True)

            # find best matching description by keyword
            desc = None
            for k, v in viz_descriptions.items():
                if k in name:
                    desc = v
                    break
            if not desc:
                desc = "Visualization generated during EDA. Inspect the image to understand distributional or temporal patterns."
            col.caption(desc)


def compute_forecast_metrics(forecast_df: pd.DataFrame) -> pd.DataFrame:
    if forecast_df.empty or "actual" not in forecast_df.columns or "forecast" not in forecast_df.columns:
        return pd.DataFrame()
    usable = forecast_df.dropna(subset=["actual", "forecast"]).copy()
    if usable.empty:
        return pd.DataFrame()
    error = usable["actual"] - usable["forecast"]
    metrics = pd.DataFrame(
        [
            {"Metric": "MAE", "Value": float(np.abs(error).mean())},
            {"Metric": "RMSE", "Value": float(np.sqrt((error ** 2).mean()))},
            {"Metric": "MAPE %", "Value": float((np.abs(error / np.clip(usable["actual"], 1, None))).mean() * 100)},
        ]
    )
    return metrics


def train_and_refresh(mode: str, model: str, horizon: int, seasonal_period: int, save_model: bool) -> None:
    with st.spinner("Training SARIMA / ARIMA model and refreshing outputs..."):
        buffer = io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(buffer):
            forecast_main.run_pipeline(
                mode=mode,
                model_type=model,
                horizon=horizon,
                seasonal_period=seasonal_period,
                save_model=save_model,
                show_progress=False,
            )
        st.session_state["last_console"] = buffer.getvalue()
        st.session_state["last_run"] = {"mode": mode, "model": model, "horizon": horizon, "seasonal_period": seasonal_period}


def train_and_stream(mode: str, model: str, horizon: int, seasonal_period: int, save_model: bool) -> None:
    """Run training in a thread and stream stdout/stderr into the Streamlit UI in real time."""
    q: "queue.Queue[str]" = queue.Queue()

    class QueueWriter:
        def __init__(self, q: "queue.Queue[str]"):
            self.q = q

        def write(self, s: str) -> None:
            if s:
                try:
                    self.q.put(s)
                except Exception:
                    pass

        def flush(self) -> None:
            return

    def target() -> None:
        old_out = sys.stdout
        old_err = sys.stderr
        sys.stdout = QueueWriter(q)
        sys.stderr = QueueWriter(q)
        try:
            forecast_main.run_pipeline(
                mode=mode,
                model_type=model,
                horizon=horizon,
                seasonal_period=seasonal_period,
                save_model=save_model,
                show_progress=True,
            )
        finally:
            sys.stdout = old_out
            sys.stderr = old_err

    thread = threading.Thread(target=target)
    thread.start()

    placeholder = st.empty()
    buf = ""
    # Poll queue and update UI until thread finishes
    while thread.is_alive() or not q.empty():
        try:
            chunk = q.get(timeout=0.1)
            buf += chunk
        except queue.Empty:
            pass
        # keep the displayed content reasonably sized
        placeholder.code(buf[-15000:])
        time.sleep(0.1)

    # drain remaining
    while not q.empty():
        buf += q.get()
    placeholder.code(buf[-15000:])
    st.session_state["last_console"] = buf
    st.session_state["last_run"] = {"mode": mode, "model": model, "horizon": horizon, "seasonal_period": seasonal_period}


def main() -> None:
    st.set_page_config(
        page_title="Capital Bikeshare Forecast Studio",
        page_icon="\U0001F6B2",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()

    assets = project_asset_paths()
    outputs_dir = assets["outputs"]
    viz_dir = assets["visualizations"]

    df = load_dataset()
    tables = build_eda_tables(df)

    st.markdown(
        """
        <div class="hero">
            <h1>Capital Bikeshare Forecast Studio</h1>
            <p>EDA, forecast training, evaluation tables, and charts in one Streamlit dashboard.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Forecast Controls")
        mode = st.selectbox("Ride mode", options=["electric_bike", "classic_bike", "total"], index=0)
        model = st.selectbox("Model", options=["sarima", "arima", "seasonal_naive"], index=0)
        horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=30, value=14, step=1)
        seasonal_period = st.slider("Seasonal period", min_value=3, max_value=14, value=7, step=1)
        save_model = st.checkbox("Save fitted model", value=True)
        train_now = st.button("Train / Refresh Forecast")

        st.divider()
        st.caption("Project assets")
        st.write(f"Dataset: `{forecast_main.locate_dataset()}`")
        st.write(f"Outputs: `{outputs_dir}`")
        st.write(f"Visualizations: `{viz_dir}`")

    if train_now:
        train_and_refresh(mode, model, horizon, seasonal_period, save_model)
        st.success("Training completed and outputs refreshed.")

    overview, eda_tab, viz_tab, forecast_tab, data_tab = st.tabs([
        "Overview",
        "EDA Tables",
        "Visualizations",
        "Forecasting",
        "Data",
    ])

    with overview:
        st.subheader("Project Summary")
        st.write(
            "This dashboard combines the exploratory analysis from the mini project with the saved SARIMA/ARIMA forecasts. "
            "Use the controls in the sidebar to retrain a model and refresh the outputs."
        )
        show_metric_row(tables["overview"])

        left, right = st.columns(2)
        with left:
            st.markdown("### EDA Snapshot")
            for key, label in [
                ("rideable_type", "Rideable Type Distribution"),
                ("member_casual", "Member vs Casual Distribution"),
                ("missing", "Missing Values"),
            ]:
                if key in tables:
                    with st.expander(label, expanded=(key != "missing")):
                        st.dataframe(tables[key], width="stretch")

        with right:
            st.markdown("### Quick Dataset Sample")
            st.dataframe(df.head(10), width="stretch")

            st.markdown("### Top Stations")
            station_tabs = st.tabs(["Start Stations", "End Stations"])
            with station_tabs[0]:
                if "top_start_stations" in tables:
                    st.dataframe(tables["top_start_stations"], width="stretch")
            with station_tabs[1]:
                if "top_end_stations" in tables:
                    st.dataframe(tables["top_end_stations"], width="stretch")

        st.markdown("### Trend Tables")
        trend_cols = st.columns(3)
        for col, key, label in zip(
            trend_cols,
            ["daily_counts", "hourly_counts", "dayofweek_counts"],
            ["Daily Counts", "Hourly Counts", "Day-of-Week Counts"],
        ):
            with col:
                if key in tables:
                    st.caption(label)
                    st.dataframe(tables[key], width="stretch")

    with eda_tab:
        st.subheader("EDA Tables")
        st.write(
            "Below are the tabular summaries produced during exploratory data analysis. Expand each section for details and quick interpretation notes."
        )
        eda_descriptions = {
            "missing": "Columns with missing values and their counts — helps identify which fields need imputation or filtering.",
            "rideable_type": "Counts per bike type (electric vs classic). Useful for mode-specific demand analysis.",
            "member_casual": "Breakdown of members vs casual riders to understand user composition.",
            "top_start_stations": "Most frequently used start stations — helpful for station-level demand insights.",
            "top_end_stations": "Most frequently used end stations — shows popular destinations.",
            "duration_summary": "Summary statistics of trip durations to understand typical trip lengths and outliers.",
            "daily_counts": "Daily aggregated ride counts showing overall trend and day-to-day variability.",
            "hourly_counts": "Counts per hour of day summarizing diurnal patterns and peak hours.",
            "dayofweek_counts": "Aggregate rides by weekday to identify weekday/weekend patterns.",
            "correlations": "Pairwise correlations of numeric features — useful to spot multicollinearity or relationships.",
        }
        for key, label in [
            ("missing", "Missing Values"),
            ("rideable_type", "Rideable Type Distribution"),
            ("member_casual", "Member vs Casual Distribution"),
            ("top_start_stations", "Top Start Stations"),
            ("top_end_stations", "Top End Stations"),
            ("duration_summary", "Ride Duration Summary"),
            ("daily_counts", "Daily Ride Counts"),
            ("hourly_counts", "Hourly Ride Counts"),
            ("dayofweek_counts", "Day-of-Week Ride Counts"),
            ("correlations", "Correlation Matrix"),
        ]:
            if key in tables:
                with st.expander(label, expanded=(key in ["missing", "rideable_type"])):
                    # short descriptive note for context
                    desc = eda_descriptions.get(key, "")
                    if desc:
                        st.caption(desc)
                    st.dataframe(tables[key], width="stretch")

    with viz_tab:
        st.subheader("Saved Visualizations")
        viz_paths = sorted(viz_dir.glob("*.png"))
        render_visualization_grid(viz_paths)

    with forecast_tab:
        st.subheader("Forecast Outputs")
        forecast_file = outputs_dir / f"forecast_{mode}_{model}.csv"
        tuning_file = outputs_dir / f"tuning_{mode}_{model}.csv"
        backtest_file = outputs_dir / f"backtest_{mode}_{model}.csv"
        plot_file = outputs_dir / f"forecast_{mode}_{model}.png"

        # retrain button inside Forecasting tab (streams backend logs live)
        st.markdown("#### Retrain model")
        if st.button("Retrain (live)"):
            train_and_stream(mode, model, horizon, seasonal_period, save_model)

        if forecast_file.exists():
            forecast_df = load_csv(str(forecast_file))
            st.markdown("#### Forecast Table")
            st.dataframe(forecast_df, width="stretch")

            metric_df = compute_forecast_metrics(forecast_df)
            if not metric_df.empty:
                st.markdown("#### Holdout Metrics")
                show_metric_row(metric_df)

            if plot_file.exists():
                st.markdown("#### Forecast Chart")
                st.image(str(plot_file), use_container_width=True)

            st.download_button(
                "Download forecast CSV",
                data=forecast_df.to_csv(index=False).encode("utf-8"),
                file_name=forecast_file.name,
                mime="text/csv",
            )
        else:
            st.info("No forecast file found yet. Use the sidebar button to train and generate one.")

        if tuning_file.exists():
            st.markdown("#### Model Tuning Table")
            st.dataframe(load_csv(str(tuning_file)), width="stretch")
        if backtest_file.exists():
            st.markdown("#### Rolling Backtest")
            st.dataframe(load_csv(str(backtest_file)), width="stretch")

        if st.session_state.get("last_console"):
            with st.expander("Console output", expanded=False):
                st.code(st.session_state["last_console"][:15000])

    with data_tab:
        st.subheader("Raw / Processed Data")
        st.write("First 25 rows of the loaded dataset")
        st.dataframe(df.head(25), width="stretch")

        st.markdown("#### Descriptive Statistics")
        try:
            numeric_desc = df.describe(include=[np.number]).transpose().reset_index().rename(columns={"index": "column"})
            st.markdown("**Numeric summary**")
            st.dataframe(numeric_desc, width="stretch")

            categorical_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("string")]
            if categorical_cols:
                cat_summary = []
                for col in categorical_cols[:12]:
                    cat_summary.append(
                        {
                            "column": col,
                            "non_null": int(df[col].notna().sum()),
                            "unique": int(df[col].nunique(dropna=True)),
                            "top_value": str(df[col].mode(dropna=True).iloc[0]) if not df[col].mode(dropna=True).empty else "",
                        }
                    )
                st.markdown("**Categorical summary**")
                st.dataframe(pd.DataFrame(cat_summary), width="stretch")
        except Exception:
            st.info("Could not compute full descriptive stats for this dataset.")

        st.markdown("#### Files in Project")
        project_files = []
        for folder in [PROJECT_ROOT / "App", PROJECT_ROOT / "Dataset", PROJECT_ROOT / "Visualizations", PROJECT_ROOT / "Outputs", PROJECT_ROOT / "For Report"]:
            if folder.exists():
                for file_path in sorted(folder.glob("*")):
                    if file_path.is_file():
                        project_files.append({"Folder": folder.name, "File": file_path.name})
        st.dataframe(pd.DataFrame(project_files), width="stretch")


if __name__ == "__main__":
    main()
