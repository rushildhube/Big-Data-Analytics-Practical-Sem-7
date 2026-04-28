"""
Time series forecasting for Capital Bikeshare traffic.

This script builds a daily ride-count series for a selected `rideable_type`,
searches for the best ARIMA or SARIMA configuration with a small AIC-based
grid search, evaluates the fitted model on a holdout window, and saves:

- forecast CSV with prediction intervals
- tuning results CSV
- forecast plot PNG
- optional pickled fitted model

Defaults are chosen for the dataset in this workspace and for short monthly
history, so the script stays fast and readable.
"""

from __future__ import annotations

import argparse
import importlib
import os
import pickle
import warnings
from dataclasses import dataclass
from datetime import timedelta
from itertools import product
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning  # type: ignore
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    ConvergenceWarning = Warning


def tqdm(iterable: Iterable, enabled: bool = False, **kwargs: object) -> Iterable:
    """Return a tqdm progress iterator only when enabled and installed."""
    if not enabled:
        return iterable
    try:
        module = importlib.import_module("tqdm.auto")
        return module.tqdm(iterable, **kwargs)
    except Exception:
        return iterable


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    aic: float


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dataset_candidates() -> list[str]:
    root = project_root()
    return [
        os.path.join(root, "Dataset", "capitalbikeshare_processed.csv"),
        os.path.join(root, "Dataset", "202603-capitalbikeshare-tripdata.csv"),
        os.path.join(os.getcwd(), "Dataset", "capitalbikeshare_processed.csv"),
        os.path.join(os.getcwd(), "Dataset", "202603-capitalbikeshare-tripdata.csv"),
    ]


def output_directory() -> str:
    path = os.path.join(project_root(), "Outputs")
    os.makedirs(path, exist_ok=True)
    return path


def locate_dataset() -> str:
    for candidate in dataset_candidates():
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        "No dataset found. Place the CSV in Mini Project/Dataset or create the processed file first."
    )


def load_data(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        frame["started_at"] = pd.to_datetime(frame["started_at"], errors="coerce")
        frame = frame.dropna(subset=["started_at"])
        frame["date"] = frame["started_at"].dt.floor("D")
    else:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.floor("D")
    return frame


def daily_series(frame: pd.DataFrame, mode: str = "electric_bike") -> pd.DataFrame:
    if mode.lower() == "total":
        subset = frame.copy()
    else:
        subset = frame.loc[frame["rideable_type"].astype(str).str.lower() == mode.lower()].copy()

    if subset.empty:
        raise ValueError(f"No rows found for rideable_type='{mode}'.")

    series = subset.groupby("date").size().rename("count").sort_index()
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
    series = series.reindex(full_index, fill_value=0)
    series.index.name = "date"
    return series.to_frame()


def holdout_split(series: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(series) <= horizon + 7:
        raise ValueError("Not enough history for the requested holdout horizon.")
    return series.iloc[:-horizon].copy(), series.iloc[-horizon:].copy()


def metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    aligned = pd.concat([actual.rename("actual"), predicted.rename("predicted")], axis=1).dropna()
    error = aligned["actual"] - aligned["predicted"]
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mape = float(np.mean(np.abs(error / np.clip(aligned["actual"], 1, None))) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def pretty_print_section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def maybe_import_statsmodels():
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
        return SARIMAX
    except Exception:
        return None


def candidate_grid(model_type: str, seasonal_period: int) -> list[tuple[tuple[int, int, int], tuple[int, int, int, int]]]:
    if model_type == "arima":
        orders = [(0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 1), (2, 1, 2)]
        return [(order, (0, 0, 0, 0)) for order in orders]

    seasonal_orders = [
        (0, 0, 0, seasonal_period),
        (0, 1, 0, seasonal_period),
        (0, 1, 1, seasonal_period),
        (1, 0, 0, seasonal_period),
        (1, 1, 0, seasonal_period),
        (1, 1, 1, seasonal_period),
    ]
    orders = [(0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 1)]
    return [(order, seasonal_order) for order, seasonal_order in product(orders, seasonal_orders)]


def fit_candidate(train: pd.Series, order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int]):
    SARIMAX = maybe_import_statsmodels()
    if SARIMAX is None:
        return None

    model = SARIMAX(
        train.astype(float),
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def tune_model(
    train: pd.Series,
    model_type: str,
    seasonal_period: int,
    show_progress: bool = False,
) -> tuple[ModelSpec | None, object | None, pd.DataFrame]:
    rows = []
    best_result = None
    best_spec = None

    candidates = candidate_grid(model_type, seasonal_period)
    for order, seasonal_order in tqdm(candidates, enabled=show_progress, desc=f"Tuning {model_type.upper()}"):
        try:
            result = fit_candidate(train, order, seasonal_order)
            if result is None:
                break
            rows.append(
                {
                    "model_type": model_type,
                    "order": str(order),
                    "seasonal_order": str(seasonal_order),
                    "aic": float(result.aic),
                }
            )
            if best_result is None or result.aic < best_result.aic:
                best_result = result
                best_spec = ModelSpec(model_type, order, seasonal_order, float(result.aic))
        except Exception:
            continue

    tuning_table = pd.DataFrame(rows).sort_values("aic") if rows else pd.DataFrame(columns=["model_type", "order", "seasonal_order", "aic"])
    return best_spec, best_result, tuning_table


def forecast_from_result(result, steps: int, index: pd.DatetimeIndex) -> pd.DataFrame:
    forecast = result.get_forecast(steps=steps)
    mean = forecast.predicted_mean.clip(lower=0)
    confidence = forecast.conf_int(alpha=0.05)

    lower_col = confidence.columns[0]
    upper_col = confidence.columns[1]
    frame = pd.DataFrame(
        {
            "forecast": np.asarray(mean),
            "lower": np.asarray(confidence[lower_col]),
            "upper": np.asarray(confidence[upper_col]),
        },
        index=index,
    )
    frame.index.name = "date"
    return frame


def seasonal_naive_forecast(train: pd.Series, horizon: int, seasonal_period: int) -> pd.Series:
    values = []
    start = train.index.max() + timedelta(days=1)
    history = train.to_dict()
    for offset in range(horizon):
        target_date = start + timedelta(days=offset)
        source_date = target_date - timedelta(days=seasonal_period)
        values.append(float(history.get(source_date, train.iloc[-min(seasonal_period, len(train)) :].mean())))
    return pd.Series(values, index=pd.date_range(start, periods=horizon, freq="D"), name="forecast")


def fit_fixed_spec(train: pd.Series, spec: ModelSpec) -> object | None:
    result = fit_candidate(train, spec.order, spec.seasonal_order)
    return result


def rolling_backtest(train: pd.Series, spec: ModelSpec | None, seasonal_period: int, folds: int = 3) -> pd.DataFrame:
    backtest_rows = []
    if len(train) < seasonal_period * 2:
        return pd.DataFrame(columns=["fold", "MAE", "RMSE", "MAPE"])

    horizon = min(max(3, seasonal_period), max(1, len(train) // 4))
    folds = max(1, min(folds, (len(train) - horizon) // horizon))
    if folds <= 0:
        folds = 1

    for fold in range(folds):
        train_end = len(train) - horizon * (folds - fold)
        if train_end <= horizon:
            continue
        fold_train = train.iloc[:train_end]
        fold_test = train.iloc[train_end: train_end + horizon]

        if spec is None:
            pred = seasonal_naive_forecast(fold_train, len(fold_test), seasonal_period)
        else:
            result = fit_fixed_spec(fold_train, spec)
            if result is None:
                pred = seasonal_naive_forecast(fold_train, len(fold_test), seasonal_period)
            else:
                pred = forecast_from_result(result, len(fold_test), fold_test.index)["forecast"]

        score = metrics(fold_test, pred)
        backtest_rows.append({"fold": fold + 1, **score})

    return pd.DataFrame(backtest_rows)


def save_plot(train: pd.Series, test: pd.Series, forecast_frame: pd.DataFrame, mode: str, model_type: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(train.index, train.values, color="black", linewidth=1.8, label="train")
    ax.plot(test.index, test.values, color="gray", linewidth=1.8, label="holdout")
    ax.plot(forecast_frame.index, forecast_frame["forecast"], color="#1f77b4", linewidth=2.2, label=f"{model_type} forecast")
    ax.fill_between(
        forecast_frame.index,
        forecast_frame["lower"],
        forecast_frame["upper"],
        color="#1f77b4",
        alpha=0.18,
        label="95% CI",
    )
    ax.set_title(f"Capital Bikeshare traffic forecast - {mode} ({model_type.upper()})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Ride count")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()

    plot_path = os.path.join(out_dir, f"forecast_{mode}_{model_type}.png")
    plt.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def print_table(title: str, frame: pd.DataFrame) -> None:
    pretty_print_section(title)
    if frame.empty:
        print("No rows to display.")
    else:
        print(frame.to_string(index=False))


def run_pipeline(mode: str, model_type: str, horizon: int, seasonal_period: int, save_model: bool, show_progress: bool) -> None:
    dataset_path = locate_dataset()
    print(f"Using dataset: {dataset_path}")

    frame = load_data(dataset_path)
    daily = daily_series(frame, mode=mode)
    train_df, test_df = holdout_split(daily, horizon)
    train = train_df["count"]
    test = test_df["count"]

    pretty_print_section("DATA SUMMARY")
    print(f"Mode: {mode}")
    print(f"Daily observations: {len(daily)}")
    print(f"Training rows: {len(train)}")
    print(f"Holdout rows: {len(test)}")
    print(f"Date range: {daily.index.min().date()} -> {daily.index.max().date()}")
    print(f"Seasonal period: {seasonal_period}")

    pretty_print_section(f"TUNING {model_type.upper()}")
    best_spec, best_result, tuning_table = tune_model(train, model_type, seasonal_period, show_progress=show_progress)
    if best_spec is None or best_result is None:
        print("Statsmodels unavailable or all fits failed. Falling back to seasonal naive forecast.")
        forecast_series = seasonal_naive_forecast(train, horizon, seasonal_period)
        forecast_frame = pd.DataFrame(
            {
                "forecast": forecast_series,
                "lower": forecast_series * 0.9,
                "upper": forecast_series * 1.1,
            }
        )
        chosen_label = "seasonal_naive"
    else:
        print(f"Best spec: order={best_spec.order}, seasonal_order={best_spec.seasonal_order}, AIC={best_spec.aic:.2f}")
        forecast_frame = forecast_from_result(best_result, horizon, pd.date_range(train.index.max() + timedelta(days=1), periods=horizon, freq="D"))
        chosen_label = f"{model_type}_best"

    print_table("TOP TUNED MODELS", tuning_table.head(10))

    holdout_scores = metrics(test, forecast_frame["forecast"])
    pretty_print_section("HOLDOUT EVALUATION")
    print(f"MAE : {holdout_scores['MAE']:.2f}")
    print(f"RMSE: {holdout_scores['RMSE']:.2f}")
    print(f"MAPE: {holdout_scores['MAPE']:.2f}%")

    backtest = rolling_backtest(train, best_spec if best_result is not None else None, seasonal_period, folds=3)
    print_table("ROLLING BACKTEST", backtest)

    outputs = output_directory()
    forecast_csv = os.path.join(outputs, f"forecast_{mode}_{model_type}.csv")
    tuning_csv = os.path.join(outputs, f"tuning_{mode}_{model_type}.csv")
    backtest_csv = os.path.join(outputs, f"backtest_{mode}_{model_type}.csv")
    model_path = os.path.join(outputs, f"{model_type}_{mode}.pkl")

    result_table = forecast_frame.copy()
    result_table["actual"] = test.reindex(result_table.index)
    result_table.to_csv(forecast_csv, index_label="date")
    tuning_table.to_csv(tuning_csv, index=False)
    backtest.to_csv(backtest_csv, index=False)

    if save_model and best_result is not None:
        try:
            with open(model_path, "wb") as handle:
                pickle.dump(best_result, handle)
        except Exception as exc:
            print(f"Warning: could not save fitted model: {exc}")

    plot_path = save_plot(train, test, forecast_frame, mode, model_type, outputs)

    pretty_print_section("FILES WRITTEN")
    print(f"Forecast CSV : {forecast_csv}")
    print(f"Tuning CSV   : {tuning_csv}")
    print(f"Backtest CSV : {backtest_csv}")
    print(f"Plot PNG     : {plot_path}")
    if save_model and best_result is not None:
        print(f"Model PKL    : {model_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capital Bikeshare time series training and forecasting")
    parser.add_argument("--mode", default="electric_bike", help="rideable_type to forecast, or 'total'")
    parser.add_argument("--model", default="sarima", choices=["sarima", "arima", "seasonal_naive"], help="forecast model to tune")
    parser.add_argument("--horizon", type=int, default=14, help="forecast horizon in days")
    parser.add_argument("--seasonal-period", type=int, default=7, help="seasonal period for SARIMA")
    parser.add_argument("--save-model", action="store_true", help="save the fitted model pickle")
    parser.add_argument("--progress", action="store_true", help="show tuning progress bars")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(
        mode=args.mode,
        model_type=args.model,
        horizon=args.horizon,
        seasonal_period=args.seasonal_period,
        save_model=args.save_model,
        show_progress=args.progress,
    )


if __name__ == "__main__":
    main()
