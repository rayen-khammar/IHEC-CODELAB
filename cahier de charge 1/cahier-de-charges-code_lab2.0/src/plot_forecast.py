from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from infer_five_day_forecast import (
    FeatureConfig,
    load_bvmt_files,
    normalize_market_data,
    load_models,
    recursive_forecast,
    drop_non_trading_days,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 5-day recursive forecast.")
    parser.add_argument("--company", required=True, help="Company name (e.g., BIAT)")
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon in days")
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional start date (dd/mm/yyyy). Forecast starts after this date.",
    )

    args = parser.parse_args()

    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_paths = [
        path
        for path in (
            [p for p in os.listdir(workspace_root)]
        )
    ]

    files = []
    for name in os.listdir(workspace_root):
        if name.startswith("histo_cotation_") and (name.endswith(".csv") or name.endswith(".txt")):
            files.append(os.path.join(workspace_root, name))

    raw = load_bvmt_files(sorted(files))
    normalized = normalize_market_data(raw)

    company_df = normalized[normalized["COMPANY"].str.upper() == args.company.upper()].copy()
    if company_df.empty:
        raise ValueError(f"Company '{args.company}' not found in data.")

    company_df = company_df.sort_values("DATE").reset_index(drop=True)

    if args.start_date:
        start_date = pd.to_datetime(args.start_date, dayfirst=True, errors="coerce")
        if pd.isna(start_date):
            raise ValueError("Invalid --start-date. Use dd/mm/yyyy format.")
        max_date = company_df["DATE"].max()
        if start_date > max_date:
            raise ValueError(
                f"Start date is beyond available data (max {max_date.strftime('%d/%m/%Y')})."
            )
        company_df = company_df[company_df["DATE"] <= start_date].copy()
        if company_df.empty:
            raise ValueError("No data available on or before the selected start date.")

    company_df = drop_non_trading_days(company_df)
    if company_df.empty:
        raise ValueError("No trading rows available after filtering non-trading days.")

    model_dir = os.path.join(workspace_root, "models", args.company)
    models = load_models(model_dir)

    config = FeatureConfig(n_lags=5, volatility_window=10)
    forecast_df = recursive_forecast(models, company_df, args.horizon, config)

    last_date = company_df["DATE"].iloc[-1]
    forecast_dates = pd.date_range(last_date, periods=args.horizon + 1, freq="B")[1:]

    plt.figure(figsize=(10, 5))
    plt.plot(company_df["DATE"].iloc[-30:], company_df["CLOSE"].iloc[-30:], label="History (last 30)")
    plt.plot(forecast_dates, forecast_df["pred_close"], label="Forecast", marker="o")
    title_suffix = f" (as of {last_date.strftime('%d/%m/%Y')})"
    plt.title(f"{args.company} - {args.horizon}-Day Forecast{title_suffix}")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.legend()
    plt.tight_layout()

    output_dir = os.path.join(workspace_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.company}_forecast_{args.horizon}d.png")
    plt.savefig(output_path)
    plt.close()

    print(f"Saved forecast plot to: {output_path}")


if __name__ == "__main__":
    main()
