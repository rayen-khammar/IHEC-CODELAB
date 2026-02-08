from __future__ import annotations

import os
import pandas as pd

from infer_five_day_forecast import (
    FeatureConfig,
    load_bvmt_files,
    normalize_market_data,
    load_models,
    recursive_forecast,
)


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    company = "BIAT"

    mc_path = os.path.join(root, "multi_company_features.csv")
    df = pd.read_csv(mc_path)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["COMPANY"] = df["COMPANY"].astype(str).str.strip()

    biat = df[df["COMPANY"].str.upper() == company].copy()
    biat = biat.dropna(subset=["DATE", "CLOSE", "TARGET_NEXT_CLOSE"])
    biat = biat.sort_values("DATE").reset_index(drop=True)

    if biat.empty:
        print("Company not found in multi_company_features.csv")
        print("Available companies:", sorted(df["COMPANY"].dropna().unique().tolist())[:20])
        return

    print("multi_company_features last date:", biat["DATE"].iloc[-1])
    print("multi_company_features last CLOSE:", biat["CLOSE"].iloc[-1])
    print("multi_company_features last TARGET_NEXT_CLOSE:", biat["TARGET_NEXT_CLOSE"].iloc[-1])

    files = [
        os.path.join(root, name)
        for name in os.listdir(root)
        if name.startswith("histo_cotation_") and (name.endswith(".csv") or name.endswith(".txt"))
    ]

    raw = load_bvmt_files(sorted(files))
    normalized = normalize_market_data(raw)

    hist = normalized[normalized["COMPANY"].str.upper() == company].sort_values("DATE").reset_index(drop=True)

    models = load_models(os.path.join(root, "models", company))
    fc = FeatureConfig(n_lags=5, volatility_window=10)
    forecast = recursive_forecast(models, hist, 5, fc)

    print("\nRecursive forecast (5 steps):")
    print(forecast.to_string(index=False))


if __name__ == "__main__":
    main()
