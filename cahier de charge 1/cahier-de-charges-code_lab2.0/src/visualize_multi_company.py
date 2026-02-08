from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["COMPANY"] = df["COMPANY"].astype(str).str.strip()
    return df


def company_correlation(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).dropna()
    corr = numeric.corr()[["TARGET_NEXT_CLOSE"]].sort_values("TARGET_NEXT_CLOSE", ascending=False)
    return corr


def plot_company(df: pd.DataFrame, company: str, output_dir: str) -> None:
    subset = df[df["COMPANY"].str.upper() == company.upper()].copy()
    if subset.empty:
        print(f"No data for {company}")
        return

    subset = subset.sort_values("DATE")

    plt.figure(figsize=(12, 5))
    plt.plot(subset["DATE"], subset["CLOSE"], label="Close")
    plt.plot(subset["DATE"], subset["ROLLING_MEAN"], label="Rolling Mean")
    plt.title(f"{company} - Close and Rolling Mean")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{company}_close_rolling.png"))
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.bar(subset["DATE"], subset["VOLUME"], color="#4C78A8")
    plt.title(f"{company} - Volume")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{company}_volume.png"))
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(subset["DATE"], subset["ROLLING_VOL"], color="#F58518")
    plt.title(f"{company} - Rolling Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{company}_rolling_vol.png"))
    plt.close()


def main() -> None:
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_path = os.path.join(workspace_root, "multi_company_features.csv")
    output_dir = os.path.join(workspace_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    df = load_features(features_path)

    companies: List[str] = sorted(df["COMPANY"].dropna().unique().tolist())

    print("Companies in dataset:")
    print(companies)

    for company in companies:
        print(f"\nCorrelation with TARGET_NEXT_CLOSE for {company}:")
        corr = company_correlation(df[df["COMPANY"].str.upper() == company.upper()])
        print(corr.to_string())
        plot_company(df, company, output_dir)

    print(f"\nSaved plots to: {output_dir}")


if __name__ == "__main__":
    main()
