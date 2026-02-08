from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance


def load_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["COMPANY"] = df["COMPANY"].astype(str).str.strip()
    return df


def correlation_with_target(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).dropna()
    corr = numeric.corr()[["TARGET_NEXT_CLOSE"]].sort_values("TARGET_NEXT_CLOSE", ascending=False)
    return corr


def permutation_importance_summary(df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    feature_cols = [
        "OPEN",
        "HIGH",
        "LOW",
        "CLOSE",
        "VOLUME",
        "LOG_RETURN",
        "ROLLING_VOL",
        "VOLUME_CHANGE",
        "ROLLING_MEAN",
        "ROLLING_MEDIAN",
        "ROLLING_STD",
    ] + [col for col in df.columns if col.endswith("_LAG_1") or col.endswith("_LAG_2")
         or col.endswith("_LAG_3") or col.endswith("_LAG_4") or col.endswith("_LAG_5")]

    modeling_df = df[feature_cols + ["TARGET_NEXT_CLOSE"]].dropna()
    if modeling_df.empty:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])

    X = modeling_df[feature_cols].to_numpy()
    y = modeling_df["TARGET_NEXT_CLOSE"].to_numpy()

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)

    perm = permutation_importance(model, X, y, n_repeats=10, random_state=42)

    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    return importance.head(top_n)


def main() -> None:
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_path = os.path.join(workspace_root, "multi_company_features.csv")

    df = load_features(features_path)
    companies: List[str] = sorted(df["COMPANY"].dropna().unique().tolist())

    for company in companies:
        subset = df[df["COMPANY"].str.upper() == company.upper()].copy()
        print(f"\n=== {company} ===")
        print("Top correlations with TARGET_NEXT_CLOSE:")
        corr = correlation_with_target(subset).head(15)
        print(corr.to_string())

        print("\nTop permutation importances:")
        importance = permutation_importance_summary(subset, top_n=12)
        print(importance.to_string(index=False))


if __name__ == "__main__":
    main()
