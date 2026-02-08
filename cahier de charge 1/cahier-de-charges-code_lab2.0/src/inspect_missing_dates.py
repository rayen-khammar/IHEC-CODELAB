from __future__ import annotations

import glob
import os
from typing import List

import pandas as pd


def _standardize_columns(columns: List[str]) -> List[str]:
    return [col.strip().upper().replace(" ", "_") for col in columns]


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", engine="python")
    df.columns = _standardize_columns(df.columns)
    return df


def _load_txt(path: str) -> pd.DataFrame:
    df = pd.read_fwf(path, skiprows=[1])
    df.columns = _standardize_columns(df.columns)
    return df


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    column_map = {
        "SEANCE": "DATE",
        "LIB_VAL": "COMPANY",
        "CODE_VAL": "CODE",
        "C_GR_RLC": "GROUPE",
        "OUVERTURE": "OPEN",
        "PLUS_HAUT": "HIGH",
        "PLUS_BAS": "LOW",
        "CLOTURE": "CLOSE",
        "QUANTITE_NEGOCIEE": "VOLUME",
        "NB_TRAN": "NB_TRANSACTION",
        "VALEUR": "COMPANY",
        "CODE": "CODE",
    }
    available_map = {key: value for key, value in column_map.items() if key in df.columns}
    df = df.rename(columns=available_map)

    required = ["DATE", "CLOSE", "COMPANY", "CODE"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")
    df["COMPANY"] = df["COMPANY"].astype(str).str.strip()
    df["CODE"] = df["CODE"].astype(str).str.strip()

    df = df.dropna(subset=["DATE", "CLOSE"]).reset_index(drop=True)
    return df


def load_all_market_files(workspace_root: str) -> pd.DataFrame:
    data_paths = sorted(
        glob.glob(os.path.join(workspace_root, "histo_cotation_*.csv"))
        + glob.glob(os.path.join(workspace_root, "histo_cotation_*.txt"))
    )

    frames = []
    for path in data_paths:
        try:
            if path.lower().endswith(".csv"):
                df = _load_csv(path)
            else:
                df = _load_txt(path)
            df = _normalize(df)
            df["SOURCE_FILE"] = os.path.basename(path)
            frames.append(df)
        except ValueError as exc:
            print(f"Skipping {os.path.basename(path)}: {exc}")

    if not frames:
        raise ValueError("No compatible input files found.")

    return pd.concat(frames, ignore_index=True)


def report_missing_dates(df: pd.DataFrame) -> None:
    df = df.copy()
    df = df[(df["DATE"].dt.year >= 2018) & (df["DATE"].dt.year <= 2020)]

    if df.empty:
        print("No rows for 2018-2020 in the parsed files.")
        return

    companies = sorted(df["COMPANY"].unique())
    print(f"Companies found in 2018-2020: {len(companies)}")

    for company in companies:
        subset = df[df["COMPANY"] == company].sort_values("DATE")
        dates = subset["DATE"].drop_duplicates().reset_index(drop=True)

        if len(dates) < 2:
            continue

        gaps = dates.diff().dt.days
        large_gaps = gaps[gaps > 3]

        print(f"\n{company}:")
        print(f"  rows: {len(subset)}")
        print(f"  date range: {dates.min().date()} -> {dates.max().date()}")
        print(f"  gaps > 3 days: {len(large_gaps)}")
        if not large_gaps.empty:
            largest_gap = large_gaps.max()
            gap_date = dates[large_gaps.idxmax()]
            print(f"  largest gap: {largest_gap} days (ending at {gap_date.date()})")

        year_counts = subset["DATE"].dt.year.value_counts().sort_index()
        print("  rows per year:")
        for year, count in year_counts.items():
            print(f"    {year}: {count}")


def main() -> None:
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full = load_all_market_files(workspace_root)
    report_missing_dates(full)


if __name__ == "__main__":
    main()
