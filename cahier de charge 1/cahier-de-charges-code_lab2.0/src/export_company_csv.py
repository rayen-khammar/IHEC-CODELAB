from __future__ import annotations

import glob
import os
import pandas as pd


def _standardize_columns(columns):
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

    required = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "COMPANY", "CODE"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")
    for col in ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(" ", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["COMPANY"] = df["COMPANY"].astype(str).str.strip()
    df["CODE"] = df["CODE"].astype(str).str.strip()

    df = df.dropna(subset=["DATE", "CLOSE"]).reset_index(drop=True)
    return df


def main() -> None:
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

    full = pd.concat(frames, ignore_index=True)

    company = "BIAT"
    company_df = full[full["COMPANY"].str.upper() == company.upper()].copy()
    if company_df.empty:
        raise ValueError(f"Company '{company}' not found in input files.")

    company_df = company_df.sort_values("DATE").reset_index(drop=True)

    output_path = os.path.join(workspace_root, "biat_all_years.csv")
    company_df.to_csv(output_path, index=False)

    print(f"Exported {len(company_df)} rows to {output_path}")


if __name__ == "__main__":
    main()
