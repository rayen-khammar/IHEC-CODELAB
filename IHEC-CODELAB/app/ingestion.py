import json
from datetime import datetime, timezone
import pandas as pd
from sqlalchemy import text

from app.scraper import scrape_bvmt_table
from app.cleaner import clean_number, clean_text
from app.database import engine


def standardize(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]

    # 🔥 Colonnes obligatoires (ta forme exacte)
    required_cols = [
        "SEANCE", "GROUPE", "CODE", "VALEUR",
        "OUVERTURE", "CLOTURE", "PLUS_BAS", "PLUS_HAUT",
        "QUANTITE_NEGOCIEE", "NB_TRANSACTION", "CAPITAUX", "IND_RES"
    ]

    # si une colonne n'existe pas, on la crée vide
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    out = pd.DataFrame()

    # timestamp (pour temps réel)
    out["TIMESTAMP"] = datetime.now(timezone.utc).isoformat()

    # textes
    out["SEANCE"] = df["SEANCE"].apply(clean_text)
    out["GROUPE"] = df["GROUPE"].apply(clean_text)
    out["CODE"] = df["CODE"].apply(clean_text)
    out["VALEUR"] = df["VALEUR"].apply(clean_text)
    out["IND_RES"] = df["IND_RES"].apply(clean_text)

    # numériques
    out["OUVERTURE"] = df["OUVERTURE"].apply(clean_number)
    out["CLOTURE"] = df["CLOTURE"].apply(clean_number)
    out["PLUS_BAS"] = df["PLUS_BAS"].apply(clean_number)
    out["PLUS_HAUT"] = df["PLUS_HAUT"].apply(clean_number)
    out["QUANTITE_NEGOCIEE"] = df["QUANTITE_NEGOCIEE"].apply(clean_number)
    out["NB_TRANSACTION"] = df["NB_TRANSACTION"].apply(clean_number)
    out["CAPITAUX"] = df["CAPITAUX"].apply(clean_number)

    # supprimer lignes vides
    out = out.dropna(subset=["CODE", "VALEUR"])
    out = out[out["CODE"].astype(str).str.len() > 0]

    # debug raw
    out["RAW_JSON"] = json.dumps(df_raw.head(50).to_dict(orient="records"), ensure_ascii=False)

    return out


def insert_ticks(df: pd.DataFrame):
    sql = text("""
        INSERT INTO ticks(
            timestamp, seance, groupe, code, valeur,
            ouverture, cloture, plus_bas, plus_haut,
            quantite_negociee, nb_transaction, capitaux,
            ind_res, raw_json
        )
        VALUES (
            :timestamp, :seance, :groupe, :code, :valeur,
            :ouverture, :cloture, :plus_bas, :plus_haut,
            :quantite_negociee, :nb_transaction, :capitaux,
            :ind_res, :raw_json
        )
    """)

    with engine.connect() as conn:
        for _, row in df.iterrows():
            conn.execute(sql, {
                "timestamp": row["TIMESTAMP"],
                "seance": row["SEANCE"],
                "groupe": row["GROUPE"],
                "code": row["CODE"],
                "valeur": row["VALEUR"],
                "ouverture": row["OUVERTURE"],
                "cloture": row["CLOTURE"],
                "plus_bas": row["PLUS_BAS"],
                "plus_haut": row["PLUS_HAUT"],
                "quantite_negociee": row["QUANTITE_NEGOCIEE"],
                "nb_transaction": row["NB_TRANSACTION"],
                "capitaux": row["CAPITAUX"],
                "ind_res": row["IND_RES"],
                "raw_json": row["RAW_JSON"]
            })
        conn.commit()


def run_one_cycle(url: str) -> pd.DataFrame:
    df_raw = scrape_bvmt_table(url)
    df = standardize(df_raw)
    insert_ticks(df)
    return df
