from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from app.config import DB_URL

engine: Engine = create_engine(DB_URL, echo=False)

def init_db():
    create_sql = """
    CREATE TABLE IF NOT EXISTS ticks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,

        seance TEXT,
        groupe TEXT,
        code TEXT,
        valeur TEXT,

        ouverture REAL,
        cloture REAL,
        plus_bas REAL,
        plus_haut REAL,

        quantite_negociee REAL,
        nb_transaction REAL,
        capitaux REAL,

        ind_res TEXT,
        raw_json TEXT
    );
    """
    with engine.connect() as conn:
        conn.execute(text(create_sql))
        conn.commit()
