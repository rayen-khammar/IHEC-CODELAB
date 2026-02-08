from fastapi import FastAPI
from sqlalchemy import text
from app.database import engine, init_db

app = FastAPI(title="BVMT Live API")

@app.on_event("startup")
def startup():
    init_db()

@app.get("/")
def root():
    return {"status": "ok", "message": "BVMT Live API running"}

@app.get("/codes")
def codes():
    q = text("""
        SELECT DISTINCT code
        FROM ticks
        WHERE code IS NOT NULL
        ORDER BY code ASC
        LIMIT 500
    """)
    with engine.connect() as conn:
        rows = conn.execute(q).fetchall()

    return [r[0] for r in rows]

@app.get("/latest")
def latest(code: str):
    q = text("""
        SELECT *
        FROM ticks
        WHERE code = :code
        ORDER BY id DESC
        LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(q, {"code": code}).fetchone()

    return dict(row._mapping) if row else {}

@app.get("/history")
def history(code: str, limit: int = 200):
    q = text("""
        SELECT *
        FROM ticks
        WHERE code = :code
        ORDER BY id DESC
        LIMIT :limit
    """)
    with engine.connect() as conn:
        rows = conn.execute(q, {"code": code, "limit": limit}).fetchall()

    return [dict(r._mapping) for r in rows]
