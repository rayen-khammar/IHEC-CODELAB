# IHEC-CODELAB — BVMT Intelligent Trading Assistant (Hackathon Prototype)

This repo is a **system integration demo**. It wires price prediction, sentiment, anomaly detection, and a decision engine into an end‑to‑end trading assistant. Models are **not retrained**; the focus is robust orchestration and safe fallbacks.

## What’s inside (high‑level)

- **Decision engine**: [decision_engine.py](decision_engine.py)
- **Demo runner**: [demo_decision_engine.py](demo_decision_engine.py)
- **Cahier data + models**: cahier-de-charges-code_lab2.0/
- **Live BVMT scraper + API**: IHEC-CODELAB/app/
- **Frontend demo** (static): projet/
- **Anomaly detection**: anomalie/
- **XAI helpers**: xai_module/
- **Scraped news**: scraped_data/

## Repository layout (key files)

### Core integration
- [decision_engine.py](decision_engine.py)
	- Safe signal normalization
	- Mean–variance utility scoring
	- Anomaly penalty
	- Portfolio projection
	- Structured explanation

- [demo_decision_engine.py](demo_decision_engine.py)
	- Uses **cahier** data only (no BVMT scraping)
	- Default demo: **ATTIJARI BANK**
	- Alerts demo: **BIAT** (or SFBT)

### Cahier dataset + models
- cahier-de-charges-code_lab2.0/
	- histo_cotation_*.csv/txt (source market data)
	- models/<COMPANY>/ (trained models: xgboost.json, gradient_boosting.pkl)
	- src/prepare_multi_company_dataset.py (dataset prep)
	- src/analyze_multi_company_features.py (feature importance)
	- src/infer_five_day_forecast.py (forecast + model loading)

### Live API (optional)
- [IHEC-CODELAB/app/api.py](IHEC-CODELAB/app/api.py)
	- FastAPI endpoints + decision demo
	- CORS enabled for the frontend

- [IHEC-CODELAB/app/ingestion.py](IHEC-CODELAB/app/ingestion.py)
	- BVMT table scraping pipeline

### Frontend (static)
- projet/
	- index.html, dashboard.html, stock.html, alerts.html, portfolio.html
	- JS: [projet/js/stock.js](projet/js/stock.js) now calls /decision_demo

## Quickstart (demo from cahier data)

1) **Install dependencies** (all required in one place):

- Base (from IHEC-CODELAB/requirements.txt)
	- requests, beautifulsoup4, lxml, pandas, sqlalchemy, fastapi, uvicorn, python-dotenv
- Additional for demo engine:
	- joblib, scikit-learn, textblob
	- xgboost (optional)

2) **Run API**

- Start FastAPI server:
	- IHEC-CODELAB/run_api.py

3) **Open frontend**

- Serve static site from projet/ (e.g., VS Code Live Server or python http.server)
- Visit http://localhost:5500/stock.html

4) **Decision demo endpoint**

- http://localhost:8000/decision_demo?company=ATTIJARI%20BANK&alert_stock=BIAT

## API endpoints (local)

- /decision_demo
	- Params: company, alert_stock, cash, holdings, horizon
	- Uses cahier data and trained models

- /codes, /latest, /history
	- BVMT live data (if BVMT_URL is set to an actual cotation table page)

## Why the UI can appear blank

If the Stock page shows “—” values, it’s usually one of these:

1) **API not running** on port 8000
2) **CORS blocked** (fixed in [IHEC-CODELAB/app/api.py](IHEC-CODELAB/app/api.py))
3) **Frontend not served** (opening file:// directly can block requests)
4) **No models available** for selected company

## Data + Modeling approach (summary)

1) **Data normalization** (cahier files → standardized schema)
2) **Feature engineering** (lags, rolling stats, volatility)
3) **Forecast generation** (5‑day recursive prediction)
4) **Signal fusion** in `decision_engine.py`
5) **Anomaly detection** (market + model error)
6) **Portfolio projection** (cash + holdings using forecasted prices)

## Notes

- This is a hackathon prototype; modules are defensive and do not assume correctness.
- Live BVMT scraping is optional. The demo uses **cahier** historical data by default.
