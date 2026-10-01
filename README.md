# ML-bomeli-api

Production Machine Learning & Predictive Forecasting REST API for Bomeli Motorcycle Financing.

## Overview
This service provides AI-driven financial risk modeling and portfolio forecasting:
- **90-Day Delinquency Migration Hazard**: Calibrated GBDT classification model predicting 3-month default risk.
- **Early Settlement Buyout Propensity**: HistGBM classifier identifying accounts eligible for Option Contract early settlements.
- **Markov Roll-Rate Migration Matrix**: Branch-level transition dynamics (Active &rarr; Delinquent &rarr; Default / Cure).
- **Dynamic Cash Inflow Cone**: Multi-branch 90-day cash projection with behavioral slippage tiers.
- **Inventory Velocity Engine**: Stockout velocity model forecasting sales rates and days-to-zero.

---

## Deployment on Render

This repository is pre-configured with a [`Dockerfile`](Dockerfile) and [`render.yaml`](render.yaml) for 1-click deployment on [Render](https://render.com).

### 1. Create Web Service
1. Log in to [dashboard.render.com](https://dashboard.render.com).
2. Click **New +** &rarr; **Web Service**.
3. Connect this GitHub repository (`ML-bomeli-api`).
4. Set:
   - **Environment / Runtime**: `Docker`
   - **Region**: `Singapore` (or nearest to your MySQL database)
   - **Plan**: `Free`
5. Configure Environment Variables:
   - `DB_HOST`: Hostinger remote MySQL host / IP
   - `DB_USER`: Remote database username
   - `DB_PASSWORD`: Remote database password
   - `DB_NAME`: Database name (e.g. `bomeli_db1`)
   - `DB_PORT`: `3306`
   - `ML_API_KEY`: Custom secret token to secure endpoints

---

## REST Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service info, database connectivity status, model readiness |
| `GET` | `/health` | Liveness health check |
| `GET` | `/docs` | Interactive Swagger UI API documentation |
| `POST` | `/api/run_forecast` | Executes full portfolio forecast & updates MySQL |
| `POST` | `/api/predict/default_hazard` | Single-account 90-day delinquency hazard scoring |
| `POST` | `/api/predict/early_settlement` | Single-account early settlement propensity scoring |
| `POST` | `/api/train` | Triggers background model re-calibration |

---

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```
