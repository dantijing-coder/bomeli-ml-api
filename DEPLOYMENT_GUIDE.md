# Bomeli ML Engine: Production Deployment Guide (Hostinger + Hugging Face / Render)

This step-by-step guide explains how to deploy the **Bomeli Machine Learning & Predictive Forecasting Engine** to **Hugging Face Spaces** or **Render** and link it with your PHP web application on **Hostinger**.

---

## 1. System Architecture

```
┌────────────────────────────────────────────────────────┐
│               HOSTINGER WEB HOSTING                    │
│  - PHP 8.2+ / Apache Web Application                   │
│  - Hostinger MySQL Database (bomeli_db1)               │
│  - User Interface (pages/ai-sales-insights.php)        │
└──────────────────────────┬─────────────────────────────┘
                           │ 
          1. Triggers HTTP POST /api/run_forecast
          2. Receives updated predictions
                           ▼
┌────────────────────────────────────────────────────────┐
│       HUGGING FACE SPACES / RENDER CLOUD               │
│  - Python 3.11 FastAPI Web Service (ml_engine/app.py)  │
│  - Calibrated GBDT (90-Day Delinquency Hazard Model)   │
│  - HistGBM (Option Contract Early Buyout Model)        │
│  - Markov Roll-Rate Migration Matrix (Model 5)         │
│  - Directly queries & updates Hostinger MySQL DB       │
└────────────────────────────────────────────────────────┘
```

---

## 2. Prerequisites & Prepared Files

All deployment files have been prepared and configured in `ml_engine/`:
- [`app.py`](file:///c:/xampp/htdocs/Bomeli1/ml_engine/app.py): Production FastAPI application with endpoints for health checks, live batch forecasting, individual hazard scoring, and re-training.
- [`requirements.txt`](file:///c:/xampp/htdocs/Bomeli1/ml_engine/requirements.txt): Pinned dependencies (`fastapi`, `uvicorn`, `scikit-learn`, `pandas`, `joblib`, `pymysql`, `cryptography`, etc.).
- [`Dockerfile`](file:///c:/xampp/htdocs/Bomeli1/ml_engine/Dockerfile): Optimized Python 3.11 container for Hugging Face Spaces / Render.
- [`.env.example`](file:///c:/xampp/htdocs/Bomeli1/ml_engine/.env.example): Reference configuration template for cloud environment secrets.
- [`config/ml_api_config.php`](file:///c:/xampp/htdocs/Bomeli1/config/ml_api_config.php): PHP API bridge linking Hostinger to Cloud ML.

---

## 3. Step 1: Configure Hostinger Remote MySQL Access

Because the Python ML engine runs on cloud infrastructure (Hugging Face / Render), it needs remote permission to connect to your Hostinger MySQL database.

1. Log in to your **Hostinger hPanel**.
2. Navigate to **Databases** &rarr; **Remote MySQL**.
3. Under **Create Remote MySQL Connection**:
   - **Host (IP)**: Enter `%` (allows any remote IP with valid username/password) OR specify the IP range of your cloud provider.
   - **Database**: Select your active Bomeli database (e.g., `u123456789_bomeli_db`).
   - Click **Create**.
4. Note your database credentials from Hostinger:
   - **`DB_HOST`**: Database server IP or hostname (e.g. `srv1234.hstgr.io` or your domain name).
   - **`DB_USER`**: Database username (e.g. `u123456789_bomeli`).
   - **`DB_PASSWORD`**: Database password.
   - **`DB_NAME`**: Database name (e.g. `u123456789_bomeli_db`).
   - **`DB_PORT`**: `3306`.

---

## 4. Step 2A: Deploy on Hugging Face Spaces (Recommended & Free)

Hugging Face Spaces provides **free, persistent 16GB RAM CPU compute** with SSL/HTTPS.

### 1. Create Space
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. **Space Name**: e.g. `bomeli-ml-engine`.
3. **License**: `MIT` or `Apache 2.0`.
4. **Space SDK**: Select **Docker** &rarr; **Blank**.
5. **Space Hardware**: Free CPU (16GB RAM, 2 vCPU).
6. **Visibility**: `Public` (recommended) or `Private`.
7. Click **Create Space**.

### 2. Upload ML Files
Clone the Space repository locally using Git or upload the files directly via the Hugging Face web interface. Upload the entire contents of `ml_engine/`:
```
├── app.py
├── config.py
├── run_predictive_engine.py
├── simulate_portfolio.py
├── train_models.py
├── requirements.txt
├── Dockerfile
├── dml/
│   ├── __init__.py
│   ├── cash_engine.py
│   ├── markov_engine.py
│   ├── model_registry.py
│   ├── survival_engine.py
│   └── velocity_engine.py
├── models/
│   ├── default_hazard_model.joblib
│   ├── early_settlement_model.joblib
│   └── markov_matrix.json
└── data/
    ├── simulated_portfolio.csv
    └── simulated_historical_snapshots.json
```

### 3. Set Environment Secrets
In your Hugging Face Space:
1. Click **Settings** &rarr; scroll to **Variables and secrets**.
2. Click **New secret** and add:
   - `DB_HOST` = `<your_hostinger_db_host>`
   - `DB_USER` = `<your_hostinger_db_user>`
   - `DB_PASSWORD` = `<your_hostinger_db_password>`
   - `DB_NAME` = `<your_hostinger_db_name>`
   - `DB_PORT` = `3306`
   - `ML_API_KEY` = `<your_custom_secret_api_key>` (e.g. `bomeli_sec_9948a73b4e0192df8c21`)
3. Hugging Face will automatically trigger a build.
4. When the status turns to **Running** (green indicator), your API is live!
5. Copy your public Space URL (e.g. `https://yourusername-bomeli-ml-engine.hf.space`).

---

## 5. Step 2B: Alternative — Deploy on Render

If you prefer Render:

1. Sign in to [dashboard.render.com](https://dashboard.render.com).
2. Click **New +** &rarr; **Web Service**.
3. Connect your GitHub repository containing the Bomeli project.
4. Configure settings:
   - **Name**: `bomeli-ml-engine`
   - **Root Directory**: `ml_engine`
   - **Environment**: `Docker` (or `Python 3`)
   - **Plan**: `Free`
5. Under **Environment Variables**, add:
   - `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, `DB_PORT`, `ML_API_KEY`.
6. Click **Create Web Service**.
7. Render will build and deploy your container. Once live, copy your Render URL (e.g. `https://bomeli-ml-engine.onrender.com`).

---

## 6. Step 3: Link Hostinger Web Application to Cloud ML

1. In your Hostinger web hosting file manager (or local repository before uploading to Hostinger), open [`config/ml_api_config.php`](file:///c:/xampp/htdocs/Bomeli1/config/ml_api_config.php).
2. Set your live Cloud ML URL and API Key:

```php
// config/ml_api_config.php

define('ML_API_URL', 'https://yourusername-bomeli-ml-engine.hf.space');
define('ML_API_KEY', 'bomeli_sec_9948a73b4e0192df8c21');
```

3. Save and upload `config/ml_api_config.php` to Hostinger.

---

## 7. Step 4: Verification & Live Testing

1. Open your browser and navigate to your Hostinger web app:
   `https://your-domain.com/pages/ai-sales-insights.php`
2. Look at the top toolbar:
   - You will see the **AI Engine: Calibrated • Active** status badge.
   - Click **[Architecture Dossier]** to open the Model Specifications modal.
3. Click the **Refresh AI Engine** button:
   - Hostinger PHP sends an authenticated request to your Cloud ML API.
   - The cloud engine executes the pipeline, updates MySQL, and returns confirmation.
   - The page refreshes with real-time predictions, updated cash forecasts, and risk tiers!

---

## 8. Step 5: Automated Recurring Cron Job

To automatically re-calculate forecasts on the 1st of every month or every midnight:

### In Hostinger hPanel:
1. Go to **Advanced** &rarr; **Cron Jobs**.
2. **Cron Type**: Custom.
3. **Command**:
   ```bash
   curl -s -X POST https://yourusername-bomeli-ml-engine.hf.space/api/run_forecast -H "X-API-Key: bomeli_sec_9948a73b4e0192df8c21" > /dev/null 2>&1
   ```
4. **Schedule**: `0 0 1 * *` (Runs at 00:00 on Day 1 of every month) or `0 0 * * *` (Daily midnight).
5. Click **Save**.

---

## 9. Summary of ML API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service info, database connectivity status, and model metadata |
| `GET` | `/health` | Liveness health check |
| `POST` | `/api/run_forecast` | Executes full multi-branch forecasting & updates MySQL |
| `POST` | `/api/predict/default_hazard` | Single-account 90-day default hazard inference |
| `POST` | `/api/predict/early_settlement` | Single-account early settlement propensity inference |
| `POST` | `/api/train` | Re-trains and calibrates models on simulated dataset |
