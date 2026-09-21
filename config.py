"""
ml_engine/config.py
Configuration and database connection helper for Bomeli ML Predictive Engine.
Supports both local XAMPP environment and cloud deployments (Hugging Face Spaces / Render).
"""

import os
import pymysql

# Load local .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Database Configuration with Cloud / Environment Variable Overrides
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'bomeli_db1'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor,
    'autocommit': True,
    'connect_timeout': 10
}

# Optional API Security Key for Cloud Deployments
ML_API_KEY = os.getenv('ML_API_KEY', '')

def get_db_connection():
    """Returns a new PyMySQL connection using active DB_CONFIG."""
    return pymysql.connect(**DB_CONFIG)

def get_active_rate_package(conn=None):
    """
    Fetches the currently active rate package from the database.
    Falls back to Standard Factory Rates (1.26, 1.48, 1.72) if not found.
    """
    should_close = False
    if conn is None:
        try:
            conn = get_db_connection()
            should_close = True
        except Exception as e:
            print(f"[config] Notice: Could not connect to DB for rate package ({e}), using default fallback.")
            return _default_rate_package()

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, package_name, rate_1_yr, rate_2_yr, rate_3_yr, is_active
                FROM rate_packages
                WHERE is_active = 1
                ORDER BY id DESC
                LIMIT 1
            """)
            row = cur.fetchone()
            if row:
                return {
                    'id': int(row['id']),
                    'package_name': str(row['package_name']),
                    'rate_1_yr': float(row['rate_1_yr']),
                    'rate_2_yr': float(row['rate_2_yr']),
                    'rate_3_yr': float(row['rate_3_yr']),
                    'rates_by_years': {
                        1: float(row['rate_1_yr']),
                        2: float(row['rate_2_yr']),
                        3: float(row['rate_3_yr'])
                    },
                    'rates_by_months': {
                        12: float(row['rate_1_yr']),
                        24: float(row['rate_2_yr']),
                        36: float(row['rate_3_yr'])
                    }
                }
    except Exception as e:
        print(f"[config] Notice: Could not read rate_packages ({e}), falling back to default.")
    finally:
        if should_close:
            conn.close()

    return _default_rate_package()

def _default_rate_package():
    return {
        'id': 1,
        'package_name': 'Default Factory Standard',
        'rate_1_yr': 1.26,
        'rate_2_yr': 1.48,
        'rate_3_yr': 1.72,
        'rates_by_years': {1: 1.26, 2: 1.48, 3: 1.72},
        'rates_by_months': {12: 1.26, 24: 1.48, 36: 1.72}
    }
