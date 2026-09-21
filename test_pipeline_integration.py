"""
ml_engine/test_pipeline_integration.py
End-to-end integration and verification test for Bomeli ML Predictive Engine.
"""

import os
import sys
import json
import joblib
import pymysql
import pandas as pd

from config import DATA_DIR, MODELS_DIR, get_db_connection, get_active_rate_package

def test_integration():
    print("=== STARTING INTEGRATION TESTS ===")

    # 1. Test Rate Package
    conn = get_db_connection()
    rate_pkg = get_active_rate_package(conn)
    assert rate_pkg is not None, "Failed to get active rate package"
    assert 'rates_by_years' in rate_pkg, "Missing rates_by_years"
    print(f"[PASS] Active Rate Package: '{rate_pkg['package_name']}' - 1Yr={rate_pkg['rate_1_yr']}, 2Yr={rate_pkg['rate_2_yr']}, 3Yr={rate_pkg['rate_3_yr']}")

    # 2. Test Simulated Data File
    sim_csv = os.path.join(DATA_DIR, 'simulated_portfolio.csv')
    assert os.path.exists(sim_csv), f"Simulated data file missing: {sim_csv}"
    df_sim = pd.read_csv(sim_csv)
    assert len(df_sim) >= 500, f"Expected at least 500 records, got {len(df_sim)}"
    print(f"[PASS] Simulated Dataset: {len(df_sim)} records verified in {sim_csv}")

    # 3. Test Trained Models
    hazard_path = os.path.join(MODELS_DIR, 'default_hazard_model.joblib')
    early_path = os.path.join(MODELS_DIR, 'early_settlement_model.joblib')
    markov_path = os.path.join(MODELS_DIR, 'markov_matrix.json')

    assert os.path.exists(hazard_path), "Missing default_hazard_model.joblib"
    assert os.path.exists(early_path), "Missing early_settlement_model.joblib"
    assert os.path.exists(markov_path), "Missing markov_matrix.json"

    hazard_model = joblib.load(hazard_path)
    early_model = joblib.load(early_path)

    # Test sample prediction
    sample_feat = pd.DataFrame([{
        'term_progress_ratio': 0.50,
        'dti_ratio': 0.22,
        'rebate_streak': 6,
        'late_penalty_streak': 0,
        'overdue_count': 0,
        'partial_payment_ratio': 1.0,
        'on_time_reliability': 1.0,
        'ci_negative_flags': 0,
        'has_phone_bounce': 0,
        'length_of_stay_years': 8.0,
        'employment_type': 'Employed (Private/Gov)',
        'residential_ownership': 'Owned (Clean Title)'
    }])
    p_def = hazard_model.predict_proba(sample_feat)[0, 1]
    assert 0.0 <= p_def <= 1.0, "Hazard prob out of range"
    print(f"[PASS] Model 1 Test Inference: sample prime borrower default probability = {p_def:.4f}")

    # 4. Test MySQL Ingestion
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS c FROM ai_predictive_insights")
        ins_count = cur.fetchone()['c']
        assert ins_count > 0, "No records in ai_predictive_insights"

        cur.execute("SELECT COUNT(*) AS c FROM ai_portfolio_monthly_snapshots")
        snap_count = cur.fetchone()['c']
        assert snap_count > 0, "No records in ai_portfolio_monthly_snapshots"

        cur.execute("SELECT COUNT(*) AS c FROM ai_account_risk_scores")
        score_count = cur.fetchone()['c']
        print(f"[PASS] MySQL Tables Verified: {ins_count} insights rows, {snap_count} snapshots rows, {score_count} account risk scores.")

        # Test Live Forecast Ingestion Query for current month (e.g. 2026-09)
        cur.execute("""
            SELECT 
                f.forecast_month,
                s.total_scheduled_due,
                s.total_actual_collected,
                JSON_UNQUOTE(JSON_EXTRACT(f.cash_forecast_json, '$.ai_expected_collected')) AS ai_expected_collected
            FROM ai_predictive_insights f
            JOIN ai_portfolio_monthly_snapshots s 
                ON s.snapshot_month = f.forecast_month AND (s.branch_id IS NULL AND f.branch_id IS NULL)
            WHERE f.forecast_month = '2026-09'
            LIMIT 1
        """)
        live_row = cur.fetchone()
        assert live_row is not None, "Failed Live Forecast query for 2026-09"
        print(f"[PASS] Live Forecast Query for 2026-09: AI Expected = PHP {float(live_row['ai_expected_collected']):,.2f} (Actual MTD: PHP {float(live_row['total_actual_collected']):,.2f})")

    conn.close()
    print("\n=== ALL INTEGRATION TESTS PASSED PERFECTLY ===")

if __name__ == '__main__':
    test_integration()
