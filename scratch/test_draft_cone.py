import sys
import os
import json
from datetime import datetime

# Insert workspace root and ml_engine to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ML_DIR = os.path.join(BASE_DIR, 'ml_engine')
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, ML_DIR)

from dml.cash_engine import CashEngine, PaymentStreamBreakdown

engine = CashEngine()

# Test sample accounts
accounts = [
    {'sale_id': 1, 'monthly_amortization': 8285.00, 'term_months': 12, 'paid_terms': 10, 'default_probability': 0.02, 'status': 'active'},
    {'sale_id': 2, 'monthly_amortization': 3305.56, 'term_months': 36, 'paid_terms': 4, 'default_probability': 0.12, 'status': 'active'},
    {'sale_id': 3, 'monthly_amortization': 4500.00, 'term_months': 24, 'paid_terms': 12, 'default_probability': 0.05, 'status': 'active'},
    {'sale_id': 4, 'monthly_amortization': 5200.00, 'term_months': 36, 'paid_terms': 2, 'default_probability': 0.45, 'status': 'delinquent', 'total_arrears': 10400.0}
]

stream = PaymentStreamBreakdown(
    total_collected_mtd=123425.0,
    regular_collected_mtd=8285.0,
    advance_collected_mtd=16570.0,
    early_settlement_collected_mtd=96570.0,
    partial_collected_mtd=2000.0,
    regular_account_count=1,
    advance_account_count=1,
    early_settlement_count=1,
    partial_account_count=1
)

cone = engine.compute_cash_forecast_cone(
    accounts=accounts,
    actual_stream=stream,
    projected_sales_units=6,
    actual_sales_units=4,
    forecast_month='2026-09'
)

print(json.dumps(cone, indent=2))
