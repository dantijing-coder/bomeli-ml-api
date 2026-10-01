"""
ml_engine/synthesize_training_dataset.py

Generates an independent, isolated simulated dataset of 3,500 financing loan records
specifically for training and benchmarking Bomeli's Machine Learning models.

SAFETY:
- Strictly saves to ml_engine/data/simulated_training_dataset.csv.
- Completely isolated from rawdata/seed_data/ and does NOT touch the live database.
- Conforms to the full feature schema: demographics, contract terms, payment trajectory,
  streaks, arrears, and ground truth target variables.
"""

import os
import random
import numpy as np
import pandas as pd

# Set deterministic seed for reproducible training
random.seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(DATA_DIR, 'simulated_training_dataset.csv')

NUM_RECORDS = 3500

BRANCHES = [
    {'branch_id': 1, 'branch_name': 'LALA', 'traffic_factor': 1.15},
    {'branch_id': 2, 'branch_name': 'KAPATAGAN', 'traffic_factor': 1.05},
    {'branch_id': 3, 'branch_name': 'TUBOD', 'traffic_factor': 0.85}
]

MODELS_CATALOG = [
    {'brand': 'HONDA', 'model_code': 'CLICK 125I', 'category': 'Scooter', 'base_price': 81400.0, 'base_vel': 4.5},
    {'brand': 'HONDA', 'model_code': 'CLICK 160', 'category': 'Scooter', 'base_price': 122900.0, 'base_vel': 4.0},
    {'brand': 'HONDA', 'model_code': 'PCX160', 'category': 'Maxi-Scooter', 'base_price': 134900.0, 'base_vel': 2.8},
    {'brand': 'HONDA', 'model_code': 'ADV160', 'category': 'Adventure Scooter', 'base_price': 166900.0, 'base_vel': 2.2},
    {'brand': 'HONDA', 'model_code': 'BEAT PREMIUM', 'category': 'Scooter', 'base_price': 72400.0, 'base_vel': 4.8},
    {'brand': 'HONDA', 'model_code': 'TMX125 ALPHA', 'category': 'Backbone/Utility', 'base_price': 57900.0, 'base_vel': 4.5},
    {'brand': 'HONDA', 'model_code': 'WAVE RSX', 'category': 'Underbone', 'base_price': 64900.0, 'base_vel': 3.8},
    {'brand': 'KAWASAKI', 'model_code': 'P24KK (BARAKO II)', 'category': 'Backbone/Tricycle', 'base_price': 68500.0, 'base_vel': 4.2},
    {'brand': 'KAWASAKI', 'model_code': 'CT100B', 'category': 'Backbone/Utility', 'base_price': 54000.0, 'base_vel': 3.5},
    {'brand': 'KAWASAKI', 'model_code': 'ROUSER NS125', 'category': 'Sport Backbone', 'base_price': 82000.0, 'base_vel': 2.4},
    {'brand': 'YAMAHA', 'model_code': 'MIO SPORTY', 'category': 'Scooter', 'base_price': 73900.0, 'base_vel': 5.0},
    {'brand': 'YAMAHA', 'model_code': 'MIO GEAR 125', 'category': 'Scooter', 'base_price': 79400.0, 'base_vel': 4.2},
    {'brand': 'YAMAHA', 'model_code': 'MIO FAZZIO', 'category': 'Retro Scooter', 'base_price': 93900.0, 'base_vel': 3.2},
    {'brand': 'YAMAHA', 'model_code': 'NMAX 155', 'category': 'Maxi-Scooter', 'base_price': 151900.0, 'base_vel': 3.5},
    {'brand': 'YAMAHA', 'model_code': 'AEROX 155', 'category': 'Sport Scooter', 'base_price': 125400.0, 'base_vel': 3.4},
    {'brand': 'YAMAHA', 'model_code': 'SNIPER 155', 'category': 'Sport Underbone', 'base_price': 123900.0, 'base_vel': 3.0},
    {'brand': 'SUZUKI', 'model_code': 'RAIDER 150 FI', 'category': 'Sport Underbone', 'base_price': 119900.0, 'base_vel': 3.2},
    {'brand': 'SUZUKI', 'model_code': 'SMASH 115', 'category': 'Underbone', 'base_price': 62400.0, 'base_vel': 4.0},
    {'brand': 'SUZUKI', 'model_code': 'BURGMAN STREET 125', 'category': 'Maxi-Scooter', 'base_price': 83400.0, 'base_vel': 3.6}
]

EMPLOYMENT_PROFILES = [
    {'type': 'Employed (Private/Gov)', 'inc_mean': 32000, 'inc_std': 8000, 'def_prop': 0.08, 'early_prop': 0.18},
    {'type': 'Public School Teacher', 'inc_mean': 38000, 'inc_std': 7000, 'def_prop': 0.04, 'early_prop': 0.22},
    {'type': 'OFW Remittance / Dependent', 'inc_mean': 52000, 'inc_std': 14000, 'def_prop': 0.06, 'early_prop': 0.28},
    {'type': 'BPO / Call Center Agent', 'inc_mean': 30000, 'inc_std': 6000, 'def_prop': 0.10, 'early_prop': 0.14},
    {'type': 'Sari-Sari / Small Retail', 'inc_mean': 24000, 'inc_std': 5000, 'def_prop': 0.15, 'early_prop': 0.10},
    {'type': 'Tricycle / PUV Operator', 'inc_mean': 20000, 'inc_std': 4000, 'def_prop': 0.20, 'early_prop': 0.06},
    {'type': 'Agri / Farmer / Fisherfolk', 'inc_mean': 18000, 'inc_std': 4500, 'def_prop': 0.22, 'early_prop': 0.05}
]

RESIDENTIAL_PROFILES = [
    'Owned (Clean Title)', 'Owned (Inherited/Tax Dec)', 'Living with Parents/Relatives', 'Rented (Apartment/House)'
]

FACTOR_RATES = {12: 1.26, 24: 1.48, 36: 1.72}

def generate_dataset():
    print(f"[synthesize] Generating {NUM_RECORDS:,} synthetic financing records...")
    records = []

    for i in range(1, NUM_RECORDS + 1):
        branch = random.choice(BRANCHES)
        model = random.choice(MODELS_CATALOG)
        emp = random.choice(EMPLOYMENT_PROFILES)
        res = random.choice(RESIDENTIAL_PROFILES)
        civil = random.choice(['Single', 'Married', 'Married', 'Widowed'])

        monthly_income = max(12000.0, round(float(np.random.normal(emp['inc_mean'], emp['inc_std'])), 2))
        stay_years = round(float(np.random.gamma(shape=3.0, scale=2.5)), 1)
        stay_years = min(35.0, max(1.0, stay_years))

        # Loan contract terms (12, 24, 36 months)
        term_months = random.choices([12, 24, 36], weights=[0.30, 0.35, 0.35])[0]
        factor_rate = FACTOR_RATES[term_months]
        gross_price = model['base_price']
        dp_ratio = random.choice([0.10, 0.12, 0.15, 0.20, 0.25, 0.30])
        down_payment = round(gross_price * dp_ratio, 2)
        principal = round(gross_price - down_payment, 2)
        pn_value = round(principal * factor_rate, 2)
        base_monthly = round(pn_value / term_months, 2)
        monthly_amortization = round(base_monthly + 200.0, 2)

        dti_ratio = round(monthly_amortization / monthly_income, 4)

        # Simulation of payment progress & tenure
        elapsed_terms = random.randint(1, term_months)
        term_progress_ratio = round(elapsed_terms / term_months, 4)

        # Behavioral simulation
        # Baseline probability of default influenced by DTI, income, employment, residential status
        risk_score = (
            (dti_ratio * 1.4) +
            (0.12 if 'Rented' in res else (0.05 if 'Parents' in res else 0.0)) +
            (emp['def_prop'] * 1.4) -
            (0.10 if stay_years >= 5.0 else 0.0)
        )
        risk_score = max(0.02, min(0.95, risk_score))

        # Behavioral metrics
        if risk_score > 0.46:
            # High risk
            on_time_reliability = round(float(np.random.uniform(0.20, 0.65)), 3)
            rebate_streak = random.choice([0, 0, 1])
            late_penalty_streak = random.choice([1, 2, 3, 4])
            overdue_count = random.choice([1, 2, 3, 4])
            partial_payment_ratio = round(float(np.random.uniform(0.15, 0.45)), 3)
            ci_negative_flags = 1 if random.random() < 0.35 else 0
            has_phone_bounce = 1 if random.random() < 0.25 else 0
            current_status = 'defaulted' if overdue_count >= 2 else 'delinquent'
            will_default_90d = 1 if (random.random() < 0.82) else 0
            lifecycle_outcome = 'defaulted' if will_default_90d else 'completed'
            early_settlement_target = 0
        elif risk_score > 0.26:
            # Medium risk
            on_time_reliability = round(float(np.random.uniform(0.60, 0.85)), 3)
            rebate_streak = random.choice([0, 1, 2, 3])
            late_penalty_streak = random.choice([0, 1, 2])
            overdue_count = random.choice([0, 1])
            partial_payment_ratio = round(float(np.random.uniform(0.0, 0.15)), 3)
            ci_negative_flags = 1 if random.random() < 0.10 else 0
            has_phone_bounce = 0
            current_status = 'delinquent' if overdue_count == 1 else 'active'
            will_default_90d = 1 if (random.random() < 0.18) else 0
            early_settlement_target = 1 if (elapsed_terms >= 5 and rebate_streak >= 2 and random.random() < 0.18) else 0
            lifecycle_outcome = 'early_settled' if early_settlement_target else ('defaulted' if will_default_90d else 'completed')
        else:
            # Low risk / Prime
            on_time_reliability = round(float(np.random.uniform(0.85, 1.00)), 3)
            rebate_streak = random.randint(1, max(1, min(12, elapsed_terms)))
            late_penalty_streak = 0
            overdue_count = 0
            partial_payment_ratio = 0.0
            ci_negative_flags = 0
            has_phone_bounce = 0
            current_status = 'active'
            will_default_90d = 0
            early_settlement_target = 1 if (elapsed_terms >= 5 and rebate_streak >= 3 and random.random() < 0.32) else 0
            lifecycle_outcome = 'early_settled' if early_settlement_target else 'completed'

        paid_terms_count = max(0, elapsed_terms - overdue_count)
        total_arrears = round(overdue_count * monthly_amortization, 2)
        total_paid_sum = round(paid_terms_count * base_monthly, 2)

        # Realization rate (actual collections / scheduled due) for cash flow
        season_month = random.randint(1, 12)
        season_mult = 1.15 if season_month in [11, 12] else (1.05 if season_month in [3, 4] else (0.88 if season_month in [1, 6] else 1.0))
        realization_rate = round(float(np.clip((on_time_reliability * season_mult) + np.random.normal(0, 0.03), 0.50, 1.10)), 4)

        # Inventory velocity simulation
        avg_days_on_lot = round(float(np.random.uniform(10, 65)), 1)
        trailing_sales = max(1, int(round(model['base_vel'] * branch['traffic_factor'] * season_mult * (1.15 if avg_days_on_lot < 25 else 0.85))))
        monthly_velocity = round(float(np.clip(trailing_sales + np.random.normal(0, 0.35), 0.5, 9.0)), 2)

        records.append({
            'synthetic_loan_id': f"SYNTH-{i:05d}",
            'branch_id': branch['branch_id'],
            'branch_name': branch['branch_name'],
            'brand': model['brand'],
            'model_code': model['model_code'],
            'vehicle_category': model['category'],
            'base_price': gross_price,
            'down_payment': down_payment,
            'down_payment_pct': dp_ratio,
            'principal': principal,
            'term_months': term_months,
            'monthly_amortization': monthly_amortization,
            'dti_ratio': dti_ratio,
            'monthly_income': monthly_income,
            'employment_type': emp['type'],
            'residential_ownership': res,
            'civil_status': civil,
            'length_of_stay_years': stay_years,
            'paid_terms_count': paid_terms_count,
            'term_progress_ratio': term_progress_ratio,
            'rebate_streak': rebate_streak,
            'late_penalty_streak': late_penalty_streak,
            'overdue_count': overdue_count,
            'total_arrears': total_arrears,
            'partial_payment_ratio': partial_payment_ratio,
            'on_time_reliability': on_time_reliability,
            'ci_negative_flags': ci_negative_flags,
            'has_phone_bounce': has_phone_bounce,
            'current_status': current_status,
            'avg_days_on_lot': avg_days_on_lot,
            'season_month': season_month,
            'trailing_sales_count': trailing_sales,
            'monthly_sales_velocity': monthly_velocity,
            'realization_rate': realization_rate,
            'early_settlement_target': early_settlement_target,
            'will_default_90d': will_default_90d,
            'lifecycle_outcome': lifecycle_outcome
        })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[synthesize] Successfully saved {len(df):,} records to {OUTPUT_CSV}")
    print(f"[synthesize] Target distributions:")
    print(f"  will_default_90d:        {df['will_default_90d'].value_counts(normalize=True).to_dict()}")
    print(f"  early_settlement_target: {df['early_settlement_target'].value_counts(normalize=True).to_dict()}")
    print(f"  lifecycle_outcome:       {df['lifecycle_outcome'].value_counts().to_dict()}")
    return df

if __name__ == '__main__':
    generate_dataset()
