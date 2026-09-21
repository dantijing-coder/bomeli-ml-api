"""
ml_engine/simulate_portfolio.py
Generates 1,000 to 50,000 realistic synthetic motorcycle financing accounts and
multi-month payment trajectories for training and evaluating Bomeli's ML models.

Includes:
- 10 comprehensive Philippine borrower archetypes (BPO, OFW, Sari-Sari, Tricycle, Gov/Teachers, etc.)
- Granular payment behaviors (Advance prepayments, Rebate streaks, Partial crumbs, Chronic overdue, Severe default)
- 12 trailing historical monthly ground-truth snapshots with seasonal calibration
- Full vehicle model catalog across major brands (Honda, Yamaha, Kawasaki, Suzuki)

SAFETY: All simulated data is stored in files under ml_engine/data/ (CSV and JSON)
and NEVER inserted into live sales or customer database tables.
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Import local config
from config import DATA_DIR, get_active_rate_package, get_db_connection

def fetch_reference_models_and_branches():
    """Fetches real vehicle models and branches from DB to ground synthetic data in real catalog."""
    branches = [
        {'branch_id': 1, 'name': 'LALA'},
        {'branch_id': 2, 'name': 'KAPATAGAN'},
        {'branch_id': 3, 'name': 'TUBOD'}
    ]
    models = [
        {'brand': 'HONDA', 'model_code': 'CLICK 125I', 'base_price': 81400.0, 'category': 'Scooter'},
        {'brand': 'HONDA', 'model_code': 'CLICK 160', 'base_price': 122900.0, 'category': 'Scooter'},
        {'brand': 'HONDA', 'model_code': 'PCX160', 'base_price': 134900.0, 'category': 'Maxi-Scooter'},
        {'brand': 'HONDA', 'model_code': 'ADV160', 'base_price': 166900.0, 'category': 'Adventure Scooter'},
        {'brand': 'HONDA', 'model_code': 'BEAT PREMIUM', 'base_price': 72400.0, 'category': 'Scooter'},
        {'brand': 'HONDA', 'model_code': 'TMX125 ALPHA', 'base_price': 57900.0, 'category': 'Backbone/Utility'},
        {'brand': 'HONDA', 'model_code': 'WAVE RSX', 'base_price': 64900.0, 'category': 'Underbone'},
        {'brand': 'KAWASAKI', 'model_code': 'P24KK (BARAKO II)', 'base_price': 68500.0, 'category': 'Backbone/Tricycle'},
        {'brand': 'KAWASAKI', 'model_code': 'CT100B', 'base_price': 54000.0, 'category': 'Backbone/Utility'},
        {'brand': 'KAWASAKI', 'model_code': 'ROUSER NS125', 'base_price': 82000.0, 'category': 'Sport Backbone'},
        {'brand': 'YAMAHA', 'model_code': 'MIO SPORTY', 'base_price': 73900.0, 'category': 'Scooter'},
        {'brand': 'YAMAHA', 'model_code': 'MIO GEAR 125', 'base_price': 79400.0, 'category': 'Scooter'},
        {'brand': 'YAMAHA', 'model_code': 'MIO FAZZIO', 'base_price': 93900.0, 'category': 'Retro Scooter'},
        {'brand': 'YAMAHA', 'model_code': 'NMAX 155', 'base_price': 151900.0, 'category': 'Maxi-Scooter'},
        {'brand': 'YAMAHA', 'model_code': 'AEROX 155', 'base_price': 125400.0, 'category': 'Sport Scooter'},
        {'brand': 'YAMAHA', 'model_code': 'SNIPER 155', 'base_price': 123900.0, 'category': 'Sport Underbone'},
        {'brand': 'SUZUKI', 'model_code': 'RAIDER 150 FI', 'base_price': 119900.0, 'category': 'Underbone'},
        {'brand': 'SUZUKI', 'model_code': 'SMASH 115', 'base_price': 62400.0, 'category': 'Underbone'},
        {'brand': 'SUZUKI', 'model_code': 'BURGMAN STREET 125', 'base_price': 83400.0, 'category': 'Maxi-Scooter'}
    ]

    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT branch_id, name FROM branches ORDER BY branch_id ASC")
            db_branches = cur.fetchall()
            if db_branches:
                branches = [{'branch_id': int(b['branch_id']), 'name': b['name']} for b in db_branches]

            cur.execute("SELECT model_id, brand, model_code FROM vehicle_models")
            db_models = cur.fetchall()
            if db_models:
                mapped = []
                for m in db_models:
                    code = (m['model_code'] or '').upper()
                    price = 85000.0
                    category = 'Commuter'
                    if '160' in code or 'NMAX' in code or 'PCX' in code or 'ADV' in code or '155' in code:
                        price = 135000.0
                        category = 'Maxi-Scooter'
                    elif '125' in code or 'MIO' in code or 'BURGMAN' in code:
                        price = 82000.0
                        category = 'Scooter'
                    elif '100' in code or 'P24' in code or 'CT' in code or 'BARAKO' in code or 'TMX' in code:
                        price = 62000.0
                        category = 'Backbone/Tricycle'
                    elif 'RAIDER' in code or 'SNIPER' in code:
                        price = 120000.0
                        category = 'Sport Underbone'
                    elif 'SMASH' in code or 'WAVE' in code or 'BEAT' in code:
                        price = 68000.0
                        category = 'Underbone'
                    mapped.append({'brand': m['brand'], 'model_code': m['model_code'], 'base_price': price, 'category': category})
                if mapped:
                    existing_codes = {m['model_code'].upper() for m in mapped}
                    for ref_m in models:
                        if ref_m['model_code'].upper() not in existing_codes:
                            mapped.append(ref_m)
                    models = mapped
        conn.close()
    except Exception as e:
        print(f"[simulate] Note: Using built-in reference models ({e}).")

    return branches, models

def generate_simulated_portfolio(n_samples=10000, seed=42):
    """
    Simulates n_samples loan accounts across Philippine motorcycle financing characteristics.
    Uses active rate package multipliers for financing calculations.
    """
    random.seed(seed)
    np.random.seed(seed)

    rate_pkg = get_active_rate_package()
    rates_by_years = rate_pkg['rates_by_years']
    print(f"[simulate] Using Rate Package '{rate_pkg['package_name']}': 1yr={rates_by_years[1]}, 2yr={rates_by_years[2]}, 3yr={rates_by_years[3]}")

    branches, models = fetch_reference_models_and_branches()
    branch_ids = [b['branch_id'] for b in branches]

    # Expanded 10 Borrower Livelihood Archetypes in Philippine Context
    employment_types = [
        ('Government / Public Teacher / Uniformed', 0.14, 0.035, (30000, 75000)),
        ('BPO / Tech / Remote Specialist', 0.12, 0.045, (26000, 80000)),
        ('Private Corporate / Bank / Office Staff', 0.16, 0.065, (22000, 58000)),
        ('OFW Remittance / Seafarer Family', 0.13, 0.030, (40000, 110000)),
        ('Micro-Business / Sari-Sari / Eatery Owner', 0.14, 0.110, (24000, 65000)),
        ('Tricycle / Habal-Habal / Courier Driver', 0.13, 0.165, (16000, 36000)),
        ('Skilled Tradesman / Welder / Carpenter', 0.06, 0.135, (18000, 42000)),
        ('Agriculture / Rice / Coconut Farmer', 0.05, 0.180, (14000, 38000)),
        ('Healthcare / Nurse / Medical Staff', 0.04, 0.040, (28000, 65000)),
        ('Retail Store / Security / Service Crew', 0.03, 0.125, (15000, 30000))
    ]
    emp_labels = [e[0] for e in employment_types]
    emp_weights = [e[1] for e in employment_types]
    emp_risk_map = {e[0]: e[2] for e in employment_types}
    emp_income_map = {e[0]: e[3] for e in employment_types}

    residence_types = [
        ('Owned (Clean Land Title)', 0.44, 0.04),
        ('Living with Parents / Family Compound', 0.32, 0.08),
        ('Rented House / Apartment', 0.18, 0.19),
        ('Informal Settler / Mortgaged / Tenant', 0.06, 0.24)
    ]
    res_labels = [r[0] for r in residence_types]
    res_weights = [r[1] for r in residence_types]
    res_risk_map = {r[0]: r[2] for r in residence_types}

    payment_preferences = ['Cash at Counter', 'GCash / Maya', 'Bank Transfer', 'Field Agent Collector']
    payment_pref_weights = [0.45, 0.35, 0.12, 0.08]

    first_names = [
        'Juan', 'Maria', 'Jose', 'Mark', 'Rodel', 'Ana', 'Grace', 'Bryan', 'Jomar', 'Lito',
        'Elena', 'Christian', 'Reynaldo', 'Gemma', 'Arnel', 'Dante', 'Aileen', 'Melvin', 'Rowena', 'Jerome',
        'Cesar', 'Maricel', 'Rommel', 'Bernadette', 'Danilo', 'Fatima', 'Eduardo', 'Joy', 'Michael', 'Shirley',
        'Kenneth', 'Marilou', 'Alexander', 'Angelica', 'Ramil', 'Cristina', 'Noel', 'Jennifer', 'Rolando', 'Karen'
    ]
    last_names = [
        'Dela Cruz', 'Santos', 'Reyes', 'Bautista', 'Villanueva', 'Fernandez', 'Alcantara', 'Mendoza',
        'Retiza', 'Navarro', 'Torres', 'Garcia', 'Ramos', 'Aquino', 'Castillo', 'Espino', 'Tolentino',
        'Macapagal', 'Valdez', 'Salazar', 'Mercado', 'De Leon', 'Magno', 'Tan', 'Lim', 'Abad', 'Perez'
    ]

    records = []

    # Simulation date reference (September 2026)
    ref_date = datetime(2026, 9, 20)

    for i in range(1, n_samples + 1):
        account_no = f"ACC-{2024000 + i}"
        customer_name = f"{random.choice(last_names)}, {random.choice(first_names)}"
        branch = random.choice(branches)
        model = random.choice(models)

        # Vehicle pricing & down payment
        srp = model['base_price'] + random.choice([-3000, -1500, 0, 1500, 3000])
        # Down payment: 10% to 28%
        down_pct = random.uniform(0.10, 0.28)
        down_payment = round(srp * down_pct, -2)
        principal_loan = srp - down_payment

        # Term selection (1, 2, or 3 years)
        term_years = random.choices([1, 2, 3], weights=[0.20, 0.55, 0.25])[0]
        term_months = term_years * 12
        factor_rate = rates_by_years.get(term_years, 1.48)

        total_payable = round(principal_loan * factor_rate, 2)
        monthly_amort = round(total_payable / term_months, 2)

        # Borrower Profile
        emp_type = random.choices(emp_labels, weights=emp_weights)[0]
        res_type = random.choices(res_labels, weights=res_weights)[0]
        res_years = round(random.lognormvariate(1.8, 0.7), 1)  # Median ~6 years

        # Monthly income tailored to livelihood
        inc_min, inc_max = emp_income_map[emp_type]
        monthly_income = round(random.uniform(inc_min, inc_max), -2)
        dti_ratio = round(monthly_amort / max(monthly_income, 1), 3)

        # Payment preference & demographics
        payment_pref = random.choices(payment_preferences, weights=payment_pref_weights)[0]
        marital_status = random.choices(['Married', 'Single', 'Head of Household'], weights=[0.60, 0.30, 0.10])[0]
        dependents = random.choices([0, 1, 2, 3, 4, 5], weights=[0.15, 0.25, 0.35, 0.15, 0.07, 0.03])[0]

        # Credit investigation flags
        ci_negative_flags = random.choices([0, 1, 2, 3], weights=[0.72, 0.18, 0.07, 0.03])[0]
        has_phone_bounce = 1 if (ci_negative_flags >= 2 and random.random() < 0.35) else 0

        # Loan elapsed months (how far into the term is the borrower?)
        current_term = random.randint(1, term_months)
        remaining_terms = term_months - current_term

        # Determine Borrower Behavioral Risk Segment
        # Latent risk probability formula calibrated to Philippine motorcycle financing portfolio
        emp_factor = emp_risk_map[emp_type] * 1.8       # 0.05 to 0.32
        res_factor = res_risk_map[res_type] * 1.2       # 0.05 to 0.29
        dti_factor = 0.25 if dti_ratio > 0.38 else (0.12 if dti_ratio > 0.28 else 0.02)
        ci_factor = ci_negative_flags * 0.14            # 0.0 to 0.42
        res_yr_factor = 0.12 if res_years < 2.0 else (-0.05 if res_years > 5.0 else 0.0)
        dep_factor = 0.08 if dependents >= 4 else 0.0
        bounce_factor = 0.18 if has_phone_bounce else 0.0

        raw_risk = (
            emp_factor + res_factor + dti_factor +
            ci_factor + res_yr_factor + dep_factor +
            bounce_factor + random.gauss(0, 0.04)
        )
        base_risk = max(0.01, min(0.99, raw_risk))

        # Behavioral Archetype Assignment & Detailed Trajectory Simulation
        if base_risk < 0.32:
            if ('OFW' in emp_type or 'Government' in emp_type or 'BPO' in emp_type) and random.random() < 0.40:
                archetype = 'ADVANCE_PREPAYER'
                advance_terms_paid = min(random.randint(1, 4), remaining_terms)
                rebate_streak = min(current_term + advance_terms_paid, random.randint(6, 24))
                late_penalty_streak = 0
                overdue_count = 0
                partial_payment_ratio = 1.0 + (advance_terms_paid / max(current_term, 1))
                on_time_reliability = 1.0
                current_status = 'active'
                will_default_90d = 0
                early_settlement_eligible = 1 if (current_term in [10, 11, 12, 22, 23, 24] and dti_ratio < 0.22) else 0
            else:
                archetype = 'PRIME_CONSISTENT'
                advance_terms_paid = 0
                rebate_streak = min(current_term, random.randint(4, 20))
                late_penalty_streak = 0
                overdue_count = 0
                partial_payment_ratio = 1.0
                on_time_reliability = round(random.uniform(0.94, 1.0), 3)
                current_status = 'active'
                will_default_90d = 0
                early_settlement_eligible = 1 if (current_term in [10, 11, 12, 22, 23, 24] and dti_ratio < 0.25) else 0

        elif base_risk < 0.60:
            if ('Farmer' in emp_type or 'Agriculture' in emp_type) and random.random() < 0.60:
                archetype = 'SEASONAL_HARVEST'
                advance_terms_paid = random.choice([0, 1])
                rebate_streak = random.randint(0, 2)
                late_penalty_streak = random.choice([0, 1])
                overdue_count = random.choice([0, 1])
                partial_payment_ratio = round(random.uniform(0.88, 1.10), 2)
                on_time_reliability = round(random.uniform(0.75, 0.90), 3)
                current_status = 'delinquent' if overdue_count > 0 else 'active'
                will_default_90d = 1 if (random.random() < 0.12) else 0
                early_settlement_eligible = 0
            else:
                archetype = 'PRIME_CONSISTENT'
                advance_terms_paid = 0
                rebate_streak = random.randint(1, 5)
                late_penalty_streak = random.choice([0, 1])
                overdue_count = 0
                partial_payment_ratio = round(random.uniform(0.95, 1.0), 2)
                on_time_reliability = round(random.uniform(0.82, 0.94), 3)
                current_status = 'active'
                will_default_90d = 1 if (random.random() < 0.08) else 0
                early_settlement_eligible = 0

        elif base_risk < 0.78:
            if ('Tricycle' in emp_type or 'Sari-Sari' in emp_type or 'Tradesman' in emp_type) and random.random() < 0.50:
                archetype = 'CRUMB_PARTIAL'
                advance_terms_paid = 0
                rebate_streak = 0
                late_penalty_streak = random.randint(1, 3)
                overdue_count = random.randint(1, 2)
                partial_payment_ratio = round(random.uniform(0.55, 0.85), 2)
                on_time_reliability = round(random.uniform(0.50, 0.72), 3)
                current_status = 'delinquent'
                will_default_90d = 1 if (random.random() < 0.45) else 0
                early_settlement_eligible = 0
            else:
                archetype = 'CHRONIC_OVERDUE'
                advance_terms_paid = 0
                rebate_streak = 0
                late_penalty_streak = random.randint(2, 4)
                overdue_count = random.randint(1, 2)
                partial_payment_ratio = round(random.uniform(0.60, 0.90), 2)
                on_time_reliability = round(random.uniform(0.40, 0.65), 3)
                current_status = 'delinquent'
                will_default_90d = 1 if (random.random() < 0.55) else 0
                early_settlement_eligible = 0

        else:
            archetype = 'SEVERE_DEFAULT_HAZARD'
            advance_terms_paid = 0
            rebate_streak = 0
            late_penalty_streak = random.randint(3, 8)
            overdue_count = random.randint(2, 5)
            partial_payment_ratio = round(random.uniform(0.0, 0.45), 2)
            on_time_reliability = round(random.uniform(0.10, 0.38), 3)
            current_status = 'defaulted' if overdue_count >= 3 else 'delinquent'
            will_default_90d = 1 if (random.random() < 0.92) else 0
            early_settlement_eligible = 0

        # Payments calculations
        total_paid_gross = round(current_term * monthly_amort * partial_payment_ratio, 2)
        total_penalties_paid = round(late_penalty_streak * 200.0, 2)
        total_paid_net = max(0.0, total_paid_gross - total_penalties_paid)
        principal_paid = min(principal_loan, round(total_paid_net / factor_rate, 2))
        remaining_principal = max(0.0, round(principal_loan - principal_paid, 2))
        total_arrears = round(overdue_count * (monthly_amort + 200.0), 2)

        # Early buyout valuation if settled now
        if current_term < 12:
            early_factor = rates_by_years.get(1, 1.26)
        elif current_term < 24:
            early_factor = rates_by_years.get(2, 1.48)
        else:
            early_factor = rates_by_years.get(3, 1.72)
        early_settlement_amount = round(remaining_principal * early_factor, 2)
        early_savings = max(0.0, round((remaining_principal * factor_rate) - early_settlement_amount, 2))

        # Propensity label
        if early_settlement_eligible and on_time_reliability >= 0.95 and dti_ratio < 0.20:
            buyout_propensity = 'HIGH'
        elif early_settlement_eligible and on_time_reliability >= 0.88:
            buyout_propensity = 'MEDIUM'
        else:
            buyout_propensity = 'LOW'

        record = {
            'account_no': account_no,
            'customer_name': customer_name,
            'branch_id': branch['branch_id'],
            'branch_name': branch['name'],
            'brand': model['brand'],
            'model_code': model['model_code'],
            'vehicle_category': model.get('category', 'Commuter'),
            'gross_principal_amount': srp,
            'down_payment': down_payment,
            'principal_loan_amount': principal_loan,
            'term_years': term_years,
            'term_months': term_months,
            'factor_rate': factor_rate,
            'total_payable': total_payable,
            'monthly_amortization': monthly_amort,
            'current_term': current_term,
            'remaining_terms': remaining_terms,
            'term_progress_ratio': round(current_term / term_months, 3),
            'monthly_income': monthly_income,
            'dti_ratio': dti_ratio,
            'employment_type': emp_type,
            'residential_ownership': res_type,
            'length_of_stay_years': res_years,
            'marital_status': marital_status,
            'dependents_count': dependents,
            'payment_method_preference': payment_pref,
            'behavioral_archetype': archetype,
            'ci_negative_flags': ci_negative_flags,
            'has_phone_bounce': has_phone_bounce,
            'rebate_streak': rebate_streak,
            'late_penalty_streak': late_penalty_streak,
            'overdue_count': overdue_count,
            'partial_payment_ratio': partial_payment_ratio,
            'on_time_reliability': on_time_reliability,
            'current_status': current_status,
            'total_paid_gross': total_paid_gross,
            'total_penalties_paid': total_penalties_paid,
            'total_paid_net': total_paid_net,
            'principal_paid': principal_paid,
            'remaining_principal': remaining_principal,
            'total_arrears': total_arrears,
            'early_settlement_amount': early_settlement_amount,
            'early_savings': early_savings,
            'buyout_propensity': buyout_propensity,
            # Target labels for ML models
            'will_default_90d': will_default_90d,
            'early_settlement_target': 1 if buyout_propensity in ['HIGH', 'MEDIUM'] else 0
        }
        records.append(record)

    df = pd.DataFrame(records)
    csv_path = os.path.join(DATA_DIR, 'simulated_portfolio.csv')
    df.to_csv(csv_path, index=False)
    print(f"[simulate] Saved {len(df):,} simulated loan records to: {csv_path}")

    # Generate Trailing Historical Monthly Snapshots (12 trailing months for full seasonal cycle)
    snapshots = generate_historical_snapshots(df, branches, ref_date)
    json_path = os.path.join(DATA_DIR, 'simulated_historical_snapshots.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(snapshots, f, indent=2)
    print(f"[simulate] Saved 12-month trailing historical ground-truth snapshots to: {json_path}")

    return df, snapshots

def generate_historical_snapshots(df, branches, ref_date):
    """
    Generates 12 trailing monthly ground-truth snapshots (e.g. 2025-10 to 2026-09)
    per branch and Consolidated Network. Incorporates Philippine macroeconomic seasons
    (13th-month bonuses in Dec, school enrollments in Jun, harvest cycles in Mar/Oct).
    Enables dual-lens auditing ('Historical Audit Mode') with verifiable reality checks.
    """
    snapshots = []

    # 12 trailing months leading to current (October 2025 to September 2026)
    months = []
    for i in range(12, -1, -1):
        m_date = ref_date - relativedelta(months=i)
        months.append((m_date.strftime('%Y-%m'), m_date.month))

    # All branches + Global scope (None)
    scopes = [{'branch_id': None, 'name': 'Consolidated Network (Global)'}]
    for b in branches:
        scopes.append({'branch_id': b['branch_id'], 'name': b['name']})

    for m_idx, (month_str, cal_month) in enumerate(months):
        # Base efficiency starts at ~90.0% and gradually improves with operations
        base_efficiency = 90.0 + (m_idx * 0.5)

        # Seasonal calibration adjustments
        seasonal_bump = 0.0
        if cal_month == 12:      # December: 13th-month bonus & OFW homecoming
            seasonal_bump = +4.5
        elif cal_month == 1:     # January: Post-holiday cashflow recovery
            seasonal_bump = -2.0
        elif cal_month in [3, 4]: # March/April: Harvest season liquidity
            seasonal_bump = +2.0
        elif cal_month == 6:     # June: School tuition & enrollment expenses
            seasonal_bump = -3.2
        elif cal_month == 9:     # September: Steady pre-ber months
            seasonal_bump = +1.0

        for scope in scopes:
            b_id = scope['branch_id']
            if b_id is None:
                sub_df = df
            else:
                sub_df = df[df['branch_id'] == b_id]

            active_count = len(sub_df[sub_df['current_status'] == 'active'])
            delinquent_count = len(sub_df[sub_df['current_status'] == 'delinquent'])
            defaulted_count = len(sub_df[sub_df['current_status'] == 'defaulted'])

            # Contractual scheduled due
            scheduled_due = round(float(sub_df['monthly_amortization'].sum() * random.uniform(0.95, 1.05)), 2)

            # Collection efficiency % with bounds
            eff_pct = round(min(99.2, max(82.0, base_efficiency + seasonal_bump + random.uniform(-1.2, 1.2))), 2)
            actual_collected = round(scheduled_due * (eff_pct / 100.0), 2)

            # High precision ML forecast (historical AI forecast)
            pred_error_pct = random.uniform(0.6, 2.4)
            pred_sign = random.choice([-1, 1])
            predicted_collected = round(actual_collected * (1.0 + (pred_sign * pred_error_pct / 100.0)), 2)
            realized_accuracy_pct = round(100.0 - abs((actual_collected - predicted_collected) / max(actual_collected, 1) * 100.0), 1)

            snapshots.append({
                'snapshot_month': month_str,
                'branch_id': b_id,
                'branch_name': scope['name'],
                'active_accounts_count': active_count,
                'delinquent_count': delinquent_count,
                'defaulted_count': defaulted_count,
                'total_scheduled_due': scheduled_due,
                'total_actual_collected': actual_collected,
                'collection_efficiency_pct': eff_pct,
                # Historical AI forecast that was generated on that date
                'historical_ai_predicted_collected': predicted_collected,
                'historical_accuracy_pct': realized_accuracy_pct
            })

    return snapshots

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simulate motorcycle financing portfolio for Bomeli ML Engine.")
    parser.add_argument('--samples', type=int, default=10000, help="Number of simulated loans (1000 to 50000).")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    n = max(1000, min(50000, args.samples))
    print(f"=== BOMELI ML ENGINE: SIMULATING {n:,} LOAN PORTFOLIO LIFECYCLES ===")
    df, snapshots = generate_simulated_portfolio(n_samples=n, seed=args.seed)
    print("=== SIMULATION COMPLETED SUCCESSFULLY ===")
