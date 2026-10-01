"""
ml_engine/synthesize_survival_dataset.py

Generates an isolated, independent simulated dataset of 4,000 loan-lifecycle
records for training the Loan Survival Analysis ML model (converting
survival_engine.py from a static constant table to a real trained classifier).

SAFETY:
- Strictly saves to ml_engine/data/simulated_survival_dataset.csv.
- Completely isolated -- never reads simulated_portfolio.csv, simulated_training_dataset.csv,
  rawdata/, seed_data/, or the live database.
- Schema is directly inspired by simulated_portfolio.csv column patterns (current_status,
  term_months, on_time_reliability, dti_ratio, overdue_count, rebate_streak, etc.)
  but all records are freshly synthesized from scratch with seed=251.

Each row represents a COMPLETED or MONITORED loan account and its final lifecycle outcome.

Features:
  - term_months              (int)   : contract length (12, 24, or 36)
  - term_progress_ratio      (float) : how far through the contract (0.05 - 1.0)
  - current_term             (int)   : current installment term number
  - dti_ratio                (float) : monthly amortization / monthly income
  - on_time_reliability      (float) : fraction of payments made on time
  - overdue_count            (int)   : number of currently overdue terms
  - rebate_streak            (int)   : consecutive months claimed on-time rebate
  - partial_payment_ratio    (float) : fraction of payments that were partial/crumb
  - ci_negative_flags        (int)   : 0 or 1
  - has_phone_bounce         (int)   : 0 or 1
  - employment_type          (str)   : categorical employment profile
  - residential_ownership    (str)   : categorical residency status
  - monthly_income           (float) : borrower monthly income (PHP)
  - dependents_count         (int)   : number of financial dependents
  - marital_status           (str)   : Single / Married / Widowed

Targets:
  - lifecycle_outcome        (str)   : 'completed' | 'early_settled' | 'defaulted'
  - time_to_event_terms      (int)   : number of terms until event (for survival curve)
  - hazard_phase             (str)   : 'onboarding_risk' | 'core_stability' | 'option_window' | 'completion_phase'
  - completion_rate          (float) : per-archetype completion probability (for Kaplan-Meier)
"""

import os
import random
import numpy as np
import pandas as pd

# Deterministic seed -- DIFFERENT from all other synthesizers (42 and 137)
random.seed(251)
np.random.seed(251)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(DATA_DIR, 'simulated_survival_dataset.csv')

NUM_RECORDS = 4000

FACTOR_RATES = {12: 1.26, 24: 1.48, 36: 1.72}

# Empirical employment profiles (matching simulated_portfolio.csv domain)
EMPLOYMENT_PROFILES = [
    {'type': 'Employed (Private/Gov)',   'inc_mean': 32000, 'inc_std': 8000,  'def_prop': 0.08, 'early_prop': 0.18},
    {'type': 'Public School Teacher',    'inc_mean': 38000, 'inc_std': 7000,  'def_prop': 0.04, 'early_prop': 0.22},
    {'type': 'OFW Remittance / Dependent','inc_mean': 52000, 'inc_std': 14000, 'def_prop': 0.06, 'early_prop': 0.28},
    {'type': 'BPO / Call Center Agent',  'inc_mean': 30000, 'inc_std': 6000,  'def_prop': 0.10, 'early_prop': 0.14},
    {'type': 'Sari-Sari / Small Retail', 'inc_mean': 24000, 'inc_std': 5000,  'def_prop': 0.15, 'early_prop': 0.10},
    {'type': 'Tricycle / PUV Operator',  'inc_mean': 20000, 'inc_std': 4000,  'def_prop': 0.20, 'early_prop': 0.06},
    {'type': 'Agri / Farmer / Fisherfolk','inc_mean': 18000, 'inc_std': 4500, 'def_prop': 0.22, 'early_prop': 0.05},
]

RESIDENTIAL_PROFILES = [
    'Owned (Clean Title)', 'Owned (Inherited/Tax Dec)',
    'Living with Parents/Relatives', 'Rented (Apartment/House)'
]

MARITAL_STATUS = ['Single', 'Married', 'Married', 'Widowed']

# Term-level empirical hazard windows (matching survival_engine.py domain constants)
TERM_HAZARD_CONFIG = {
    12: {'completion_base': 0.854, 'early_base': 0.058, 'peak_hazard': (2, 4),  'option_window': (10, 11)},
    24: {'completion_base': 0.816, 'early_base': 0.072, 'peak_hazard': (2, 5),  'option_window': (11, 13)},
    36: {'completion_base': 0.778, 'early_base': 0.085, 'peak_hazard': (2, 6),  'option_window': (12, 25)},
}


def classify_hazard_phase(current_term: int, term_months: int) -> str:
    """Assigns the lifecycle risk phase based on current term position."""
    if current_term <= 4:
        return 'onboarding_risk'
    elif current_term >= (term_months - 3):
        return 'completion_phase'
    config = TERM_HAZARD_CONFIG.get(term_months, TERM_HAZARD_CONFIG[24])
    ow_start, ow_end = config['option_window']
    if ow_start <= current_term <= ow_end:
        return 'option_window'
    return 'core_stability'


def generate_dataset():
    print(f"[survival-synth] Generating {NUM_RECORDS:,} loan lifecycle records...")
    records = []

    for i in range(1, NUM_RECORDS + 1):
        emp = random.choice(EMPLOYMENT_PROFILES)
        res = random.choice(RESIDENTIAL_PROFILES)
        civil = random.choice(MARITAL_STATUS)
        dependents = random.choices([0, 1, 2, 3, 4], weights=[0.20, 0.30, 0.28, 0.15, 0.07])[0]

        monthly_income = max(12000.0, round(float(np.random.normal(emp['inc_mean'], emp['inc_std'])), 2))
        stay_years = min(35.0, max(1.0, round(float(np.random.gamma(shape=3.0, scale=2.5)), 1)))

        term_months = random.choices([12, 24, 36], weights=[0.30, 0.35, 0.35])[0]
        config = TERM_HAZARD_CONFIG[term_months]

        # Elapsed term (current position in the contract)
        elapsed_terms = random.randint(1, term_months)
        term_progress_ratio = round(elapsed_terms / term_months, 4)
        hazard_phase = classify_hazard_phase(elapsed_terms, term_months)

        # Loan financials
        base_price = float(np.random.uniform(55000.0, 170000.0))
        dp_ratio = random.choice([0.10, 0.12, 0.15, 0.20, 0.25, 0.30])
        principal = round(base_price * (1.0 - dp_ratio), 2)
        pn_value = round(principal * FACTOR_RATES[term_months], 2)
        monthly_amort = round(pn_value / term_months + 200.0, 2)
        dti_ratio = round(monthly_amort / monthly_income, 4)

        # Risk score drives behavioral simulation
        risk_score = (
            (dti_ratio * 1.4)
            + (0.12 if 'Rented' in res else (0.05 if 'Parents' in res else 0.0))
            + (emp['def_prop'] * 1.4)
            - (0.10 if stay_years >= 5.0 else 0.0)
            + (0.05 if dependents >= 3 else 0.0)
        )
        risk_score = max(0.02, min(0.95, risk_score))

        # Behavioral metrics based on risk segment
        if risk_score > 0.46:
            on_time_rel   = round(float(np.random.uniform(0.20, 0.65)), 3)
            rebate_streak = random.choice([0, 0, 1])
            overdue_count = random.choice([1, 2, 3, 4])
            partial_ratio = round(float(np.random.uniform(0.15, 0.45)), 3)
            ci_flags      = 1 if random.random() < 0.35 else 0
            phone_bounce  = 1 if random.random() < 0.25 else 0
        elif risk_score > 0.26:
            on_time_rel   = round(float(np.random.uniform(0.60, 0.85)), 3)
            rebate_streak = random.choice([0, 1, 2, 3])
            overdue_count = random.choice([0, 1])
            partial_ratio = round(float(np.random.uniform(0.0, 0.15)), 3)
            ci_flags      = 1 if random.random() < 0.10 else 0
            phone_bounce  = 0
        else:
            on_time_rel   = round(float(np.random.uniform(0.85, 1.00)), 3)
            rebate_streak = random.randint(1, max(1, min(12, elapsed_terms)))
            overdue_count = 0
            partial_ratio = 0.0
            ci_flags      = 0
            phone_bounce  = 0

        # Deterministic outcome assignment — hard rules for strong separating signal
        if overdue_count >= 3 or (partial_ratio > 0.25 and on_time_rel < 0.45):
            # Strongly defaulted
            outcome = random.choices(['completed', 'early_settled', 'defaulted'],
                                     weights=[0.03, 0.01, 0.96], k=1)[0]
        elif overdue_count == 2 or (partial_ratio > 0.15 and on_time_rel < 0.60):
            # High risk
            outcome = random.choices(['completed', 'early_settled', 'defaulted'],
                                     weights=[0.12, 0.02, 0.86], k=1)[0]
        elif overdue_count == 1 or (partial_ratio > 0.05 and on_time_rel < 0.75):
            # Elevated risk
            outcome = random.choices(['completed', 'early_settled', 'defaulted'],
                                     weights=[0.55, 0.08, 0.37], k=1)[0]
        elif rebate_streak >= 5 and overdue_count == 0 and on_time_rel >= 0.88:
            # Prime early-settler
            outcome = random.choices(['completed', 'early_settled', 'defaulted'],
                                     weights=[0.55, 0.41, 0.04], k=1)[0]
        elif overdue_count == 0 and on_time_rel >= 0.80 and ci_flags == 0:
            # Prime completer
            outcome = random.choices(['completed', 'early_settled', 'defaulted'],
                                     weights=[0.88, 0.08, 0.04], k=1)[0]
        else:
            # General medium band
            p_complete = config['completion_base'] - risk_score * 0.35
            p_early    = config['early_base'] + (1.0 - risk_score) * 0.06
            p_default  = max(0.05, 1.0 - p_complete - p_early)
            total_p    = p_complete + p_early + p_default
            outcome = random.choices(
                ['completed', 'early_settled', 'defaulted'],
                weights=[p_complete / total_p, p_early / total_p, p_default / total_p],
                k=1
            )[0]


        # Time-to-event: how many terms remain until the event
        if outcome == 'defaulted':
            # Defaults typically happen in the peak hazard window
            ph_start, ph_end = config['peak_hazard']
            event_term = random.randint(
                max(elapsed_terms, ph_start),
                min(term_months, max(elapsed_terms + 1, ph_end + 3))
            )
        elif outcome == 'early_settled':
            ow_start, ow_end = config['option_window']
            lo = max(elapsed_terms, ow_start)
            hi = max(lo, min(term_months, ow_end + 2))
            event_term = random.randint(lo, hi)
        else:
            event_term = term_months

        time_to_event_terms = max(0, event_term - elapsed_terms)

        # Per-archetype completion rate for Kaplan-Meier target
        base_completion = config['completion_base']
        completion_rate = float(np.clip(
            base_completion - risk_score * 0.28 + float(np.random.normal(0, 0.03)),
            0.40, 0.97
        ))

        records.append({
            'survival_id':           f"SURV-{i:05d}",
            'term_months':           term_months,
            'current_term':          elapsed_terms,
            'term_progress_ratio':   term_progress_ratio,
            'dti_ratio':             dti_ratio,
            'monthly_income':        monthly_income,
            'monthly_amortization':  monthly_amort,
            'on_time_reliability':   on_time_rel,
            'overdue_count':         overdue_count,
            'rebate_streak':         rebate_streak,
            'partial_payment_ratio': partial_ratio,
            'ci_negative_flags':     ci_flags,
            'has_phone_bounce':      phone_bounce,
            'employment_type':       emp['type'],
            'residential_ownership': res,
            'marital_status':        civil,
            'dependents_count':      dependents,
            'length_of_stay_years':  stay_years,
            'hazard_phase':          hazard_phase,
            # Targets
            'lifecycle_outcome':     outcome,
            'time_to_event_terms':   time_to_event_terms,
            'completion_rate':       round(completion_rate, 4),
        })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[survival-synth] Saved {len(df):,} records to {OUTPUT_CSV}")
    print(f"[survival-synth] lifecycle_outcome distribution:")
    print(df['lifecycle_outcome'].value_counts(normalize=True).to_string())
    print(f"[survival-synth] hazard_phase distribution:")
    print(df['hazard_phase'].value_counts().to_string())
    return df


if __name__ == '__main__':
    generate_dataset()
