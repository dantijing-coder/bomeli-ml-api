"""
ml_engine/synthesize_markov_dataset.py

Generates an isolated 5,000-row synthetic dataset for training the
ML-powered Delinquency State Transition Model (Model 9).

DESIGN PHILOSOPHY — Archetype-First Generation:
  Each record is generated from a RISK ARCHETYPE (e.g., high-risk active,
  low-risk active, severe-delinquent, mild-delinquent) so that the feature
  distributions are class-conditionally separated. This gives the classifier
  clean separating hyperplanes to learn from.

  Labels are derived DETERMINISTICALLY from archetypes, ensuring strong
  correlation between features and labels — without information leakage.

Three transition problems:
  (a) Active       -> Delinquent?   binary label: label_active_to_delinquent
  (b) Delinquent   -> Default?      binary label: label_delinquent_to_default
  (c) Delinquent   -> Cured?        binary label: label_delinquent_to_cured

SAFETY:
- Seed = 389. Completely isolated from other synthesizers (42, 137, 251).
- Never reads simulated_portfolio.csv or simulated_training_dataset.csv.
- Saves to ml_engine/data/simulated_markov_transitions_dataset.csv.
"""

import os
import random
import numpy as np
import pandas as pd

random.seed(389)
np.random.seed(389)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(DATA_DIR, 'simulated_markov_transitions_dataset.csv')

NUM_RECORDS = 5000

EMPLOYMENT_PROFILES = [
    {'type': 'Employed (Private/Gov)',     'base_def_risk': 0.08, 'cure_prop': 0.65},
    {'type': 'Public School Teacher',      'base_def_risk': 0.04, 'cure_prop': 0.78},
    {'type': 'OFW Remittance / Dependent', 'base_def_risk': 0.06, 'cure_prop': 0.72},
    {'type': 'BPO / Call Center Agent',    'base_def_risk': 0.10, 'cure_prop': 0.60},
    {'type': 'Sari-Sari / Small Retail',   'base_def_risk': 0.16, 'cure_prop': 0.50},
    {'type': 'Tricycle / PUV Operator',    'base_def_risk': 0.22, 'cure_prop': 0.42},
    {'type': 'Agri / Farmer / Fisherfolk', 'base_def_risk': 0.25, 'cure_prop': 0.38},
]

RESIDENTIAL = [
    {'ownership': 'Owned (Clean Title)',          'stability': 0.92},
    {'ownership': 'Owned (Inherited/Tax Dec)',     'stability': 0.85},
    {'ownership': 'Living with Parents/Relatives', 'stability': 0.75},
    {'ownership': 'Rented (Apartment/House)',      'stability': 0.62},
]

SEASON_CURE_BOOST = {
    1: -0.04, 2: -0.02, 3: 0.02, 4: 0.03, 5: 0.01,  6: -0.03,
    7: -0.01, 8: -0.01, 9: 0.01, 10: 0.02, 11: 0.06, 12: 0.08
}


# ──────────────────────────────────────────────────────────────────────────────
# ARCHETYPE-FIRST GENERATION
# ──────────────────────────────────────────────────────────────────────────────

ACTIVE_ARCHETYPES = {
    # Archetype name: (label, feature_distribution)
    'prime_stable': {
        'label': 0,   # will NOT slip to delinquent
        'on_time_rel':   (0.88, 1.00),
        'overdue_count': [0],
        'rebate_streak': (3, 10),
        'partial_ratio': (0.00, 0.03),
        'dti_adj':       -0.08,
        'ci_flag_prob':  0.02,
    },
    'low_risk_stable': {
        'label': 0,
        'on_time_rel':   (0.78, 0.92),
        'overdue_count': [0],
        'rebate_streak': (1, 5),
        'partial_ratio': (0.00, 0.06),
        'dti_adj':       -0.03,
        'ci_flag_prob':  0.05,
    },
    'moderate_risk': {
        'label': 0,   # mostly safe — small chance still assigned 0
        'on_time_rel':   (0.65, 0.82),
        'overdue_count': [0, 1],
        'rebate_streak': (0, 3),
        'partial_ratio': (0.02, 0.10),
        'dti_adj':       0.05,
        'ci_flag_prob':  0.12,
    },
    'at_risk': {
        'label': 1,   # will slip to delinquent
        'on_time_rel':   (0.45, 0.70),
        'overdue_count': [1],
        'rebate_streak': (0, 2),
        'partial_ratio': (0.05, 0.18),
        'dti_adj':       0.10,
        'ci_flag_prob':  0.20,
    },
    'high_risk_slipper': {
        'label': 1,
        'on_time_rel':   (0.30, 0.58),
        'overdue_count': [1, 1, 1],   # always overdue
        'rebate_streak': (0, 1),
        'partial_ratio': (0.08, 0.30),
        'dti_adj':       0.18,
        'ci_flag_prob':  0.35,
    },
}

# Archetype mix: (archetype_name, weight)
ACTIVE_MIX = [
    ('prime_stable',       0.28),
    ('low_risk_stable',    0.30),
    ('moderate_risk',      0.17),
    ('at_risk',            0.14),
    ('high_risk_slipper',  0.11),
]


DELINQUENT_ARCHETYPES = {
    'mild_curable': {
        'label_default': 0, 'label_cure': 1,
        'overdue_count': [1],
        'on_time_rel':   (0.55, 0.75),
        'rebate_streak': (1, 3),
        'partial_ratio': (0.00, 0.12),
    },
    'moderate_uncertain': {
        'label_default': 0, 'label_cure': 0,
        'overdue_count': [2],
        'on_time_rel':   (0.38, 0.60),
        'rebate_streak': (0, 2),
        'partial_ratio': (0.05, 0.22),
    },
    'severe_rolling': {
        'label_default': 1, 'label_cure': 0,
        'overdue_count': [2, 3],
        'on_time_rel':   (0.20, 0.48),
        'rebate_streak': [0],
        'partial_ratio': (0.15, 0.40),
    },
    'critical_default': {
        'label_default': 1, 'label_cure': 0,
        'overdue_count': [3, 4],
        'on_time_rel':   (0.12, 0.38),
        'rebate_streak': [0],
        'partial_ratio': (0.28, 0.60),
    },
}

DELINQUENT_MIX = [
    ('mild_curable',       0.28),
    ('moderate_uncertain', 0.30),
    ('severe_rolling',     0.26),
    ('critical_default',   0.16),
]


def pick_archetype(mix):
    names, weights = zip(*mix)
    return random.choices(names, weights=weights, k=1)[0]


def sample_range(lo_hi):
    lo, hi = lo_hi
    return round(float(np.random.uniform(lo, hi)), 3)


def make_shared_features(emp, res, month):
    monthly_income = max(14000.0, float(np.random.normal(
        loc=28000 if emp['base_def_risk'] < 0.12 else 20000,
        scale=6000
    )))
    term_months = random.choices([12, 24, 36], weights=[0.30, 0.35, 0.35])[0]
    principal   = float(np.random.uniform(50000, 160000))
    factor      = {12: 1.26, 24: 1.48, 36: 1.72}[term_months]
    amort       = round(principal * factor / term_months, 2)
    base_dti    = round(amort / max(monthly_income, 1000), 4)
    length_stay = min(30.0, max(1.0, float(np.random.gamma(3.0, 2.5))))
    dependents  = random.choices([0, 1, 2, 3, 4], weights=[0.18, 0.30, 0.30, 0.15, 0.07])[0]
    term_prog   = round(float(np.random.uniform(0.05, 0.95)), 3)
    return {
        'monthly_income': round(monthly_income, 2),
        'monthly_amortization': amort,
        '_base_dti': base_dti,
        'term_months': term_months,
        'term_progress_ratio': term_prog,
        'length_of_stay_years': round(length_stay, 1),
        'dependents_count': dependents,
        'employment_type': emp['type'],
        'residential_ownership': res['ownership'],
        'season_month': month,
    }


def make_active_record(emp, res, month, i):
    arch_name = pick_archetype(ACTIVE_MIX)
    arch      = ACTIVE_ARCHETYPES[arch_name]
    shared    = make_shared_features(emp, res, month)
    dti       = round(min(0.80, max(0.05, shared['_base_dti'] + arch['dti_adj'])), 4)

    otr_lo, otr_hi = arch['on_time_rel']
    overdue = random.choice(arch['overdue_count'])
    reb_lo, reb_hi = arch['rebate_streak']
    rebate  = random.randint(reb_lo, reb_hi)
    pr_lo, pr_hi = arch['partial_ratio']

    amort = shared['monthly_amortization']
    feat = {
        'transition_id':         f"MRK-{i:05d}",
        'current_state':         'active',
        'term_months':           shared['term_months'],
        'term_progress_ratio':   shared['term_progress_ratio'],
        'dti_ratio':             dti,
        'on_time_reliability':   round(float(np.random.uniform(otr_lo, otr_hi)), 3),
        'overdue_count':         overdue,
        'rebate_streak':         rebate,
        'partial_payment_ratio': round(float(np.random.uniform(pr_lo, pr_hi)), 3),
        'total_arrears':         0.0 if overdue == 0 else round(amort * overdue * 0.85, 2),
        'ci_negative_flags':     1 if random.random() < arch['ci_flag_prob'] else 0,
        'has_phone_bounce':      1 if random.random() < (arch['ci_flag_prob'] * 0.4) else 0,
        'length_of_stay_years':  shared['length_of_stay_years'],
        'dependents_count':      shared['dependents_count'],
        'employment_type':       shared['employment_type'],
        'residential_ownership': shared['residential_ownership'],
        'season_month':          shared['season_month'],
        'label_active_to_delinquent': arch['label'],
        'label_delinquent_to_default': 0,
        'label_delinquent_to_cured':   0,
    }
    # Small noise: 5% mislabel probability to avoid overfit
    if random.random() < 0.05:
        feat['label_active_to_delinquent'] = 1 - feat['label_active_to_delinquent']
    return feat


def make_delinquent_record(emp, res, month, i):
    arch_name = pick_archetype(DELINQUENT_MIX)
    arch      = DELINQUENT_ARCHETYPES[arch_name]
    shared    = make_shared_features(emp, res, month)
    dti       = round(min(0.80, max(0.05, shared['_base_dti'] + 0.08)), 4)

    overdue = random.choice(arch['overdue_count'])
    otr_lo, otr_hi = arch['on_time_rel']
    if isinstance(arch['rebate_streak'], list):
        rebate = random.choice(arch['rebate_streak'])
    else:
        reb_lo, reb_hi = arch['rebate_streak']
        rebate = random.randint(reb_lo, reb_hi)
    pr_lo, pr_hi = arch['partial_ratio']

    amort = shared['monthly_amortization']
    feat = {
        'transition_id':         f"MRK-{i:05d}",
        'current_state':         'delinquent',
        'term_months':           shared['term_months'],
        'term_progress_ratio':   shared['term_progress_ratio'],
        'dti_ratio':             dti,
        'on_time_reliability':   round(float(np.random.uniform(otr_lo, otr_hi)), 3),
        'overdue_count':         overdue,
        'rebate_streak':         rebate,
        'partial_payment_ratio': round(float(np.random.uniform(pr_lo, pr_hi)), 3),
        'total_arrears':         round(amort * overdue * float(np.random.uniform(0.90, 1.30)), 2),
        'ci_negative_flags':     1 if random.random() < (0.10 + overdue * 0.08) else 0,
        'has_phone_bounce':      1 if random.random() < (0.05 + overdue * 0.06) else 0,
        'length_of_stay_years':  shared['length_of_stay_years'],
        'dependents_count':      shared['dependents_count'],
        'employment_type':       shared['employment_type'],
        'residential_ownership': shared['residential_ownership'],
        'season_month':          shared['season_month'],
        'label_active_to_delinquent': 0,
        'label_delinquent_to_default': arch['label_default'],
        'label_delinquent_to_cured':   arch['label_cure'],
    }
    # 5% mislabel noise
    if random.random() < 0.05:
        feat['label_delinquent_to_default'] = 1 - feat['label_delinquent_to_default']
    if random.random() < 0.05:
        feat['label_delinquent_to_cured'] = 1 - feat['label_delinquent_to_cured']
    return feat


def generate_dataset():
    print(f"[markov-synth] Generating {NUM_RECORDS:,} state-transition records (archetype-first)...")
    records = []
    active_n   = int(NUM_RECORDS * 0.76)
    delinq_n   = NUM_RECORDS - active_n

    for i in range(1, active_n + 1):
        emp   = random.choice(EMPLOYMENT_PROFILES)
        res   = random.choice(RESIDENTIAL)
        month = random.randint(1, 12)
        records.append(make_active_record(emp, res, month, i))

    for i in range(active_n + 1, NUM_RECORDS + 1):
        emp   = random.choice(EMPLOYMENT_PROFILES)
        res   = random.choice(RESIDENTIAL)
        month = random.randint(1, 12)
        records.append(make_delinquent_record(emp, res, month, i))

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)

    active_df = df[df['current_state'] == 'active']
    delinq_df = df[df['current_state'] == 'delinquent']

    print(f"[markov-synth] Saved {len(df):,} records to {OUTPUT_CSV}")
    print(f"[markov-synth] State distribution: active={len(active_df):,}  delinquent={len(delinq_df):,}")
    print(f"[markov-synth] Label A (Active->Delinquent):   {active_df['label_active_to_delinquent'].mean():.1%} will slip")
    print(f"[markov-synth] Label B (Delinquent->Default):  {delinq_df['label_delinquent_to_default'].mean():.1%} will default")
    print(f"[markov-synth] Label C (Delinquent->Cured):    {delinq_df['label_delinquent_to_cured'].mean():.1%} will cure")
    return df


if __name__ == '__main__':
    generate_dataset()
