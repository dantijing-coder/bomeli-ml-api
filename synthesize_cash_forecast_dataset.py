"""
ml_engine/synthesize_cash_forecast_dataset.py

Generates an isolated, independent simulated dataset of 4,000 monthly
portfolio cash-flow snapshots for training the Portfolio Cash Forecasting
ML model (converting cash_engine.py from rule-based to ML).

SAFETY:
- Strictly saves to ml_engine/data/simulated_cash_forecast_dataset.csv.
- Completely isolated from rawdata/seed_data/ and does NOT touch the live database.
- Follows the exact schema and domain patterns of simulated_portfolio.csv
  as reference, but generates a SEPARATE file with different random seeds.
- Never reads or imports simulated_portfolio.csv or simulated_training_dataset.csv.

Features match what CashEngine.compute_cash_forecast_cone() already receives:
  - portfolio_size          (int)    : total active accounts this month
  - pct_active              (float)  : % accounts in 'active' status
  - pct_delinquent          (float)  : % accounts in 'delinquent' status
  - pct_defaulted           (float)  : % accounts in 'defaulted' or worse
  - avg_dti                 (float)  : portfolio average debt-to-income ratio
  - avg_on_time_reliability (float)  : portfolio average on-time reliability score
  - total_arrears           (float)  : total overdue arrears across portfolio
  - scheduled_due           (float)  : total contractual amount due this month
  - avg_term_progress       (float)  : average term completion ratio across portfolio
  - season_month            (int)    : calendar month (1-12) -- seasonality driver
  - projected_units_sold    (int)    : new showroom bookings this month
  - trailing_cure_rate      (float)  : % delinquent accounts that cured last month
  - new_accounts_ratio      (float)  : % accounts that are < 4 terms old

Targets:
  - collection_realization_rate  (float) : actual_collected / scheduled_due (0.55 - 1.10)
  - stress_factor                (float) : pessimistic collection factor (0.70 - 0.95)
"""

import os
import random
import numpy as np
import pandas as pd

# Deterministic seed -- DIFFERENT from synthesize_training_dataset.py (which uses 42)
random.seed(137)
np.random.seed(137)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(DATA_DIR, 'simulated_cash_forecast_dataset.csv')

NUM_RECORDS = 4000

# Philippine motorcycle dealership monthly seasonality indices
SEASONALITY = {
    1: 0.94, 2: 0.97, 3: 1.03, 4: 1.05, 5: 1.02, 6: 0.96,
    7: 0.98, 8: 0.97, 9: 1.01, 10: 1.04, 11: 1.14, 12: 1.18
}

BRANCHES = [
    {'branch_id': 1, 'name': 'LALA',      'size_factor': 1.15},
    {'branch_id': 2, 'name': 'KAPATAGAN', 'size_factor': 1.05},
    {'branch_id': 3, 'name': 'TUBOD',     'size_factor': 0.85},
]

# Portfolio archetypes: (name, weight, pct_active_mu, pct_del_mu, pct_def_mu,
#                        dti_mu, otr_mu, cure_rate_mu)
PORTFOLIO_ARCHETYPES = [
    ('prime_healthy',    0.30, 0.85, 0.10, 0.03, 0.18, 0.92, 0.72),
    ('moderate_stress',  0.35, 0.72, 0.18, 0.07, 0.25, 0.80, 0.58),
    ('elevated_risk',    0.20, 0.60, 0.26, 0.11, 0.32, 0.68, 0.42),
    ('distressed',       0.15, 0.48, 0.32, 0.18, 0.40, 0.54, 0.28),
]


def generate_dataset():
    print(f"[cash-synth] Generating {NUM_RECORDS:,} cash forecast snapshots...")
    records = []

    archetype_weights = [a[1] for a in PORTFOLIO_ARCHETYPES]

    for i in range(1, NUM_RECORDS + 1):
        branch = random.choice(BRANCHES)
        season_month = random.randint(1, 12)
        sea_idx = SEASONALITY[season_month]

        archetype = random.choices(PORTFOLIO_ARCHETYPES, weights=archetype_weights, k=1)[0]
        (arch_name, _, pct_act_mu, pct_del_mu, pct_def_mu,
         dti_mu, otr_mu, cure_mu) = archetype

        portfolio_size = max(20, int(np.random.normal(
            loc=350 * branch['size_factor'],
            scale=80 * branch['size_factor']
        )))

        pct_active     = float(np.clip(np.random.normal(pct_act_mu, 0.06), 0.35, 0.97))
        pct_delinquent = float(np.clip(np.random.normal(pct_del_mu, 0.04), 0.01, 0.45))
        pct_defaulted  = float(np.clip(np.random.normal(pct_def_mu, 0.03), 0.00, 0.30))
        total_pct = pct_active + pct_delinquent + pct_defaulted
        if total_pct > 1.0:
            scale = 1.0 / total_pct
            pct_active     *= scale
            pct_delinquent *= scale
            pct_defaulted  *= scale

        avg_dti              = float(np.clip(np.random.normal(dti_mu, 0.05), 0.05, 0.65))
        avg_on_time_rel      = float(np.clip(np.random.normal(otr_mu, 0.07), 0.20, 1.00))
        trailing_cure_rate   = float(np.clip(np.random.normal(cure_mu, 0.08), 0.10, 0.90))
        avg_term_progress    = float(np.clip(np.random.uniform(0.10, 0.92), 0.05, 0.98))
        new_accounts_ratio   = float(np.clip(np.random.beta(2.5, 8.0), 0.02, 0.45))

        avg_monthly_amort = float(np.random.uniform(2500.0, 8500.0))
        scheduled_due     = round(portfolio_size * avg_monthly_amort, 2)
        total_arrears     = round(
            portfolio_size * pct_delinquent * avg_monthly_amort
            * float(np.random.uniform(1.0, 3.2)), 2
        )

        projected_units = max(0, int(np.random.normal(
            loc=8 * branch['size_factor'] * sea_idx,
            scale=3
        )))

        # Target 1: Collection Realization Rate
        base_rate = (
            avg_on_time_rel * 0.55
            + (1.0 - pct_delinquent) * 0.25
            + (1.0 - pct_defaulted * 2.5) * 0.15
            + trailing_cure_rate * 0.05
        )
        base_rate *= (0.85 + (sea_idx - 0.90) * 0.60)
        base_rate -= new_accounts_ratio * 0.06
        base_rate -= max(0.0, avg_dti - 0.25) * 0.18
        base_rate += float(np.random.normal(0.0, 0.025))
        collection_realization_rate = float(np.clip(base_rate, 0.55, 1.10))

        # Target 2: Stress Factor (pessimistic multiplier)
        base_stress = (
            0.91
            - (pct_defaulted * 1.2)
            - (pct_delinquent * 0.35)
            + (trailing_cure_rate * 0.06)
            - max(0.0, avg_dti - 0.30) * 0.10
        )
        base_stress += float(np.random.normal(0.0, 0.02))
        stress_factor = float(np.clip(base_stress, 0.70, 0.95))

        records.append({
            'snapshot_id':               f"CASH-{i:05d}",
            'branch_id':                 branch['branch_id'],
            'branch_name':               branch['name'],
            'portfolio_archetype':       arch_name,
            'season_month':              season_month,
            'portfolio_size':            portfolio_size,
            'pct_active':                round(pct_active, 4),
            'pct_delinquent':            round(pct_delinquent, 4),
            'pct_defaulted':             round(pct_defaulted, 4),
            'avg_dti':                   round(avg_dti, 4),
            'avg_on_time_reliability':   round(avg_on_time_rel, 4),
            'avg_term_progress':         round(avg_term_progress, 4),
            'trailing_cure_rate':        round(trailing_cure_rate, 4),
            'new_accounts_ratio':        round(new_accounts_ratio, 4),
            'scheduled_due':             scheduled_due,
            'total_arrears':             total_arrears,
            'projected_units_sold':      projected_units,
            'collection_realization_rate': round(collection_realization_rate, 4),
            'stress_factor':             round(stress_factor, 4),
        })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[cash-synth] Saved {len(df):,} records to {OUTPUT_CSV}")
    print(f"[cash-synth] collection_realization_rate:"
          f" mean={df['collection_realization_rate'].mean():.3f}"
          f"  std={df['collection_realization_rate'].std():.3f}"
          f"  range=[{df['collection_realization_rate'].min():.3f}"
          f", {df['collection_realization_rate'].max():.3f}]")
    print(f"[cash-synth] stress_factor:              "
          f" mean={df['stress_factor'].mean():.3f}"
          f"  std={df['stress_factor'].std():.3f}"
          f"  range=[{df['stress_factor'].min():.3f}"
          f", {df['stress_factor'].max():.3f}]")
    return df


if __name__ == '__main__':
    generate_dataset()
