#!/usr/bin/env python3
"""
ml_engine/train_velocity_model.py
Retrains the Showroom Stock Velocity model (v2) on a RESCALED simulated panel.

Why v2: the v1 regressor was trained on simulated data where a model sold ~13 units per
quarter per branch, while the live dealership sells 0-3. At live scale v1 collapsed to ~2.2
for almost every model. v1 was also fed a hard-coded category/price at inference.

Methodology (unchanged rule: training uses SIMULATED data only):
  1. Synthesize many independent "simulated dealerships" (3 branches x catalog models x 48 months)
     at the live dealership's scale (~10 releases/month network-wide, most model-branch-months = 0),
     with model popularity drift, price sensitivity, seasonality, promo spikes, and stock-outs.
  2. Build features with dml.velocity_features (identical code is used at inference).
  3. Train a Poisson HistGradientBoostingRegressor (count target).
  4. Holdout = 20% of simulated dealerships never seen in training.
  5. Live benchmark = the real sales history, used ONLY to measure error (never to fit).
Both holdouts are compared to a naive baseline (average of the last 3 months).

Writes:
  data/simulated_velocity_panel.csv
  models/inventory_velocity_model.joblib   (bundle: {'version': 2, 'pipeline', ...})
  models/benchmark_report.json             (velocity entry replaced)
"""

import os
import sys
import json
import argparse
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import pymysql
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, MODELS_DIR, DB_CONFIG
from dml.velocity_features import SalesPanel, NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES, month_add
from dml.velocity_engine import CATALOG_DEFAULTS, resolve_model_specs
from dml.forecast_helpers import SEASONAL_INDEX

PANEL_PATH = os.path.join(DATA_DIR, 'simulated_velocity_panel.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'inventory_velocity_model.joblib')
BENCHMARK_REPORT_PATH = os.path.join(MODELS_DIR, 'benchmark_report.json')

# Scale calibration: the live network releases ~10 units/month across 3 branches.
# (Only these aggregate constants reflect the dealership; no live rows are used for fitting.)
NETWORK_MONTHLY_UNITS = 10.0
BRANCH_SHARES = [0.40, 0.31, 0.29]
BRANCHES = ['B1', 'B2', 'B3']
SIM_MONTHS = 48
WARMUP_MONTHS = 12   # months needed for lag_12 before rows become training examples


def simulate_dealership(rng, world_id, catalog):
    """One simulated dealership: returns (sales events, inventory units, row keys)."""
    n_models = len(catalog)
    world_scale = rng.uniform(0.6, 1.6)                            # busier / quieter dealerships
    shares = rng.dirichlet(np.array(BRANCH_SHARES) * 40)
    base_pop = rng.lognormal(mean=0.0, sigma=0.8, size=n_models)   # some models are hits, some are dead
    dead = rng.random(n_models) < 0.25                              # ~a quarter of the catalog barely sells
    base_pop[dead] *= 0.08
    price = np.array([c['price'] for c in catalog])
    base_pop *= (price / 90000.0) ** -0.6                            # cheaper units move faster
    log_pop = np.log(base_pop)
    branch_taste = rng.normal(0, 0.35, size=(3, n_models))          # local preferences per branch

    start = datetime(2020, 1, 1)
    months = [(start.replace(year=start.year + (start.month - 1 + i) // 12, month=(start.month - 1 + i) % 12 + 1)).strftime('%Y-%m')
              for i in range(SIM_MONTHS)]

    stock = rng.poisson(2.0, size=(3, n_models))
    sales, inventory = [], []
    uid = 0
    unit_queue = {(b, k): [] for b in range(3) for k in range(n_models)}

    def add_units(b, k, count, month):
        nonlocal uid
        for _ in range(count):
            uid += 1
            unit_queue[(b, k)].append(uid)
            inventory.append({'unit_id': f'{world_id}-{uid}', 'branch_id': BRANCHES[b],
                              'model_code': catalog[k]['code'], 'date_added': month_add(month, -1) + '-15'})

    for b in range(3):
        for k in range(n_models):
            add_units(b, k, int(stock[b, k]), months[0])

    for m in months:
        log_pop += rng.normal(0, 0.12, size=n_models)                  # popularity drifts month to month
        pop = np.exp(log_pop)
        share = pop / pop.sum()
        promo = 2.0 if rng.random() < 0.05 else 1.0                     # occasional promo / fiesta spike
        season = SEASONAL_INDEX[int(m[5:7])]
        for b in range(3):
            for k in range(n_models):
                lam = NETWORK_MONTHLY_UNITS * world_scale * shares[b] * share[k] * np.exp(branch_taste[b, k]) * season * promo
                on_hand = len(unit_queue[(b, k)])
                demand = rng.poisson(lam)
                sold = min(demand, on_hand)                             # can't sell what isn't on the floor
                for i in range(sold):
                    u = unit_queue[(b, k)].pop(0)
                    day = int(rng.integers(1, 28))
                    sales.append({'branch_id': BRANCHES[b], 'model_code': catalog[k]['code'],
                                  'date': f'{m}-{day:02d}', 'unit_id': f'{world_id}-{u}'})
                # Restock policy: replenish low floors, sometimes late
                if len(unit_queue[(b, k)]) <= 1 and rng.random() < 0.6:
                    add_units(b, k, int(rng.integers(1, 4)), month_add(m, 1))
        # units added during month m are dated inside m (visible from m+1)
    return sales, inventory, months


def build_rows(panel, branches, catalog, months, world_id=None):
    rows = []
    for m in months:
        for b in branches:
            for c in catalog:
                f = panel.features(b, c['code'], m, c['brand'], c['category'], c['price'])
                y = panel.units(b, c['code'], m)
                if f['stock_start'] == 0 and f['lag_6'] == 0 and y == 0:
                    continue    # never on the floor, never sold: nothing to predict
                f.update({'target_units': y, 'month': m, 'branch': b, 'model_code': c['code']})
                if world_id is not None:
                    f['world_id'] = world_id
                rows.append(f)
    return rows


def synthesize_panel(n_worlds, seed):
    rng = np.random.default_rng(seed)
    catalog = [{'code': code, 'brand': spec['brand'], 'category': spec['category'], 'price': spec['price']}
               for code, spec in CATALOG_DEFAULTS.items()]

    all_rows = []
    for w in range(n_worlds):
        sales, inventory, months = simulate_dealership(rng, w, catalog)
        panel = SalesPanel(sales, inventory)
        all_rows.extend(build_rows(panel, BRANCHES, catalog, months[WARMUP_MONTHS:], world_id=w))
    return pd.DataFrame(all_rows)


def load_live_panel():
    """Live history for the benchmark ONLY (never used to fit)."""
    conn = pymysql.connect(**DB_CONFIG)
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("""
        SELECT s.unit_id, u.branch_id, vm.model_code, vm.brand, DATE(s.created_at) AS d
        FROM sales s
        JOIN inventory_units u ON s.unit_id = u.unit_id
        JOIN vehicle_models vm ON u.model_id = vm.model_id
        WHERE s.status NOT IN ('cancelled', 'pending', 'pending_approval')
    """)
    sales = [{'branch_id': r['branch_id'], 'model_code': r['model_code'], 'date': str(r['d']), 'unit_id': r['unit_id']}
             for r in cur.fetchall()]
    cur.execute("""
        SELECT u.unit_id, u.branch_id, vm.model_code, vm.brand, DATE(u.date_added) AS d
        FROM inventory_units u JOIN vehicle_models vm ON u.model_id = vm.model_id
    """)
    inv_rows = cur.fetchall()
    inventory = [{'unit_id': r['unit_id'], 'branch_id': r['branch_id'], 'model_code': r['model_code'], 'date_added': str(r['d'])}
                 for r in inv_rows]
    cur.execute("SELECT branch_id FROM branches")
    branches = [r['branch_id'] for r in cur.fetchall()]
    conn.close()

    catalog = {}
    for r in inv_rows:
        spec = resolve_model_specs(r['model_code'], r['brand'])
        catalog[r['model_code']] = {'code': r['model_code'], 'brand': r['brand'], 'category': spec['category'], 'price': spec['price']}

    first = min(s['date'] for s in sales)[:7]
    last_complete = month_add(datetime.now().strftime('%Y-%m'), -1)
    months = []
    m = month_add(first, 12)
    while m <= last_complete:
        months.append(m)
        m = month_add(m, 1)
    return pd.DataFrame(build_rows(SalesPanel(sales, inventory), branches, list(catalog.values()), months))


def evaluate(pipeline, df):
    y = df['target_units'].values
    pred = np.clip(pipeline.predict(df[ALL_FEATURES]), 1e-6, None)
    naive = np.clip(df['lag_3'].values / 3.0, 1e-6, None)
    return {
        'rows': int(len(df)),
        'mae': round(float(mean_absolute_error(y, pred)), 4),
        'naive_mae': round(float(mean_absolute_error(y, naive)), 4),
        'poisson_deviance': round(float(mean_poisson_deviance(y, pred)), 4),
        'naive_poisson_deviance': round(float(mean_poisson_deviance(y, naive)), 4),
        'tolerance_accuracy_pct': round(float(np.mean(np.abs(y - pred) <= 1.0)) * 100, 2),
        'mean_actual': round(float(y.mean()), 4),
        'mean_predicted': round(float(pred.mean()), 4),
        'pred_p10_p50_p90': [round(float(q), 3) for q in np.percentile(pred, [10, 50, 90])],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worlds', type=int, default=60)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    print('[1/4] Synthesizing rescaled simulated velocity panel...')
    panel = synthesize_panel(args.worlds, args.seed)
    panel.to_csv(PANEL_PATH, index=False)
    print(f'      {len(panel):,} rows from {args.worlds} simulated dealerships -> {PANEL_PATH}')
    print(f'      mean units per model-branch-month: {panel.target_units.mean():.3f}')

    rng = np.random.default_rng(args.seed)
    worlds = np.arange(args.worlds)
    rng.shuffle(worlds)
    test_worlds = set(worlds[: max(1, args.worlds // 5)])
    train_df = panel[~panel.world_id.isin(test_worlds)]
    test_df = panel[panel.world_id.isin(test_worlds)]

    print(f'[2/4] Training Poisson gradient boosting on {len(train_df):,} rows ({args.worlds - len(test_worlds)} dealerships)...')
    pipeline = Pipeline([
        ('prep', ColumnTransformer([
            ('num', 'passthrough', NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES),
        ])),
        ('reg', HistGradientBoostingRegressor(loss='poisson', max_iter=300, learning_rate=0.05,
                                              max_depth=6, min_samples_leaf=40, l2_regularization=1.0,
                                              random_state=args.seed)),
    ])
    pipeline.fit(train_df[ALL_FEATURES], train_df['target_units'])

    print('[3/4] Evaluating...')
    sim_metrics = evaluate(pipeline, test_df)
    live_df = load_live_panel()
    live_metrics = evaluate(pipeline, live_df) if len(live_df) else {}
    for name, mt in [('Simulated holdout', sim_metrics), ('Live history (test only)', live_metrics)]:
        print(f'      {name}: rows={mt.get("rows")} MAE={mt.get("mae")} (naive {mt.get("naive_mae")}) '
              f'deviance={mt.get("poisson_deviance")} (naive {mt.get("naive_poisson_deviance")}) '
              f'mean actual={mt.get("mean_actual")} mean pred={mt.get("mean_predicted")} p10/50/90={mt.get("pred_p10_p50_p90")}')

    print('[4/4] Saving model bundle and benchmark entry...')
    bundle = {
        'version': 2,
        'pipeline': pipeline,
        'features': ALL_FEATURES,
        'target': 'units sold of one model at one branch in the month',
        'trained_on': os.path.basename(PANEL_PATH),
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    joblib.dump(bundle, MODEL_PATH)

    entry = {
        'model_name': 'Showroom Inventory Velocity Regressor',
        'algorithm': 'Poisson HistGradientBoostingRegressor (v2, lag features)',
        'dataset_source': os.path.basename(PANEL_PATH),
        'train_samples': int(len(train_df)),
        'test_samples': int(len(test_df)),
        'mae': sim_metrics['mae'],
        'naive_baseline_mae': sim_metrics['naive_mae'],
        'poisson_deviance': sim_metrics['poisson_deviance'],
        'naive_baseline_poisson_deviance': sim_metrics['naive_poisson_deviance'],
        'tolerance_accuracy_pct': sim_metrics['tolerance_accuracy_pct'],
        'live_benchmark_test_only': live_metrics,
        'target_window_met': sim_metrics['tolerance_accuracy_pct'] >= 75.0,
    }
    report = {}
    if os.path.exists(BENCHMARK_REPORT_PATH):
        with open(BENCHMARK_REPORT_PATH, 'r', encoding='utf-8') as f:
            report = json.load(f)
    models = [m for m in report.get('models', []) if m.get('model_name') != entry['model_name']]
    models.insert(2, entry)
    report['models'] = models
    with open(BENCHMARK_REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f'      Saved {MODEL_PATH}')


if __name__ == '__main__':
    main()
