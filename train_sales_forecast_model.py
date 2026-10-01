#!/usr/bin/env python3
"""
ml_engine/train_sales_forecast_model.py
Trains the Monthly Sales Forecast model: units sold per month for a branch or the whole network,
1-12 months ahead. It replaces the seasonal rule behind the sales goal and the 2-month sales outlook
(the forecast for next month becomes next month's goal).

Methodology (same rule as the velocity model: training uses SIMULATED data only):
  1. Source: data/simulated_velocity_panel.csv (simulated dealerships at the live dealership's scale,
     built by train_velocity_model.py). Units are summed per (dealership, branch, month) and per
     (dealership, month) for the network, giving 4 monthly series per simulated dealership.
  2. For every anchor month and horizon 1..12, features come from dml.sales_forecast.build_features
     (identical code is used at inference) and the label is the units sold in the target month.
  3. Model: Poisson HistGradientBoostingRegressor (count target).
  4. Holdout = 20% of simulated dealerships never seen in training.
  5. Live benchmark = the real monthly sales history, used ONLY to measure error (never to fit).
Both are compared to the seasonal rule it replaces (dml.forecast_helpers.estimate_monthly_units).

Writes:
  models/sales_forecast_model.joblib
  models/benchmark_report.json   (Monthly Sales Forecast entry added / replaced)
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, MODELS_DIR, DB_CONFIG
from dml.sales_forecast import FEATURES, MAX_HORIZON, MODEL_FILENAME, build_features
from dml.forecast_helpers import estimate_monthly_units, months_after

PANEL_PATH = os.path.join(DATA_DIR, 'simulated_velocity_panel.csv')
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)
BENCHMARK_REPORT_PATH = os.path.join(MODELS_DIR, 'benchmark_report.json')
MODEL_NAME = 'Monthly Sales Forecast Regressor'
MIN_HISTORY = 3   # anchor months need at least 3 known months behind them
SHOWN_HORIZON = 2 # the page shows the next 2 months; the headline test scores exactly those


def series_rows(units: dict, months: list, key: dict) -> list:
    """Every (anchor, horizon) pair whose target month is inside the series."""
    rows = []
    first = months[0]
    for i, anchor in enumerate(months):
        if i + 1 < MIN_HISTORY:
            continue
        for h in range(1, MAX_HORIZON + 1):
            if i + h >= len(months):
                break
            target = months[i + h]
            f = build_features(units, anchor, h, first)
            # The rule being replaced, for the baseline
            f['rule_units'] = estimate_monthly_units({m: units.get(m, 0) for m in months[:i + 1]}, months_after(anchor, 1)[0], target)
            f.update(key)
            f.update({'anchor': anchor, 'target': target, 'label_units': float(units.get(target, 0))})
            rows.append(f)
    return rows


def build_dataset(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    branch_m = panel.groupby(['world_id', 'branch', 'month'])['target_units'].sum()
    net_m = panel.groupby(['world_id', 'month'])['target_units'].sum()
    for w, wdf in panel.groupby('world_id'):
        months = sorted(wdf['month'].unique())
        for b in sorted(wdf['branch'].unique()):
            units = {m: float(branch_m.get((w, b, m), 0)) for m in months}
            rows.extend(series_rows(units, months, {'world_id': w, 'scope': b}))
        units = {m: float(net_m.get((w, m), 0)) for m in months}
        rows.extend(series_rows(units, months, {'world_id': w, 'scope': 'network'}))
    return pd.DataFrame(rows)


def load_live_dataset() -> pd.DataFrame:
    """Live monthly units per branch and network - benchmark ONLY (never used to fit)."""
    conn = pymysql.connect(**DB_CONFIG)
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("""
        SELECT u.branch_id, DATE_FORMAT(s.created_at, '%Y-%m') AS m, COUNT(*) AS units
        FROM sales s JOIN inventory_units u ON s.unit_id = u.unit_id
        WHERE s.status NOT IN ('cancelled', 'pending', 'pending_approval')
        GROUP BY u.branch_id, m
    """)
    data = cur.fetchall()
    conn.close()
    if not data:
        return pd.DataFrame()
    cur_month = datetime.now().strftime('%Y-%m')
    first = min(r['m'] for r in data)
    months = [first]
    while months[-1] < cur_month:
        months.append(months_after(months[-1], 1)[0])
    months = [m for m in months if m < cur_month]      # closed months only
    by_branch = {}
    for r in data:
        by_branch.setdefault(r['branch_id'], {})[r['m']] = float(r['units'])
    rows = []
    for b, units in by_branch.items():
        rows.extend(series_rows(units, months, {'scope': f'branch {b}'}))
    net = {}
    for units in by_branch.values():
        for m, v in units.items():
            net[m] = net.get(m, 0.0) + v
    rows.extend(series_rows(net, months, {'scope': 'network'}))
    return pd.DataFrame(rows)


def evaluate(pipeline, df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    y = df['label_units'].values
    pred = np.clip(pipeline.predict(df[FEATURES]), 0, None)
    rule = df['rule_units'].values
    out = {
        'rows': int(len(df)),
        'mae': round(float(mean_absolute_error(y, pred)), 4),
        'seasonal_rule_mae': round(float(mean_absolute_error(y, rule)), 4),
        'within_1_unit_pct': round(float(np.mean(np.abs(y - np.floor(pred + 0.5)) <= 1)) * 100, 2),
        'within_2_units_pct': round(float(np.mean(np.abs(y - np.floor(pred + 0.5)) <= 2)) * 100, 2),
        'mean_actual': round(float(y.mean()), 3),
        'mean_predicted': round(float(pred.mean()), 3),
    }
    # The horizons the page shows (next month and the month after)
    for h in (1, 2):
        sel = df['horizon'].values == h
        if sel.any():
            out[f'mae_h{h}'] = round(float(mean_absolute_error(y[sel], pred[sel])), 4)
            out[f'seasonal_rule_mae_h{h}'] = round(float(mean_absolute_error(y[sel], rule[sel])), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    print('[1/4] Building monthly sales series from the simulated panel...')
    if not os.path.exists(PANEL_PATH):
        sys.exit(f'Missing {PANEL_PATH}: run train_velocity_model.py first to synthesize it.')
    panel = pd.read_csv(PANEL_PATH, usecols=['world_id', 'branch', 'month', 'target_units'])
    data = build_dataset(panel)
    n_worlds = data['world_id'].nunique()
    print(f'      {len(data):,} (anchor, horizon) rows from {n_worlds} simulated dealerships')

    rng = np.random.default_rng(args.seed)
    worlds = np.array(sorted(data['world_id'].unique()))
    rng.shuffle(worlds)
    test_worlds = set(worlds[: max(1, n_worlds // 5)])
    train_df = data[~data.world_id.isin(test_worlds)]
    test_df = data[data.world_id.isin(test_worlds)]

    print(f'[2/4] Training Poisson gradient boosting on {len(train_df):,} rows ({n_worlds - len(test_worlds)} dealerships)...')
    pipeline = Pipeline([
        ('reg', HistGradientBoostingRegressor(loss='poisson', max_iter=300, learning_rate=0.05,
                                              max_depth=5, min_samples_leaf=60, l2_regularization=1.0,
                                              random_state=args.seed)),
    ])
    pipeline.fit(train_df[FEATURES], train_df['label_units'])

    print('[3/4] Evaluating...')
    # Headline test = the horizons the page shows (next month and the month after). The model is still
    # trained on 1-12 so the engine can forecast through December; the all-horizon score is kept too.
    shown = lambda df: df[df['horizon'] <= SHOWN_HORIZON] if len(df) else df
    sim_all = evaluate(pipeline, test_df)
    sim = evaluate(pipeline, shown(test_df))
    try:
        live_df = load_live_dataset()
        live_all = evaluate(pipeline, live_df)
        live = evaluate(pipeline, shown(live_df))
    except Exception as e:
        print(f'      Live benchmark skipped: {e}')
        live, live_all = {}, {}
    for name, mt in [('Simulated holdout', sim), ('Live history (test only)', live)]:
        print(f'      {name}: rows={mt.get("rows")} MAE={mt.get("mae")} (seasonal rule {mt.get("seasonal_rule_mae")}) '
              f'next month MAE={mt.get("mae_h1")} (rule {mt.get("seasonal_rule_mae_h1")}) '
              f'within 2 units={mt.get("within_2_units_pct")}%')

    print('[4/4] Saving model bundle and benchmark entry...')
    joblib.dump({
        'version': 1,
        'pipeline': pipeline,
        'features': FEATURES,
        'max_horizon': MAX_HORIZON,
        'target': 'units sold in the target month (branch or network)',
        'trained_on': os.path.basename(PANEL_PATH),
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }, MODEL_PATH)

    entry = {
        'model_name': MODEL_NAME,
        'algorithm': 'Poisson HistGradientBoostingRegressor',
        'dataset_source': os.path.basename(PANEL_PATH) + ' (summed to branch & network months)',
        'train_samples': int(len(train_df)),
        'test_samples': int(sim.get('rows', 0)),        # 1-2 months ahead, held-out dealerships
        'test_horizon_months': SHOWN_HORIZON,
        'mae_units_per_month': sim.get('mae'),
        'mae_next_month': sim.get('mae_h1'),
        'mae_two_months_ahead': sim.get('mae_h2'),
        'seasonal_rule_mae': sim.get('seasonal_rule_mae'),
        'seasonal_rule_mae_next_month': sim.get('seasonal_rule_mae_h1'),
        'within_1_unit_pct': sim.get('within_1_unit_pct'),
        'within_2_units_pct': sim.get('within_2_units_pct'),
        'all_horizons_1_12': {'rows': sim_all.get('rows'), 'mae': sim_all.get('mae'), 'seasonal_rule_mae': sim_all.get('seasonal_rule_mae')},
        'live_benchmark_test_only': live,
        'live_benchmark_all_horizons': {'rows': live_all.get('rows'), 'mae': live_all.get('mae'), 'seasonal_rule_mae': live_all.get('seasonal_rule_mae')},
        'honest_critique': (
            f"Next-month MAE {sim.get('mae_h1')} units vs {sim.get('seasonal_rule_mae_h1')} for the seasonal rule it replaces. "
            "Learns level, momentum and season from the unit history only, so promotions or stock-outs it has never seen "
            "in the history are not anticipated."
        ),
    }
    report = {}
    if os.path.exists(BENCHMARK_REPORT_PATH):
        with open(BENCHMARK_REPORT_PATH, 'r', encoding='utf-8') as f:
            report = json.load(f)
    report['models'] = [m for m in report.get('models', []) if m.get('model_name') != MODEL_NAME] + [entry]
    with open(BENCHMARK_REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f'      Saved {MODEL_PATH}')


if __name__ == '__main__':
    main()
