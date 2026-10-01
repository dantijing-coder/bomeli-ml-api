#!/usr/bin/env python3
"""
ml_engine/train_branch_expansion.py
Branch Expansion Opportunity model - "which town should be our next branch?" (3-year plan)

Methodology (same rule as the other models): TRAIN on simulated data, SCORE live data.

  Training (simulated: data/simulated_geospatial_borrowers.csv)
    A. Credit model (borrower level, HistGradientBoostingClassifier): P(default/bad) from distance &
       travel time to the servicing branch, road quality, income, debt-to-income, down payment %,
       term, engine size, job sector, address migration. Repayment outcomes are NOT inputs.
    B. Demand model (town-year level, Poisson GLM): new borrowers per year from population,
       commercial density, banking infrastructure, road quality and distance to the nearest branch.
       Validated leave-one-town-out.

  Scoring (live DB)
    Live clients are grouped by the town on their profile. A town OUTSIDE the existing branch towns
    is only evaluated once it has enough live evidence (MIN_LIVE_CLIENTS over MIN_HISTORY_MONTHS).
    Until then it is reported on a watchlist with its progress - no simulated ranking is shown.
    Town reference attributes (population, commerce/banking scores, distance to nearest branch)
    come from the geospatial dataset's town table.

Writes models/branch_expansion_model.joblib and models/branch_expansion_report.json
"""

import os
import sys
import json
import re
from datetime import datetime, date

import joblib
import numpy as np
import pandas as pd
import pymysql
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, MODELS_DIR, DB_CONFIG
from dml.forecast_helpers import SEASONAL_INDEX

GEO_PATH = os.path.join(DATA_DIR, 'simulated_geospatial_borrowers.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'branch_expansion_model.joblib')
REPORT_PATH = os.path.join(MODELS_DIR, 'branch_expansion_report.json')

CREDIT_NUM = ['nearest_branch_distance_km', 'travel_time_minutes', 'road_quality_score', 'monthly_income',
              'debt_to_income_ratio', 'down_payment_pct', 'term_months', 'engine_cc', 'is_address_migrated']
CREDIT_CAT = ['employment_sector']
DEMAND_FEATURES = ['log_population', 'commercial_density_score', 'banking_infrastructure_score',
                   'road_quality_score', 'median_distance_km']

# Evidence required before a non-branch town is evaluated
MIN_LIVE_CLIENTS = 20
MIN_HISTORY_MONTHS = 12
# A town whose clients already live within this distance of a branch is "served", not a candidate
SERVED_RADIUS_KM = 5.0
RAMP_UP = [0.60, 0.85, 1.00]
GROWTH_CLIP = 0.15
SCORE_WEIGHTS = {'incremental_demand': 0.35, 'credit_quality': 0.25, 'growth': 0.15, 'market': 0.15, 'underserved': 0.10}
BAD_STATUSES = ('defaulted', 'repossessed', 'pre_repossession', 'repo', 'terminated', 'lost_vehicle')

SECTOR_KEYWORDS = [
    ('Transport', ['tricycle', 'puv', 'driver', 'truck', 'operator', 'habal']),
    ('Gov', ['teacher', 'gov', 'public school']),
    ('Agri-Labor', ['farmer', 'fisher', 'agri', 'laborer']),
    ('Retail', ['sari', 'retail', 'vendor', 'store', 'business']),
    ('Tech/Remote', ['bpo', 'call center', 'freelance', 'remote']),
    ('Remittance', ['ofw', 'remittance']),
    ('Health', ['nurse', 'health', 'medical']),
    ('Uniformed', ['police', 'military', 'bfp', 'army']),
    ('Construction', ['construction', 'carpenter', 'mason']),
]


# ─────────────────────────────── training (simulated) ───────────────────────────────

def load_geo():
    df = pd.read_csv(GEO_PATH)
    df['year'] = df['account_no'].str.extract(r'-(\d{4})-')[0].astype(int)
    df['town'] = df['present_municipality'].str.strip()
    return df


def train_credit_model(df, seed=42):
    X = df[CREDIT_NUM + CREDIT_CAT]
    y = df['is_default_or_bad'].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    pipe = Pipeline([
        ('prep', ColumnTransformer([
            ('num', 'passthrough', CREDIT_NUM),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CREDIT_CAT),
        ])),
        ('clf', HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_depth=4,
                                               min_samples_leaf=40, l2_regularization=1.0,
                                               class_weight='balanced', random_state=seed)),
    ])
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    metrics = {
        'train_samples': int(len(X_tr)), 'test_samples': int(len(X_te)),
        'roc_auc': round(float(roc_auc_score(y_te, proba)), 4),
        'balanced_accuracy_pct': round(float(balanced_accuracy_score(y_te, proba >= 0.5)) * 100, 2),
    }
    pipe.fit(X, y)
    return pipe, metrics


# Towns our clients come from that are outside the simulated geospatial dataset.
# Approximate public figures - verify before relying on them.
EXTRA_TOWNS = {
    'Aurora': {  # Aurora, Zamboanga del Sur - across the provincial border from Kapatagan
        'population': 48000, 'commercial_density_score': 72, 'banking_infrastructure_score': 62,
        'road_quality_score': 74.0, 'median_distance_km': 26.0, 'median_travel_min': 48.0,
        'nearest_branch': 'Kapatagan',
    },
}


def town_reference(df):
    """Per-town reference attributes (the 'gazetteer')."""
    ref = df.groupby('town').agg(
        population=('territory_population', 'first'),
        commercial_density_score=('commercial_density_score', 'first'),
        banking_infrastructure_score=('banking_infrastructure_score', 'first'),
        road_quality_score=('road_quality_score', 'median'),
        median_distance_km=('nearest_branch_distance_km', 'median'),
        median_travel_min=('travel_time_minutes', 'median'),
        nearest_branch=('assigned_branch_name', lambda s: s.mode().iat[0]),
    )
    extra = pd.DataFrame.from_dict({k: v for k, v in EXTRA_TOWNS.items() if k not in ref.index}, orient='index')
    ref = pd.concat([ref, extra]) if len(extra) else ref
    ref['log_population'] = np.log(ref['population'])
    return ref


def train_demand_model(df, ref):
    ty = df.groupby(['town', 'year']).size().rename('new_borrowers').reset_index()
    ty = ty.merge(ref, left_on='town', right_index=True)

    def make():
        return Pipeline([('scale', StandardScaler()), ('glm', PoissonRegressor(alpha=0.05, max_iter=1000))])

    errors, rel = [], []
    for town in ty['town'].unique():
        tr, te = ty[ty.town != town], ty[ty.town == town]
        pred = make().fit(tr[DEMAND_FEATURES], tr['new_borrowers']).predict(te[DEMAND_FEATURES])
        errors.extend(np.abs(pred - te['new_borrowers']))
        rel.extend(np.abs(pred - te['new_borrowers']) / np.maximum(te['new_borrowers'], 1))
    model = make().fit(ty[DEMAND_FEATURES], ty['new_borrowers'])
    return model, {
        'rows': int(len(ty)), 'validation': 'leave-one-town-out',
        'mae_borrowers_per_year': round(float(np.mean(errors)), 2),
        'median_abs_pct_error': round(float(np.median(rel)) * 100, 1),
        'standardized_coefficients': dict(zip(DEMAND_FEATURES, [round(float(c), 4) for c in model.named_steps['glm'].coef_])),
    }


# ─────────────────────────────── live data ───────────────────────────────

def norm(s):
    return re.sub(r'\s+', ' ', str(s or '')).strip().lower()


def match_town(value, towns_lower):
    v = norm(value)
    if not v:
        return None
    if v in towns_lower:
        return towns_lower[v]
    for key, town in towns_lower.items():          # "Lala, Lanao del Norte" -> Lala
        if re.search(r'\b' + re.escape(key) + r'\b', v):
            return town
    return None


def sector_for(text):
    t = norm(text)
    for sector, words in SECTOR_KEYWORDS:
        if any(w in t for w in words):
            return sector
    return 'Retail'


def engine_cc(model_code):
    m = re.search(r'(\d{3})', str(model_code or ''))
    return int(m.group(1)) if m else 125


def load_live(ref):
    conn = pymysql.connect(**DB_CONFIG)
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT branch_id, name FROM branches")
    branches = cur.fetchall()
    cur.execute("""
        SELECT s.sale_id, s.created_at, s.status, s.monthly_amortization, s.down_payment,
               s.gross_principal_amount, s.term_months, s.application_data,
               cp.city, cp.permanent_city, cp.monthly_gross_income, cp.employment_status,
               vm.model_code, b.name AS branch_name
        FROM sales s
        LEFT JOIN customer_profiles cp ON cp.customer_id = s.customer_id
        LEFT JOIN inventory_units u ON u.unit_id = s.unit_id
        LEFT JOIN vehicle_models vm ON vm.model_id = u.model_id
        LEFT JOIN branches b ON b.branch_id = u.branch_id
        WHERE s.status NOT IN ('cancelled', 'pending', 'pending_approval')
    """)
    rows = cur.fetchall()
    conn.close()

    towns_lower = {t.lower(): t for t in ref.index}
    recs, unmatched = [], {}
    for r in rows:
        town = match_town(r['city'], towns_lower)
        if not town:
            key = str(r['city'] or '(blank)').strip() or '(blank)'
            unmatched[key] = unmatched.get(key, 0) + 1
            continue
        app = {}
        try:
            app = json.loads(r['application_data']) if r['application_data'] else {}
        except Exception:
            pass
        income = float(r['monthly_gross_income'] or app.get('monthly_gross_income') or app.get('totalIncome') or 0) or 25000.0
        amort = float(r['monthly_amortization'] or 0)
        gross = float(r['gross_principal_amount'] or 0)
        perm_town = match_town(r['permanent_city'], towns_lower)
        recs.append({
            'town': town,
            'created_at': r['created_at'],
            'year': r['created_at'].year,
            'is_bad': int(str(r['status']).lower() in BAD_STATUSES),
            'monthly_income': income,
            'debt_to_income_ratio': round(amort / income * 100.0, 1) if income > 0 else 0.0,
            'down_payment_pct': round(float(r['down_payment'] or 0) / gross, 3) if gross > 0 else 0.15,
            'term_months': int(r['term_months'] or 24),
            'engine_cc': engine_cc(r['model_code']),
            'is_address_migrated': int(bool(perm_town and perm_town != town)),
            'employment_sector': sector_for(r['employment_status'] or app.get('employment_status')),
            'branch_name': r['branch_name'],
        })
    live = pd.DataFrame(recs)
    branch_towns = {match_town(b['name'], towns_lower) for b in branches} - {None}
    return live, branch_towns, unmatched


# ─────────────────────────────── scoring ───────────────────────────────

def annualized_by_year(tdf, today, start):
    """
    Clients per calendar year, scaled to 12 months for partial years:
    the first year of operation (from `start`) and the current year (to today).
    """
    out = {}
    for y, n in tdf.groupby('year').size().items():
        y = int(y)
        first_m = start.month if y == start.year else 1
        last_m = today.month if y == today.year else 12
        months = max(1, last_m - first_m + 1)
        out[y] = round(n * 12.0 / months, 1)
    return out


def calibrate_to_live(live, ref, branch_towns, credit_model, demand_model, in_town_distance, in_town_travel,
                      today, start):
    """
    The models learn relative effects (distance, market, income...) from simulated data whose scale is not ours.
    Rescale both to the live network using the existing branch towns as the reference:
      demand_scale   = actual new clients/yr in branch towns / GLM prediction for those towns
      bad_rate_scale = actual default/repo share of branch-town clients / credit-model prediction for them
    """
    towns = [t for t in branch_towns if len(live) and (live['town'] == t).any()]
    if not towns:
        return {'demand_scale': 1.0, 'bad_rate_scale': 1.0, 'reference_towns': []}
    actual, pred = [], []
    for t in towns:
        by_year = annualized_by_year(live[live['town'] == t], today, start)
        actual.append(float(np.mean(list(by_year.values()))))
        feat = pd.DataFrame([ref.loc[t, DEMAND_FEATURES].to_dict()]).assign(median_distance_km=in_town_distance)
        pred.append(float(demand_model.predict(feat)[0]))
    demand_scale = sum(actual) / max(1e-6, sum(pred))

    bt = live[live['town'].isin(towns)]
    X = bt.assign(nearest_branch_distance_km=in_town_distance, travel_time_minutes=in_town_travel,
                  road_quality_score=bt['town'].map(ref['road_quality_score']))[CREDIT_NUM + CREDIT_CAT]
    p_bad = float(credit_model.predict_proba(X)[:, 1].mean())
    a_bad = float(bt['is_bad'].mean())
    # A branch network with almost no defaults should not zero the scale; keep it within a sane band
    bad_rate_scale = float(np.clip(a_bad / max(1e-6, p_bad), 0.05, 2.0))
    return {
        'demand_scale': round(demand_scale, 4), 'bad_rate_scale': round(bad_rate_scale, 4),
        'reference_towns': towns,
        'reference_actual_clients_per_year': round(sum(actual) / len(actual), 1),
        'reference_model_clients_per_year': round(sum(pred) / len(pred), 1),
        'reference_actual_bad_rate_pct': round(a_bad * 100, 1),
        'reference_model_bad_rate_pct': round(p_bad * 100, 1),
    }


MIN_GROWTH_BASE = 5          # fewer clients than this in the earlier 12 months -> no % (1 -> 4 is not "+300%")
SEASON_MEAN = sum(SEASONAL_INDEX.values()) / 12.0


def month_key(d):
    return f"{d.year:04d}-{d.month:02d}"


def add_months(key, n):
    y, m = int(key[:4]), int(key[5:7]) - 1 + n
    return f"{y + m // 12:04d}-{m % 12 + 1:02d}"


def monthly_counts(tdf, start_key, end_key):
    """New clients per calendar month from start_key to end_key inclusive (zeros filled)."""
    counts = tdf['created_at'].map(month_key).value_counts().to_dict()
    out, k = {}, start_key
    while k <= end_key:
        out[k] = int(counts.get(k, 0))
        k = add_months(k, 1)
    return out


def trailing_growth(by_month, today, ops_start_key):
    """
    Year-over-year change: the last 12 months (including this one) vs the 12 before them.
    Returns (recent, prior, growth or None). None when there is not 24 months of operating history
    or the earlier window is too small for a percentage to mean anything.
    """
    last_full = month_key(today)
    recent_keys = [add_months(last_full, -i) for i in range(12)]
    prior_keys = [add_months(last_full, -12 - i) for i in range(12)]
    recent = sum(by_month.get(k, 0) for k in recent_keys)
    prior = sum(by_month.get(k, 0) for k in prior_keys)
    if prior_keys[-1] < ops_start_key or prior < MIN_GROWTH_BASE:
        return recent, prior, None
    return recent, prior, recent / prior - 1


def spread_by_month(yearly_rate, year, from_key):
    """Spreads a yearly figure over that year's months (from from_key on) using the seasonal index."""
    out, k = {}, max(from_key, f"{year}-01")
    while k <= f"{year}-12":
        out[k] = round(yearly_rate / 12.0 * SEASONAL_INDEX[int(k[5:7])] / SEASON_MEAN, 2)
        k = add_months(k, 1)
    return out


def growth_rate(by_year):
    ys = sorted(by_year)
    if len(ys) < 2 or by_year[ys[0]] <= 0 or by_year[ys[-1]] <= 0:
        return 0.0
    return (by_year[ys[-1]] / by_year[ys[0]]) ** (1 / (len(ys) - 1)) - 1


def pct_rank(series):
    return series.rank(pct=True, method='average').fillna(0.5)


def build_report(seed=42):
    today = date.today()
    geo = load_geo()
    ref = town_reference(geo)
    credit_model, credit_metrics = train_credit_model(geo, seed)
    demand_model, demand_metrics = train_demand_model(geo, ref)
    live, branch_towns, unmatched = load_live(ref)

    in_town_distance = float(ref.loc[list(branch_towns), 'median_distance_km'].median()) if branch_towns else 1.2
    in_town_travel = float(ref.loc[list(branch_towns), 'median_travel_min'].median()) if branch_towns else 10.0

    towns = []
    operations_start = live['created_at'].min() if len(live) else datetime.now()
    calib = calibrate_to_live(live, ref, branch_towns, credit_model, demand_model, in_town_distance, in_town_travel,
                              today, operations_start)
    ops_key, cur_key = month_key(operations_start), month_key(today)
    for town, tdf in (live.groupby('town') if len(live) else []):
        st = ref.loc[town]
        by_year = annualized_by_year(tdf, today, operations_start)
        by_month = monthly_counts(tdf, ops_key, cur_key)
        recent_12m, prior_12m, growth = trailing_growth(by_month, today, ops_key)
        first_sale = tdf['created_at'].min()
        history_months = (today.year - first_sale.year) * 12 + today.month - first_sale.month
        is_branch = town in branch_towns
        is_served = (not is_branch) and st['median_distance_km'] < SERVED_RADIUS_KM
        eligible = len(tdf) >= MIN_LIVE_CLIENTS and history_months >= MIN_HISTORY_MONTHS
        status = 'existing_branch' if is_branch else ('served' if is_served else ('candidate' if eligible else 'watch'))

        row = {
            'town': town, 'status': status, 'nearest_branch': st['nearest_branch'],
            'live_clients': int(len(tdf)), 'history_months': int(history_months),
            'clients_by_year': by_year, 'clients_by_month': by_month,
            'recent_12m': int(recent_12m), 'prior_12m': int(prior_12m),
            'growth_pct': None if growth is None else round(growth * 100, 1),
            'live_bad_rate_pct': round(float(tdf['is_bad'].mean()) * 100, 1),
            'population': int(st['population']),
            'median_distance_km': round(float(st['median_distance_km']), 1),
            'median_travel_min': round(float(st['median_travel_min']), 0),
            'commercial_density_score': int(st['commercial_density_score']),
            'banking_infrastructure_score': int(st['banking_infrastructure_score']),
            'progress': {'clients': int(len(tdf)), 'clients_needed': MIN_LIVE_CLIENTS,
                         'months': int(history_months), 'months_needed': MIN_HISTORY_MONTHS},
        }

        if status == 'candidate':
            X_now = tdf.assign(nearest_branch_distance_km=st['median_distance_km'],
                               travel_time_minutes=st['median_travel_min'],
                               road_quality_score=st['road_quality_score'])[CREDIT_NUM + CREDIT_CAT]
            X_branch = X_now.assign(nearest_branch_distance_km=min(st['median_distance_km'], in_town_distance),
                                    travel_time_minutes=min(st['median_travel_min'], in_town_travel))
            feat_b = pd.DataFrame([st[DEMAND_FEATURES].to_dict()]).assign(
                median_distance_km=min(st['median_distance_km'], in_town_distance))
            # Today's demand is what the town actually produced over the last 12 full months. With a branch:
            # the simulated-trained GLM rescaled to what our own branch towns really produce, never below today.
            base_year = today.year
            d_now = float(recent_12m)
            d_branch = max(d_now, float(demand_model.predict(feat_b)[0]) * calib['demand_scale'])
            g = float(np.clip(growth or 0.0, -GROWTH_CLIP, GROWTH_CLIP))
            # Ramp-up applies to the extra clients a new branch attracts, not to the clients the town already sends
            projection = [{'year': base_year + i + 1,
                           'new_clients': int(round((d_now + (d_branch - d_now) * ramp) * (1 + g) ** (i + 1)))}
                          for i, ramp in enumerate(RAMP_UP)]
            bad_now = float(credit_model.predict_proba(X_now)[:, 1].mean()) * calib['bad_rate_scale']
            bad_branch = float(credit_model.predict_proba(X_branch)[:, 1].mean()) * calib['bad_rate_scale']
            row.update({
                'bad_rate_now_pct': round(min(1.0, bad_now) * 100, 1),
                'bad_rate_with_branch_pct': round(min(1.0, bad_branch) * 100, 1),
                'demand_now_per_year': round(d_now, 1),
                'demand_with_branch_per_year': round(d_branch, 1),
                'incremental_demand_per_year': round(max(0.0, d_branch - d_now), 1),
                'projection_3yr': projection,
            })
            # Until a branch opens (rest of this year) the town keeps sending what it sends today
            row['projection_by_year'] = {p['year']: p['new_clients'] for p in projection}
            rest_rate = d_now
        elif status == 'existing_branch':
            g = float(np.clip(growth or 0.0, -GROWTH_CLIP, GROWTH_CLIP))
            row['projection_by_year'] = {today.year + i: round(recent_12m * (1 + g) ** i, 1) for i in (1, 2, 3)}
            rest_rate = float(recent_12m)
        else:
            rest_rate = None    # watchlist / close-to-a-branch towns get no projection

        if rest_rate is not None:
            nxt = add_months(cur_key, 1)
            pm = spread_by_month(rest_rate, today.year, nxt)
            for y, v in row['projection_by_year'].items():
                pm.update(spread_by_month(float(v), int(y), nxt))
            row['projection_by_month'] = pm
        # Year view: whole-year totals; the current year = recorded so far + projected rest (or annualized)
        yt = {}
        for k, n in by_month.items():
            yt[int(k[:4])] = yt.get(int(k[:4]), 0) + n
        if rest_rate is not None:
            yt[today.year] = round(yt.get(today.year, 0) + sum(v for k, v in row['projection_by_month'].items()
                                                              if k.startswith(str(today.year))), 1)
        else:
            yt[today.year] = by_year.get(today.year, 0.0)
        if operations_start.year in yt and operations_start.year < today.year:
            yt[operations_start.year] = by_year.get(operations_start.year, yt[operations_start.year])
        row['year_totals'] = yt
        towns.append(row)

    # Score candidates against each other AND the existing branch towns (so a lone candidate is not 100 by default)
    cand = [t for t in towns if t['status'] == 'candidate']
    if cand:
        pool = pd.DataFrame([t for t in towns if t['status'] in ('candidate', 'existing_branch')])
        pool['incremental'] = pool.get('incremental_demand_per_year', pd.Series(0, index=pool.index)).fillna(0)
        pool['bad'] = pool.get('bad_rate_with_branch_pct', pool['live_bad_rate_pct']).fillna(pool['live_bad_rate_pct'])
        comp = pd.DataFrame({
            'incremental_demand': pct_rank(pool['incremental']),
            'credit_quality': pct_rank(-pool['bad']),
            'growth': pct_rank(pool['growth_pct'].astype(float).fillna(0.0)),
            'market': pct_rank(pool['commercial_density_score'] + pool['banking_infrastructure_score']),
            'underserved': pct_rank(pool['median_distance_km']),
        })
        pool['score'] = sum(comp[k] * w for k, w in SCORE_WEIGHTS.items()) * 100
        for t in cand:
            idx = pool.index[pool['town'] == t['town']][0]
            t['score'] = round(float(pool.loc[idx, 'score']), 1)
            t['score_components'] = {k: round(float(comp.loc[idx, k]) * 100) for k in SCORE_WEIGHTS}
            t['recommendation'] = 'Strong candidate' if t['score'] >= 70 else ('Monitor' if t['score'] >= 45 else 'Not yet')

    # Existing-branch benchmark with a 3-year trend line (so the reference line spans the whole chart)
    bench = [t for t in towns if t['status'] == 'existing_branch']
    bench_years = sorted({y for t in bench for y in t['clients_by_year']})
    bench_avg = {y: round(float(np.mean([t['clients_by_year'].get(y, 0) for t in bench])), 1) for y in bench_years}
    bench_growth = float(np.clip(growth_rate(bench_avg), -GROWTH_CLIP, GROWTH_CLIP)) if bench_avg else 0.0
    last = bench_years[-1] if bench_years else today.year
    bench_proj = {last + i: round(bench_avg[last] * (1 + bench_growth) ** i, 1) for i in range(1, 4)} if bench_avg else {}
    # Same benchmark on the month/year views the chart uses: the average of the branch towns
    avg_of = lambda key: ({k: round(float(np.mean([t.get(key, {}).get(k, 0) for t in bench])), 2)
                           for k in sorted({k for t in bench for k in t.get(key, {})})} if bench else {})
    bench_by_month, bench_proj_month, bench_year_totals = avg_of('clients_by_month'), avg_of('projection_by_month'), avg_of('year_totals')
    bench_proj_year = avg_of('projection_by_year')

    watch = sorted([t for t in towns if t['status'] == 'watch'], key=lambda t: -t['live_clients'])
    report = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'purpose': 'Branch expansion evaluation for the 3-year plan',
        'method': 'Trained on simulated geospatial borrowers; scored on live client records',
        'readiness': {
            'eligible_towns': len(cand),
            'watch_towns': len(watch),
            'min_live_clients': MIN_LIVE_CLIENTS,
            'min_history_months': MIN_HISTORY_MONTHS,
            'live_clients_matched': int(len(live)),
            'unmatched_city_values': [{'value': k, 'clients': v} for k, v in sorted(unmatched.items(), key=lambda kv: -kv[1])],
            'status': 'ready' if cand else 'insufficient_live_data',
        },
        'assumptions': {
            'served_radius_km': SERVED_RADIUS_KM, 'in_town_distance_km': round(in_town_distance, 2),
            'ramp_up': RAMP_UP, 'growth_clip_pct': int(GROWTH_CLIP * 100), 'score_weights': SCORE_WEIGHTS,
            'partial_years_annualized': True,
            'operations_start': str(operations_start)[:10],
            'calibration': calib,
        },
        'models': {
            'credit': {'algorithm': 'HistGradientBoostingClassifier (balanced)', 'trained_on': 'simulated', **credit_metrics},
            'demand': {'algorithm': 'Poisson GLM (standardized features)', 'trained_on': 'simulated', **demand_metrics},
        },
        'benchmark': {'towns': [t['town'] for t in bench], 'avg_by_year': bench_avg,
                      'projection': bench_proj, 'growth_pct': round(bench_growth * 100, 1),
                      'by_month': bench_by_month, 'projection_by_month': bench_proj_month,
                      'year_totals': bench_year_totals, 'projection_by_year': bench_proj_year},
        'timeline': {'current_month': cur_key, 'operations_start_month': ops_key,
                     'growth_window': 'last 12 months (including this month) vs the 12 months before', 'min_growth_base': MIN_GROWTH_BASE},
        'towns': sorted(towns, key=lambda t: ({'candidate': 0, 'watch': 1, 'served': 2, 'existing_branch': 3}[t['status']],
                                              -(t.get('score') or 0), -t['live_clients'])),
    }
    joblib.dump({'version': 2, 'credit_model': credit_model, 'demand_model': demand_model,
                 'credit_features': CREDIT_NUM + CREDIT_CAT, 'demand_features': DEMAND_FEATURES}, MODEL_PATH)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    return report


if __name__ == '__main__':
    rep = build_report()
    r = rep['readiness']
    print(f"Status: {r['status']} | eligible towns: {r['eligible_towns']} | watchlist: {r['watch_towns']} | "
          f"live clients matched to a town: {r['live_clients_matched']}")
    print(f"Unmatched city values: {r['unmatched_city_values']}")
    print(f"Credit model AUC={rep['models']['credit']['roc_auc']} | Demand LOTO MAE={rep['models']['demand']['mae_borrowers_per_year']}/yr")
    print(f"Benchmark avg/yr: {rep['benchmark']['avg_by_year']} -> projection {rep['benchmark']['projection']}")
    for t in rep['towns']:
        print(f"  {t['town']:<12} {t['status']:<16} live={t['live_clients']} by_year={t['clients_by_year']} bad={t['live_bad_rate_pct']}%")
    print(f"Saved {REPORT_PATH}")
