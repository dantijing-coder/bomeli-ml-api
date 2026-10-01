"""
Monthly Sales Forecast model (units sold per month for a branch or the whole network).

Shared by training (train_sales_forecast_model.py) and inference (forecast_helpers.build_forward_forecast)
so both build identical features. The model predicts units sold `horizon` months after an anchor month,
using only the unit history up to and including the anchor month.

  label:    units sold in the target month
  features: see FEATURES below (NaN = not enough history; the gradient-boosting model handles it natively)
"""

import math
import os
from typing import Dict, List, Optional

from dateutil.relativedelta import relativedelta
from datetime import datetime

from .forecast_helpers import SEASONAL_INDEX

MODEL_FILENAME = 'sales_forecast_model.joblib'
MAX_HORIZON = 12

FEATURES = [
    'units_anchor',        # units in the anchor month (live month: max(sold, goal x attainment))
    'units_lag_1',         # units the month before the anchor
    'units_lag_2',         # units two months before the anchor
    'mean_3',              # average of the 3 months ending at the anchor
    'mean_6',              # average of the 6 months ending at the anchor
    'mean_12',             # average of the 12 months ending at the anchor (overall level)
    'same_month_last_year',# units in the target month one year earlier
    'horizon',             # months ahead (1 = next month)
    'target_month',        # calendar month being forecast (1-12)
    'season_target',       # seasonal index of the target month
    'season_anchor',       # seasonal index of the anchor month
]


def _shift(month: str, n: int) -> str:
    return (datetime.strptime(month + '-01', '%Y-%m-%d') + relativedelta(months=n)).strftime('%Y-%m')


def build_features(units: Dict[str, float], anchor: str, horizon: int,
                   first_month: Optional[str] = None) -> Dict[str, float]:
    """
    units: month -> units sold (months missing after `first_month` count as 0; before it, unknown).
    anchor: last month whose units are known (YYYY-MM). horizon: months ahead of the anchor.
    """
    first = first_month or (min(units) if units else anchor)

    def u(m):
        if m < first or m > anchor:
            return math.nan
        return float(units.get(m, 0.0))

    def mean_back(n):
        vals = [u(_shift(anchor, -i)) for i in range(n)]
        vals = [v for v in vals if not math.isnan(v)]
        return sum(vals) / len(vals) if vals else math.nan

    target = _shift(anchor, horizon)
    t_month = int(target[5:7])
    return {
        'units_anchor': u(anchor),
        'units_lag_1': u(_shift(anchor, -1)),
        'units_lag_2': u(_shift(anchor, -2)),
        'mean_3': mean_back(3),
        'mean_6': mean_back(6),
        'mean_12': mean_back(12),
        'same_month_last_year': u(_shift(target, -12)),
        'horizon': float(horizon),
        'target_month': float(t_month),
        'season_target': SEASONAL_INDEX.get(t_month, 1.0),
        'season_anchor': SEASONAL_INDEX.get(int(anchor[5:7]), 1.0),
    }


_BUNDLE = None
_LOADED = False


def load_model(models_dir: Optional[str] = None):
    """Returns the trained bundle, or None when it has not been trained yet (callers fall back to the rule)."""
    global _BUNDLE, _LOADED
    if _LOADED and models_dir is None:
        return _BUNDLE
    path = os.path.join(models_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'),
                        MODEL_FILENAME)
    bundle = None
    if os.path.exists(path):
        try:
            import joblib
            bundle = joblib.load(path)
        except Exception as e:
            print(f"[sales_forecast] Warning: could not load {path}: {e}")
    if models_dir is None:
        _BUNDLE, _LOADED = bundle, True
    return bundle


def predict_units(units: Dict[str, float], anchor: str, target_months: List[str]) -> Optional[Dict[str, float]]:
    """Model forecast for each target month (all after `anchor`), or None if the model is unavailable."""
    bundle = load_model()
    if not bundle:
        return None
    import pandas as pd
    first = min(units) if units else anchor
    rows, keep = [], []
    for m in target_months:
        h = (int(m[:4]) - int(anchor[:4])) * 12 + int(m[5:7]) - int(anchor[5:7])
        if 1 <= h <= bundle.get('max_horizon', MAX_HORIZON):
            rows.append(build_features(units, anchor, h, first))
            keep.append(m)
    if not rows:
        return {}
    preds = bundle['pipeline'].predict(pd.DataFrame(rows)[bundle.get('features', FEATURES)])
    return {m: max(0.0, float(p)) for m, p in zip(keep, preds)}
