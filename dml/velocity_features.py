"""
Shared feature builder for the Stock Velocity model (v2).

The same code builds features for:
  - the simulated training panel (train_velocity_model.py)
  - the live-data benchmark (test only, never training)
  - inference in run_walk_forward_pipeline.py and run_predictive_engine.py

Unit of prediction: units of one model sold at one branch in one calendar month,
using only information available before that month starts.
"""

from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, Optional, Any

from dateutil.relativedelta import relativedelta

NUMERIC_FEATURES = [
    'lag_1',          # units sold last month
    'lag_3',          # units sold over the last 3 months
    'lag_6',          # units sold over the last 6 months
    'lag_12',         # units sold in the same month last year
    'net_lag_3',      # this model, all branches, last 3 months (network demand)
    'branch_lag_3',   # all models at this branch, last 3 months (branch traffic)
    'stock_start',    # units of this model on the branch floor when the month starts
    'base_price',
    'season_month',
]
CATEGORICAL_FEATURES = ['brand', 'vehicle_category']
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def month_add(month: str, n: int) -> str:
    return (datetime.strptime(month + '-01', '%Y-%m-%d') + relativedelta(months=n)).strftime('%Y-%m')


class SalesPanel:
    """
    Monthly unit counts per (branch, model), plus unit-level stock history.

    sales:     iterable of {'branch_id', 'model_code', 'date' (YYYY-MM-DD), 'unit_id' (optional)}
    inventory: iterable of {'unit_id', 'branch_id', 'model_code', 'date_added' (YYYY-MM-DD)}
    """

    def __init__(self, sales: Iterable[Dict[str, Any]], inventory: Iterable[Dict[str, Any]] = ()):
        self.bm = defaultdict(int)       # (branch, model, month) -> units
        self.net = defaultdict(int)      # (model, month) -> units
        self.branch = defaultdict(int)   # (branch, month) -> units
        self.sold_date: Dict[Any, str] = {}
        for s in sales:
            m = str(s['date'])[:7]
            b, k = s['branch_id'], s['model_code']
            if b is None or not k:
                continue
            self.bm[(b, k, m)] += 1
            self.net[(k, m)] += 1
            self.branch[(b, m)] += 1
            if s.get('unit_id') is not None:
                self.sold_date[s['unit_id']] = str(s['date'])[:10]

        self.units_by_bm = defaultdict(list)   # (branch, model) -> [(date_added, unit_id)]
        for u in inventory:
            if u.get('branch_id') is None or not u.get('model_code'):
                continue
            self.units_by_bm[(u['branch_id'], u['model_code'])].append(
                (str(u.get('date_added') or '2000-01-01')[:10], u.get('unit_id'))
            )

    def units(self, b, k, m) -> int:
        return self.bm.get((b, k, m), 0)

    def _window(self, table, key_prefix, m, n) -> int:
        return sum(table.get(key_prefix + (month_add(m, -i),), 0) for i in range(1, n + 1))

    def stock_start(self, b, k, m) -> int:
        start = m + '-01'
        return sum(
            1 for (added, uid) in self.units_by_bm.get((b, k), [])
            if added < start and (self.sold_date.get(uid) is None or self.sold_date[uid] >= start)
        )

    def features(self, b, k, m, brand: str, vehicle_category: str, base_price: float,
                 stock_override: Optional[int] = None) -> Dict[str, Any]:
        return {
            'lag_1': self._window(self.bm, (b, k), m, 1),
            'lag_3': self._window(self.bm, (b, k), m, 3),
            'lag_6': self._window(self.bm, (b, k), m, 6),
            'lag_12': self.units(b, k, month_add(m, -12)),
            'net_lag_3': self._window(self.net, (k,), m, 3),
            'branch_lag_3': self._window(self.branch, (b,), m, 3),
            'stock_start': self.stock_start(b, k, m) if stock_override is None else int(stock_override),
            'base_price': float(base_price),
            'season_month': int(m[5:7]),
            'brand': str(brand or 'UNKNOWN').upper(),
            'vehicle_category': str(vehicle_category or 'Scooter'),
        }
