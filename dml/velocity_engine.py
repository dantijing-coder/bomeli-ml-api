"""
Hierarchical Branch-Model Inventory Velocity & Stockout Engine.
DML Deployment Module for Bomeli Dealership.

Calculates individualized monthly sales velocity per vehicle model per branch,
using empirical Bayes shrinkage to blend local sales data with network-wide baselines.
"""

from datetime import date, datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any


@dataclass
class BranchModelVelocityStats:
    model_code: str
    brand: str
    branch_id: int
    branch_name: str
    available_stock: int
    committed_stock: int
    historical_sales_count: int
    monthly_velocity: float
    days_to_depletion: int
    avg_days_on_lot: float
    stock_status: str
    recommended_transfer: Optional[str] = None


class VelocityEngine:
    def __init__(
        self,
        lookback_days: int = 90,
        min_velocity_floor: float = 0.25,
        shrinkage_pseudocount: float = 2.0,
        critical_threshold_days: int = 14,
        warning_threshold_days: int = 30
    ):
        self.lookback_days = lookback_days
        self.min_velocity_floor = min_velocity_floor
        self.shrinkage_pseudocount = shrinkage_pseudocount
        self.critical_threshold_days = critical_threshold_days
        self.warning_threshold_days = warning_threshold_days

    def compute_branch_model_velocity(
        self,
        inventory: List[Dict[str, Any]],
        historical_sales: List[Dict[str, Any]],
        branches: List[Dict[str, Any]],
        as_of_date: Optional[date] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Computes granular inventory velocity per (branch, model).
        Returns: (velocity_list, transfer_recommendations)
        """
        today = as_of_date or date.today()
        branch_map = {b['branch_id']: b['name'] for b in branches}

        # 1. Aggregate historical sales per branch and model in lookback window
        lookback_months = max(self.lookback_days / 30.0, 1.0)
        sales_by_bm: Dict[Tuple[int, str], int] = {}
        sales_by_branch: Dict[int, int] = {}
        sales_by_model: Dict[str, int] = {}
        total_network_sales = 0

        for s in historical_sales:
            s_date = s.get('sale_date') or s.get('created_at')
            if isinstance(s_date, datetime):
                s_date = s_date.date()

            # Filter within lookback window if date is provided
            if s_date:
                days_ago = (today - s_date).days
                if days_ago > self.lookback_days or days_ago < 0:
                    continue

            b_id = s.get('branch_id')
            m_code = s.get('model_code', 'UNKNOWN')
            if not b_id or not m_code:
                continue

            sales_by_bm[(b_id, m_code)] = sales_by_bm.get((b_id, m_code), 0) + 1
            sales_by_branch[b_id] = sales_by_branch.get(b_id, 0) + 1
            sales_by_model[m_code] = sales_by_model.get(m_code, 0) + 1
            total_network_sales += 1

        # 2. Aggregate current inventory stats per (branch, model)
        inv_stats: Dict[Tuple[int, str], Dict[str, Any]] = {}
        for item in inventory:
            b_id = item.get('branch_id')
            m_code = item.get('model_code', 'UNKNOWN')
            if not b_id or not m_code:
                continue

            key = (b_id, m_code)
            if key not in inv_stats:
                inv_stats[key] = {
                    'model_code': m_code,
                    'brand': item.get('brand', 'UNKNOWN'),
                    'branch_id': b_id,
                    'branch_name': branch_map.get(b_id, f"Branch {b_id}"),
                    'available': 0,
                    'reserved': 0,
                    'released': 0,
                    'sold_inventory_units': 0,
                    'total_days_on_lot': 0,
                    'available_count': 0
                }

            st = str(item.get('status', '')).lower()
            if st == 'available':
                inv_stats[key]['available'] += 1
                inv_stats[key]['available_count'] += 1
                added = item.get('date_added')
                if added:
                    if isinstance(added, datetime):
                        added = added.date()
                    days = (today - added).days
                    inv_stats[key]['total_days_on_lot'] += max(0, days)
            elif st == 'reserved':
                inv_stats[key]['reserved'] += 1
            elif st == 'released':
                inv_stats[key]['released'] += 1
            elif st == 'sold':
                inv_stats[key]['sold_inventory_units'] += 1

        # Ensure all models with historical sales or inventory are evaluated
        all_keys = set(inv_stats.keys()).union(sales_by_bm.keys())

        velocity_list = []
        donor_pool = []
        recipient_pool = []

        for (b_id, m_code) in all_keys:
            st = inv_stats.get((b_id, m_code), {
                'model_code': m_code,
                'brand': 'UNKNOWN',
                'branch_id': b_id,
                'branch_name': branch_map.get(b_id, f"Branch {b_id}"),
                'available': 0,
                'reserved': 0,
                'released': 0,
                'total_days_on_lot': 0,
                'available_count': 0
            })

            # Check historical sales count
            n_sales = sales_by_bm.get((b_id, m_code), 0)
            if n_sales == 0 and st['sold_inventory_units'] > 0:
                n_sales = st['sold_inventory_units']

            # Local empirical velocity (units/month)
            v_local = n_sales / lookback_months

            # Global model velocity (units/month)
            model_total_sales = sales_by_model.get(m_code, 0)
            v_global_model = model_total_sales / lookback_months

            # Branch traffic weight (share of total dealership sales)
            branch_total_sales = sales_by_branch.get(b_id, 0)
            branch_weight = (branch_total_sales / max(total_network_sales, 1)) if total_network_sales > 0 else (1.0 / max(len(branches), 1))

            # Expected prior velocity for this branch-model combination
            v_prior = max(self.min_velocity_floor, v_global_model * branch_weight)

            # Hierarchical Bayesian Shrinkage Weight (lambda)
            # High sample size -> lambda near 1.0 (uses true local speed)
            # Low sample size -> lambda near 0.0 (smoothly blends with prior)
            shrinkage_lambda = n_sales / (n_sales + self.shrinkage_pseudocount)
            monthly_velocity = (shrinkage_lambda * v_local) + ((1.0 - shrinkage_lambda) * v_prior)
            monthly_velocity = max(self.min_velocity_floor, round(monthly_velocity, 2))

            avail = st['available']
            committed = st['reserved'] + st['released']
            # Committed units count as near-sold, soft buffer
            effective_avail = avail + (committed * 0.3)

            daily_velocity = monthly_velocity / 30.0
            if daily_velocity > 0 and (avail > 0 or committed > 0):
                days_to_depletion = round(effective_avail / daily_velocity)
            else:
                days_to_depletion = 999 if avail > 0 else 0
            days_to_depletion = min(days_to_depletion, 999)

            avg_dol = round(st['total_days_on_lot'] / max(st['available_count'], 1), 1)

            # Status classification
            if days_to_depletion == 0 and avail == 0:
                stock_status = 'Stockout'
            elif days_to_depletion <= self.critical_threshold_days:
                stock_status = 'Critical'
            elif days_to_depletion <= self.warning_threshold_days:
                stock_status = 'Low'
            elif days_to_depletion <= 45:
                stock_status = 'Watch'
            else:
                stock_status = 'OK'

            entry = {
                'model_code': m_code,
                'brand': st['brand'],
                'branch_id': b_id,
                'branch_name': st['branch_name'],
                'available_stock': avail,
                'committed_stock': committed,
                'historical_sales_count': n_sales,
                'monthly_velocity': monthly_velocity,
                'days_to_depletion': days_to_depletion,
                'avg_days_on_lot': avg_dol,
                'stock_status': stock_status,
                'recommended_transfer': None
            }
            velocity_list.append(entry)

            # Classify for cross-branch transfer pairing
            if days_to_depletion <= self.warning_threshold_days and (avail + committed) > 0:
                recipient_pool.append(entry)
            elif avail >= 2 and avg_dol >= 30:
                donor_pool.append(entry)

        # 3. Intelligent Cross-Branch Transfer Matching
        # Pair critical branches with donor branches having excess/slow-moving stock of the SAME model
        transfer_recs = []
        for rec in recipient_pool:
            m_code = rec['model_code']
            rec_b_id = rec['branch_id']
            rec_b_name = rec['branch_name']

            matching_donors = [
                d for d in donor_pool
                if d['model_code'] == m_code and d['branch_id'] != rec_b_id and d['available_stock'] >= 2
            ]

            if matching_donors:
                # Pick donor with highest days on lot or most excess stock
                donor = max(matching_donors, key=lambda d: d['avg_days_on_lot'])
                transfer_qty = min(2, donor['available_stock'] - 1)
                if transfer_qty > 0:
                    transfer_msg = (
                        f"Move {transfer_qty} unit{'s' if transfer_qty > 1 else ''} of {m_code} "
                        f"from {donor['branch_name']} (low velocity / {donor['avg_days_on_lot']:.0f}d on lot) "
                        f"to {rec_b_name} ({rec['days_to_depletion']}d until stockout)"
                    )
                    rec['recommended_transfer'] = transfer_msg
                    transfer_recs.append({
                        'model_code': m_code,
                        'source_branch_id': donor['branch_id'],
                        'source_branch_name': donor['branch_name'],
                        'target_branch_id': rec_b_id,
                        'target_branch_name': rec_b_name,
                        'transfer_quantity': transfer_qty,
                        'reason': transfer_msg
                    })

        # Sort by urgency: Critical/Low first, then days to depletion ascending
        urgency_order = {'Stockout': 0, 'Critical': 1, 'Low': 2, 'Watch': 3, 'OK': 4}
        velocity_list.sort(key=lambda x: (urgency_order.get(x['stock_status'], 5), x['days_to_depletion']))

        return velocity_list, transfer_recs
