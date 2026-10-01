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


CATALOG_DEFAULTS = {
    'CLICK 125I': {'brand': 'HONDA', 'category': 'Scooter', 'price': 81400.0},
    'CLICK 160': {'brand': 'HONDA', 'category': 'Scooter', 'price': 122900.0},
    'PCX160': {'brand': 'HONDA', 'category': 'Maxi-Scooter', 'price': 134900.0},
    'ADV160': {'brand': 'HONDA', 'category': 'Adventure Scooter', 'price': 166900.0},
    'BEAT PREMIUM': {'brand': 'HONDA', 'category': 'Scooter', 'price': 72400.0},
    'BEAT': {'brand': 'HONDA', 'category': 'Scooter', 'price': 72400.0},
    'TMX125 ALPHA': {'brand': 'HONDA', 'category': 'Backbone/Utility', 'price': 57900.0},
    'TMX125': {'brand': 'HONDA', 'category': 'Backbone/Utility', 'price': 57900.0},
    'WAVE RSX': {'brand': 'HONDA', 'category': 'Underbone', 'price': 64900.0},
    'P24KK (BARAKO II)': {'brand': 'KAWASAKI', 'category': 'Backbone/Tricycle', 'price': 68500.0},
    'BARAKO': {'brand': 'KAWASAKI', 'category': 'Backbone/Tricycle', 'price': 68500.0},
    'CT100B': {'brand': 'KAWASAKI', 'category': 'Backbone/Utility', 'price': 54000.0},
    'ROUSER NS125': {'brand': 'KAWASAKI', 'category': 'Sport Backbone', 'price': 82000.0},
    'MIO SPORTY': {'brand': 'YAMAHA', 'category': 'Scooter', 'price': 73900.0},
    'MIO GEAR 125': {'brand': 'YAMAHA', 'category': 'Scooter', 'price': 79400.0},
    'MIO FAZZIO': {'brand': 'YAMAHA', 'category': 'Retro Scooter', 'price': 93900.0},
    'NMAX 155': {'brand': 'YAMAHA', 'category': 'Maxi-Scooter', 'price': 151900.0},
    'NMAX': {'brand': 'YAMAHA', 'category': 'Maxi-Scooter', 'price': 151900.0},
    'AEROX 155': {'brand': 'YAMAHA', 'category': 'Sport Scooter', 'price': 125400.0},
    'AEROX': {'brand': 'YAMAHA', 'category': 'Sport Scooter', 'price': 125400.0},
    'SNIPER 155': {'brand': 'YAMAHA', 'category': 'Sport Underbone', 'price': 123900.0},
    'RAIDER 150 FI': {'brand': 'SUZUKI', 'category': 'Sport Underbone', 'price': 119900.0},
    'RAIDER 150': {'brand': 'SUZUKI', 'category': 'Sport Underbone', 'price': 119900.0},
    'SMASH 115': {'brand': 'SUZUKI', 'category': 'Underbone', 'price': 62400.0},
    'BURGMAN STREET 125': {'brand': 'SUZUKI', 'category': 'Maxi-Scooter', 'price': 83400.0},
}


def resolve_model_specs(model_code: str, fallback_brand: str = 'UNKNOWN') -> Dict[str, Any]:
    code_clean = (model_code or '').strip().upper()
    if code_clean in CATALOG_DEFAULTS:
        return CATALOG_DEFAULTS[code_clean]
    for k, v in CATALOG_DEFAULTS.items():
        if k in code_clean or code_clean in k:
            return v
    brand = fallback_brand if fallback_brand and fallback_brand != 'UNKNOWN' else 'HONDA'
    if any(tag in code_clean for tag in ['160', 'NMAX', 'PCX', 'ADV']):
        return {'brand': brand, 'category': 'Maxi-Scooter', 'price': 135000.0}
    elif any(tag in code_clean for tag in ['125', 'MIO', 'BEAT', 'CLICK']):
        return {'brand': brand, 'category': 'Scooter', 'price': 82000.0}
    elif any(tag in code_clean for tag in ['TMX', 'CT', 'BARAKO', 'P24']):
        return {'brand': brand, 'category': 'Backbone/Utility', 'price': 58000.0}
    return {'brand': brand, 'category': 'Commuter', 'price': 75000.0}


class VelocityEngine:
    def __init__(
        self,
        lookback_days: int = 90,
        min_velocity_floor: float = 0.25,
        shrinkage_pseudocount: float = 2.0,
        critical_threshold_days: int = 14,
        warning_threshold_days: int = 30,
        model_registry: Optional[Any] = None
    ):
        self.lookback_days = lookback_days
        self.min_velocity_floor = min_velocity_floor
        self.shrinkage_pseudocount = shrinkage_pseudocount
        self.critical_threshold_days = critical_threshold_days
        self.warning_threshold_days = warning_threshold_days

        if model_registry is not None:
            self.model_registry = model_registry
        else:
            try:
                from dml.model_registry import ModelRegistry
                self.model_registry = ModelRegistry()
            except Exception:
                self.model_registry = None

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

        # Stock Velocity v2 panel (full history, not just the lookback window)
        use_v2 = bool(self.model_registry and getattr(self.model_registry, 'velocity_model_version', 1) >= 2)
        if use_v2:
            from dml.velocity_features import SalesPanel
            v2_panel = SalesPanel(sales=[
                {'branch_id': s.get('branch_id'), 'model_code': s.get('model_code'),
                 'date': str(s.get('sale_date') or s.get('created_at'))[:10]}
                for s in historical_sales if (s.get('sale_date') or s.get('created_at'))
            ])
            current_month = today.strftime('%Y-%m')

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

            # Resolve vehicle specifications (brand, category, base price)
            specs = resolve_model_specs(m_code, st.get('brand', 'UNKNOWN'))
            brand_resolved = st.get('brand') if st.get('brand') and st['brand'] != 'UNKNOWN' else specs['brand']
            category = specs['category']
            base_price = specs['price']
            avg_dol = round(st['total_days_on_lot'] / max(st['available_count'], 1), 1)

            # Check historical sales count
            n_sales = sales_by_bm.get((b_id, m_code), 0)
            if n_sales == 0 and st.get('sold_inventory_units', 0) > 0:
                n_sales = st['sold_inventory_units']

            # Global model velocity (units/month)
            model_total_sales = sales_by_model.get(m_code, 0)
            v_global_model = model_total_sales / lookback_months

            # Branch traffic weight (share of total dealership sales)
            branch_total_sales = sales_by_branch.get(b_id, 0)
            branch_weight = (branch_total_sales / max(total_network_sales, 1)) if total_network_sales > 0 else (1.0 / max(len(branches), 1))

            # Expected prior velocity for this branch-model combination
            v_prior = max(self.min_velocity_floor, v_global_model * branch_weight)

            if use_v2:
                # Poisson GBM on lag features; stock = units on the floor right now
                feats = v2_panel.features(b_id, m_code, current_month, brand_resolved, category, base_price,
                                          stock_override=st['available'] + st['reserved'])
                monthly_velocity = round(self.model_registry.predict_velocity(feats), 3)
            # Zero-sales network floor: If model has 0 sales across entire network, assign conservative floor
            elif n_sales == 0 and model_total_sales == 0:
                monthly_velocity = self.min_velocity_floor
            elif self.model_registry and self.model_registry.inventory_velocity_model:
                pred_vel = self.model_registry.predict_inventory_velocity(
                    brand=brand_resolved,
                    vehicle_category=category,
                    base_price=base_price,
                    branch_name=st['branch_name'],
                    avg_days_on_lot=avg_dol,
                    trailing_sales_count=n_sales,
                    season_month=today.month
                )
                monthly_velocity = max(self.min_velocity_floor, round(pred_vel, 2))
            else:
                # Hierarchical Bayesian Shrinkage fallback
                v_local = n_sales / lookback_months
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
            # 999 is the dashboard's "no stock" sentinel; slow movers with stock cap at 998
            days_to_depletion = min(days_to_depletion, 998 if avail > 0 else 999)

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
                'brand': brand_resolved,
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
