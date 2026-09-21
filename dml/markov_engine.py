"""
Delinquency Roll-Rate & Borrower Reliability Engine.
DML Deployment Module for Bomeli Dealership.

Computes transition roll-rates:
- P(Active -> Delinquent) [At Risk of Delay]
- P(Delinquent -> Default) [Repossession Risk]
- P(Cure -> Active) [Expected to Catch Up]
Blends empirical live loan behavior with calibrated baseline priors.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass
class RollRateTransitionMatrix:
    p_active_to_delinquent: float
    p_delinquent_to_default: float
    p_cure_to_active: float
    blended_source: str
    sample_size: int


class MarkovEngine:
    DEFAULT_BASELINE = {
        'active_to_delinquent': 0.060,
        'delinquent_to_default': 0.225,
        'cure_to_active': 0.700
    }

    def __init__(self, baseline_matrix: Optional[Dict[str, float]] = None):
        self.baseline = baseline_matrix or self.DEFAULT_BASELINE

    def compute_roll_rates(
        self,
        accounts: List[Dict[str, Any]]
    ) -> RollRateTransitionMatrix:
        """
        Computes empirical roll-rate transition probabilities from a list of accounts,
        blending with baseline prior based on sample size.
        """
        n = len(accounts)
        if n == 0:
            return RollRateTransitionMatrix(
                p_active_to_delinquent=0.0,
                p_delinquent_to_default=0.0,
                p_cure_to_active=0.0,
                blended_source='No Accounts (Clean State)',
                sample_size=0
            )

        active_cnt = sum(1 for a in accounts if a.get('status') == 'active')
        delinq_cnt = sum(1 for a in accounts if a.get('status') == 'delinquent')
        default_cnt = sum(1 for a in accounts if a.get('status') == 'defaulted')

        # Accounts with overdue count > 0 among active
        at_risk_active = sum(1 for a in accounts if a.get('status') == 'active' and int(a.get('overdue_count', 0)) > 0)
        
        # Accounts with severe arrears (overdue >= 2 terms)
        severe_delinq = sum(1 for a in accounts if a.get('status') == 'delinquent' and int(a.get('overdue_count', 0)) >= 2)

        # Accounts that recently paid an overdue term
        cured_cnt = sum(1 for a in accounts if a.get('status') == 'delinquent' and float(a.get('total_paid_sum', 0)) > 0)

        # Empirical probabilities
        p_act_del = (at_risk_active / max(active_cnt, 1)) if active_cnt > 0 else self.baseline['active_to_delinquent']
        p_del_def = (severe_delinq / max(delinq_cnt, 1)) if delinq_cnt > 0 else self.baseline['delinquent_to_default']
        p_cure = (cured_cnt / max(delinq_cnt, 1)) if delinq_cnt > 0 else self.baseline['cure_to_active']

        # Determine blending weight
        if n >= 5:
            w_live = 0.80
            source_lbl = f'Live Data Dominant ({n} accounts)'
        elif n >= 2:
            w_live = 0.40
            source_lbl = f'Partially Blended ({n} accounts)'
        else:
            w_live = 0.15
            source_lbl = f'Baseline Prior Guided ({n} account)'

        p_a_d = round((w_live * p_act_del) + ((1.0 - w_live) * self.baseline['active_to_delinquent']), 4)
        p_d_f = round((w_live * p_del_def) + ((1.0 - w_live) * self.baseline['delinquent_to_default']), 4)
        p_c_a = round((w_live * p_cure) + ((1.0 - w_live) * self.baseline['cure_to_active']), 4)

        return RollRateTransitionMatrix(
            p_active_to_delinquent=p_a_d,
            p_delinquent_to_default=p_d_f,
            p_cure_to_active=p_c_a,
            blended_source=source_lbl,
            sample_size=n
        )

    def compute_branch_breakdown(
        self,
        accounts_by_branch: Dict[int, List[Dict[str, Any]]],
        branches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Computes per-branch transition roll-rates for inclusion in consolidated views.
        """
        branch_map = {b['branch_id']: b['name'] for b in branches}
        breakdown = []

        for b in branches:
            b_id = b['branch_id']
            b_name = b['name']
            b_accs = accounts_by_branch.get(b_id, [])
            rr = self.compute_roll_rates(b_accs)

            breakdown.append({
                'branch_id': b_id,
                'branch_name': b_name,
                'account_count': len(b_accs),
                'active_count': sum(1 for a in b_accs if a.get('status') == 'active'),
                'delinquent_count': sum(1 for a in b_accs if a.get('status') == 'delinquent'),
                'p_active_to_delinquent': round(rr.p_active_to_delinquent * 100.0, 1),
                'p_delinquent_to_default': round(rr.p_delinquent_to_default * 100.0, 1),
                'p_cure_to_active': round(rr.p_cure_to_active * 100.0, 1),
                'data_source': rr.blended_source
            })

        return breakdown
