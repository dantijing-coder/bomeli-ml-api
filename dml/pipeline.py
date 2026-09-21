"""
Predictive Forecasting Pipeline Orchestrator.
DML Deployment Module for Bomeli Dealership.

Coordinates the entire start-of-month operational forecast generation.
"""

import json
from datetime import date, datetime
from typing import Dict, List, Optional, Any, Tuple

from .velocity_engine import VelocityEngine
from .cash_engine import CashEngine
from .markov_engine import MarkovEngine
from .model_registry import ModelRegistry


class PredictivePipeline:
    def __init__(
        self,
        velocity_engine: Optional[VelocityEngine] = None,
        cash_engine: Optional[CashEngine] = None,
        markov_engine: Optional[MarkovEngine] = None,
        model_registry: Optional[ModelRegistry] = None
    ):
        self.velocity_engine = velocity_engine or VelocityEngine()
        self.cash_engine = cash_engine or CashEngine()
        self.markov_engine = markov_engine or MarkovEngine()
        self.model_registry = model_registry or ModelRegistry()

    def run_inference_on_accounts(
        self,
        accounts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Scores all accounts with default hazard, early settlement, and macro action."""
        scored = []
        for acc in accounts:
            a = dict(acc)
            p_def, tier = self.model_registry.predict_default_hazard(a)
            p_early, is_early = self.model_registry.predict_early_settlement(a)
            macro_type, macro_label = self.model_registry.assign_macro_action(a, tier, is_early)

            a['default_probability'] = p_def
            a['hazard_tier'] = tier
            a['early_settlement_probability'] = p_early
            a['is_early_settlement_candidate'] = is_early
            a['macro_type'] = macro_type
            a['macro_label'] = macro_label
            scored.append(a)
        return scored

    def build_executive_summary(
        self,
        scope_name: str,
        cash_data: Dict[str, Any],
        roll_rates: Any,
        velocity_list: List[Dict[str, Any]],
        accounts: List[Dict[str, Any]]
    ) -> str:
        """Synthesizes clean, plain-language operational executive summary with zero jargon."""
        expected = cash_data.get('ai_expected_collected', 0.0)
        scheduled = cash_data.get('contractual_scheduled', 0.0)
        actual_mtd = cash_data.get('actual_collected_mtd', 0.0)
        proj_units = cash_data.get('projected_units_sold', 0)
        act_units = cash_data.get('actual_units_sold', 0)

        n_acc = len(accounts)
        n_act = sum(1 for a in accounts if a.get('status') == 'active')
        n_del = sum(1 for a in accounts if a.get('status') == 'delinquent')

        low_stock_models = [v['model_code'] for v in velocity_list if v.get('stock_status') in ['Critical', 'Low']]
        low_stock_str = ", ".join(low_stock_models[:3]) if low_stock_models else "None"

        if n_acc == 0 and scheduled == 0:
            return (
                f"For {scope_name}, there are currently no active financing accounts. "
                f"Month-to-date showroom sales stand at {act_units} units booked against a start-of-month target of {proj_units} units. "
                f"Cash collections recorded to date total ₱{actual_mtd:,.2f}."
            )

        return (
            f"For {scope_name}, our operational collection target for this month is ₱{expected:,.2f} "
            f"out of ₱{scheduled:,.2f} in scheduled installment payments across {n_acc} active customer accounts "
            f"({n_act} on time, {n_del} with past due terms). "
            f"Month-to-date collections stand at ₱{actual_mtd:,.2f}. "
            f"On the showroom floor, we have booked {act_units} units sold against our monthly target of {proj_units} units. "
            f"Showroom inventory alerts highlight {len(low_stock_models)} model(s) running low on stock ({low_stock_str})."
        )

    def build_risk_observations(
        self,
        accounts: List[Dict[str, Any]],
        velocity_list: List[Dict[str, Any]],
        transfer_recs: List[Dict[str, Any]]
    ) -> List[str]:
        """Synthesizes actionable bullet observations for dealership managers."""
        obs = []
        n_acc = len(accounts)

        crit_accounts = [a for a in accounts if a.get('hazard_tier') == 'critical']
        if crit_accounts:
            obs.append(
                f"{len(crit_accounts)} borrower account(s) have reached critical arrears (3+ terms past due). "
                f"Immediate field verification recommended before 90-day repossession threshold."
            )

        mod_accounts = [a for a in accounts if a.get('hazard_tier') in ['high', 'moderate']]
        if mod_accounts:
            obs.append(
                f"{len(mod_accounts)} account(s) show initial payment delays. Automated SMS reminders and follow-up calls will maintain on-time recovery."
            )

        early_cand = [a for a in accounts if a.get('is_early_settlement_candidate')]
        if early_cand:
            obs.append(
                f"{len(early_cand)} prime borrower(s) have completed over 6 consecutive terms with zero late penalties and qualify for Option Contract early settlement buyout discounts."
            )

        crit_stock = [v for v in velocity_list if v.get('stock_status') == 'Critical']
        if crit_stock:
            models = ", ".join(f"{v['model_code']} ({v['days_to_depletion']}d)" for v in crit_stock[:2])
            obs.append(f"Showroom stock depletion warning: {models} are within critical stockout windows.")

        if transfer_recs:
            rec = transfer_recs[0]
            obs.append(f"Inventory rebalancing opportunity: {rec.get('reason', '')}.")

        if not obs:
            obs.append("All portfolio metrics and showroom inventory levels are stable within standard operating benchmarks.")

        return obs
