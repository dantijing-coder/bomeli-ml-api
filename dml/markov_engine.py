"""
Delinquency Roll-Rate & Borrower Reliability Engine.
DML Deployment Module for Bomeli Dealership.

Computes transition roll-rates:
- P(Active -> Delinquent) [At Risk of Delay]
- P(Delinquent -> Default) [Repossession Risk]
- P(Cure -> Active) [Expected to Catch Up]

ML Integration (Model 9):
- MarkovEngine now loads markov_transition_model.joblib (3 binary HistGBT classifiers)
  trained on simulated_markov_transitions_dataset.csv (5,000 per-account transition records).
- When the model is loaded, per-account transition probabilities are predicted from
  observable borrower features (DTI, overdue_count, rebate_streak, on_time_reliability, etc.)
  and averaged across the portfolio to produce feature-aware transition rates.
- This replaces the original frequency-ratio approach:
    Statistical: P(A->D) = count(at_risk_active) / count(active)
    ML:          P(A->D) = mean(model.predict_proba(account_features)[:,1]) across active accounts
- Falls back gracefully to Bayesian-blended statistical rates if model not loaded.
- The markov_matrix.json artifact is retained as the cold-start reference.
"""

import os
import json
import joblib
import pandas as pd
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

    def __init__(self, baseline_matrix: Optional[Dict[str, Any]] = None, models_dir: Optional[str] = None):
        base = dict(self.DEFAULT_BASELINE)
        self.transition_matrix = None
        if baseline_matrix and isinstance(baseline_matrix, dict):
            for k, v in baseline_matrix.items():
                if k == 'transition_matrix' and isinstance(v, dict):
                    self.transition_matrix = v
                    continue
                if isinstance(v, (int, float, str)):
                    try:
                        f_val = float(v)
                        norm_k = k.replace('_prob', '')
                        base[norm_k] = f_val
                        base[k] = f_val
                    except (ValueError, TypeError):
                        pass
        self.baseline = base
        self._ml_model = None
        self._load_ml_model(models_dir)

    def _load_ml_model(self, models_dir: Optional[str] = None):
        """Loads the trained ML Markov transition model bundle if available."""
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, 'models')
        model_path = os.path.join(models_dir, 'markov_transition_model.joblib')
        if os.path.exists(model_path):
            try:
                self._ml_model = joblib.load(model_path)
            except Exception as e:
                print(f"[MarkovEngine] Warning: Failed to load markov_transition_model: {e}")
                self._ml_model = None

    def _extract_account_features(self, account: Dict[str, Any], season_month: int = 6) -> Optional[Dict[str, Any]]:
        """Extracts a feature dictionary for a single account matching the training schema."""
        try:
            app_data = account.get('application_data', {}) or {}
            if isinstance(app_data, str):
                try:
                    app_data = json.loads(app_data)
                except Exception:
                    app_data = {}

            monthly_income = 25000.0
            for k in ['totalIncome', 'monthly_gross_income', 'salary', 'monthly_income']:
                val = app_data.get(k) or account.get(k)
                if val:
                    try:
                        v = float(str(val).replace(',', ''))
                        if v > 0:
                            monthly_income = v
                            break
                    except Exception:
                        pass

            amort       = float(account.get('monthly_amortization', account.get('monthly_amort', 3000.0)) or 3000.0)
            term_months = int(account.get('term_months', account.get('terms', 24)) or 24)
            paid_terms  = int(account.get('paid_terms_count', account.get('terms_paid', 0)) or 0)
            overdue     = int(account.get('overdue_count', 0) or 0)
            curr_term   = paid_terms + overdue + 1
            prog        = round(curr_term / max(term_months, 1), 4)
            dti         = round(amort / max(monthly_income, 1000.0), 4)
            arrears     = float(account.get('total_arrears', 0.0) or 0.0)

            return {
                'term_months':           term_months,
                'term_progress_ratio':   prog,
                'dti_ratio':             dti,
                'on_time_reliability':   float(account.get('on_time_reliability', 0.80) or 0.80),
                'overdue_count':         overdue,
                'rebate_streak':         int(account.get('rebate_streak', 0) or 0),
                'partial_payment_ratio': float(account.get('partial_payment_ratio', 0.0) or 0.0),
                'total_arrears':         arrears,
                'ci_negative_flags':     int(account.get('ci_negative_flags', 0) or 0),
                'has_phone_bounce':      int(account.get('has_phone_bounce', 0) or 0),
                'length_of_stay_years':  float(app_data.get('lengthOfStayYears', app_data.get('years_at_address', 3.0)) or 3.0),
                'dependents_count':      int(app_data.get('dependents_count', 0) or 0),
                'season_month':          int(season_month),
                'employment_type':       str(app_data.get('employment_status', 'Employed (Private/Gov)') or 'Employed (Private/Gov)'),
                'residential_ownership': str(app_data.get('residential_ownership', 'Owned (Clean Title)') or 'Owned (Clean Title)'),
            }
        except Exception:
            return None

    def _build_account_features(self, account: Dict[str, Any], season_month: int = 6) -> Optional[pd.DataFrame]:
        """Constructs a feature row for a single account matching the training schema."""
        feat = self._extract_account_features(account, season_month)
        return pd.DataFrame([feat]) if feat else None

    def _ml_compute_rates(
        self,
        active_accounts: List[Dict[str, Any]],
        delinq_accounts: List[Dict[str, Any]],
        season_month: int = 6
    ) -> Optional[Dict[str, float]]:
        """
        Uses Model 4 to compute per-account transition probabilities
        in vectorized batches and average them across each cohort.

        Returns dict with keys: active_to_delinquent, delinquent_to_default, cure_to_active
        or None if ML inference fails.
        """
        if self._ml_model is None:
            return None

        slip_clf = self._ml_model.get('slip_classifier')
        roll_clf = self._ml_model.get('roll_classifier')
        cure_clf = self._ml_model.get('cure_classifier')

        try:
            # 4a: P(Active -> Delinquent) — batched average across active accounts
            p_slip = None
            if active_accounts and slip_clf is not None:
                active_rows = [self._extract_account_features(a, season_month) for a in active_accounts]
                active_rows = [r for r in active_rows if r is not None]
                if active_rows:
                    df_act = pd.DataFrame(active_rows)
                    probs = slip_clf.predict_proba(df_act)[:, 1]
                    p_slip = float(probs.mean())

            # 4b + 4c: P(Delinquent -> Default) and P(Delinquent -> Cured)
            p_roll = None
            p_cure = None
            if delinq_accounts and (roll_clf is not None or cure_clf is not None):
                delinq_rows = [self._extract_account_features(a, season_month) for a in delinq_accounts]
                delinq_rows = [r for r in delinq_rows if r is not None]
                if delinq_rows:
                    df_del = pd.DataFrame(delinq_rows)
                    if roll_clf is not None:
                        p_roll = float(roll_clf.predict_proba(df_del)[:, 1].mean())
                    if cure_clf is not None:
                        p_cure = float(cure_clf.predict_proba(df_del)[:, 1].mean())

            if p_slip is None and p_roll is None and p_cure is None:
                return None

            base_ad = self.baseline.get('active_to_delinquent', 0.060)
            base_dd = self.baseline.get('delinquent_to_default', 0.225)
            base_ca = self.baseline.get('cure_to_active', 0.700)
            return {
                'active_to_delinquent': round(max(0.01, min(0.60, p_slip if p_slip is not None else base_ad)), 4),
                'delinquent_to_default': round(max(0.01, min(0.95, p_roll if p_roll is not None else base_dd)), 4),
                'cure_to_active': round(max(0.05, min(0.95, p_cure if p_cure is not None else base_ca)), 4),
            }
        except Exception as e:
            print(f"[MarkovEngine] ML rate inference failed: {e}")
            return None

    def compute_roll_rates(
        self,
        accounts: List[Dict[str, Any]],
        season_month: int = 6
    ) -> RollRateTransitionMatrix:
        """
        Computes transition roll-rate probabilities.

        When Model 4 is loaded: uses per-account ML predictions averaged across
        active and delinquent cohorts (feature-aware transition rates).

        When model is absent or inference fails: falls back to Bayesian-blended
        empirical frequency ratios (the original statistical method).
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

        def _get_status(a):
            return str(a.get('current_status') or a.get('account_status') or a.get('status') or 'active').lower()

        active_accs = [a for a in accounts if _get_status(a) == 'active']
        delinq_accs = [a for a in accounts if _get_status(a) in ('delinquent', 'pre_repossession', 'overdue')]

        # -- Try ML inference first (Model 4) --
        if self._ml_model is not None and (active_accs or delinq_accs):
            ml_rates = self._ml_compute_rates(active_accs, delinq_accs, season_month)
            if ml_rates is not None:
                return RollRateTransitionMatrix(
                    p_active_to_delinquent=ml_rates['active_to_delinquent'],
                    p_delinquent_to_default=ml_rates['delinquent_to_default'],
                    p_cure_to_active=ml_rates['cure_to_active'],
                    blended_source=f'ML Markov Model ({n} accounts, season={season_month})',
                    sample_size=n
                )

        # -- Statistical fallback: Bayesian-blended frequency ratios --
        active_cnt  = len(active_accs)
        delinq_cnt  = len(delinq_accs)

        at_risk_active = sum(1 for a in active_accs if int(a.get('overdue_count', 0)) > 0)
        severe_delinq  = sum(1 for a in delinq_accs if int(a.get('overdue_count', 0)) >= 2)
        cured_cnt      = sum(1 for a in delinq_accs if float(a.get('total_paid_sum', 0)) > 0)

        base_ad = self.baseline.get('active_to_delinquent', 0.060)
        base_dd = self.baseline.get('delinquent_to_default', 0.225)
        base_ca = self.baseline.get('cure_to_active', 0.700)

        p_act_del = (at_risk_active / max(active_cnt, 1)) if active_cnt > 0 else base_ad
        p_del_def = (severe_delinq / max(delinq_cnt, 1)) if delinq_cnt > 0 else base_dd
        p_cure    = (cured_cnt / max(delinq_cnt, 1)) if delinq_cnt > 0 else base_ca

        if n >= 5:
            w_live, source_lbl = 0.80, f'Statistical Blend ({n} accounts)'
        elif n >= 2:
            w_live, source_lbl = 0.40, f'Statistical Partially Blended ({n} accounts)'
        else:
            w_live, source_lbl = 0.15, f'Baseline Prior Guided ({n} account)'

        p_a_d = round((w_live * p_act_del) + ((1.0 - w_live) * base_ad), 4)
        p_d_f = round((w_live * p_del_def) + ((1.0 - w_live) * base_dd), 4)
        p_c_a = round((w_live * p_cure) + ((1.0 - w_live) * base_ca), 4)

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
        branches: List[Dict[str, Any]],
        season_month: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Computes per-branch transition roll-rates for inclusion in consolidated views.
        Passes season_month to compute_roll_rates for seasonality-aware ML inference.
        """
        breakdown = []
        for b in branches:
            b_id   = b['branch_id']
            b_name = b['name']
            b_accs = accounts_by_branch.get(b_id, [])
            rr     = self.compute_roll_rates(b_accs, season_month=season_month)

            breakdown.append({
                'branch_id':                  b_id,
                'branch_name':                b_name,
                'account_count':              len(b_accs),
                'active_count':               sum(1 for a in b_accs if a.get('status') == 'active'),
                'delinquent_count':           sum(1 for a in b_accs if a.get('status') == 'delinquent'),
                'p_active_to_delinquent':     round(rr.p_active_to_delinquent * 100.0, 1),
                'p_delinquent_to_default':    round(rr.p_delinquent_to_default * 100.0, 1),
                'p_cure_to_active':           round(rr.p_cure_to_active * 100.0, 1),
                'data_source':                rr.blended_source
            })

        return breakdown
