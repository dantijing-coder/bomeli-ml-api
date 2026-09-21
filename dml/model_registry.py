"""
Model Registry & Inference Pipeline.
DML Deployment Module for Bomeli Dealership.

Handles loading, validating, and executing inference pipelines for:
- Model 1: 90-Day Default Hazard Classifier (Calibrated)
- Model 3: Option Contract Early Settlement Propensity Classifier
- Macro Action Recommendation Dispatcher
"""

import os
import json
from datetime import datetime, date
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class ModelRegistry:
    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, 'models')
        self.models_dir = models_dir

        self.default_model = None
        self.early_settlement_model = None
        self.markov_matrix = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads serialized models if present on disk, with self-healing fallback."""
        def_path = os.path.join(self.models_dir, 'default_hazard_model.joblib')
        if os.path.exists(def_path):
            try:
                self.default_model = joblib.load(def_path)
            except Exception as e:
                print(f"[registry] Warning: Failed to load default hazard model: {e}")
                self.default_model = None

        early_path = os.path.join(self.models_dir, 'early_settlement_model.joblib')
        if os.path.exists(early_path):
            try:
                self.early_settlement_model = joblib.load(early_path)
            except Exception as e:
                print(f"[registry] Warning: Failed to load early settlement model: {e}")
                self.early_settlement_model = None

        markov_path = os.path.join(self.models_dir, 'markov_matrix.json')
        if os.path.exists(markov_path):
            try:
                with open(markov_path, 'r') as f:
                    self.markov_matrix = json.load(f)
            except Exception as e:
                print(f"[registry] Warning: Failed to load Markov matrix: {e}")
                self.markov_matrix = None

        # Self-healing fallback: If models failed to load or are incompatible with runtime scikit-learn,
        # automatically re-train in-process so service operates without interruption.
        if self.default_model is None or self.early_settlement_model is None:
            try:
                print("[registry] Models missing or version incompatible. Auto-calibrating in-process...")
                import train_models
                df = train_models.load_simulated_dataset()
                if df is not None:
                    train_models.train_default_hazard_model(df, target_accuracy=0.81)
                    train_models.train_early_settlement_model(df, target_accuracy=0.81)
                    train_models.compute_markov_roll_rates(df)
                    if os.path.exists(def_path):
                        self.default_model = joblib.load(def_path)
                    if os.path.exists(early_path):
                        self.early_settlement_model = joblib.load(early_path)
                    if os.path.exists(markov_path):
                        with open(markov_path, 'r') as f:
                            self.markov_matrix = json.load(f)
                    print("[registry] Auto-calibration complete. All models loaded and ready!")
            except Exception as train_err:
                print(f"[registry] Warning: Auto-calibration failed: {train_err}")

    def extract_features(self, account: Dict[str, Any]) -> pd.DataFrame:
        """Extracts engineered feature vector for Model 1 and Model 3."""
        app_data = account.get('application_data', {})
        if isinstance(app_data, str):
            try:
                app_data = json.loads(app_data)
            except:
                app_data = {}

        income = 25000.0
        for k in ['totalIncome', 'monthly_gross_income', 'salary']:
            val = app_data.get(k)
            if val:
                try:
                    clean_val = float(str(val).replace(',', ''))
                    if clean_val > 0:
                        income = clean_val
                        break
                except:
                    pass

        try:
            amort = float(account.get('monthly_amortization') or 3000.0)
        except:
            amort = 3000.0
        dti = round(amort / max(income, 1000.0), 4)

        try:
            years_res = float(app_data.get('lengthOfStayYears') or app_data.get('years_at_address') or 3.0)
        except:
            years_res = 3.0

        try:
            years_emp = float(app_data.get('serviceYears') or app_data.get('years_employed') or 2.0)
        except:
            years_emp = 2.0

        try:
            overdue_cnt = int(account.get('overdue_count') or 0)
        except:
            overdue_cnt = 0

        try:
            paid_cnt = int(account.get('paid_terms_count') or 0)
        except:
            paid_cnt = 0

        try:
            arrears = float(account.get('total_arrears') or 0.0)
        except:
            arrears = 0.0

        try:
            term_mo = int(account.get('term_months') or 24)
        except:
            term_mo = 24

        emp_type = str(app_data.get('employment_status') or app_data.get('employment_status_choice') or 'Regular')
        res_type = str(app_data.get('residential_ownership') or app_data.get('residential_ownership_choice') or 'Owned')
        ci_rem = str(account.get('ci_status') or 'approved').lower()

        # Rebate streak and behavioral heuristics
        rebate_streak = max(0, min(paid_cnt - overdue_cnt, 12)) if overdue_cnt == 0 else 0
        term_progress = round(paid_cnt / max(term_mo, 1), 3)
        partial_ratio = 1.0 if overdue_cnt == 0 else max(0.2, 1.0 - (overdue_cnt * 0.25))
        on_time_rel = round(max(0.1, 1.0 - (overdue_cnt / max(paid_cnt + overdue_cnt, 1))), 3)
        ci_flags = 1 if overdue_cnt >= 2 or ci_rem not in ['approved', 'clean'] else 0
        phone_bounce = 0

        df = pd.DataFrame([{
            # Features for Model 1 (Default Hazard Pipeline)
            'term_progress_ratio': term_progress,
            'dti_ratio': dti,
            'rebate_streak': rebate_streak,
            'late_penalty_streak': overdue_cnt,
            'overdue_count': overdue_cnt,
            'partial_payment_ratio': partial_ratio,
            'on_time_reliability': on_time_rel,
            'ci_negative_flags': ci_flags,
            'has_phone_bounce': phone_bounce,
            'length_of_stay_years': years_res,
            'employment_type': emp_type,
            'residential_ownership': res_type,
            # Features for Model 3 (Early Settlement)
            'monthly_income': income,
            # Legacy/Convenience fields
            'residency_type': res_type,
            'years_at_residence': years_res,
            'years_at_employment': years_emp,
            'term_months': term_mo,
            'monthly_amortization': amort,
            'has_partial_payment': 1 if arrears > 0 and arrears < amort else 0,
            'current_overdue_count': overdue_cnt,
            'total_arrears_amount': arrears,
            'ci_remarks': 'clean' if ci_rem == 'approved' else 'minor_notes',
            'season_month': datetime.now().month,
            'paid_terms_count': paid_cnt
        }])

        return df

    def predict_default_hazard(self, account: Dict[str, Any]) -> Tuple[float, str]:
        """
        Infers 90-Day Default Hazard probability and assigns hazard tier.
        Returns: (probability, hazard_tier)
        """
        if self.default_model is not None:
            try:
                df = self.extract_features(account)
                prob = float(self.default_model.predict_proba(df)[0][1])
            except Exception as e:
                prob = self._heuristic_default_prob(account)
        else:
            prob = self._heuristic_default_prob(account)

        prob = min(max(prob, 0.001), 0.999)

        if prob >= 0.70:
            tier = 'critical'
        elif prob >= 0.40:
            tier = 'high'
        elif prob >= 0.15:
            tier = 'moderate'
        else:
            tier = 'low'

        return round(prob, 4), tier

    def predict_early_settlement(self, account: Dict[str, Any]) -> Tuple[float, bool]:
        """
        Infers Option Contract Early Settlement propensity.
        Returns: (propensity_score, is_candidate)
        """
        if self.early_settlement_model is not None:
            try:
                df = self.extract_features(account)
                prob = float(self.early_settlement_model.predict_proba(df)[0][1])
            except:
                prob = self._heuristic_early_prob(account)
        else:
            prob = self._heuristic_early_prob(account)

        prob = round(min(max(prob, 0.0), 1.0), 4)
        paid_cnt = int(account.get('paid_terms_count', 0))
        is_candidate = bool(prob >= 0.65 and paid_cnt >= 6)

        return prob, is_candidate

    def assign_macro_action(
        self,
        account: Dict[str, Any],
        hazard_tier: str,
        is_early_candidate: bool
    ) -> Tuple[str, str]:
        """
        Assigns tactical operational macro action for branch staff.
        Returns: (macro_type, macro_label)
        """
        overdue_cnt = int(account.get('overdue_count', 0))
        arrears = float(account.get('total_arrears', 0.0))
        cust_name = account.get('customer_name', 'Borrower')

        if hazard_tier == 'critical' or overdue_cnt >= 3:
            return 'FIELD_CHECK', f"Dispatch field verification for {cust_name} ({overdue_cnt} terms overdue, ₱{arrears:,.2f} arrears)"
        elif hazard_tier == 'high' or overdue_cnt == 2:
            return 'SMS_INTERVENTION', f"Send formal demand notice to {cust_name} (₱{arrears:,.2f} past due)"
        elif hazard_tier == 'moderate' or overdue_cnt == 1:
            return 'SMS_REMINDER', f"Send gentle SMS payment reminder to {cust_name}"
        elif is_early_candidate:
            return 'EARLY_SETTLEMENT', f"Offer Option Contract early settlement buyout discount to {cust_name}"
        else:
            return 'ROUTINE_SERVICING', f"Routine account servicing — on track"

    def _heuristic_default_prob(self, account: Dict[str, Any]) -> float:
        try:
            overdue = int(account.get('overdue_count') or 0)
        except:
            overdue = 0
        if overdue >= 3:
            return 0.85
        elif overdue == 2:
            return 0.55
        elif overdue == 1:
            return 0.25
        return 0.015

    def _heuristic_early_prob(self, account: Dict[str, Any]) -> float:
        try:
            paid = int(account.get('paid_terms_count') or 0)
            overdue = int(account.get('overdue_count') or 0)
        except:
            paid, overdue = 0, 0
        if overdue == 0 and paid in [11, 12, 13, 23, 24, 25]:
            return 0.82
        elif overdue == 0 and paid >= 6:
            return 0.55
        return 0.10