"""
4-Tier Payment Stream & Dynamic Cash Flow Forecasting Engine.
DML Deployment Module for Bomeli Dealership.

Isolates Regular Amortization, Advance Prepayments, Option Contract Early Buyouts,
and Partials, and produces realistic non-linear 3-month cash inflow projection cones
accounting for multi-term cohort decay (12, 24, 36 months), regional seasonality,
holiday bonus surges (13th-month pay), and forecasted showroom new sales bookings.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dateutil.relativedelta import relativedelta


# Calibrated Philippine retail motorcycle empirical monthly seasonality indices
PHILIPPINE_MOTORCYCLE_SEASONALITY = {
    1: {'index': 0.94, 'name': 'Jan', 'note': 'Post-Holiday Liquidity Tightening (-6%)'},
    2: {'index': 0.97, 'name': 'Feb', 'note': 'Post-Holiday Liquidity Recovery'},
    3: {'index': 1.03, 'name': 'Mar', 'note': 'Graduation & Dry-Season Commercial (+3%)'},
    4: {'index': 1.05, 'name': 'Apr', 'note': 'Summer Commercial Activity Peak (+5%)'},
    5: {'index': 1.02, 'name': 'May', 'note': 'Mid-Year Commercial Activity (+2%)'},
    6: {'index': 0.96, 'name': 'Jun', 'note': 'School Tuition Outflows Dip (-4%)'},
    7: {'index': 0.98, 'name': 'Jul', 'note': 'Mid-Year Monsoon Season'},
    8: {'index': 0.97, 'name': 'Aug', 'note': 'Monsoon Commercial Lull (-3%)'},
    9: {'index': 1.01, 'name': 'Sep', 'note': 'Start of "Ber" Months Pre-Holiday Demand'},
    10: {'index': 1.04, 'name': 'Oct', 'note': 'Regional Harvest Liquidity (+4%)'},
    11: {'index': 1.14, 'name': 'Nov', 'note': '13th-Month Pay Early Bonus Surge (+14%)'},
    12: {'index': 1.18, 'name': 'Dec', 'note': 'Peak 13th-Month & Early Buyouts (+18%)'},
}


@dataclass
class PaymentStreamBreakdown:
    total_collected_mtd: float
    regular_collected_mtd: float
    advance_collected_mtd: float
    early_settlement_collected_mtd: float
    partial_collected_mtd: float
    regular_account_count: int
    advance_account_count: int
    early_settlement_count: int
    partial_account_count: int


class CashEngine:
    def __init__(self, pessimistic_stress_factor: float = 0.88):
        self.pessimistic_stress_factor = pessimistic_stress_factor

    def classify_payment_streams(
        self,
        payments: List[Dict[str, Any]],
        active_sales_map: Optional[Dict[int, float]] = None
    ) -> PaymentStreamBreakdown:
        """
        Classifies raw payment rows into 4 isolated financial streams:
        1. Regular Amortization (current term dues)
        2. Advance Prepayments (future term dues)
        3. Option Contract Early Settlements (full loan buyouts)
        4. Partials (crumbs under monthly term amount)
        """
        active_sales_map = active_sales_map or {}
        
        regular_collected = 0.0
        advance_collected = 0.0
        early_settlement_collected = 0.0
        partial_collected = 0.0

        regular_accounts = set()
        advance_accounts = set()
        early_accounts = set()
        partial_accounts = set()

        for p in payments:
            amt = float(p.get('amount_paid', 0.0))
            if amt <= 0:
                continue

            sale_id = p.get('sale_id')
            notes = str(p.get('notes', '')).lower()
            amort = float(active_sales_map.get(sale_id, p.get('monthly_amortization', 0.0)))
            if amort <= 0:
                amort = float(p.get('monthly_amortization', 0.0))

            # Stream A: Option Contract Early Settlement
            if 'option contract' in notes or 'early settlement' in notes or 'buyout' in notes or 'settled' in notes:
                early_settlement_collected += amt
                if sale_id:
                    early_accounts.add(sale_id)
            # Stream B: Advance Prepayment
            elif 'advance' in notes or (amort > 0 and amt >= (amort * 1.5)):
                advance_collected += amt
                if sale_id:
                    advance_accounts.add(sale_id)
            # Stream C: Partial / Crumb Payment
            elif amort > 0 and amt < (amort * 0.85):
                partial_collected += amt
                if sale_id:
                    partial_accounts.add(sale_id)
            # Stream D: Regular Amortization
            else:
                regular_collected += amt
                if sale_id:
                    regular_accounts.add(sale_id)

        total_collected = round(
            regular_collected + advance_collected + early_settlement_collected + partial_collected,
            2
        )

        return PaymentStreamBreakdown(
            total_collected_mtd=total_collected,
            regular_collected_mtd=round(regular_collected, 2),
            advance_collected_mtd=round(advance_collected, 2),
            early_settlement_collected_mtd=round(early_settlement_collected, 2),
            partial_collected_mtd=round(partial_collected, 2),
            regular_account_count=len(regular_accounts),
            advance_account_count=len(advance_accounts),
            early_settlement_count=len(early_accounts),
            partial_account_count=len(partial_accounts)
        )

    def compute_cash_forecast_cone(
        self,
        accounts: List[Dict[str, Any]],
        actual_stream: PaymentStreamBreakdown,
        projected_sales_units: int = 0,
        actual_sales_units: int = 0,
        forecast_month: Optional[str] = None,
        cure_to_active_prob: float = 0.58
    ) -> Dict[str, Any]:
        """
        Computes realistic, dynamic 3-month forecast cone with multi-term cohort decay,
        regional seasonality cycles, holiday bonus surges, and confidence intervals.
        """
        # Parse reference start date
        if forecast_month:
            try:
                base_dt = datetime.strptime(forecast_month, '%Y-%m')
            except Exception:
                base_dt = datetime.now()
        else:
            base_dt = datetime.now()

        # Generate 3 calendar month points
        month_dts = [base_dt + relativedelta(months=i) for i in range(3)]
        labels = [m.strftime('%b %Y') for m in month_dts]
        season_factors = [PHILIPPINE_MOTORCYCLE_SEASONALITY.get(m.month, {'index': 1.0})['index'] for m in month_dts]
        season_notes = [PHILIPPINE_MOTORCYCLE_SEASONALITY.get(m.month, {'note': 'Normal Run-Rate'})['note'] for m in month_dts]

        # Base month seasonality normalizer
        base_season_index = max(0.5, season_factors[0])

        if not accounts:
            contractual = 0.0
            ai_expected = 0.0
            pessimistic = 0.0
            three_month_expected = [0.0, 0.0, 0.0]
            three_month_scheduled = [0.0, 0.0, 0.0]
            three_month_optimistic = [0.0, 0.0, 0.0]
            three_month_pessimistic = [0.0, 0.0, 0.0]
            maturing_counts = [0, 0, 0]
            realization = 100.0 if actual_stream.total_collected_mtd > 0 else 0.0
            accuracy = 100.0 if actual_stream.total_collected_mtd == 0 else 0.0
        else:
            # 1. Evaluate Month 1 (Current Baseline)
            sched_m1 = sum(float(a.get('monthly_amortization', a.get('monthly_amort', 0.0))) for a in accounts)
            exp_m1_sum = 0.0
            for a in accounts:
                amort = float(a.get('monthly_amortization', a.get('monthly_amort', 0.0)))
                p_def = float(a.get('default_probability', 0.05))
                # Realization factor based on default probability
                realiz = max(0.20, 1.0 - (p_def * 0.75))
                exp_m1_sum += amort * realiz
                # Arrears cure contribution
                arrears = float(a.get('total_arrears', 0.0))
                if arrears > 0 and cure_to_active_prob > 0:
                    exp_m1_sum += arrears * cure_to_active_prob * 0.40

            if exp_m1_sum <= 0 or (sched_m1 > 0 and exp_m1_sum > sched_m1 * 1.08):
                exp_m1_sum = sched_m1 * 0.915

            contractual = round(sched_m1, 2)
            ai_expected = round(exp_m1_sum, 2)
            pessimistic = round(ai_expected * self.pessimistic_stress_factor, 2)

            # 2. Multi-Month Roll-Forward Simulation (Months 1, 2, 3)
            three_month_expected = []
            three_month_scheduled = []
            three_month_optimistic = []
            three_month_pessimistic = []
            maturing_counts = []

            for step in range(3):
                m_factor = season_factors[step] / base_season_index
                sched_step = 0.0
                exp_step = 0.0
                matured_this_step = 0

                for a in accounts:
                    amort = float(a.get('monthly_amortization', a.get('monthly_amort', 0.0)))
                    tot_terms = int(a.get('term_months', a.get('terms', a.get('total_terms', 36))))
                    paid_terms = int(a.get('paid_terms', a.get('terms_paid', 0))) + step
                    p_def = float(a.get('default_probability', 0.05))

                    # Check if contract has completed/matured
                    if paid_terms >= tot_terms:
                        matured_this_step += 1
                        continue  # Loan finished, zero scheduled due

                    sched_step += amort
                    # Base continuation hazard
                    continuation = max(0.20, 1.0 - (p_def * (0.75 + step * 0.03)))
                    
                    # Onboarding vs Mature stage adjustment
                    if paid_terms <= 4:
                        continuation *= 0.96  # Early term friction
                    elif paid_terms >= 10:
                        continuation = min(1.0, continuation * 1.02)  # Established habit

                    exp_step += amort * continuation

                # Add expected recurring inflow from projected new sales bookings
                if step > 0 and projected_sales_units > 0:
                    # Projected units adjusted by monthly seasonality
                    step_units = max(1, round(projected_sales_units * (season_factors[step] / base_season_index)))
                    # Average downpayment collection & 1st month installment realization
                    avg_downpayment = 14500.0
                    avg_monthly_amort = 3600.0
                    new_sales_cash = (step_units * avg_downpayment * 0.40) + (step_units * avg_monthly_amort * 0.90)
                    exp_step += new_sales_cash

                # Apply calendar monthly seasonality multiplier to recurring portfolio cashflow
                exp_step_seasonal = exp_step * m_factor

                # Upper / Lower confidence envelope
                if step == 0:
                    exp_final = ai_expected
                    opt_final = round(exp_final * 1.06, 2)
                    pess_final = round(exp_final * self.pessimistic_stress_factor, 2)
                else:
                    exp_final = round(exp_step_seasonal, 2)
                    # Optimistic includes seasonal bonus accelerations & early buyout options
                    holiday_boost = 1.08 if month_dts[step].month in [11, 12] else 1.04
                    opt_final = round(exp_final * (1.05 + step * 0.03) * holiday_boost, 2)
                    # Pessimistic factors in delay slippage
                    pess_final = round(exp_final * max(0.72, self.pessimistic_stress_factor - step * 0.04), 2)

                three_month_expected.append(exp_final)
                three_month_scheduled.append(round(sched_step, 2))
                three_month_optimistic.append(opt_final)
                three_month_pessimistic.append(pess_final)
                maturing_counts.append(matured_this_step)

            # Calculate MTD Realization and Accuracy
            realization = round((actual_stream.total_collected_mtd / ai_expected) * 100.0, 1) if ai_expected > 0 else 100.0
            accuracy = round(max(0.0, 100.0 - abs(ai_expected - actual_stream.regular_collected_mtd) / ai_expected * 100.0), 1) if ai_expected > 0 else 100.0

        # Sales unit metrics
        sales_realization = round((actual_sales_units / max(projected_sales_units, 1)) * 100.0, 1) if projected_sales_units > 0 else 100.0
        sales_accuracy = round(max(0.0, 100.0 - abs(projected_sales_units - actual_sales_units) / max(projected_sales_units, 1) * 100.0), 1)

        active_cnt = sum(1 for a in accounts if a.get('status') == 'active' or a.get('current_status') == 'active')
        delinq_cnt = sum(1 for a in accounts if a.get('status') == 'delinquent' or a.get('current_status') == 'delinquent')
        default_cnt = sum(1 for a in accounts if a.get('status') == 'defaulted' or a.get('current_status') == 'defaulted')

        return {
            'contractual_scheduled': round(contractual, 2),
            'ai_expected_collected': round(ai_expected, 2),
            'pessimistic_collected': round(pessimistic, 2),
            'pending_collections': round(max(0.0, ai_expected - actual_stream.total_collected_mtd), 2),
            'three_month_projection': three_month_expected,
            'three_month_labels': labels,
            'three_month_scheduled': three_month_scheduled,
            'three_month_optimistic': three_month_optimistic,
            'three_month_pessimistic': three_month_pessimistic,
            'three_month_seasonal_factors': season_factors,
            'three_month_notes': season_notes,
            'maturing_accounts_count': maturing_counts,
            'account_count_active': active_cnt,
            'account_count_delinquent': delinq_cnt,
            'account_count_defaulted': default_cnt,
            'account_count_total': len(accounts),
            'actual_collected_mtd': actual_stream.total_collected_mtd,
            'regular_collected_mtd': actual_stream.regular_collected_mtd,
            'advance_collected_mtd': actual_stream.advance_collected_mtd,
            'early_settlement_collected_mtd': actual_stream.early_settlement_collected_mtd,
            'partial_collected_mtd': actual_stream.partial_collected_mtd,
            'regular_account_count': actual_stream.regular_account_count,
            'advance_account_count': actual_stream.advance_account_count,
            'early_settlement_count': actual_stream.early_settlement_count,
            'partial_account_count': actual_stream.partial_account_count,
            'collection_realization_pct': realization,
            'recurring_accuracy_pct': accuracy,
            'projected_units_sold': projected_sales_units,
            'actual_units_sold': actual_sales_units,
            'sales_realization_pct': sales_realization,
            'sales_accuracy_pct': sales_accuracy
        }
