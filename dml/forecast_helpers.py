"""
Shared forecasting helpers used by both the walk-forward pipeline and the
live predictive engine so that the stored payloads have identical semantics.

  - round_half_up(): 8.4 -> 8, 8.5 -> 9 (Python's round() is banker's rounding)
  - account_status_from_ledger(): active / delinquent / defaulted as of a cut-off date
  - compute_transition_rates(): true Markov roll-rates (status entering month -> status now),
    pooled over whatever accounts are passed in, smoothed toward a baseline
  - build_forward_forecast(): Existing Pipeline (booked installment schedule x realization)
    + Expected New Sales (projected installment units x avg amortization x realization)
  - assumed_month_close(): units the live month is assumed to close at (TARGET_ATTAINMENT of goal)
  - project_portfolio_health(): rolls on-time / behind / default counts forward with the Markov rates
  - forecast_models_by_month(): splits each forecast month's expected units across models
"""

import math
from datetime import datetime
from typing import Dict, List, Any, Iterable, Optional, Tuple

from dateutil.relativedelta import relativedelta

# Calendar seasonality for Philippine motorcycle retail (shared with the walk-forward pipeline)
SEASONAL_INDEX = {
    1: 0.85, 2: 0.92, 3: 1.15, 4: 1.12, 5: 1.05, 6: 0.88,
    7: 0.94, 8: 0.98, 9: 1.08, 10: 1.14, 11: 1.22, 12: 1.32
}

MARKOV_BASELINE = {
    'active_to_delinquent_prob': 0.08,
    'delinquent_to_default_prob': 0.22,
    'cure_to_active_prob': 0.58,
}

# Pseudo-count used to shrink small-sample transition rates toward the baseline
TRANSITION_PSEUDOCOUNT = 5.0

# Share of the monthly sales goal the live month is assumed to reach when forecasting the
# months after it (a goal is rarely hit in full; units already sold are never discounted)
TARGET_ATTAINMENT = 0.80


def round_half_up(value: float) -> int:
    """Rounds to the nearest whole number with .5 always going up (8.5 -> 9)."""
    return int(math.floor(float(value or 0.0) + 0.5))


def payoff_readiness(model_probability: float, paid_terms: int, term_months: int, overdue_terms: int) -> float:
    """
    Ranks early-payoff candidates. The early-settlement classifier saturates (most steady payers get the
    same leaf probability), so it is weighted by how far into the term the borrower is - a borrower with
    3 terms left is far more likely to settle now than one with 20 left - and discounted if they are behind.
    Returns 0..1.
    """
    progress = min(1.0, max(0.0, paid_terms / max(1, term_months)))
    behind = 0.6 if overdue_terms > 0 else 1.0
    return round(min(1.0, max(0.0, float(model_probability) * (0.55 + 0.45 * progress) * behind)), 4)


def status_from_overdue(overdue_terms: int) -> str:
    if overdue_terms <= 0:
        return 'active'
    if overdue_terms <= 2:
        return 'delinquent'
    return 'defaulted'


def account_status_from_ledger(payments: Iterable[Dict[str, Any]],
                               installments: Iterable[Dict[str, Any]],
                               cutoff: str,
                               inclusive: bool) -> Tuple[str, int]:
    """
    Waterfalls cash paid up to `cutoff` (YYYY-MM-DD) across installments due up to `cutoff`
    and returns (status, overdue_terms). `inclusive` controls <= vs < on both dates.
    Payment dicts need 'p_date_str', 'amount_paid', 'rebate_amount'; installment dicts
    need 'due_date_str', 'amount_due'.
    """
    if inclusive:
        paid = sum(float(p['amount_paid']) + float(p.get('rebate_amount') or 0.0)
                   for p in payments if p['p_date_str'] <= cutoff)
        due = [i for i in installments if i['due_date_str'] <= cutoff]
    else:
        paid = sum(float(p['amount_paid']) + float(p.get('rebate_amount') or 0.0)
                   for p in payments if p['p_date_str'] < cutoff)
        due = [i for i in installments if i['due_date_str'] < cutoff]

    remaining = paid
    overdue = 0
    for inst in sorted(due, key=lambda x: x['due_date_str']):
        amt = float(inst['amount_due'])
        if remaining >= amt - 10.0:
            remaining -= amt
        else:
            overdue += 1
            remaining = 0.0
    return status_from_overdue(overdue), overdue


def compute_transition_rates(pairs: List[Tuple[str, str]],
                             baseline: Optional[Dict[str, float]] = None,
                             pseudocount: float = TRANSITION_PSEUDOCOUNT) -> Dict[str, Any]:
    """
    pairs: list of (status_entering, status_now) for every account in scope.
    Returns empirical Markov roll-rates smoothed toward `baseline`:
        rate = (events + k * baseline) / (at_risk + k)
    so a branch with 3 delinquent accounts does not swing to 0% / 100%.
    """
    base = dict(MARKOV_BASELINE)
    if baseline:
        for key in base:
            if baseline.get(key) is not None:
                base[key] = float(baseline[key])

    from_active = [now for (start, now) in pairs if start == 'active']
    from_delinq = [now for (start, now) in pairs if start == 'delinquent']

    n_active_start = len(from_active)
    n_delinq_start = len(from_delinq)
    a_to_d_events = sum(1 for now in from_active if now in ('delinquent', 'defaulted'))
    d_to_def_events = sum(1 for now in from_delinq if now == 'defaulted')
    cure_events = sum(1 for now in from_delinq if now == 'active')

    k = pseudocount
    a_to_d = (a_to_d_events + k * base['active_to_delinquent_prob']) / (n_active_start + k)
    d_to_def = (d_to_def_events + k * base['delinquent_to_default_prob']) / (n_delinq_start + k)
    cure = (cure_events + k * base['cure_to_active_prob']) / (n_delinq_start + k)

    now_states = [now for (_, now) in pairs]
    return {
        'active_to_delinquent_prob': round(a_to_d, 3),
        'delinquent_to_default_prob': round(d_to_def, 3),
        'cure_to_active_prob': round(cure, 3),
        'transition_counts': {
            'active_start': n_active_start,
            'delinquent_start': n_delinq_start,
            'active_to_delinquent': a_to_d_events,
            'delinquent_to_default': d_to_def_events,
            'delinquent_to_active': cure_events,
        },
        'active_count': sum(1 for s in now_states if s == 'active'),
        'delinquent_count': sum(1 for s in now_states if s == 'delinquent'),
        'defaulted_count': sum(1 for s in now_states if s == 'defaulted'),
        'total_accounts': len(pairs),
        'method': 'empirical_transition_smoothed',
    }


def estimate_monthly_units(monthly_actual_units: Dict[str, int],
                           before_month: str,
                           target_month: str,
                           lookback: int = 3,
                           fallback: float = 10.0) -> float:
    """
    Seasonal-naive unit estimate for `target_month`: de-seasonalized average of the
    `lookback` calendar months strictly before `before_month`, re-seasonalized.
    """
    ref = datetime.strptime(before_month + '-01', '%Y-%m-%d')
    deseason = []
    for i in range(1, lookback + 1):
        m_dt = ref - relativedelta(months=i)
        m_key = m_dt.strftime('%Y-%m')
        if m_key in monthly_actual_units:
            deseason.append(monthly_actual_units[m_key] / SEASONAL_INDEX.get(m_dt.month, 1.0))
    base = (sum(deseason) / len(deseason)) if deseason else fallback
    t_month = int(target_month[5:7])
    return max(0.0, base * SEASONAL_INDEX.get(t_month, 1.0))


def build_forward_forecast(current_month: str,
                           horizon_months: List[str],
                           scheduled_by_month: Dict[str, float],
                           realization_rate: float,
                           projected_units_current: float,
                           actual_units_current: int,
                           monthly_actual_units: Dict[str, int],
                           installment_share: float,
                           avg_new_amortization: float) -> List[Dict[str, Any]]:
    """
    Builds the forward cash forecast for every month in `horizon_months` (all after current_month).

      existing_pipeline  = booked installment_schedule due that month x realization_rate
      expected_new_sales = first/ongoing amortizations of installment units expected to be
                           sold between now and the month before (first due date is ~1 month
                           after release), x realization_rate
      ai_expected        = existing_pipeline + expected_new_sales

    The current month is assumed to close at TARGET_ATTAINMENT of its goal (never below what is
    already sold), and that assumed close joins the trailing months the unit estimate is built on,
    so the forecast moves with this month's goal. Closed months pass projected == actual.

    Down payments are not recorded in `payments`, so they are not counted as collections.
    """
    rows = []
    assumed_close = assumed_month_close(projected_units_current, actual_units_current)
    # Units still expected to close in the current month
    remaining_current = max(0.0, assumed_close - float(actual_units_current))
    cumulative_new_units = remaining_current * installment_share
    units_basis = dict(monthly_actual_units)
    units_basis[current_month] = assumed_close
    next_after_current = months_after(current_month, 1)[0]

    # Trained Monthly Sales Forecast model (dml/sales_forecast.py); seasonal rule if it is not trained yet
    try:
        from .sales_forecast import predict_units
        model_units = predict_units(units_basis, current_month, list(horizon_months)) or {}
    except Exception as e:
        print(f"[forecast] Sales forecast model unavailable, using seasonal rule: {e}")
        model_units = {}

    for idx, m in enumerate(horizon_months):
        existing_sched = float(scheduled_by_month.get(m, 0.0))
        existing_exp = existing_sched * realization_rate

        # Units sold up to the end of the previous month have their first amortization due in `m`
        new_amort_sched = cumulative_new_units * avg_new_amortization
        new_exp = new_amort_sched * realization_rate

        units_source = 'model' if m in model_units else 'seasonal_rule'
        units_this_month = model_units[m] if m in model_units else estimate_monthly_units(units_basis, next_after_current, m)
        # Installment accounts opened since the previous point (rest of the live month for the first row)
        new_accounts = (remaining_current if idx == 0 else 0.0) * installment_share + units_this_month * installment_share

        rows.append({
            'month': m,
            'label': datetime.strptime(m + '-01', '%Y-%m-%d').strftime('%b %Y'),
            'scheduled_existing': round(existing_sched, 2),
            'existing_pipeline': round(existing_exp, 2),
            'expected_new_sales_cash': round(new_exp, 2),
            'expected_new_installment_units': round(cumulative_new_units, 1),
            'expected_units_sold': round_half_up(units_this_month),
            'expected_new_accounts': round(new_accounts, 1),
            'units_source': units_source,
            'scheduled_total': round(existing_sched + new_amort_sched, 2),
            'ai_expected': round(existing_exp + new_exp, 2),
        })

        cumulative_new_units += units_this_month * installment_share

    return rows


def assumed_month_close(projected_units: float, actual_units: float,
                        attainment: float = TARGET_ATTAINMENT) -> int:
    """Units the month is assumed to close at: `attainment` of the goal, never below units already sold."""
    return max(int(actual_units or 0), round_half_up(float(projected_units or 0) * attainment))


def project_portfolio_health(active: int, delinquent: int, defaulted: int,
                             rates: Dict[str, Any],
                             forward_rows: List[Dict[str, Any]],
                             months: int = 2) -> List[Dict[str, Any]]:
    """
    Rolls borrower counts forward one month at a time with the scope's Markov roll-rates:
        on time  <- on time that keep paying + behind that catch up + new installment accounts
        behind   <- on time that slip + behind that neither catch up nor default
        default  <- default (absorbing) + behind that roll into default
    New installment accounts come from the same sales forecast as the cash outlook.
    """
    p_ad = float(rates.get('active_to_delinquent_prob') or 0.0)
    p_dd = float(rates.get('delinquent_to_default_prob') or 0.0)
    p_cure = float(rates.get('cure_to_active_prob') or 0.0)
    stay_d = max(0.0, 1.0 - p_dd - p_cure)

    a, d, x = float(active), float(delinquent), float(defaulted)
    out = []
    for row in forward_rows[:months]:
        new = float(row.get('expected_new_accounts') or 0.0)
        a, d, x = a * (1.0 - p_ad) + d * p_cure + new, a * p_ad + d * stay_d, x + d * p_dd
        ra, rd, rx = round_half_up(a), round_half_up(d), round_half_up(x)
        out.append({'month': row['month'], 'label': row['label'],
                    'active': ra, 'delinquent': rd, 'defaulted': rx, 'total': ra + rd + rx})
    return out


def forecast_models_by_month(velocity_items: List[Dict[str, Any]],
                             forward_rows: List[Dict[str, Any]],
                             months: int = 2) -> List[Dict[str, Any]]:
    """
    Returns copies of the velocity rows with `forecast_by_month`: each forecast month's expected
    units (from the sales forecast, which assumes the goal attainment above) split across models
    by their share of the velocity model's next-month forecast. The model lines therefore add
    up to the Sales This Month outlook instead of drifting from it.
    """
    def rate(v):
        return float(v.get('next_month_forecast', v.get('monthly_sales_rate_raw', v.get('monthly_sales_rate', 0))) or 0.0)

    total = sum(rate(v) for v in velocity_items)
    out = []
    for v in velocity_items:
        share = rate(v) / total if total > 0 else 0.0
        item = dict(v)
        item['forecast_by_month'] = [
            {'month': r['month'], 'units': round(float(r.get('expected_units_sold') or 0) * share, 2)}
            for r in forward_rows[:months]
        ]
        out.append(item)
    return out


def months_after(month: str, count: int) -> List[str]:
    base = datetime.strptime(month + '-01', '%Y-%m-%d')
    return [(base + relativedelta(months=i)).strftime('%Y-%m') for i in range(1, count + 1)]


def forecast_horizon(month: str, min_months: int = 2) -> List[str]:
    """Months after `month` through December of the same year, at least `min_months` long."""
    m_num = int(month[5:7])
    return months_after(month, max(min_months, 12 - m_num))
