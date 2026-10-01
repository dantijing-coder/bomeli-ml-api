"""
Month-scoped payment stream classifier (shared by both pipelines).

Splits the cash a borrower paid inside ONE calendar month into two streams:
  - current: cash that settled what was owed up to the month end
             (this month's term, overdue arrears, penalties, partials)
  - ahead:   cash that went beyond everything owed through the month end,
             including Option Contract payoffs (paid off = paid ahead)

Ledger rule per sale:
  owed_through_end = sum(amount_due of terms due <= month end)
                     - credit applied before the month (paid - penalty + rebate)
  Each payment's credit (paid - penalty + rebate) first fills owed_through_end;
  only the overflow is 'ahead'. Nothing outside [m_start, m_end] is ever read
  as this month's cash, so prior-month payoffs cannot bleed forward.
"""
from typing import Any, Dict, Iterable, List

PAYOFF_MARKERS = ('option contract', 'early settlement', 'early settled', 'buyout', 'early payoff')
AHEAD_TOLERANCE = 1.0  # ignore sub-peso rounding overflow


def is_payoff(notes: Any) -> bool:
    n = str(notes or '').lower()
    return any(k in n for k in PAYOFF_MARKERS)


def _f(v: Any) -> float:
    try:
        return float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _day(v: Any) -> str:
    return str(v)[:10]


def split_month_payments(
    payments: Iterable[Dict[str, Any]],
    installments: Iterable[Dict[str, Any]],
    m_start: str,
    m_end: str,
) -> Dict[str, Any]:
    """
    payments:     payment rows (sale_id, payment_date, amount_paid, rebate_amount,
                  penalty_amount, notes) for the scope. Rows before m_start are
                  used only as prior ledger credit; rows after m_end are ignored.
    installments: schedule rows (sale_id, due_date, amount_due) for the scope.
    m_start/m_end: 'YYYY-MM-DD' inclusive bounds of the month.
    """
    due_by_sale: Dict[Any, float] = {}
    for i in installments:
        if _day(i['due_date']) <= m_end:
            due_by_sale[i['sale_id']] = due_by_sale.get(i['sale_id'], 0.0) + _f(i['amount_due'])

    prior_credit: Dict[Any, float] = {}
    in_month: Dict[Any, List[Dict[str, Any]]] = {}
    for p in payments:
        d = _day(p['payment_date'])
        if d > m_end:
            continue
        sid = p['sale_id']
        if d < m_start:
            prior_credit[sid] = prior_credit.get(sid, 0.0) + max(
                0.0, _f(p['amount_paid']) - _f(p.get('penalty_amount')) + _f(p.get('rebate_amount')))
        else:
            in_month.setdefault(sid, []).append(p)

    current_total = ahead_total = 0.0
    current_accounts, ahead_accounts = set(), set()
    by_sale: Dict[Any, Dict[str, float]] = {}

    for sid, plist in in_month.items():
        owed = max(0.0, due_by_sale.get(sid, 0.0) - prior_credit.get(sid, 0.0))
        s_cur = s_ahead = 0.0
        for p in sorted(plist, key=lambda r: str(r['payment_date'])):
            paid = _f(p['amount_paid'])
            if paid <= 0:
                continue
            if is_payoff(p.get('notes')):
                s_ahead += paid
                owed = 0.0
                continue
            penalty = min(paid, _f(p.get('penalty_amount')))
            credit = (paid - penalty) + _f(p.get('rebate_amount'))
            overflow = max(0.0, credit - owed)
            owed = max(0.0, owed - credit)
            ahead_cash = min(paid - penalty, overflow) if overflow > AHEAD_TOLERANCE else 0.0
            s_ahead += ahead_cash
            s_cur += paid - ahead_cash

        by_sale[sid] = {'current': round(s_cur, 2), 'ahead': round(s_ahead, 2)}
        current_total += s_cur
        ahead_total += s_ahead
        if s_cur > 0:
            current_accounts.add(sid)
        if s_ahead > 0:
            ahead_accounts.add(sid)

    return {
        'total': round(current_total + ahead_total, 2),
        'current': round(current_total, 2),
        'ahead': round(ahead_total, 2),
        'current_count': len(current_accounts),
        'ahead_count': len(ahead_accounts),
        'by_sale': by_sale,
    }


def to_cash_forecast_fields(split: Dict[str, Any]) -> Dict[str, Any]:
    """Maps a split onto the cash_forecast_json keys the dashboard reads.
    Paid-off cash lives inside 'advance'; the early/partial keys stay at 0 for
    backward compatibility with older readers."""
    return {
        'actual_collected_mtd': split['total'],
        'regular_collected_mtd': split['current'],
        'advance_collected_mtd': split['ahead'],
        'early_settlement_collected_mtd': 0.0,
        'partial_collected_mtd': 0.0,
        'regular_account_count': split['current_count'],
        'advance_account_count': split['ahead_count'],
        'early_settlement_count': 0,
        'partial_account_count': 0,
    }
