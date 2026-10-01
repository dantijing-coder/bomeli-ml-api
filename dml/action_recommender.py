"""
Recommended Actions builder.

Turns the numbers already computed for a scope (network or one branch) into a short,
prioritized to-do list. Every action names who/what, how much, and why - so a manager can act
on it without reading the rest of the dashboard.

Priorities: high (act this week), medium (this month), low (keep an eye on it).
"""

import math
from typing import Any, Dict, List, Optional

PRIORITY_ORDER = {'high': 0, 'medium': 1, 'low': 2}


def _peso(v: float) -> str:
    return f"₱{v:,.0f}"


def _names(items: List[Dict[str, Any]], key: str = 'customer_name', limit: int = 3) -> str:
    names = [str(i.get(key) or '').strip() for i in items[:limit] if i.get(key)]
    extra = len(items) - len(names)
    return ', '.join(names) + (f" +{extra} more" if extra > 0 else '')


def build_actions(scope_name: str,
                  accounts: List[Dict[str, Any]],
                  early_candidates: List[Dict[str, Any]],
                  inventory: List[Dict[str, Any]],
                  projected_units: int,
                  actual_units: int,
                  days_left_in_month: Optional[int],
                  collection: Dict[str, Any],
                  expected_this_month: float,
                  forward: List[Dict[str, Any]],
                  transitions: Dict[str, Any],
                  branch_breakdown: Optional[Dict[str, Dict[str, Any]]] = None,
                  max_actions: int = 8) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []

    def add(priority, type_, label, action, why):
        actions.append({'priority': priority, 'type': type_, 'label': label, 'action': action, 'why': why})

    # 1. Accounts at serious risk -> field visits
    critical = sorted(
        [a for a in accounts if a.get('current_status') == 'defaulted'
         or float(a.get('default_probability') or 0) >= 0.70],
        key=lambda a: -float(a.get('default_probability') or 0))
    if critical:
        arrears = sum(float(a.get('total_arrears') or 0) for a in critical)
        add('high', 'FIELD_CHECK', f"Field visits for {len(critical)} high-risk account{'s' if len(critical) != 1 else ''}",
            f"Visit or call {_names(critical)} to agree on a payment plan or begin unit recovery.",
            f"{_peso(arrears)} past due; these accounts have a 70%+ modeled chance of default or are already 3+ installments behind.")

    # 2. 1-2 installments behind -> reminders
    late = [a for a in accounts if a.get('current_status') == 'delinquent' and a not in critical]
    if late:
        arrears = sum(float(a.get('total_arrears') or 0) for a in late)
        add('high' if len(late) >= 5 else 'medium', 'COLLECTION_FOLLOWUP',
            f"Payment reminders to {len(late)} overdue borrower{'s' if len(late) != 1 else ''}",
            f"Send SMS/call reminders before they reach 3 missed installments ({_names(late)}).",
            f"{_peso(arrears)} past due. Accounts caught at 1-2 missed installments are far more likely to catch up than those at 3+.")

    # 3. Early payoff offers
    # Many steady payers score HIGH; a call list is only useful if it is short, so take the top 10
    hot_all = sorted([c for c in early_candidates if str(c.get('buyout_propensity', '')).upper() == 'HIGH'],
                     key=lambda c: -float(c.get('propensity_score') or c.get('settlement_probability') or 0))
    hot = hot_all[:10]
    if hot:
        cash = sum(float(c.get('buyout_quote') or 0) for c in hot)
        more = f" ({len(hot_all)} flagged in total)" if len(hot_all) > len(hot) else ''
        add('medium', 'EARLY_PAYOFF_OFFER', f"Offer payoff quotes to the top {len(hot)} likely early settler{'s' if len(hot) != 1 else ''}",
            f"Call {_names(hot)} with their discounted buyout quote.",
            f"Up to {_peso(cash)} in early cash from these {len(hot)}{more}; they pay steadily and are near the option-contract discount window.")

    # 4. Stock: likely to sell out vs. not moving
    def expected(v):
        return float(v.get('next_month_forecast', v.get('monthly_sales_rate_raw', v.get('monthly_sales_rate', 0))) or 0)

    restock = sorted([v for v in inventory
                      if expected(v) >= 0.5 and int(v.get('available_stock') or 0) < math.ceil(expected(v))],
                     key=lambda v: -expected(v))
    if restock:
        detail = ', '.join(f"{v['model_code']} ({int(v.get('available_stock') or 0)} on floor, ~{expected(v):.1f} expected)" for v in restock[:3])
        add('high' if any(int(v.get('available_stock') or 0) == 0 for v in restock) else 'medium', 'RESTOCK',
            f"Restock {len(restock)} fast-moving model{'s' if len(restock) != 1 else ''}",
            f"Order or transfer units: {detail}.",
            "Next month's expected sales exceed what is on the floor, so these sales could be lost.")

    idle = sorted([v for v in inventory if int(v.get('available_stock') or 0) >= 2 and expected(v) < 0.15],
                  key=lambda v: -int(v.get('available_stock') or 0))
    if idle:
        units = sum(int(v.get('available_stock') or 0) for v in idle)
        detail = ', '.join(f"{v['model_code']} ({int(v.get('available_stock') or 0)})" for v in idle[:3])
        add('low', 'SLOW_STOCK', f"Pause re-orders on {len(idle)} slow model{'s' if len(idle) != 1 else ''}",
            f"Hold new orders and consider transferring or promoting: {detail}.",
            f"{units} units tied up in models expected to sell less than one unit every 6 months.")

    # 5. Sales pace (live month only)
    if days_left_in_month is not None and projected_units > 0:
        gap = projected_units - actual_units
        if gap > 0:
            per_day = gap / max(days_left_in_month, 1)
            add('high' if gap >= max(3, projected_units * 0.3) else 'medium', 'SALES_PACE',
                f"Close {gap} more unit{'s' if gap != 1 else ''} to hit this month's target",
                f"{actual_units} of {projected_units} released with {days_left_in_month} day{'s' if days_left_in_month != 1 else ''} left "
                f"(~{per_day:.1f}/day). Follow up pending applications and reservations.",
                "New installment sales are part of next months' collections forecast.")

    # 6. Collections outlook
    if forward and expected_this_month > 0:
        nxt = forward[0]
        change = (float(nxt['ai_expected']) - expected_this_month) / expected_this_month * 100
        if change <= -5:
            add('medium', 'CASH_OUTLOOK', f"{nxt['label']} collections forecast down {abs(change):.0f}%",
                f"Plan for about {_peso(float(nxt['ai_expected']))} next month (existing contracts {_peso(float(nxt['existing_pipeline']))} "
                f"+ new sales {_peso(float(nxt['expected_new_sales_cash']))}).",
                "Contracts are maturing faster than new ones are being booked.")
    if collection.get('overdue_installments'):
        add('medium' if collection.get('collection_rate_pct', 100) >= 90 else 'high', 'COLLECTION_GAP',
            f"Recover {collection['overdue_installments']} unpaid installment{'s' if collection['overdue_installments'] != 1 else ''} from this month",
            f"{_peso(float(collection.get('overdue_amount') or 0))} of this month's dues is still unpaid.",
            f"Collection rate is {float(collection.get('collection_rate_pct') or 0):.1f}% (cash + on-time rebates vs collectible dues).")

    # 7. Roll-rate warnings
    cure = float(transitions.get('cure_to_active_prob') or 0)
    counts = transitions.get('transition_counts') or {}
    if counts.get('delinquent_start', 0) >= 3 and cure < 0.30:
        add('medium', 'RISK_TREND', f"Only {cure * 100:.0f}% of late accounts are catching up",
            "Offer restructuring or partial-payment arrangements to accounts 1-2 installments behind.",
            f"{counts.get('delinquent_start', 0)} accounts started the month late; few returned to good standing.")

    # 8. Network view: which branch needs attention
    if branch_breakdown:
        ranked = sorted(branch_breakdown.values(), key=lambda b: -float(b.get('a_to_d') or 0))
        if len(ranked) >= 2 and float(ranked[0].get('a_to_d') or 0) > float(ranked[-1].get('a_to_d') or 0) * 1.5:
            worst, best = ranked[0], ranked[-1]
            add('medium', 'BRANCH_FOCUS', f"Focus collections support on {worst.get('branch_name') or worst.get('name')}",
                f"Share {best.get('branch_name') or best.get('name')}'s follow-up routine with this branch.",
                f"{float(worst.get('a_to_d') or 0) * 100:.1f}% of its current payers slipped this month vs "
                f"{float(best.get('a_to_d') or 0) * 100:.1f}% at {best.get('branch_name') or best.get('name')}.")

    if not actions:
        add('low', 'ROUTINE_SERVICING', 'Routine servicing',
            f"{scope_name} is within normal ranges. Keep sending due-date reminders.", 'No risk, stock, or pacing issues detected.')

    actions.sort(key=lambda a: PRIORITY_ORDER.get(a['priority'], 3))
    return actions[:max_actions]
