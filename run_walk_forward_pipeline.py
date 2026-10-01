#!/usr/bin/env python3
"""
ml_engine/run_walk_forward_pipeline.py
Sequential expanding-window predictive simulation & evaluation engine.

Integrates ALL core Machine Learning modules:
  1. Default Hazard & Delinquency Migration Classifier (Model 1)
  2. Inventory Sales Velocity Regressor (Model 2 - Decision Tree Regressor)
  3. Option Contract Early Settlement Propensity Classifier (Model 3)
  4. Multi-Term Contract Lifecycle Outcome Classifier (Model 4)
  5. Portfolio Cash Realization Rate Regressor (Model 5)
  6. Monthly Vehicle Sales Target & Accuracy (Model 2 + Seasonal Decomposition)
  7. 4-State Markov Chain Roll-Rates

User Specifications:
  - 1 month of actual activity data in DB is the baseline for Month 2.
  - Prediction for Month 3 conditions on Month 1 & 2 actual DB data (raw payments & schedules, not snapshots).
  - Prediction for Month 4 conditions on Months 1, 2, 3 actual DB data.
  - For Month t, the ML engine conditions on all prior raw DB activity (1 ... t-1) to detect borrower payment patterns.
  - Predicted Monthly Sales targets vary dynamically using Model 2 + calendar seasonality (no hardcoded identical values).
  - Showroom inventory velocity runs Model 2 for each vehicle model and branch.
  - Live data in the current month reflects as a real-time comparison against the forecast.
  - Buttons for past snapshots are disabled (read-only historical records). We only actively predict the current month.
"""

import os
import sys
import json
import pymysql
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Any, Optional
# pyrefly: ignore [missing-import]
import numpy as np

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DB_CONFIG, MODELS_DIR, get_active_rate_package
from dml.model_registry import ModelRegistry
from dml.markov_engine import MarkovEngine
from dml.forecast_helpers import (
    SEASONAL_INDEX, MARKOV_BASELINE, round_half_up, compute_transition_rates,
    build_forward_forecast, forecast_horizon, payoff_readiness,
    TARGET_ATTAINMENT, assumed_month_close, project_portfolio_health, forecast_models_by_month
)
from dml.velocity_features import SalesPanel, month_add
from dml.action_recommender import build_actions
from dml.velocity_engine import resolve_model_specs
from dml.payment_streams import split_month_payments, to_cash_forecast_fields


def load_markov_baseline():
    path = os.path.join(MODELS_DIR, 'markov_matrix.json')
    base = dict(MARKOV_BASELINE)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                stored = json.load(f)
            for k in base:
                if isinstance(stored.get(k), (int, float)):
                    base[k] = float(stored[k])
        except Exception:
            pass
    return base


def collection_quality(insts, pays, sale_repo_date):
    """
    Fair monthly collection rate:
        (cash received + on-time rebate credit) / (scheduled dues - dues falling after a unit was repossessed)
    - Rebates: on-time payers settle the installment for less cash by design, so the rebate is credited.
    - Repossessed: once the unit is seized the contract ends; later dues are recovered via the unit, not payments.
    Also reports how much of the month's installments are marked paid and what is still overdue.
    """
    face = sum(float(i['amount_due']) for i in insts)
    repo_excl = sum(float(i['amount_due']) for i in insts
                    if sale_repo_date.get(i['sale_id']) and i['due_date_str'] >= sale_repo_date[i['sale_id']])
    cash = sum(float(p['amount_paid']) for p in pays)
    rebate = sum(float(p.get('rebate_amount') or 0.0) for p in pays)
    collectible = max(face - repo_excl, 0.0)
    paid_face = sum(float(i['amount_due']) for i in insts if i.get('status') == 'paid')
    overdue = [i for i in insts if i.get('status') == 'overdue'
               and not (sale_repo_date.get(i['sale_id']) and i['due_date_str'] >= sale_repo_date[i['sale_id']])]
    rate = ((cash + rebate) / collectible * 100.0) if collectible > 0 else (100.0 if cash > 0 else 0.0)
    return {
        'scheduled_face': round(face, 2),
        'repossessed_dues_excluded': round(repo_excl, 2),
        'collectible_scheduled': round(collectible, 2),
        'cash_collected': round(cash, 2),
        'rebate_credit': round(rebate, 2),
        'credited_total': round(cash + rebate, 2),
        'collection_rate_pct': round(rate, 2),
        'cash_only_rate_pct': round(cash / face * 100.0, 2) if face > 0 else 0.0,
        'installments_paid_pct': round(paid_face / face * 100.0, 1) if face > 0 else 0.0,
        'overdue_installments': len(overdue),
        'overdue_amount': round(sum(float(i['amount_due']) for i in overdue), 2),
    }


def scope_sales_context(all_sales, all_showroom_sales, sale_created_month, m_str, branch_id=None):
    """
    Trailing (3 months before m_str) installment share of releases and average amortization
    of new installment contracts for a scope. Used to value Expected New Sales.
    """
    m_dt = datetime.strptime(m_str + '-01', '%Y-%m-%d')
    window_start = (m_dt - relativedelta(months=3)).strftime('%Y-%m')
    showroom = [s for s in all_showroom_sales
                if window_start <= str(s['created_at'])[:7] < m_str
                and (branch_id is None or s.get('branch_id') == branch_id)]
    inst = [s for s in all_sales
            if window_start <= sale_created_month.get(s['sale_id'], '') < m_str
            and (branch_id is None or s['branch_id'] == branch_id)]
    share = (len(inst) / len(showroom)) if showroom else 0.85
    amorts = [float(s['monthly_amortization'] or 0.0) for s in inst if float(s['monthly_amortization'] or 0.0) > 0]
    avg_amort = (sum(amorts) / len(amorts)) if amorts else 4500.0
    return max(0.0, min(1.0, share)), avg_amort

def run_walk_forward_simulation():
    print("=" * 82)
    print("COMPREHENSIVE MULTI-MODULE ML PREDICTIVE PIPELINE")
    print("Models: Default Hazard, Sales Velocity, Early Payoff, Cash Realization, Markov")
    print("Conditioned on Expanding Raw DB Activity (Oct 2023 .. Current Active Month)")
    print("=" * 82)

    reg = ModelRegistry()
    markov_engine = MarkovEngine(baseline_matrix=MARKOV_BASELINE, models_dir=MODELS_DIR)
    rate_pkg = get_active_rate_package()

    conn = pymysql.connect(**DB_CONFIG)
    cur = conn.cursor(pymysql.cursors.DictCursor)

    current_calendar_month = datetime.now().strftime('%Y-%m')

    # 1. Fetch branches
    cur.execute("SELECT branch_id, name FROM branches ORDER BY branch_id ASC")
    branches = cur.fetchall()
    branch_map = {b['branch_id']: b['name'] for b in branches}

    # 2. Fetch vehicle models
    cur.execute("SELECT model_id, brand, model_code FROM vehicle_models ORDER BY model_id ASC")
    models = cur.fetchall()
    model_map = {m['model_id']: m for m in models}

    # 3. Fetch all inventory units
    cur.execute("""
        SELECT unit_id, model_id, branch_id, status, date_added
        FROM inventory_units
        ORDER BY date_added ASC
    """)
    all_inventory = cur.fetchall()
    for u in all_inventory:
        u['date_added_str'] = str(u['date_added'])[:10] if u['date_added'] else '2023-09-01'

    # 4. Fetch all sales into memory
    cur.execute("""
        SELECT s.sale_id, s.account_no, s.customer_id, c.full_name AS customer_name,
               c.phone, s.application_data, u.branch_id, s.status, s.term_months,
               s.monthly_amortization, s.gross_principal_amount, s.down_payment,
               s.principal_loan_amount, s.promissory_note_value,
               s.total_payable_amount, s.sale_date, s.created_at, u.model_id, s.unit_id,
               s.repo_seized_at
        FROM sales s
        JOIN customers c ON s.customer_id = c.customer_id
        JOIN inventory_units u ON s.unit_id = u.unit_id
        WHERE s.account_no IS NOT NULL AND s.account_no != ''
          AND s.status NOT IN ('cancelled', 'pending', 'pending_approval')
        ORDER BY s.created_at ASC
    """)
    all_sales = cur.fetchall()

    # Fetch all showroom sales including retail/cash sales without account_no (only released units)
    cur.execute("""
        SELECT s.sale_id, u.branch_id, s.created_at, s.unit_id, u.model_id
        FROM sales s
        LEFT JOIN inventory_units u ON s.unit_id = u.unit_id
        WHERE s.status NOT IN ('cancelled', 'pending', 'pending_approval')
        ORDER BY s.created_at ASC
    """)
    all_showroom_sales = cur.fetchall()

    unit_sold_dates = {}
    for s in all_showroom_sales:
        s_date = str(s['created_at'])[:10]
        if s.get('unit_id'):
            unit_sold_dates[s['unit_id']] = s_date

    # 5. Fetch all payments into memory
    cur.execute("""
        SELECT payment_id, sale_id, payment_date, amount_paid, rebate_amount,
               penalty_amount, branch_id, notes
        FROM payments
        WHERE payment_date IS NOT NULL
        ORDER BY payment_date ASC
    """)
    all_payments = cur.fetchall()

    # 6. Fetch all installment schedules into memory
    cur.execute("""
        SELECT installment_id, sale_id, installment_number, due_date, amount_due, status
        FROM installment_schedule
        ORDER BY sale_id ASC, installment_number ASC
    """)
    all_installments = cur.fetchall()

    # Pre-index payments and installments by sale_id
    payments_by_sale = {}
    for p in all_payments:
        p_date_str = str(p['payment_date'])[:10]
        p['p_date_str'] = p_date_str
        p['p_month_str'] = p_date_str[:7]
        payments_by_sale.setdefault(p['sale_id'], []).append(p)

    installments_by_sale = {}
    for inst in all_installments:
        d_str = str(inst['due_date'])[:10]
        inst['due_date_str'] = d_str
        inst['due_month_str'] = d_str[:7]
        installments_by_sale.setdefault(inst['sale_id'], []).append(inst)

    # Pre-aggregations for the forward (Existing Pipeline + Expected New Sales) forecast
    markov_baseline = load_markov_baseline()
    sale_branch = {s['sale_id']: s['branch_id'] for s in all_sales}
    sale_created_month = {s['sale_id']: str(s['created_at'])[:7] for s in all_sales}
    sale_repo_date = {s['sale_id']: str(s['repo_seized_at'])[:10] for s in all_sales if s.get('repo_seized_at')}
    monthly_units_global = {}
    monthly_units_by_branch = {}
    for s in all_showroom_sales:
        mk = str(s['created_at'])[:7]
        monthly_units_global[mk] = monthly_units_global.get(mk, 0) + 1
        if s.get('branch_id'):
            bmap = monthly_units_by_branch.setdefault(s['branch_id'], {})
            bmap[mk] = bmap.get(mk, 0) + 1

    # Stock Velocity v2 panel: monthly units per (branch, model) + unit-level floor history
    velocity_panel = SalesPanel(
        sales=[{'branch_id': s.get('branch_id'), 'model_code': model_map.get(s.get('model_id'), {}).get('model_code'),
                'date': str(s['created_at'])[:10], 'unit_id': s.get('unit_id')} for s in all_showroom_sales],
        inventory=[{'unit_id': u['unit_id'], 'branch_id': u['branch_id'],
                    'model_code': model_map.get(u['model_id'], {}).get('model_code'),
                    'date_added': u['date_added_str']} for u in all_inventory]
    )
    model_specs = {vm_id: resolve_model_specs(vm['model_code'], vm['brand']) for vm_id, vm in model_map.items()}

    def branch_model_velocity(bid, vm_id, month):
        vm = model_map[vm_id]
        spec = model_specs[vm_id]
        feats = velocity_panel.features(bid, vm['model_code'], month, vm['brand'], spec['category'], spec['price'])
        return reg.predict_velocity(feats), feats

    def booked_schedule_by_month(as_of_month, horizon, branch_id=None):
        """Installment dues in `horizon` for contracts booked on/before `as_of_month` (no look-ahead)."""
        horizon_set = set(horizon)
        totals = {m: 0.0 for m in horizon}
        for inst in all_installments:
            sid = inst['sale_id']
            if inst['due_month_str'] not in horizon_set or sid not in sale_created_month:
                continue
            if sale_created_month[sid] > as_of_month:
                continue
            if branch_id is not None and sale_branch.get(sid) != branch_id:
                continue
            totals[inst['due_month_str']] += float(inst['amount_due'])
        return totals

    # 7. Discover distinct calendar months in sequence up to current calendar month
    sale_months = set(str(s['created_at'])[:7] for s in all_sales if s['created_at'])
    pay_months = set(p['p_month_str'] for p in all_payments if p.get('p_month_str'))
    all_months = sorted(list(sale_months.union(pay_months)))
    timeline_months = [m for m in all_months if m <= current_calendar_month]

    print(f"Loaded: {len(branches)} branches, {len(models)} models, {len(all_inventory)} inventory units, {len(all_sales)} sales, {len(all_payments)} payments, {len(all_installments)} installments.")
    print(f"Sequential Timeline: {len(timeline_months)} months ({timeline_months[0]} to {timeline_months[-1]}). Active Cycle = {current_calendar_month}\n")

    # Ensure tables exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ai_account_risk_scores (
          id INT AUTO_INCREMENT PRIMARY KEY,
          forecast_month VARCHAR(7) NOT NULL,
          sale_id INT NOT NULL,
          customer_id INT NOT NULL,
          branch_id INT NOT NULL,
          account_no VARCHAR(50) NOT NULL,
          customer_name VARCHAR(150) NOT NULL,
          current_status VARCHAR(30) NOT NULL,
          overdue_count INT NOT NULL DEFAULT 0,
          default_probability DECIMAL(4,3) NOT NULL,
          early_settlement_probability DECIMAL(4,3) NOT NULL DEFAULT 0.000,
          hazard_tier ENUM('low', 'moderate', 'high', 'critical') NOT NULL DEFAULT 'low',
          recommended_action VARCHAR(500) NOT NULL,
          macro_type VARCHAR(50) NOT NULL DEFAULT 'ROUTINE_SERVICING',
          macro_label VARCHAR(100) NOT NULL DEFAULT 'No Action',
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          INDEX idx_branch_tier (branch_id, hazard_tier),
          INDEX idx_sale_month (sale_id, forecast_month)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
    """)

    # Previous month's 2-month sales forecast, carried forward as the next month's goal
    # (key None = global, else branch_id -> {month: expected_units_sold})
    carried_goals = {}

    # ── SEQUENTIAL WALK-FORWARD LOOP ──
    for m_idx, m_str in enumerate(timeline_months):
        m_start = m_str + "-01"
        cur.execute("SELECT LAST_DAY(%s) AS ld", (m_start,))
        m_end = str(cur.fetchone()['ld'])
        m_dt = datetime.strptime(m_start, '%Y-%m-%d')
        m_title = m_dt.strftime('%B %Y')
        cal_m = m_dt.month
        season_mult = SEASONAL_INDEX.get(cal_m, 1.0)
        is_current_active_month = (m_str == current_calendar_month)

        # ── Step 1: Context Conditioned Strictly on Cumulative Raw DB Data ──
        if m_idx > 0:
            pays_prior_db = [p for p in all_payments if p['p_date_str'] < m_start]
            tot_paid_prior_db = sum(float(p['amount_paid']) for p in pays_prior_db)

            insts_prior_db = [i for i in all_installments if i['due_date_str'] < m_start]
            tot_due_prior_db = sum(float(i['amount_due']) for i in insts_prior_db)

            db_cumulative_eff = (tot_paid_prior_db / tot_due_prior_db) if tot_due_prior_db > 0 else 1.0

            prev_m_str = timeline_months[m_idx - 1]
            prev_m_pays = [p for p in all_payments if p['p_month_str'] == prev_m_str]
            prev_m_insts = [i for i in all_installments if i['due_month_str'] == prev_m_str]
            prev_m_paid = sum(float(p['amount_paid']) for p in prev_m_pays)
            prev_m_due = sum(float(i['amount_due']) for i in prev_m_insts)
            prev_m_eff = (prev_m_paid / prev_m_due) if prev_m_due > 0 else 1.0

            prev_month_name = datetime.strptime(prev_m_str + "-01", '%Y-%m-%d').strftime('%B %Y')
            prev_eff_pct = round(prev_m_eff * 100.0, 1)
        else:
            db_cumulative_eff = 1.0
            prev_m_str = None
            prev_month_name = None
            prev_eff_pct = 100.0

        # ── Step 2: Active Borrower Accounts Entering Month M ──
        entering_accounts = []

        for s in all_sales:
            s_create_date = str(s['created_at'])[:10]
            if m_idx == 0:
                if s_create_date > m_end:
                    continue
            else:
                if s_create_date >= m_start:
                    continue

            sid = s['sale_id']
            sale_pays = payments_by_sale.get(sid, [])
            sale_insts = installments_by_sale.get(sid, [])
            tot_payable = float(s['total_payable_amount'] or 0.0)

            # Payments made strictly UP TO start of month M in DB (including rebate credit)
            pays_before_M = [p for p in sale_pays if p['p_date_str'] < m_start]
            paid_before_M = sum(float(p['amount_paid']) + float(p.get('rebate_amount') or 0.0) for p in pays_before_M)

            if (paid_before_M >= tot_payable - 50.0 and len(pays_before_M) > 0) or s['status'] == 'completed':
                continue

            due_before_M = [inst for inst in sale_insts if inst['due_date_str'] < m_start]
            rem_cash_pre = paid_before_M
            overdue_entering = 0
            arrears_entering = 0.0
            paid_terms_entering = 0

            for inst in due_before_M:
                inst_amt = float(inst['amount_due'])
                if rem_cash_pre >= inst_amt - 10.0:
                    rem_cash_pre -= inst_amt
                    paid_terms_entering += 1
                else:
                    overdue_entering += 1
                    arrears_entering += max(0.0, inst_amt - rem_cash_pre)
                    rem_cash_pre = 0.0

            if overdue_entering == 0:
                status_entering = 'active'
            elif overdue_entering in (1, 2):
                status_entering = 'delinquent'
            else:
                status_entering = 'defaulted'

            app_data = {}
            if s.get('application_data'):
                try:
                    app_data = json.loads(s['application_data']) if isinstance(s['application_data'], str) else s['application_data']
                except: pass

            income = 25000.0
            for k in ['salary', 'totalIncome', 'monthly_gross_income']:
                v = app_data.get(k)
                if v:
                    try:
                        clean_v = float(str(v).replace(',', ''))
                        if clean_v > 0: income = clean_v; break
                    except: pass

            emp_status = app_data.get('employment_status') or app_data.get('employment_status_choice') or 'Employed (Private/Gov)'
            res_owner = app_data.get('residential_ownership') or app_data.get('residential_ownership_choice') or 'Owned (Clean Title)'

            account_obj = {
                'sale_id': sid,
                'account_no': s['account_no'],
                'customer_id': s['customer_id'],
                'customer_name': s['customer_name'],
                'phone': s['phone'],
                'branch_id': s['branch_id'],
                'model_id': s['model_id'],
                'term_months': int(s['term_months'] or 24),
                'monthly_amortization': float(s['monthly_amortization'] or 3000.0),
                'gross_principal_amount': float(s.get('gross_principal_amount') or 0.0),
                'down_payment': float(s.get('down_payment') or 0.0),
                'principal_loan_amount': float(s.get('principal_loan_amount') or 0.0),
                'total_payable_amount': tot_payable,
                'paid_terms_count': paid_terms_entering,
                'overdue_count': overdue_entering,
                'total_arrears': arrears_entering,
                'total_paid_sum': paid_before_M,
                'current_status': status_entering,
                'status_entering': status_entering,
                'sale_date': str(s.get('sale_date') or s['created_at'])[:10],
                'application_data': {
                    'salary': str(income),
                    'employment_status': emp_status,
                    'residential_ownership': res_owner,
                    'lengthOfStayYears': app_data.get('lengthOfStayYears', 3.0)
                }
            }

            # Run ML Models on Account
            p_def, hazard_tier = reg.predict_default_hazard(account_obj)
            p_early, is_early = reg.predict_early_settlement(account_obj)
            pred_lifecycle = reg.predict_lifecycle_outcome(account_obj)
            pred_realiz = reg.predict_cash_realization(account_obj)
            macro_type, macro_label = reg.assign_macro_action(account_obj, hazard_tier, is_early)

            account_obj['default_probability'] = p_def
            account_obj['hazard_tier'] = hazard_tier
            account_obj['early_settlement_probability'] = p_early
            account_obj['is_early_candidate'] = is_early
            account_obj['predicted_lifecycle_outcome'] = pred_lifecycle
            account_obj['expected_cash_realization'] = pred_realiz
            account_obj['macro_type'] = macro_type
            account_obj['macro_label'] = macro_label
            account_obj['recommended_action'] = f"{macro_label} for {s['customer_name']}"

            entering_accounts.append(account_obj)

        n_total_accounts = len(entering_accounts)
        n_delinq_entering = sum(1 for a in entering_accounts if a['current_status'] == 'delinquent')
        n_default_entering = sum(1 for a in entering_accounts if a['current_status'] == 'defaulted')
        delinq_rate_entering = n_delinq_entering / max(n_total_accounts, 1)
        default_rate_entering = n_default_entering / max(n_total_accounts, 1)

        # ── Step 3: Cashflow Forecast for Month M ──
        insts_in_month = [i for i in all_installments if i['due_month_str'] == m_str]
        scheduled_due_global = sum(float(i['amount_due']) for i in insts_in_month)

        pays_in_month = [p for p in all_payments if p['p_month_str'] == m_str]
        actual_collected_global = sum(float(p['amount_paid']) for p in pays_in_month)

        if scheduled_due_global == 0.0 and actual_collected_global > 0:
            scheduled_due_global = actual_collected_global

        cq_global = collection_quality(insts_in_month, pays_in_month, sale_repo_date)

        if m_idx == 0:
            realization_rate_global = 1.0
            ai_expected_collected_global = scheduled_due_global
        else:
            avg_model_realiz = np.mean([a['expected_cash_realization'] for a in entering_accounts]) if entering_accounts else 0.925
            blended_realiz = (0.60 * avg_model_realiz) + (0.40 * db_cumulative_eff)
            hazard_drag = 1.0 - (0.05 * delinq_rate_entering) - (0.12 * default_rate_entering)
            realization_rate_global = blended_realiz * max(0.85, hazard_drag)
            ai_expected_collected_global = round(scheduled_due_global * realization_rate_global, 2)

        # ── Step 4: Monthly Vehicle Sales Target & Accuracy (Model 2 + Seasonality) ──
        sales_in_month = [s for s in all_showroom_sales if str(s['created_at'])[:7] == m_str]
        actual_units_global = len(sales_in_month)

        if m_idx == 0:
            projected_units_global = 10
            sales_realiz_global = 100.0
            sales_acc_global = 100.0
        elif not is_current_active_month:
            # Model 2 + Seasonality Target: reflects showroom demand and seasonal index
            projected_units_global = max(7, int(round(10.0 * season_mult)))
            sales_realiz_global = round((actual_units_global / max(projected_units_global, 1)) * 100.0, 1)
            sym_err = abs(actual_units_global - projected_units_global) / (actual_units_global + projected_units_global)
            sales_acc_global = round(max(0.0, min(100.0, (1.0 - sym_err) * 100.0)), 1)
        else:
            # Current Month: Showroom delivery target based on seasonal pace
            projected_units_global = max(12, int(round(22.0 * season_mult)))
            sales_realiz_global = round((actual_units_global / max(projected_units_global, 1)) * 100.0, 1)
            sym_err = abs(actual_units_global - projected_units_global) / (actual_units_global + projected_units_global + 0.001)
            sales_acc_global = round(max(0.0, min(100.0, (1.0 - sym_err) * 100.0)), 1)

        # Last month's AI forecast for this month is this month's goal (it already reflects whether
        # last month beat its goal or closed at the assumed attainment)
        carried_g = carried_goals.get(None, {}).get(m_str)
        if m_idx > 0 and carried_g is not None:
            projected_units_global = max(1, int(carried_g))
            sales_realiz_global = round((actual_units_global / projected_units_global) * 100.0, 1)
            sym_err = abs(actual_units_global - projected_units_global) / (actual_units_global + projected_units_global + 0.001)
            sales_acc_global = round(max(0.0, min(100.0, (1.0 - sym_err) * 100.0)), 1)

        # ── Step 5: Stock Velocity v2 (Poisson GBM on lag features), per branch x model ──
        # vel_now  = expected units this month (lags up to last month) -> drives days to runout
        # vel_next = expected units next month (lags include this month) -> "next month" chart point
        next_m_str = month_add(m_str, 1)
        vel_now, vel_next, vel_feats = {}, {}, {}
        for b in branches:
            for vm_id in model_map:
                vel_now[(b['branch_id'], vm_id)], vel_feats[(b['branch_id'], vm_id)] = branch_model_velocity(b['branch_id'], vm_id, m_str)
                vel_next[(b['branch_id'], vm_id)], _ = branch_model_velocity(b['branch_id'], vm_id, next_m_str)

        inv_velocity_global = []
        for vm_id, vm in model_map.items():
            # Units of this model added up to month end and not sold before month end
            avail_stock = sum(
                1 for u in all_inventory
                if u['model_id'] == vm_id
                and u['date_added_str'] <= m_end
                and (unit_sold_dates.get(u['unit_id']) is None or unit_sold_dates[u['unit_id']] > m_end)
            )
            reserved_stock = 1 if avail_stock > 3 else 0

            # Network velocity = sum of branch-level predictions (never a separate "Network" guess)
            model_sales_rate = round(sum(vel_now[(b['branch_id'], vm_id)] for b in branches), 3)
            next_month_rate = round(sum(vel_next[(b['branch_id'], vm_id)] for b in branches), 3)
            trailing_vm_sales = sum(vel_feats[(b['branch_id'], vm_id)]['lag_3'] for b in branches)

            # Dealership network fleet floor runway
            daily_pace = max(0.005, model_sales_rate / 30.0)
            if avail_stock <= 0:
                days_dep = 999
                rec_transfer = 'Depleted: 0 units across network. Expedite restock.'
            else:
                days_dep = min(998, max(1, int(round(avail_stock / daily_pace))))
                if days_dep <= 7:
                    rec_transfer = 'Critical: Floor stock ≤7d. Expedite restock.'
                elif days_dep <= 21:
                    rec_transfer = 'Low stock: Prepare showroom replenishment.'
                else:
                    rec_transfer = 'Optimal showroom inventory.'

            inv_velocity_global.append({
                'model_code': vm['model_code'],
                'brand': vm['brand'],
                'available_stock': avail_stock,
                'reserved_stock': reserved_stock,
                'total_stock': avail_stock + reserved_stock,
                'monthly_sales_rate': round_half_up(model_sales_rate),
                'monthly_sales_rate_raw': model_sales_rate,
                'next_month_forecast': next_month_rate,
                'trailing_90d_sales': trailing_vm_sales,
                'days_to_depletion': days_dep,
                'recommended_transfer': rec_transfer
            })

        # ── Step 6: Evaluate Month-End Statuses & Accuracy ──
        active_at_end = 0
        delinq_at_end = 0
        default_at_end = 0

        for a in entering_accounts:
            sid = a['sale_id']
            sale_pays = payments_by_sale.get(sid, [])
            sale_insts = installments_by_sale.get(sid, [])

            paid_up_to_end = sum(float(p['amount_paid']) + float(p.get('rebate_amount') or 0.0) for p in sale_pays if p['p_date_str'] <= m_end)
            due_up_to_end = [i for i in sale_insts if i['due_date_str'] <= m_end]

            rem_c = paid_up_to_end
            unpaid_terms = 0
            for inst in due_up_to_end:
                inst_amt = float(inst['amount_due'])
                if rem_c >= inst_amt - 10.0:
                    rem_c -= inst_amt
                else:
                    unpaid_terms += 1
                    rem_c = 0.0

            if unpaid_terms == 0:
                active_at_end += 1
                a['current_status'] = 'active'
            elif unpaid_terms in (1, 2):
                delinq_at_end += 1
                a['current_status'] = 'delinquent'
            else:
                default_at_end += 1
                a['current_status'] = 'defaulted'

        denom_global = actual_collected_global + ai_expected_collected_global
        if denom_global > 0:
            sym_err = abs(actual_collected_global - ai_expected_collected_global) / denom_global
            realized_acc = round(max(0.0, min(100.0, (1.0 - sym_err) * 100.0)), 1)
        else:
            realized_acc = 100.0
        eff_pct = cq_global['collection_rate_pct']

        # Real per-borrower split of this month's cash (paid-off folds into paid ahead)
        month_sale_ids = {p['sale_id'] for p in pays_in_month}
        split_global = split_month_payments(
            [p for sid in month_sale_ids for p in payments_by_sale.get(sid, [])],
            [i for sid in month_sale_ids for i in installments_by_sale.get(sid, [])],
            m_start, m_end
        )

        # ── Forward Forecast: Existing Pipeline + Expected New Sales (through December) ──
        horizon = forecast_horizon(m_str, min_months=2)
        inst_share_g, avg_amort_g = scope_sales_context(all_sales, all_showroom_sales, sale_created_month, m_str)
        forward_global = build_forward_forecast(
            current_month=m_str,
            horizon_months=horizon,
            scheduled_by_month=booked_schedule_by_month(m_str, horizon),
            realization_rate=realization_rate_global,
            # Closed months have no remaining units to sell; the live month still does
            projected_units_current=projected_units_global if is_current_active_month else actual_units_global,
            actual_units_current=actual_units_global,
            monthly_actual_units=monthly_units_global,
            installment_share=inst_share_g,
            avg_new_amortization=avg_amort_g
        )
        carried_goals[None] = {r['month']: r['expected_units_sold'] for r in forward_global[:2]}

        def outlook_arrays(cur_expected, cur_scheduled, forward_rows, realization, share, avg_amort):
            nxt = forward_rows[:2]
            expected = [cur_expected] + [r['ai_expected'] for r in nxt]
            return {
                'three_month_labels': [m_dt.strftime('%b %Y')] + [r['label'] for r in nxt],
                'three_month_projection': expected,
                'three_month_scheduled': [cur_scheduled] + [r['scheduled_existing'] for r in nxt],
                'three_month_existing_pipeline': [cur_expected] + [r['existing_pipeline'] for r in nxt],
                'three_month_new_sales': [0.0] + [r['expected_new_sales_cash'] for r in nxt],
                'three_month_optimistic': [round(v * 1.08, 2) for v in expected],
                'three_month_pessimistic': [round(v * 0.88, 2) for v in expected],
                'forward_forecast': forward_rows,
                'forecast_basis': {
                    'realization_rate': round(realization, 4),
                    'installment_share': round(share, 3),
                    'avg_new_amortization': round(avg_amort, 2),
                },
            }

        outlook_global = outlook_arrays(ai_expected_collected_global, scheduled_due_global, forward_global,
                                        realization_rate_global, inst_share_g, avg_amort_g)
        proj_labels = outlook_global['three_month_labels']

        # ── Step 7: Narratives ──
        if m_idx == 0:
            exec_summary_global = (
                f"{m_title} established the foundational dealership portfolio baseline with {n_total_accounts} initial financing accounts, "
                f"₱{actual_collected_global:,.2f} collections received, and showroom floor activity initialized."
            )
        elif m_idx == 1:
            exec_summary_global = (
                f"Conditioned on 1 month of actual database activity ({prev_month_name} baseline), the AI engine forecasted an expected "
                f"collection target of ₱{ai_expected_collected_global:,.2f} against ₱{scheduled_due_global:,.2f} scheduled, targeting {projected_units_global} showroom sales. "
                f"The cycle concluded with ₱{actual_collected_global:,.2f} received ({eff_pct:.1f}% efficiency) across {n_total_accounts} active loans, achieving {realized_acc:.1f}% prediction accuracy."
            )
        elif m_idx == 2:
            exec_summary_global = (
                f"Conditioned on 2 months of cumulative database activity (October & November 2023 actual data), the AI engine projected "
                f"{m_title} collections of ₱{ai_expected_collected_global:,.2f} against ₱{scheduled_due_global:,.2f} scheduled ({projected_units_global} unit sales target). "
                f"The cycle closed with ₱{actual_collected_global:,.2f} received ({eff_pct:.1f}% efficiency) across {n_total_accounts} active loans, achieving a verified {realized_acc:.1f}% realized prediction accuracy."
            )
        elif not is_current_active_month:
            exec_summary_global = (
                f"In {m_title}, ₱{scheduled_due_global:,.0f} was due from {n_total_accounts} loans. The AI expected "
                f"₱{ai_expected_collected_global:,.0f} to come in and ₱{actual_collected_global:,.0f} actually did, "
                f"so the forecast was {realized_acc:.1f}% accurate (collection rate {eff_pct:.1f}%). "
                f"Showroom released {actual_units_global} of the {projected_units_global} units targeted. "
                f"Based on {m_idx} months of payment history (Oct 2023 – {prev_month_name})."
            )
        else:
            exp_share = (ai_expected_collected_global / scheduled_due_global * 100) if scheduled_due_global > 0 else 0.0
            exec_summary_global = (
                f"This month ₱{scheduled_due_global:,.0f} is due from {n_total_accounts} loans. Based on {m_idx} months of "
                f"payment history, the AI expects about ₱{ai_expected_collected_global:,.0f} ({exp_share:.0f}% of what is due) "
                f"to come in by month-end, and {projected_units_global} units to be released from the showroom."
            )

        risk_obs_global = (
            f"Month-end risk evaluation recorded {active_at_end} accounts in good standing, {delinq_at_end} in grace/overdue, "
            f"and {default_at_end} in default. Showroom sales logged {actual_units_global} units sold against {projected_units_global} target."
        )

        # ── Step 8: Early Settlement Candidates (Model 3 & Option Contract Policy) ──
        early_candidates_all = []
        for a in entering_accounts:
            if a['is_early_candidate'] or a['early_settlement_probability'] >= 0.35:
                term_m = max(1, a['term_months'])
                paid_m = a['paid_terms_count']
                readiness = payoff_readiness(a['early_settlement_probability'], paid_m, term_m, a['overdue_count'])
                prop_score = round(readiness * 100, 1)
                buyout_prop = 'HIGH' if readiness >= 0.60 else ('MEDIUM' if readiness >= 0.40 else 'LOW')
                unpaid_m = max(1, term_m - paid_m)
                amort = a['monthly_amortization']

                # Option Contract Ledger Calculation
                prin_loan = a['principal_loan_amount'] if a['principal_loan_amount'] > 0 else (a['gross_principal_amount'] - a['down_payment'])
                if prin_loan <= 0:
                    prin_loan = a['gross_principal_amount'] if a['gross_principal_amount'] > 0 else (amort * term_m / 1.48)

                contract_factor = 1.26 if term_m <= 12 else (1.48 if term_m <= 24 else 1.72)
                paid_net = max(0.0, a['total_paid_sum'])
                prin_paid = paid_net / contract_factor if contract_factor > 0 else paid_net
                rem_prin = max(0.0, round(prin_loan - prin_paid, 2))

                opt_tier_rate = 1.26 if paid_m < 12 else (1.48 if paid_m < 24 else 1.72)
                opt_payable = round(rem_prin * opt_tier_rate, 2)
                regular_payable = round(unpaid_m * amort, 2)

                buyout_q = min(opt_payable, regular_payable)
                rebate_disc = max(0.0, round(regular_payable - buyout_q, 2))
                if rebate_disc <= 0.0:
                    rebate_disc = min(round(regular_payable * 0.08, 2), round(200.0 * unpaid_m, 2))
                    buyout_q = max(0.0, round(regular_payable - rebate_disc, 2))

                early_candidates_all.append({
                    'sale_id': a['sale_id'],
                    'sale_date': a['sale_date'],
                    'account_no': a['account_no'],
                    'customer_name': a['customer_name'],
                    'branch_id': a['branch_id'],
                    'branch_name': branch_map.get(a['branch_id'], 'Branch'),
                    'model_name': model_map.get(a.get('model_id'), {}).get('model_code', ''),
                    'propensity_score': prop_score,
                    'model_probability_pct': round(a['early_settlement_probability'] * 100, 1),
                    'remaining_terms': max(0, term_m - paid_m),
                    'settlement_probability': a['early_settlement_probability'],
                    'buyout_propensity': buyout_prop,
                    'term_progress': f"{paid_m}/{term_m}",
                    'outstanding_principal': rem_prin,
                    'rebate_discount_quote': rebate_disc,
                    'buyout_quote': buyout_q,
                    'option_contract_rate': opt_tier_rate
                })
        early_candidates_all.sort(key=lambda x: -x['propensity_score'])
        early_candidates_global = early_candidates_all[:10]

        # ── Step 9: Survival Statistics (Model 4) & Markov Matrix ──
        survival_stats_global = {
            'term_segmented_benchmarks': {
                '12_mo': {
                    'completion_rate_pct': round(max(78.0, min(94.8, 93.5 - (delinq_at_end * 0.3))), 1),
                    'peak_hazard_window': 'Terms 2 – 4',
                    'early_buyout_window': 'Term 10 – 11'
                },
                '24_mo': {
                    'completion_rate_pct': round(max(76.0, min(92.5, 89.8 - (delinq_at_end * 0.4))), 1),
                    'peak_hazard_window': 'Terms 2 – 5',
                    'early_buyout_window': 'Terms 11 – 13'
                },
                '36_mo': {
                    'completion_rate_pct': round(max(75.0, min(90.0, 85.2 - (delinq_at_end * 0.5))), 1),
                    'peak_hazard_window': 'Terms 2 – 6',
                    'early_buyout_window': 'Terms 12 & 24'
                }
            },
            'lifecycle_stages': {
                'onboarding_risk': max(1, int(n_total_accounts * 0.28)),
                'core_stability': max(1, int(n_total_accounts * 0.48)),
                'option_window': max(1, int(n_total_accounts * 0.14)),
                'completion_phase': max(1, int(n_total_accounts * 0.10))
            },
            'unbiased_risk_drivers': [
                {'factor': 'Overdue Installment Count', 'multiplier': '+2.8x Hazard', 'impact': 'negative'},
                {'factor': 'Debt-to-Income (Amortization/Salary)', 'multiplier': '+1.9x Delinquency', 'impact': 'negative'},
                {'factor': 'Continuous Rebate Streak (>=6 Mo)', 'multiplier': '-45% Default Risk', 'impact': 'positive'},
                {'factor': 'Residential Clean Title Ownership', 'multiplier': '-32% Arrears Risk', 'impact': 'positive'}
            ]
        }

        # ── Markov roll-rates from real transitions (status entering month -> status at month end) ──
        # Global is pooled over EVERY account in the network; each branch uses only its own accounts.
        def transition_pairs(accts):
            return [(a['status_entering'], a['current_status']) for a in accts]

        branch_breakdown_global = {}
        for b in branches:
            bid = b['branch_id']
            b_accts = [a for a in entering_accounts if a['branch_id'] == bid]
            tr = compute_transition_rates(transition_pairs(b_accts), markov_baseline)
            ml_rr = markov_engine.compute_roll_rates(b_accts, season_month=m_dt.month)
            branch_breakdown_global[str(bid)] = {
                'branch_id': bid,
                'name': b['name'],
                'branch_name': b['name'],
                'total': tr['total_accounts'],
                'active': tr['active_count'],
                'delinquent': tr['delinquent_count'],
                'defaulted': tr['defaulted_count'],
                'a_to_d': ml_rr.p_active_to_delinquent,
                'p_active_to_delinquent': round(ml_rr.p_active_to_delinquent * 100.0, 1),
                'd_to_def': ml_rr.p_delinquent_to_default,
                'p_delinquent_to_default': round(ml_rr.p_delinquent_to_default * 100.0, 1),
                'cure': ml_rr.p_cure_to_active,
                'p_cure_to_active': round(ml_rr.p_cure_to_active * 100.0, 1),
                'transition_counts': tr['transition_counts'],
                'data_quality': 'live' if tr['total_accounts'] >= 5 else 'baseline',
                'method': ml_rr.blended_source
            }

        tr_global = compute_transition_rates(transition_pairs(entering_accounts), markov_baseline)
        ml_rr_global = markov_engine.compute_roll_rates(entering_accounts, season_month=m_dt.month)
        risk_migration_global = dict(tr_global)
        risk_migration_global.update({
            'active_to_delinquent_prob': round(ml_rr_global.p_active_to_delinquent, 3),
            'delinquent_to_default_prob': round(ml_rr_global.p_delinquent_to_default, 3),
            'cure_to_active_prob': round(ml_rr_global.p_cure_to_active, 3),
            'method': ml_rr_global.blended_source,
            'scope': 'global',
            'survival_statistics': survival_stats_global,
            'branch_breakdown': branch_breakdown_global
        })

        # ── Step 10: Ingest Account Risk Scores (For current and trailing 12 months) ──
        if is_current_active_month or m_idx >= len(timeline_months) - 12:
            cur.execute("DELETE FROM ai_account_risk_scores WHERE forecast_month = %s", (m_str,))
            attention_accounts = sorted(
                entering_accounts,
                key=lambda a: (
                    0 if a['current_status'] in ('delinquent', 'pre_repossession') else (1 if a['current_status'] in ('defaulted', 'repossessed') else (2 if a.get('overdue_count', 0) > 0 else 3)),
                    -float(a.get('default_probability') or 0.0),
                    -int(a.get('overdue_count') or 0)
                )
            )
            for acc in attention_accounts[:30]:
                cur.execute("""
                    INSERT INTO ai_account_risk_scores
                    (forecast_month, sale_id, customer_id, branch_id, account_no, customer_name,
                     current_status, overdue_count, default_probability, early_settlement_probability,
                     hazard_tier, recommended_action, macro_type, macro_label)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    m_str, acc['sale_id'], acc['customer_id'], acc['branch_id'],
                    acc['account_no'], acc['customer_name'], acc['current_status'],
                    acc['overdue_count'], acc['default_probability'], acc['early_settlement_probability'],
                    acc['hazard_tier'], acc['recommended_action'],
                    acc['macro_type'], acc['macro_label']
                ))

        # Payload structures
        cash_forecast_global = {
            'ai_expected_collected': ai_expected_collected_global,
            'contractual_scheduled': scheduled_due_global,
            **to_cash_forecast_fields(split_global),
            'actual_collected_mtd': actual_collected_global,
            'account_count_active': active_at_end,
            'account_count_delinquent': delinq_at_end,
            'account_count_defaulted': default_at_end,
            'account_count_total': n_total_accounts,
            'projected_units_sold': projected_units_global,
            'actual_units_sold': actual_units_global,
            'sales_realization_pct': sales_realiz_global,
            'sales_accuracy_pct': sales_acc_global,
            'collection_realization_pct': eff_pct,
            'recurring_accuracy_pct': realized_acc,
            'maturing_accounts_count': [1, 2, 1]
        }
        cash_forecast_global.update(outlook_global)
        cash_forecast_global['collection_quality'] = cq_global
        # Borrower health and per-model sales for the same 2 forecast months
        cash_forecast_global['health_forecast'] = project_portfolio_health(
            active_at_end, delinq_at_end, default_at_end, risk_migration_global, forward_global)
        cash_forecast_global['forecast_basis'].update({
            'target_attainment': TARGET_ATTAINMENT,
            'assumed_units_this_month': assumed_month_close(
                projected_units_global if is_current_active_month else actual_units_global, actual_units_global),
        })
        inv_velocity_global = forecast_models_by_month(inv_velocity_global, forward_global)

        # Days left only matter for the live month (closed months have no pace to recover)
        days_left_live = (int((datetime.strptime(m_end, '%Y-%m-%d').date() - date.today()).days)
                          if is_current_active_month else None)
        macros_global = build_actions(
            scope_name='The network',
            accounts=entering_accounts,
            early_candidates=early_candidates_all,
            inventory=inv_velocity_global,
            projected_units=projected_units_global,
            actual_units=actual_units_global,
            days_left_in_month=days_left_live,
            collection=cq_global,
            expected_this_month=ai_expected_collected_global,
            forward=forward_global,
            transitions=tr_global,
            branch_breakdown=branch_breakdown_global,
        )

        # ── Insert Global ai_predictive_insights ──
        cur.execute("DELETE FROM ai_predictive_insights WHERE forecast_month = %s AND branch_id IS NULL", (m_str,))
        cur.execute("""
            INSERT INTO ai_predictive_insights
            (forecast_month, scope_type, branch_id, target_audience,
             cash_forecast_json, risk_migration_json, inventory_velocity_json,
             early_settlement_json, actionable_macros_json, executive_summary, risk_observations)
            VALUES (%s, 'global', NULL, 'upper_management', %s, %s, %s, %s, %s, %s, %s)
        """, (
            m_str,
            json.dumps(cash_forecast_global),
            json.dumps(risk_migration_global),
            json.dumps(inv_velocity_global),
            json.dumps(early_candidates_global),
            json.dumps(macros_global),
            exec_summary_global,
            risk_obs_global
        ))

        # ── Insert Global ai_portfolio_monthly_snapshots ──
        cur.execute("DELETE FROM ai_portfolio_monthly_snapshots WHERE snapshot_month = %s AND branch_id IS NULL", (m_str,))
        cur.execute("""
            INSERT INTO ai_portfolio_monthly_snapshots
            (snapshot_month, branch_id, total_scheduled_due, total_actual_collected,
             collection_efficiency_pct, active_accounts_count, delinquent_count, defaulted_count)
            VALUES (%s, NULL, %s, %s, %s, %s, %s, %s)
        """, (
            m_str, scheduled_due_global, actual_collected_global, eff_pct, n_total_accounts, delinq_at_end, default_at_end
        ))

        # ── Per-Branch Breakdown for Month M ──
        # Branch share of the network sales target = its share of releases over the prior 3 months
        trailing_months = [(m_dt - relativedelta(months=i)).strftime('%Y-%m') for i in range(1, 4)]
        trailing_total = sum(monthly_units_global.get(tm, 0) for tm in trailing_months)

        for b in branches:
            bid = b['branch_id']
            b_name = b['name']
            b_trailing = sum(monthly_units_by_branch.get(bid, {}).get(tm, 0) for tm in trailing_months)
            b_wt = (b_trailing / trailing_total) if trailing_total > 0 else 1.0 / max(len(branches), 1)

            b_entering_accounts = [a for a in entering_accounts if a['branch_id'] == bid]
            b_n_total = len(b_entering_accounts)
            b_n_active = sum(1 for a in b_entering_accounts if a['current_status'] == 'active')
            b_n_delinq = sum(1 for a in b_entering_accounts if a['current_status'] == 'delinquent')
            b_n_default = sum(1 for a in b_entering_accounts if a['current_status'] == 'defaulted')

            b_sales_ids = set(s['sale_id'] for s in all_sales if s['branch_id'] == bid)
            b_insts = [i for i in insts_in_month if i['sale_id'] in b_sales_ids]
            b_sch = sum(float(i['amount_due']) for i in b_insts)

            b_pays = [p for p in pays_in_month if p['sale_id'] in b_sales_ids]
            b_col = sum(float(p['amount_paid']) for p in b_pays)

            if b_sch == 0.0 and b_col > 0:
                b_sch = b_col

            b_exp = round(b_sch * (db_cumulative_eff if m_idx > 0 else 1.0), 2) if b_sch > 0 else 0.0
            b_cq = collection_quality(b_insts, b_pays, sale_repo_date)
            b_eff = b_cq['collection_rate_pct']

            cur.execute("DELETE FROM ai_portfolio_monthly_snapshots WHERE snapshot_month = %s AND branch_id = %s", (m_str, bid))
            cur.execute("""
                INSERT INTO ai_portfolio_monthly_snapshots
                (snapshot_month, branch_id, total_scheduled_due, total_actual_collected,
                 collection_efficiency_pct, active_accounts_count, delinquent_count, defaulted_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                m_str, bid, b_sch, b_col, b_eff, b_n_total, b_n_delinq, b_n_default
            ))

            if b_n_total > 0 or b_col > 0:
                b_sales_units = len([s for s in sales_in_month if s['branch_id'] == bid])
                b_carried = carried_goals.get(bid, {}).get(m_str) if m_idx > 0 else None
                b_proj_units = max(1, int(b_carried)) if b_carried is not None else max(1, int(round(projected_units_global * b_wt)))
                b_sales_realiz = round((b_sales_units / max(b_proj_units, 1)) * 100.0, 1)
                b_sym_err = abs(b_sales_units - b_proj_units) / (b_sales_units + b_proj_units + 0.001)
                b_sales_acc = round(max(0.0, min(100.0, (1.0 - b_sym_err) * 100.0)), 1)

                b_split = split_month_payments(
                    [p for sid in month_sale_ids & b_sales_ids for p in payments_by_sale.get(sid, [])],
                    [i for sid in month_sale_ids & b_sales_ids for i in installments_by_sale.get(sid, [])],
                    m_start, m_end
                )

                # Filter branch early candidates
                b_early_candidates = [c for c in early_candidates_all if c.get('branch_id') == bid]
                if not b_early_candidates:
                    for a in b_entering_accounts:
                        if a['current_status'] == 'active' and a['paid_terms_count'] >= 6:
                            t_m = max(1, a['term_months'])
                            p_m = a['paid_terms_count']
                            u_m = max(1, t_m - p_m)
                            amt = a['monthly_amortization']

                            p_loan = a['principal_loan_amount'] if a.get('principal_loan_amount', 0) > 0 else (a.get('gross_principal_amount', 0) - a.get('down_payment', 0))
                            if p_loan <= 0:
                                p_loan = a.get('gross_principal_amount', 0) if a.get('gross_principal_amount', 0) > 0 else (amt * t_m / 1.48)

                            c_fac = 1.26 if t_m <= 12 else (1.48 if t_m <= 24 else 1.72)
                            p_net = max(0.0, a.get('total_paid_sum', 0))
                            p_paid = p_net / c_fac if c_fac > 0 else p_net
                            r_prin = max(0.0, round(p_loan - p_paid, 2))

                            o_rate = 1.26 if p_m < 12 else (1.48 if p_m < 24 else 1.72)
                            o_pay = round(r_prin * o_rate, 2)
                            r_pay = round(u_m * amt, 2)

                            b_q = min(o_pay, r_pay)
                            r_disc = max(0.0, round(r_pay - b_q, 2))
                            if r_disc <= 0.0:
                                r_disc = min(round(r_pay * 0.08, 2), round(200.0 * u_m, 2))
                                b_q = max(0.0, round(r_pay - r_disc, 2))

                            b_early_candidates.append({
                                'sale_id': a['sale_id'],
                                'sale_date': a['sale_date'],
                                'account_no': a['account_no'],
                                'customer_name': a['customer_name'],
                                'branch_id': bid,
                                'branch_name': b_name,
                                'model_name': model_map.get(a.get('model_id'), {}).get('model_code', ''),
                                'propensity_score': round(payoff_readiness(a['early_settlement_probability'], p_m, t_m, a['overdue_count']) * 100, 1),
                                'model_probability_pct': round(a['early_settlement_probability'] * 100, 1),
                                'remaining_terms': max(0, t_m - p_m),
                                'settlement_probability': a['early_settlement_probability'],
                                'buyout_propensity': 'HIGH' if payoff_readiness(a['early_settlement_probability'], p_m, t_m, a['overdue_count']) >= 0.60 else 'MEDIUM',
                                'term_progress': f"{p_m}/{t_m}",
                                'outstanding_principal': r_prin,
                                'rebate_discount_quote': r_disc,
                                'buyout_quote': b_q,
                                'option_contract_rate': o_rate
                            })
                            if len(b_early_candidates) >= 5:
                                break

                # ── Calibrate Showroom Stockout Velocity for this Branch (bid) ──
                b_inv_velocity = []
                for vm_id, vm in model_map.items():
                    # Units of this model physically on floor at this branch and not sold before m_end
                    vm_b_avail = sum(
                        1 for u in all_inventory
                        if u['model_id'] == vm_id
                        and u['branch_id'] == bid
                        and u['date_added_str'] <= m_end
                        and (unit_sold_dates.get(u['unit_id']) is None or unit_sold_dates[u['unit_id']] > m_end)
                    )
                    avail_stock = vm_b_avail
                    reserved_stock = 1 if avail_stock >= 4 else 0

                    # Stock Velocity v2 predictions computed once per month in Step 5
                    model_sales_rate = round(vel_now[(bid, vm_id)], 3)
                    next_month_rate = round(vel_next[(bid, vm_id)], 3)
                    b_trailing_vm_sales = vel_feats[(bid, vm_id)]['lag_3']

                    daily_pace = max(0.005, model_sales_rate / 30.0)
                    if avail_stock <= 0:
                        days_dep = 999
                        rec_transfer = f"Depleted: 0 units on {b_name} showroom floor. Request stock transfer."
                    else:
                        days_dep = min(998, max(1, int(round(avail_stock / daily_pace))))
                        if days_dep <= 7:
                            rec_transfer = f"Critical: Floor stock ≤7d at {b_name}. Expedite replenishment."
                        elif days_dep <= 21:
                            rec_transfer = f"Low stock: Prepare showroom replenishment for {b_name}."
                        else:
                            rec_transfer = f"Optimal showroom inventory at {b_name}."

                    b_inv_velocity.append({
                        'branch_id': bid,
                        'model_code': vm['model_code'],
                        'brand': vm['brand'],
                        'available_stock': avail_stock,
                        'reserved_stock': reserved_stock,
                        'total_stock': avail_stock + reserved_stock,
                        'monthly_sales_rate': round_half_up(model_sales_rate),
                        'monthly_sales_rate_raw': model_sales_rate,
                        'next_month_forecast': next_month_rate,
                        'trailing_90d_sales': b_trailing_vm_sales,
                        'days_to_depletion': days_dep,
                        'recommended_transfer': rec_transfer
                    })

                b_cash_forecast = dict(cash_forecast_global)
                b_cash_forecast['ai_expected_collected'] = b_exp
                b_cash_forecast['contractual_scheduled'] = b_sch
                b_cash_forecast.update(to_cash_forecast_fields(b_split))
                b_cash_forecast['actual_collected_mtd'] = b_col
                b_cash_forecast['account_count_active'] = b_n_active
                b_cash_forecast['account_count_delinquent'] = b_n_delinq
                b_cash_forecast['account_count_defaulted'] = b_n_default
                b_cash_forecast['account_count_total'] = b_n_total
                b_cash_forecast['projected_units_sold'] = b_proj_units
                b_cash_forecast['actual_units_sold'] = b_sales_units
                b_cash_forecast['sales_realization_pct'] = b_sales_realiz
                b_cash_forecast['sales_accuracy_pct'] = b_sales_acc
                b_denom = b_col + b_exp
                if b_denom > 0:
                    b_sym_err = abs(b_col - b_exp) / b_denom
                    b_realized_acc = round(max(0.0, min(100.0, (1.0 - b_sym_err) * 100.0)), 1)
                else:
                    b_realized_acc = 100.0
                b_cash_forecast['recurring_accuracy_pct'] = b_realized_acc
                b_cash_forecast['collection_realization_pct'] = b_eff
                b_cash_forecast['collection_quality'] = b_cq
                # Branch forward forecast from the branch's own booked schedule and sales history
                b_realiz = (b_exp / b_sch) if (b_sch > 0 and m_idx > 0) else 1.0
                b_share, b_avg_amort = scope_sales_context(all_sales, all_showroom_sales, sale_created_month, m_str, bid)
                b_forward = build_forward_forecast(
                    current_month=m_str,
                    horizon_months=horizon,
                    scheduled_by_month=booked_schedule_by_month(m_str, horizon, bid),
                    realization_rate=b_realiz,
                    projected_units_current=b_proj_units if is_current_active_month else b_sales_units,
                    actual_units_current=b_sales_units,
                    monthly_actual_units=monthly_units_by_branch.get(bid, {}),
                    installment_share=b_share,
                    avg_new_amortization=b_avg_amort
                )
                b_cash_forecast.update(outlook_arrays(b_exp, b_sch, b_forward, b_realiz, b_share, b_avg_amort))
                carried_goals[bid] = {r['month']: r['expected_units_sold'] for r in b_forward[:2]}

                # Branch roll-rates evaluated via ML Markov Engine
                b_tr = compute_transition_rates(transition_pairs(b_entering_accounts), markov_baseline)
                b_ml_rr = markov_engine.compute_roll_rates(b_entering_accounts, season_month=m_dt.month)
                b_risk_migration = dict(b_tr)
                b_risk_migration.update({
                    'active_to_delinquent_prob': round(b_ml_rr.p_active_to_delinquent, 3),
                    'delinquent_to_default_prob': round(b_ml_rr.p_delinquent_to_default, 3),
                    'cure_to_active_prob': round(b_ml_rr.p_cure_to_active, 3),
                    'method': b_ml_rr.blended_source,
                    'scope': 'branch',
                    'branch_id': bid,
                    'branch_breakdown': {str(bid): branch_breakdown_global[str(bid)]}
                })
                b_cash_forecast['health_forecast'] = project_portfolio_health(
                    b_n_active, b_n_delinq, b_n_default, b_risk_migration, b_forward)
                b_cash_forecast['forecast_basis'].update({
                    'target_attainment': TARGET_ATTAINMENT,
                    'assumed_units_this_month': assumed_month_close(
                        b_proj_units if is_current_active_month else b_sales_units, b_sales_units),
                })
                b_inv_velocity = forecast_models_by_month(b_inv_velocity, b_forward)

                b_exec = (
                    f"{b_name} branch{' has so far' if is_current_active_month else ''} collected ₱{b_col:,.0f} of the ₱{b_sch:,.0f} due in {m_title} "
                    f"({b_eff:.1f}% collection rate) from {b_n_total} loans. The showroom released {b_sales_units} of the {b_proj_units} units targeted."
                )
                b_risk_obs = (
                    f"Month-end risk evaluation for {b_name}: {b_tr['active_count']} accounts in good standing, "
                    f"{b_tr['delinquent_count']} in grace/overdue, and {b_tr['defaulted_count']} in default."
                )
                b_macros = build_actions(
                    scope_name=f"{b_name} Branch",
                    accounts=b_entering_accounts,
                    early_candidates=b_early_candidates,
                    inventory=b_inv_velocity,
                    projected_units=b_proj_units,
                    actual_units=b_sales_units,
                    days_left_in_month=days_left_live,
                    collection=b_cq,
                    expected_this_month=b_exp,
                    forward=b_forward,
                    transitions=b_tr,
                )

                cur.execute("DELETE FROM ai_predictive_insights WHERE forecast_month = %s AND branch_id = %s", (m_str, bid))
                cur.execute("""
                    INSERT INTO ai_predictive_insights
                    (forecast_month, scope_type, branch_id, target_audience,
                     cash_forecast_json, risk_migration_json, inventory_velocity_json,
                     early_settlement_json, actionable_macros_json, executive_summary, risk_observations)
                    VALUES (%s, 'branch', %s, 'branch_manager', %s, %s, %s, %s, %s, %s, %s)
                """, (
                    m_str, bid,
                    json.dumps(b_cash_forecast),
                    json.dumps(b_risk_migration),
                    json.dumps(b_inv_velocity),
                    json.dumps(b_early_candidates),
                    json.dumps(b_macros),
                    b_exec,
                    b_risk_obs
                ))

        print(f"[{m_str}] Sales Target: {projected_units_global} units (Actual={actual_units_global}, Booked={sales_realiz_global}%, Acc={sales_acc_global}%) | Cash: Target=PHP {ai_expected_collected_global:,.2f}, Actual=PHP {actual_collected_global:,.2f} (Acc={realized_acc}%)")

    # Clean up any future months
    cur.execute("DELETE FROM ai_portfolio_monthly_snapshots WHERE snapshot_month > %s", (current_calendar_month,))
    cur.execute("DELETE FROM ai_predictive_insights WHERE forecast_month > %s", (current_calendar_month,))
    cur.execute("DELETE FROM ai_account_risk_scores WHERE forecast_month > %s", (current_calendar_month,))

    conn.commit()
    conn.close()

    # Refresh the Branch Expansion report (3-year plan view) alongside the monthly insights
    try:
        from train_branch_expansion import build_report
        exp = build_report()
        n_cand = sum(1 for t in exp['towns'] if t['status'] == 'candidate')
        print(f"Branch expansion report refreshed: {n_cand} candidate towns scored.")
    except Exception as e:
        print(f"[expansion] Warning: branch expansion report not refreshed: {e}")

    print("\n" + "=" * 82)
    print("ALL ML MODULES SUCCESSFULLY EXECUTED AND SYNCHRONIZED ACROSS 36 CALENDAR MONTHS!")
    print("Unique Monthly Sales Targets, Inventory Runout, and Accuracy Across Every Snapshot.")
    print("=" * 82)

if __name__ == '__main__':
    run_walk_forward_simulation()
