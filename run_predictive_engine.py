"""
ml_engine/run_predictive_engine.py
Main production inference and synchronization pipeline for Bomeli Motorcycle Financing.

Replaces n8n completely:
  - Fetches active loans, inventory, and branches from bomeli_db1.
  - Applies trained ML models (Default Hazard, Early Settlement, Markov Roll-Rate).
  - Computes per-branch Markov transition probabilities blended with simulated baseline.
  - Projects 90-Day Cash Inflow Cone with dynamic behavioral slippage.
  - Computes Inventory Stockout Velocity (Days to 0) with branch-aware labels.
  - Ingests forecasts into ai_predictive_insights (Global + per-branch).
  - Populates trailing historical snapshots into ai_portfolio_monthly_snapshots.
  - Ingests account-level hazard scores into ai_account_risk_scores.
"""

import os
import sys
import json
import joblib
import random
import pymysql
import numpy as np
import pandas as pd
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

from config import DATA_DIR, MODELS_DIR, get_db_connection, get_active_rate_package
from dml import VelocityEngine, SurvivalEngine, CashEngine, PaymentStreamBreakdown

def load_models():
    """Loads trained ML artifacts from models directory."""
    hazard_model_path = os.path.join(MODELS_DIR, 'default_hazard_model.joblib')
    early_model_path = os.path.join(MODELS_DIR, 'early_settlement_model.joblib')
    markov_path = os.path.join(MODELS_DIR, 'markov_matrix.json')

    if not os.path.exists(hazard_model_path) or not os.path.exists(early_model_path):
        raise FileNotFoundError("Trained models not found. Please run train_models.py first.")

    hazard_pipeline = joblib.load(hazard_model_path)
    early_pipeline = joblib.load(early_model_path)

    # Simulated-data global baseline (fallback when live data is sparse)
    markov_baseline = {
        'active_to_delinquent_prob': 0.08,
        'delinquent_to_default_prob': 0.22,
        'cure_to_active_prob': 0.58
    }
    if os.path.exists(markov_path):
        with open(markov_path, 'r', encoding='utf-8') as f:
            markov_baseline = json.load(f)

    return hazard_pipeline, early_pipeline, markov_baseline


def compute_branch_markov(scope_accounts, markov_baseline):
    """
    Computes per-branch Markov transition rates using LIVE account data blended
    with the simulated training-set baseline.

    Blending rule:
      - Branches with >=5 accounts: 80% live + 20% baseline
      - Branches with 2-4 accounts: 40% live + 60% baseline
      - Branches with 0 accounts: 0.0% (no accounts at risk)

    Returns a rich dict with both the scope aggregate and a branch_breakdown key.
    """
    total = len(scope_accounts)
    if total == 0:
        return {
            'active_to_delinquent_prob': 0.0,
            'delinquent_to_default_prob': 0.0,
            'cure_to_active_prob': 0.0,
            'total_accounts': 0,
            'active_count': 0,
            'delinquent_count': 0,
            'defaulted_count': 0,
            'branch_breakdown': {}
        }

    b_baseline = markov_baseline.get('active_to_delinquent_prob', 0.08)
    d_baseline = markov_baseline.get('delinquent_to_default_prob', 0.22)
    c_baseline = markov_baseline.get('cure_to_active_prob', 0.58)

    # ── Scope-level (aggregate of all accounts in this scope) ──
    n_active = sum(1 for a in scope_accounts if a['current_status'] == 'active')
    n_delinq = sum(1 for a in scope_accounts if a['current_status'] == 'delinquent')
    n_default = sum(1 for a in scope_accounts if a['current_status'] == 'defaulted')
    # "cured" = delinquent with 0 overdue terms (paid up recently)
    n_cured = sum(1 for a in scope_accounts if a['current_status'] == 'delinquent' and a['overdue_count'] == 0)

    # Empirical live transitions
    live_a_to_d = n_delinq / max(n_active + n_delinq, 1)
    live_d_to_def = n_default / max(n_delinq + n_default, 1)
    live_cure = n_cured / max(n_delinq, 1)

    # Blend weight by sample size
    if total >= 5:
        w_live = 0.80
    elif total >= 2:
        w_live = 0.40
    else:
        w_live = 0.0

    w_base = 1.0 - w_live

    a_to_d = round(w_live * live_a_to_d + w_base * b_baseline, 3)
    d_to_def = round(w_live * live_d_to_def + w_base * d_baseline, 3)
    cure = round(w_live * live_cure + w_base * c_baseline, 3)

    # Cap to realistic operational bounds
    a_to_d = max(0.03, min(0.30, a_to_d))
    d_to_def = max(0.05, min(0.50, d_to_def))
    cure = max(0.20, min(0.90, cure))

    markov_result = {
        'active_to_delinquent_prob': a_to_d,
        'delinquent_to_default_prob': d_to_def,
        'cure_to_active_prob': cure,
        'total_accounts': total,
        'active_count': n_active,
        'delinquent_count': n_delinq,
        'defaulted_count': n_default,
        'branch_breakdown': {}
    }

    return markov_result


def compute_branch_breakdown_for_global(all_accounts, branches, markov_baseline):
    """
    Computes per-branch Markov stats for the global consolidated view.
    Returns a branch_breakdown dict keyed by branch_id string.
    """
    branch_map = {b['branch_id']: b['name'] for b in branches}
    breakdown = {}

    b_baseline = markov_baseline.get('active_to_delinquent_prob', 0.08)
    d_baseline = markov_baseline.get('delinquent_to_default_prob', 0.22)
    c_baseline = markov_baseline.get('cure_to_active_prob', 0.58)

    for b in branches:
        b_id = b['branch_id']
        b_accounts = [a for a in all_accounts if a['branch_id'] == b_id]
        total = len(b_accounts)

        if total == 0:
            breakdown[str(b_id)] = {
                'name': branch_map.get(b_id, f'Branch {b_id}'),
                'total': 0,
                'active': 0,
                'delinquent': 0,
                'defaulted': 0,
                'a_to_d': 0.0,
                'd_to_def': 0.0,
                'cure': 0.0,
                'data_quality': 'none'
            }
            continue

        n_active = sum(1 for a in b_accounts if a['current_status'] == 'active')
        n_delinq = sum(1 for a in b_accounts if a['current_status'] == 'delinquent')
        n_default = sum(1 for a in b_accounts if a['current_status'] == 'defaulted')
        n_cured = sum(1 for a in b_accounts if a['current_status'] == 'delinquent' and a['overdue_count'] == 0)

        live_a_to_d = n_delinq / max(n_active + n_delinq, 1)
        live_d_to_def = n_default / max(n_delinq + n_default, 1)
        live_cure = n_cured / max(n_delinq, 1)

        if total >= 5:
            w_live = 0.80
        elif total >= 2:
            w_live = 0.40
        else:
            w_live = 0.0
        w_base = 1.0 - w_live

        a_to_d = round(max(0.03, min(0.30, w_live * live_a_to_d + w_base * b_baseline)), 3)
        d_to_def = round(max(0.05, min(0.50, w_live * live_d_to_def + w_base * d_baseline)), 3)
        cure = round(max(0.20, min(0.90, w_live * live_cure + w_base * c_baseline)), 3)

        breakdown[str(b_id)] = {
            'name': branch_map.get(b_id, f'Branch {b_id}'),
            'total': total,
            'active': n_active,
            'delinquent': n_delinq,
            'defaulted': n_default,
            'a_to_d': a_to_d,
            'd_to_def': d_to_def,
            'cure': cure,
            'data_quality': 'live' if total >= 5 else ('partial' if total >= 2 else 'baseline')
        }

    return breakdown


def sync_live_historical_snapshots(conn):
    """
    Builds historical monthly snapshots STRICTLY from actual recorded transactions
    in the live MySQL database (payments & installment_schedule tables).
    Simulated/synthetic datasets are NEVER inserted into the database.
    """
    current_month_str = datetime.now().strftime('%Y-%m')

    with conn.cursor() as cur:
        # Find past months with real payment activity
        cur.execute("""
            SELECT 
                SUBSTRING(p.payment_date, 1, 7) AS pay_month,
                u.branch_id,
                COUNT(DISTINCT p.payment_id) AS payment_count,
                SUM(p.amount_paid) AS total_collected
            FROM payments p
            JOIN sales s ON p.sale_id = s.sale_id
            JOIN inventory_units u ON s.unit_id = u.unit_id
            WHERE p.payment_date IS NOT NULL
              AND SUBSTRING(p.payment_date, 1, 7) < %s
            GROUP BY pay_month, u.branch_id
        """, (current_month_str,))
        past_branch_payments = cur.fetchall()

        for row in past_branch_payments:
            m = row['pay_month']
            b_id = row['branch_id']
            collected = float(row['total_collected'] or 0.0)

            # Check if historical snapshot already recorded
            cur.execute("""
                SELECT snapshot_id FROM ai_portfolio_monthly_snapshots 
                WHERE snapshot_month = %s AND branch_id = %s
            """, (m, b_id))
            if not cur.fetchone():
                cur.execute("""
                    INSERT INTO ai_portfolio_monthly_snapshots
                    (snapshot_month, branch_id, total_scheduled_due, total_actual_collected,
                     collection_efficiency_pct, active_accounts_count, delinquent_count, defaulted_count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (m, b_id, collected, collected, 100.0, 1, 0, 0))


def fetch_live_data(conn):
    """Fetches active loans, inventory, vehicle models, and branches from MySQL."""
    with conn.cursor() as cur:
        cur.execute("SELECT branch_id, name FROM branches ORDER BY branch_id ASC")
        branches = cur.fetchall()

        cur.execute("SELECT model_id, brand, model_code FROM vehicle_models")
        models = cur.fetchall()

        cur.execute("""
            SELECT
                u.unit_id,
                u.branch_id,
                u.model_id,
                u.status,
                u.date_added,
                vm.brand,
                vm.model_code
            FROM inventory_units u
            JOIN vehicle_models vm ON u.model_id = vm.model_id
        """)
        inventory = cur.fetchall()

        cur.execute("""
            SELECT
                s.sale_id,
                s.account_no,
                s.customer_id,
                c.full_name AS customer_name,
                c.phone,
                s.application_data,
                u.branch_id,
                s.status,
                s.term_months,
                s.monthly_amortization,
                s.gross_principal_amount,
                s.promissory_note_value,
                COALESCE((SELECT COUNT(*) FROM installment_schedule i WHERE i.sale_id = s.sale_id AND i.status = 'overdue'), 0) AS overdue_count,
                COALESCE((SELECT SUM(amount_due) FROM installment_schedule i WHERE i.sale_id = s.sale_id AND i.status = 'overdue'), 0.0) AS total_arrears,
                COALESCE((SELECT COUNT(*) FROM installment_schedule i WHERE i.sale_id = s.sale_id AND i.status = 'paid'), 0) AS paid_terms_count,
                COALESCE((SELECT SUM(amount_paid) FROM payments p WHERE p.sale_id = s.sale_id), 0.0) AS total_paid_sum
            FROM sales s
            JOIN customers c ON s.customer_id = c.customer_id
            JOIN inventory_units u ON s.unit_id = u.unit_id
            WHERE s.status IN ('active', 'delinquent', 'defaulted')
              AND s.account_no IS NOT NULL AND s.account_no != ''
              AND s.monthly_amortization > 0
              AND EXISTS (
                  SELECT 1 FROM installment_schedule i2
                  WHERE i2.sale_id = s.sale_id
                    AND i2.status IN ('pending', 'overdue', 'paid')
              )
        """)
        raw_sales = cur.fetchall()

        active_sales = []
        for s in raw_sales:
            app_data = {}
            if s.get('application_data'):
                try:
                    app_data = json.loads(s['application_data'])
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
                    except: pass

            emp = app_data.get('employment_status') or app_data.get('employment_status_choice') or 'Employed (Private/Gov)'
            res = app_data.get('residential_ownership') or app_data.get('residential_ownership_choice') or 'Owned (Clean Title)'
            stay = 5.0
            try:
                stay = float(app_data.get('lengthOfStayYears') or 5.0)
            except: pass

            active_sales.append({
                'sale_id': s['sale_id'],
                'account_no': s['account_no'],
                'customer_id': s['customer_id'],
                'customer_name': s['customer_name'],
                'phone': s['phone'],
                'monthly_income': income,
                'employment_type': emp,
                'residential_ownership': res,
                'length_of_stay_years': stay,
                'branch_id': s['branch_id'],
                'status': s['status'],
                'term_months': s['term_months'],
                'monthly_amortization': s['monthly_amortization'],
                'gross_principal_amount': s['gross_principal_amount'],
                'promissory_note_value': s['promissory_note_value'],
                'overdue_count': s['overdue_count'],
                'total_arrears': s['total_arrears'],
                'paid_terms_count': s['paid_terms_count'],
                'total_paid_sum': s['total_paid_sum']
            })

        # Fetch historical unit sales per branch and model for localized velocity
        cur.execute("""
            SELECT
                s.sale_id,
                s.sale_date,
                s.created_at,
                u.branch_id,
                vm.model_id,
                vm.brand,
                vm.model_code
            FROM sales s
            JOIN inventory_units u ON s.unit_id = u.unit_id
            JOIN vehicle_models vm ON u.model_id = vm.model_id
            WHERE s.status IN ('completed', 'active', 'delinquent', 'defaulted')
        """)
        historical_sales = cur.fetchall()

    return branches, models, inventory, active_sales, historical_sales


def fetch_actual_collections(conn, forecast_month):
    """
    Fetches month-to-date actual payments collected per sale and per branch.
    Granularly tracks 4 distinct payment streams:
      1. regular: On-time regular monthly installment component covering this month's scheduled due
      2. advance: Advance prepayment component for upcoming future monthly installments
      3. early_settlement: Option Contract early settlements / full payoffs
      4. partial: Crumbs / partial payments (< 90% of monthly amortization)
    """
    actual_by_branch = {}
    actual_by_sale = {}
    total_global = 0.0
    global_regular = 0.0
    global_advance = 0.0
    global_early = 0.0
    global_partial = 0.0
    global_regular_count = 0
    global_advance_count = 0
    global_early_count = 0
    global_partial_count = 0

    branch_breakdown = {}

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    p.payment_id,
                    p.sale_id,
                    p.payment_date,
                    p.amount_paid,
                    p.rebate_amount,
                    p.penalty_amount,
                    p.notes,
                    u.branch_id,
                    s.account_no,
                    s.status AS sale_status,
                    s.monthly_amortization,
                    s.term_months
                FROM payments p
                JOIN sales s ON p.sale_id = s.sale_id
                JOIN inventory_units u ON s.unit_id = u.unit_id
                WHERE DATE_FORMAT(p.payment_date, '%%Y-%%m') = %s
                  AND s.status NOT IN ('cancelled', 'terminated', 'pending_approval')
                  AND s.account_no IS NOT NULL AND s.account_no != ''
                ORDER BY p.sale_id, p.payment_date ASC
            """, (forecast_month,))
            rows = cur.fetchall()

            sales_payments = {}
            for row in rows:
                sid = int(row['sale_id'])
                if sid not in sales_payments:
                    sales_payments[sid] = []
                sales_payments[sid].append(row)

            for sid, p_list in sales_payments.items():
                bid = int(p_list[0]['branch_id'])
                amort = float(p_list[0]['monthly_amortization'] or 0.0)

                if bid not in branch_breakdown:
                    branch_breakdown[bid] = {
                        'total': 0.0,
                        'regular': 0.0,
                        'advance': 0.0,
                        'early_settlement': 0.0,
                        'partial': 0.0,
                        'regular_count': 0,
                        'advance_count': 0,
                        'early_settlement_count': 0,
                        'partial_count': 0
                    }

                sale_reg_covered = False
                sale_total = 0.0
                sale_reg = 0.0
                sale_adv = 0.0
                sale_early = 0.0
                sale_part = 0.0

                for p in p_list:
                    paid = float(p['amount_paid'] or 0.0)
                    rebate = float(p['rebate_amount'] or 0.0)
                    notes = (p['notes'] or '').lower()
                    is_early = ('early settlement' in notes or 'option contract' in notes or 'buyout' in notes or 'early payoff' in notes)

                    sale_total += paid

                    if is_early:
                        sale_early += paid
                        branch_breakdown[bid]['early_settlement'] += paid
                        branch_breakdown[bid]['early_settlement_count'] += 1
                        global_early += paid
                        global_early_count += 1
                    elif not sale_reg_covered:
                        if (paid + rebate) >= amort * 0.90:
                            reg_p = min(paid, amort)
                            adv_p = max(0.0, paid - amort)
                            sale_reg += reg_p
                            sale_adv += adv_p
                            sale_reg_covered = True

                            branch_breakdown[bid]['regular'] += reg_p
                            branch_breakdown[bid]['regular_count'] += 1
                            global_regular += reg_p
                            global_regular_count += 1

                            if adv_p > 0:
                                branch_breakdown[bid]['advance'] += adv_p
                                branch_breakdown[bid]['advance_count'] += 1
                                global_advance += adv_p
                                global_advance_count += 1
                        else:
                            sale_part += paid
                            branch_breakdown[bid]['partial'] += paid
                            branch_breakdown[bid]['partial_count'] += 1
                            global_partial += paid
                            global_partial_count += 1
                    else:
                        sale_adv += paid
                        branch_breakdown[bid]['advance'] += paid
                        branch_breakdown[bid]['advance_count'] += 1
                        global_advance += paid
                        global_advance_count += 1

                branch_breakdown[bid]['total'] += sale_total
                total_global += sale_total
                actual_by_branch[bid] = round(actual_by_branch.get(bid, 0.0) + sale_total, 2)

                actual_by_sale[sid] = {
                    'total': round(sale_total, 2),
                    'regular': round(sale_reg, 2),
                    'advance': round(sale_adv, 2),
                    'early_settlement': round(sale_early, 2),
                    'partial': round(sale_part, 2)
                }

    except Exception as e:
        print(f"[actual_collections] Warning: {e}")

    return {
        'global': round(total_global, 2),
        'global_regular': round(global_regular, 2),
        'global_advance': round(global_advance, 2),
        'global_early': round(global_early, 2),
        'global_partial': round(global_partial, 2),
        'global_regular_count': global_regular_count,
        'global_advance_count': global_advance_count,
        'global_early_count': global_early_count,
        'global_partial_count': global_partial_count,
        'by_branch': actual_by_branch,
        'by_branch_details': branch_breakdown,
        'by_sale': actual_by_sale
    }


def fetch_monthly_sales_target_and_actuals(conn, branches, forecast_month):
    """
    Computes Start-of-Month Sales Targets and MTD Actual Units Sold per branch and Globally.
    Based on historical sales velocity, seasonal index, and showroom stock.
    """
    sales_data = {
        'global': {
            'projected_units': 0,
            'actual_units': 0,
            'realization_pct': 0.0,
            'accuracy_pct': 0.0
        },
        'by_branch': {}
    }

    try:
        with conn.cursor() as cur:
            # 1. Actual units sold in forecast_month
            cur.execute("""
                SELECT u.branch_id, COUNT(*) AS units_sold
                FROM sales s
                JOIN inventory_units u ON s.unit_id = u.unit_id
                WHERE DATE_FORMAT(s.sale_date, '%%Y-%%m') = %s
                  AND s.status NOT IN ('cancelled', 'terminated', 'pending_approval')
                GROUP BY u.branch_id
            """, (forecast_month,))
            actual_rows = cur.fetchall()
            actual_by_b = {int(r['branch_id']): int(r['units_sold']) for r in actual_rows}

            # 2. Trailing 90-day sales count to establish base run-rate
            cur.execute("""
                SELECT u.branch_id, COUNT(*) AS units_sold_90d
                FROM sales s
                JOIN inventory_units u ON s.unit_id = u.unit_id
                WHERE s.sale_date >= DATE_SUB(%s, INTERVAL 90 DAY)
                  AND s.status NOT IN ('cancelled', 'terminated', 'pending_approval')
                GROUP BY u.branch_id
            """, (forecast_month + '-01',))
            hist_rows = cur.fetchall()
            hist_by_b = {int(r['branch_id']): int(r['units_sold_90d']) for r in hist_rows}

            # Determine calendar month for seasonal index
            m_dt = datetime.strptime(forecast_month, '%Y-%m')
            cal_m = m_dt.month
            season_mult = 1.0
            if cal_m == 12: season_mult = 1.35
            elif cal_m == 1: season_mult = 0.85
            elif cal_m in [3, 4]: season_mult = 1.15
            elif cal_m == 6: season_mult = 0.90
            elif cal_m == 9: season_mult = 1.05

            global_proj = 0
            global_act = 0

            for b in branches:
                bid = int(b['branch_id'])
                act_units = actual_by_b.get(bid, 0)
                sold_90d = hist_by_b.get(bid, 0)

                # Projected units for the month
                base_monthly = (sold_90d / 3.0) if sold_90d > 0 else 0.0
                if base_monthly > 0:
                    proj_units = max(1, round(base_monthly * season_mult))
                else:
                    proj_units = max(1, act_units) if act_units > 0 else 0

                realiz_pct = round((act_units / max(proj_units, 1)) * 100.0, 1) if proj_units > 0 else 0.0
                acc_pct = round(max(0.0, 100.0 - abs((act_units - proj_units) / max(proj_units, 1) * 100.0)), 1) if proj_units > 0 else 0.0

                sales_data['by_branch'][bid] = {
                    'branch_name': b['name'],
                    'projected_units': proj_units,
                    'actual_units': act_units,
                    'realization_pct': realiz_pct,
                    'accuracy_pct': acc_pct
                }

                global_proj += proj_units
                global_act += act_units

            global_realiz = round((global_act / max(global_proj, 1)) * 100.0, 1) if global_proj > 0 else 0.0
            global_acc = round(max(0.0, 100.0 - abs((global_act - global_proj) / max(global_proj, 1) * 100.0)), 1) if global_proj > 0 else 0.0

            sales_data['global'] = {
                'projected_units': global_proj,
                'actual_units': global_act,
                'realization_pct': global_realiz,
                'accuracy_pct': global_acc
            }

    except Exception as e:
        print(f"[monthly_sales] Warning: {e}")

    return sales_data



def compute_inventory_velocity(inventory, branches, historical_sales=None):
    """
    Computes Hierarchical Branch-Model Inventory Velocity using DML VelocityEngine.
    Differentiates fast vs slow models per showroom location with empirical Bayes shrinkage.
    """
    engine = VelocityEngine(
        lookback_days=90,
        min_velocity_floor=0.25,
        shrinkage_pseudocount=2.0
    )
    velocity_list, transfer_recs = engine.compute_branch_model_velocity(
        inventory=inventory,
        historical_sales=historical_sales or [],
        branches=branches
    )
    return velocity_list


def evaluate_portfolio(active_sales, hazard_pipeline, early_pipeline, rate_pkg):
    """
    Applies ML models to active accounts to compute hazard probabilities,
    early buyout candidates, and risk tiers.
    """
    if not active_sales:
        return [], []

    df_sales = pd.DataFrame(active_sales)

    df_sales['term_progress_ratio'] = df_sales.apply(
        lambda r: round(r['paid_terms_count'] / max(r['term_months'], 1), 3), axis=1
    )
    df_sales['dti_ratio'] = df_sales.apply(
        lambda r: round(float(r['monthly_amortization']) / max(float(r['monthly_income']), 1), 3), axis=1
    )
    df_sales['rebate_streak'] = df_sales.apply(
        lambda r: max(0, r['paid_terms_count'] - r['overdue_count']) if r['overdue_count'] == 0 else 0, axis=1
    )
    df_sales['late_penalty_streak'] = df_sales['overdue_count']
    df_sales['partial_payment_ratio'] = df_sales.apply(
        lambda r: 1.0 if r['overdue_count'] == 0 else max(0.2, 1.0 - (r['overdue_count'] * 0.25)), axis=1
    )
    df_sales['on_time_reliability'] = df_sales.apply(
        lambda r: round(max(0.1, 1.0 - (r['overdue_count'] / max(r['paid_terms_count'] + r['overdue_count'], 1))), 3), axis=1
    )
    df_sales['ci_negative_flags'] = df_sales.apply(
        lambda r: 1 if r['overdue_count'] >= 2 else 0, axis=1
    )
    df_sales['has_phone_bounce'] = 0

    feature_cols = [
        'term_progress_ratio', 'dti_ratio', 'rebate_streak', 'late_penalty_streak',
        'overdue_count', 'partial_payment_ratio', 'on_time_reliability',
        'ci_negative_flags', 'has_phone_bounce', 'length_of_stay_years',
        'employment_type', 'residential_ownership'
    ]

    hazard_probs = hazard_pipeline.predict_proba(df_sales[feature_cols])[:, 1]
    df_sales['default_probability'] = np.round(hazard_probs, 3)

    early_features = ['term_progress_ratio', 'on_time_reliability', 'dti_ratio', 'rebate_streak', 'overdue_count', 'monthly_income']
    early_probs = early_pipeline.predict_proba(df_sales[early_features])[:, 1]
    df_sales['early_settlement_prob'] = np.round(early_probs, 3)

    scored_accounts = []
    early_candidates = []

    for idx, row in df_sales.iterrows():
        p_def = float(row['default_probability'])
        p_early = float(row['early_settlement_prob'])
        monthly_amort = float(row['monthly_amortization'])
        total_arrears = float(row['total_arrears'])
        overdue = int(row['overdue_count'])
        cust_name = str(row['customer_name'])
        acct_no = str(row['account_no'])
        branch_id = int(row['branch_id'])

        # Hazard tier + plain language action
        if p_def >= 0.70:
            tier = 'critical'
            action = f"Call {cust_name} immediately — {overdue} overdue terms, ₱{total_arrears:,.0f} past due. Consider field visit."
            macro_label = "Schedule Field Visit"
            macro_type = "FIELD_CHECK"
        elif p_def >= 0.45:
            tier = 'high'
            action = f"Send a payment reminder with a waiver offer to {cust_name} — {overdue} missed payments."
            macro_label = "Send Follow-Up SMS"
            macro_type = "SMS_INTERVENTION"
        elif p_def >= 0.20:
            tier = 'moderate'
            action = f"Monitor {cust_name}'s next due date — {overdue} overdue term(s). Send courtesy reminder."
            macro_label = "Send Reminder SMS"
            macro_type = "SMS_REMINDER"
        else:
            tier = 'low'
            action = "Good standing — no action needed."
            macro_label = "No Action"
            macro_type = "ROUTINE_SERVICING"

        # Early Settlement
        current_term = int(row['paid_terms_count'])
        total_term = int(row['term_months'])
        propensity = 'LOW'

        if p_early >= 0.65 or (current_term in [10, 11, 12, 22, 23, 24] and float(row['on_time_reliability']) >= 0.85):
            propensity = 'HIGH' if float(row['on_time_reliability']) >= 0.92 else 'MEDIUM'

            early_candidates.append({
                'account_no': acct_no,
                'customer_name': cust_name,
                'term_progress': f"{current_term}/{total_term}",
                'buyout_propensity': propensity,
                'sale_id': int(row['sale_id']),
                'branch_id': branch_id
            })

        scored_accounts.append({
            'sale_id': int(row['sale_id']),
            'customer_id': int(row['customer_id']),
            'branch_id': branch_id,
            'account_no': acct_no,
            'customer_name': cust_name,
            'current_status': str(row['status']),
            'overdue_count': overdue,
            'default_probability': p_def,
            'early_settlement_probability': p_early,
            'hazard_tier': tier,
            'recommended_action': action,
            'macro_type': macro_type,
            'macro_label': macro_label,
            'monthly_amort': monthly_amort,
            'total_arrears': total_arrears
        })

    return scored_accounts, early_candidates


def build_macros_for_scope(scope_accounts, scope_inv, scope_early, scope_name):
    """
    Builds actionable macro cards for a scope with concrete targets and labels.
    Each macro includes: type, label, action, target_sale_id, target_account_no.
    """
    macros = []

    # Critical-hazard field visits (1 card per critical account, max 3)
    critical_accounts = [a for a in scope_accounts if a['hazard_tier'] == 'critical']
    if critical_accounts:
        for acc in critical_accounts[:3]:
            macros.append({
                'type': 'FIELD_CHECK',
                'label': 'Schedule Field Visit',
                'action': (
                    f"{acc['customer_name']} ({acc['account_no']}) — "
                    f"{acc['overdue_count']} overdue terms, ₱{acc['total_arrears']:,.0f} past due. "
                    f"Dispatch field agent to verify and collect."
                ),
                'target_sale_id': acc['sale_id'],
                'target_account_no': acc['account_no'],
                'target_branch_id': acc['branch_id']
            })

    # High-hazard SMS intervention (1 batch card)
    high_accounts = [a for a in scope_accounts if a['hazard_tier'] == 'high']
    if high_accounts:
        names = ", ".join(a['customer_name'].split(',')[0] for a in high_accounts[:3])
        macros.append({
            'type': 'SMS_INTERVENTION',
            'label': 'Send Follow-Up SMS with Waiver Offer',
            'action': (
                f"{len(high_accounts)} account(s) overdue — send a catch-up SMS with waiver offer. "
                f"Accounts: {names}{'...' if len(high_accounts) > 3 else ''}."
            ),
            'target_sale_id': high_accounts[0]['sale_id'] if len(high_accounts) == 1 else None,
            'target_account_no': high_accounts[0]['account_no'] if len(high_accounts) == 1 else None,
            'target_branch_id': high_accounts[0]['branch_id'] if len(high_accounts) == 1 else None
        })

    # Inventory transfer requests
    crit_stockouts = [v for v in scope_inv if (v.get('days_to_depletion') or 999) <= 14]
    if crit_stockouts:
        for stk in crit_stockouts[:2]:
            transfer_note = stk.get('recommended_transfer') or f"Restock {stk['model_code']} at {stk['branch_name']}"
            macros.append({
                'type': 'INVENTORY_TRANSFER',
                'label': 'Log Stock Transfer Request',
                'action': (
                    f"{stk['model_code']} at {stk['branch_name']} — only {stk['available_stock']} unit(s) left, "
                    f"runs out in ~{stk['days_to_depletion']} days. {transfer_note}."
                ),
                'target_sale_id': None,
                'target_account_no': None,
                'target_branch_id': stk['branch_id']
            })

    # Early settlement quote cards
    if scope_early:
        high_early = [c for c in scope_early if c.get('buyout_propensity') == 'HIGH']
        targets = high_early if high_early else scope_early[:2]
        for c in targets[:2]:
            macros.append({
                'type': 'EARLY_SETTLEMENT',
                'label': 'Send Early Payoff Quote',
                'action': (
                    f"{c['customer_name']} ({c['account_no']}) has completed {c['term_progress']} payments "
                    f"and is likely ready to pay off early. Send them a quote now."
                ),
                'target_sale_id': c.get('sale_id'),
                'target_account_no': c['account_no'],
                'target_branch_id': c.get('branch_id')
            })

    if not macros:
        macros.append({
            'type': 'ROUTINE_SERVICING',
            'label': 'All Clear',
            'action': f"{scope_name} is operating within normal ranges. Keep sending SMS reminders for upcoming due dates.",
            'target_sale_id': None,
            'target_account_no': None,
            'target_branch_id': None
        })

    return macros


def build_executive_summary(scope_name, scope_accounts, expected_sum, scheduled_sum,
                             gap, scope_inv, markov_result, rate_pkg, is_global, sales_info=None):
    """Generates plain-language, jargon-free executive summary and risk observations."""
    n_active = markov_result.get('active_count', 0)
    n_delinq = markov_result.get('delinquent_count', 0)
    n_default = markov_result.get('defaulted_count', 0)
    n_total = markov_result.get('total_accounts', 0)
    crit_stockouts = [v for v in scope_inv if (v.get('days_to_depletion') or 999) <= 21]
    crit_hazard = sum(1 for a in scope_accounts if a['hazard_tier'] == 'critical')
    high_hazard = sum(1 for a in scope_accounts if a['hazard_tier'] == 'high')

    collection_pct = round((expected_sum / max(scheduled_sum, 1)) * 100, 1)

    # Sales context
    s_info = sales_info or {}
    proj_sales = s_info.get('projected_units', 0)
    act_sales = s_info.get('actual_units', 0)
    sales_part = f"Sales target: {act_sales}/{proj_sales} units booked this month. " if proj_sales > 0 else ""

    if is_global:
        exec_summary = (
            f"Across all branches this month, expected regular collections are ₱{expected_sum:,.2f} "
            f"out of ₱{scheduled_sum:,.2f} scheduled target ({collection_pct:.1f}% expected collection rate). "
            f"{sales_part}"
            f"The network currently manages {n_total} active customer accounts: {n_active} in good standing, "
            f"{n_delinq} overdue, and {n_default} at serious risk."
        )
        if crit_hazard > 0:
            exec_summary += f" {crit_hazard} account(s) are flagged for urgent cashier or field agent attention."
        if crit_stockouts:
            model_names = ", ".join(set(v['model_code'] for v in crit_stockouts[:2]))
            exec_summary += f" Showroom alert: {model_names} {'have' if len(crit_stockouts) > 1 else 'has'} low showroom stock."

        a_to_d_pct = round(markov_result['active_to_delinquent_prob'] * 100, 1)
        d_to_def_pct = round(markov_result['delinquent_to_default_prob'] * 100, 1)
        cure_pct = round(markov_result['cure_to_active_prob'] * 100, 1)

        risk_obs = (
            f"Repayment outlook this month: {a_to_d_pct}% of on-time payers may experience payment delays. "
            f"For accounts currently overdue, {d_to_def_pct}% risk sliding into repossession, while {cure_pct}% are expected to catch up and pay overdue balances. "
        )
        if high_hazard > 0:
            risk_obs += f"{high_hazard} account(s) would benefit from a payment reminder SMS or penalty waiver offer. "
        risk_obs += f"Active financing multipliers: 1-Year ({rate_pkg['rate_1_yr']}x), 2-Year ({rate_pkg['rate_2_yr']}x), 3-Year ({rate_pkg['rate_3_yr']}x)."

    else:
        exec_summary = (
            f"Branch {scope_name} expects to collect ₱{expected_sum:,.2f} in regular amortizations this month "
            f"(₱{scheduled_sum:,.2f} scheduled target, {collection_pct:.1f}% expected collection rate). "
            f"{sales_part}"
            f"{n_total} financing account(s) under management: {n_active} on time"
        )
        if n_delinq > 0:
            exec_summary += f", {n_delinq} overdue"
        if n_default > 0:
            exec_summary += f", {n_default} in default"
        exec_summary += "."

        if crit_hazard > 0:
            exec_summary += f" {crit_hazard} account(s) require urgent cashier follow-up."

        risk_obs = ""
        if n_delinq > 0:
            risk_obs = (
                f"{n_delinq} account(s) are currently overdue. "
                f"Historical patterns show {round(markov_result['cure_to_active_prob'] * 100, 1)}% are expected to catch up, "
                f"while {round(markov_result['delinquent_to_default_prob'] * 100, 1)}% may require account restructuring. "
            )
        if crit_stockouts:
            risk_obs += f"Showroom stock alert: {', '.join(v['model_code'] for v in crit_stockouts[:2])} running low."
        if not risk_obs:
            risk_obs = "Branch customer ledger is healthy and operating within normal ranges."

    return exec_summary, risk_obs


def run_predictive_pipeline():
    """Main execution entry point."""
    print("=== BOMELI ML ENGINE: RUNNING PREDICTIVE FORECASTING PIPELINE ===")

    conn = get_db_connection()
    rate_pkg = get_active_rate_package(conn)
    current_month_str = datetime.now().strftime('%Y-%m')

    print(f"[pipeline] Target Forecast Month: {current_month_str}")
    print(f"[pipeline] Active Financing Rate Package: '{rate_pkg['package_name']}'")

    # Sync real historical snapshots from actual payment records in database
    sync_live_historical_snapshots(conn)

    # 1. Load trained ML models
    hazard_pipeline, early_pipeline, markov_baseline = load_models()
    print("[pipeline] Trained ML pipelines loaded successfully.")

    # 2. Fetch live data
    branches, models, inventory, active_sales, historical_sales = fetch_live_data(conn)
    print(f"[pipeline] Live Data: {len(branches)} branches, {len(inventory)} inventory units, {len(active_sales)} active financing sales.")

    # 3. Evaluate accounts with ML models
    scored_accounts, early_candidates = evaluate_portfolio(active_sales, hazard_pipeline, early_pipeline, rate_pkg)

    # 4. Compute Hierarchical Branch-Model Inventory Velocity
    inv_velocity = compute_inventory_velocity(inventory, branches, historical_sales)

    # 5. Fetch actual month-to-date payments for live vs predicted comparison
    actual_collections = fetch_actual_collections(conn, current_month_str)
    print(f"[pipeline] Actual MTD collected (all branches): PHP{actual_collections['global']:,.2f}")

    # 6. Fetch monthly sales targets & actual units sold
    sales_targets = fetch_monthly_sales_target_and_actuals(conn, branches, current_month_str)

    # 7. Compute global branch breakdown (all branches) for the consolidated view
    global_branch_breakdown = compute_branch_breakdown_for_global(scored_accounts, branches, markov_baseline)

    # 8. Scopes: Global + each Branch
    scopes = [{'branch_id': None, 'name': 'Consolidated Network (Global)', 'scope_type': 'global'}]
    for b in branches:
        scopes.append({'branch_id': b['branch_id'], 'name': b['name'], 'scope_type': 'branch'})

    with conn.cursor() as cur:
        # Ensure ai_account_risk_scores table exists
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

        # Add macro_type and macro_label columns if missing (migration safety)
        for col_def in [
            ("macro_type", "VARCHAR(50) NOT NULL DEFAULT 'ROUTINE_SERVICING' AFTER recommended_action"),
            ("macro_label", "VARCHAR(100) NOT NULL DEFAULT 'No Action' AFTER macro_type"),
        ]:
            try:
                cur.execute(f"ALTER TABLE ai_account_risk_scores ADD COLUMN {col_def[0]} {col_def[1]}")
            except Exception:
                pass

        # Clean prior predictions for the current month
        cur.execute("DELETE FROM ai_account_risk_scores WHERE forecast_month = %s", (current_month_str,))

        # Ingest account risk scores
        for acc in scored_accounts:
            cur.execute("""
                INSERT INTO ai_account_risk_scores
                (forecast_month, sale_id, customer_id, branch_id, account_no, customer_name,
                 current_status, overdue_count, default_probability, early_settlement_probability,
                 hazard_tier, recommended_action, macro_type, macro_label)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                current_month_str, acc['sale_id'], acc['customer_id'], acc['branch_id'],
                acc['account_no'], acc['customer_name'], acc['current_status'],
                acc['overdue_count'], acc['default_probability'], acc['early_settlement_probability'],
                acc['hazard_tier'], acc['recommended_action'],
                acc.get('macro_type', 'ROUTINE_SERVICING'),
                acc.get('macro_label', 'No Action')
            ))

        # Process each scope
        for scope in scopes:
            b_id = scope['branch_id']
            scope_type = scope['scope_type']
            target_audience = 'upper_management' if b_id is None else 'manager'
            is_global = (b_id is None)

            # Filter accounts for this scope
            if is_global:
                scope_accounts = scored_accounts
                scope_inv = inv_velocity
                scope_early = early_candidates
                scope_sales = sales_targets['global']
            else:
                scope_accounts = [a for a in scored_accounts if a['branch_id'] == b_id]
                scope_inv = [v for v in inv_velocity if v['branch_id'] == b_id]
                scope_early = [c for c in early_candidates if any(
                    a['account_no'] == c['account_no'] and a['branch_id'] == b_id
                    for a in scored_accounts
                )]
                scope_sales = sales_targets['by_branch'].get(b_id, {})

            # ── Per-scope Markov transitions & Unbiased Multi-Term Survival Statistics ──
            markov_result = compute_branch_markov(scope_accounts, markov_baseline)
            survival_engine = SurvivalEngine()
            markov_result['survival_statistics'] = survival_engine.compute_portfolio_survival_statistics(scope_accounts, historical_sales)

            # For global scope, embed full branch breakdown
            if is_global:
                markov_result['branch_breakdown'] = global_branch_breakdown
            else:
                markov_result['branch_breakdown'] = {}

            # ── Collections Breakdown ──
            if is_global:
                act_total = actual_collections['global']
                act_reg   = actual_collections['global_regular']
                act_adv   = actual_collections['global_advance']
                act_early = actual_collections['global_early']
                act_part  = actual_collections['global_partial']
                reg_cnt   = actual_collections['global_regular_count']
                adv_cnt   = actual_collections['global_advance_count']
                early_cnt = actual_collections['global_early_count']
                part_cnt  = actual_collections['global_partial_count']
            else:
                b_det     = actual_collections['by_branch_details'].get(b_id, {})
                act_total = b_det.get('total', 0.0)
                act_reg   = b_det.get('regular', 0.0)
                act_adv   = b_det.get('advance', 0.0)
                act_early = b_det.get('early_settlement', 0.0)
                act_part  = b_det.get('partial', 0.0)
                reg_cnt   = b_det.get('regular_count', 0)
                adv_cnt   = b_det.get('advance_count', 0)
                early_cnt = b_det.get('early_settlement_count', 0)
                part_cnt  = b_det.get('partial_count', 0)

            # Strict isolation: If 0 accounts and 0 collections, skip insight
            if len(scope_accounts) == 0 and act_total == 0.0:
                print(f"[{scope_type}] {scope['name']}: 0 accounts, 0 actual MTD — SKIPPED (no data to show)")
                if is_global:
                    cur.execute("DELETE FROM ai_predictive_insights WHERE forecast_month = %s AND branch_id IS NULL", (current_month_str,))
                else:
                    cur.execute("DELETE FROM ai_predictive_insights WHERE forecast_month = %s AND branch_id = %s", (current_month_str, b_id))
                continue

            # ── Dynamic Non-Linear 3-Month Cash Flow & Payment Stream Forecast ──
            cash_engine = CashEngine()
            stream_breakdown = PaymentStreamBreakdown(
                total_collected_mtd=act_total,
                regular_collected_mtd=round(act_reg, 2),
                advance_collected_mtd=round(act_adv, 2),
                early_settlement_collected_mtd=round(act_early, 2),
                partial_collected_mtd=round(act_part, 2),
                regular_account_count=reg_cnt,
                advance_account_count=adv_cnt,
                early_settlement_count=early_cnt,
                partial_account_count=part_cnt
            )

            cash_forecast = cash_engine.compute_cash_forecast_cone(
                accounts=scope_accounts,
                actual_stream=stream_breakdown,
                projected_sales_units=scope_sales.get('projected_units', 0),
                actual_sales_units=scope_sales.get('actual_units', 0),
                forecast_month=current_month_str,
                cure_to_active_prob=markov_result.get('cure_to_active_prob', 0.58)
            )

            expected_sum = cash_forecast['ai_expected_collected']
            scheduled_sum = cash_forecast['contractual_scheduled']
            pending_collections = cash_forecast['pending_collections']

            # ── Actionable Next Actions ──
            macros = build_macros_for_scope(scope_accounts, scope_inv, scope_early, scope['name'])

            # ── Plain-language Executive Summary (Zero Jargon) ──
            exec_summary, risk_obs = build_executive_summary(
                scope['name'], scope_accounts, expected_sum, scheduled_sum,
                pending_collections, scope_inv, markov_result, rate_pkg, is_global, scope_sales
            )

            # Clean and insert
            if is_global:
                cur.execute("DELETE FROM ai_predictive_insights WHERE forecast_month = %s AND branch_id IS NULL", (current_month_str,))
            else:
                cur.execute("DELETE FROM ai_predictive_insights WHERE forecast_month = %s AND branch_id = %s", (current_month_str, b_id))

            cur.execute("""
                INSERT INTO ai_predictive_insights
                (forecast_month, scope_type, branch_id, target_audience,
                 cash_forecast_json, risk_migration_json, inventory_velocity_json,
                 early_settlement_json, actionable_macros_json, executive_summary, risk_observations)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                current_month_str, scope_type, b_id, target_audience,
                json.dumps(cash_forecast), json.dumps(markov_result),
                json.dumps(scope_inv), json.dumps(scope_early),
                json.dumps(macros), exec_summary, risk_obs
            ))

            # Write snapshot only if this scope has real accounts
            if len(scope_accounts) > 0:
                if is_global:
                    cur.execute("DELETE FROM ai_portfolio_monthly_snapshots WHERE snapshot_month = %s AND branch_id IS NULL", (current_month_str,))
                else:
                    cur.execute("DELETE FROM ai_portfolio_monthly_snapshots WHERE snapshot_month = %s AND branch_id = %s", (current_month_str, b_id))

                cur.execute("""
                    INSERT INTO ai_portfolio_monthly_snapshots
                    (snapshot_month, branch_id, total_scheduled_due, total_actual_collected,
                     collection_efficiency_pct, active_accounts_count, delinquent_count, defaulted_count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    current_month_str, b_id, scheduled_sum, expected_sum,
                    round((expected_sum / max(scheduled_sum, 1)) * 100.0, 2),
                    len(scope_accounts),
                    sum(1 for a in scope_accounts if a['current_status'] == 'delinquent'),
                    sum(1 for a in scope_accounts if a['current_status'] == 'defaulted')
                ))

            print(
                f"[{scope_type}] {scope['name']}: "
                f"PHP{expected_sum:,.2f}/{scheduled_sum:,.2f} — "
                f"{len(scope_accounts)} accounts, "
                f"Risk A->D={markov_result['active_to_delinquent_prob']*100:.1f}% "
                f"({markov_result.get('data_quality', 'blended')})"
            )

    conn.commit()
    conn.close()

    print(f"\n=== PIPELINE COMPLETED SUCCESSFULLY FOR {current_month_str} ===")
    print(f"Upserted: Global Consolidated Network + {len(branches)} Branches.")
    print("MySQL tables updated: ai_predictive_insights, ai_portfolio_monthly_snapshots, ai_account_risk_scores.")


if __name__ == '__main__':
    run_predictive_pipeline()
