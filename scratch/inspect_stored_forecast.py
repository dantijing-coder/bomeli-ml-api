import pymysql
import json

conn = pymysql.connect(
    host='localhost',
    user='root',
    password='',
    database='bomeli_db1',
    cursorclass=pymysql.cursors.DictCursor
)

with conn.cursor() as cur:
    cur.execute("SELECT branch_id, scope_type, cash_forecast_json FROM ai_predictive_insights WHERE forecast_month = '2026-09'")
    rows = cur.fetchall()
    for r in rows:
        print(f"=== Scope: {r['scope_type']} (Branch {r['branch_id']}) ===")
        cf = json.loads(r['cash_forecast_json'])
        print(f"Labels: {cf.get('three_month_labels')}")
        print(f"Expected: {cf.get('three_month_projection')}")
        print(f"Scheduled: {cf.get('three_month_scheduled')}")
        print(f"Optimistic: {cf.get('three_month_optimistic')}")
        print(f"Pessimistic: {cf.get('three_month_pessimistic')}")
        print(f"Seasonal Notes: {cf.get('three_month_notes')}")
        print(f"Maturing: {cf.get('maturing_accounts_count')}")
        print()
