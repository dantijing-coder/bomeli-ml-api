#!/usr/bin/env python3
"""
CLI helper for fast single-account ML inference.
Accepts JSON from stdin or sys.argv[1], loads pre-trained models from model_registry,
and outputs predictions as clean JSON to stdout.
"""
import sys
import json
import os

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure directory is on python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dml.model_registry import ModelRegistry

def main():
    if len(sys.argv) > 1 and sys.argv[1].strip():
        raw_input = sys.argv[1].strip()
    else:
        raw_input = sys.stdin.read().strip()

    if not raw_input:
        print(json.dumps({"error": "No input JSON provided"}))
        sys.exit(1)

    try:
        data = json.loads(raw_input)
    except Exception as e:
        print(json.dumps({"error": f"Invalid JSON input: {str(e)}"}))
        sys.exit(1)

    try:
        reg = ModelRegistry()
        p_def, tier = reg.predict_default_hazard(data)
        p_early, is_early = reg.predict_early_settlement(data)
        outcome = reg.predict_lifecycle_outcome(data)
        realiz = reg.predict_cash_realization(data)
        m_type, m_label = reg.assign_macro_action(data, tier, is_early)

        res = {
            "default_probability": round(float(p_def), 4),
            "hazard_tier": tier,
            "early_settlement_probability": round(float(p_early), 4),
            "is_early_candidate": bool(is_early),
            "predicted_lifecycle_outcome": outcome,
            "expected_cash_realization": round(float(realiz), 4),
            "recommended_action_type": m_type,
            "recommended_action_label": m_label
        }
        print(json.dumps(res))
    except Exception as e:
        print(json.dumps({"error": f"Inference execution failed: {str(e)}"}))
        sys.exit(1)

if __name__ == '__main__':
    main()
