"""
ml_engine/train_models.py
Trains and serializes the core predictive Machine Learning models for Bomeli Financing:
  1. 90-Day Delinquency Migration & Default Hazard Classifier (Calibrated GBDT / HistGradientBoosting)
  2. Option Contract Early Settlement Propensity Classifier
  3. Delinquency Roll-Rate Markov Transition Matrix

Includes:
- Un-train / reset capabilities to ensure a clean slate before training.
- 80/20 train/test data context splitting (80% training / 20% out-of-sample testing).
- Automated evaluation and assertions ensuring test accuracy hits >= 81%+ on holdout testing data.
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Ensure local imports work regardless of execution directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, MODELS_DIR


def untrain_models():
    """
    Un-trains existing models by wiping previously serialized artifacts from the models directory.
    Ensures that subsequent training begins with a completely clean slate.
    """
    print("\n=======================================================")
    print(">>> UN-TRAINING PREVIOUS MODELS (CLEAN SLATE RESET) <<<")
    print("=======================================================")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    artifacts = [
        'default_hazard_model.joblib',
        'early_settlement_model.joblib',
        'markov_matrix.json'
    ]
    
    removed_count = 0
    for filename in artifacts:
        target_path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
                print(f" [UN-TRAIN] Removed existing model artifact: {filename}")
                removed_count += 1
            except Exception as e:
                print(f" [UN-TRAIN] Warning: Could not remove {filename}: {e}")
        else:
            print(f" [UN-TRAIN] Artifact not found (already clean): {filename}")
            
    print(f" [UN-TRAIN] Model directory successfully reset ({removed_count} artifacts removed).\n")


def load_simulated_dataset():
    """
    Loads simulated motorcycle portfolio data.
    """
    csv_path = os.path.join(DATA_DIR, 'simulated_portfolio.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Simulated data file not found at {csv_path}. Please run simulate_portfolio.py first.")
    df = pd.read_csv(csv_path)
    print(f"[train] Loaded {len(df):,} simulated records from {csv_path}")
    return df


def train_default_hazard_model(df, target_accuracy=0.81, random_state=42):
    """
    Trains Model 1: Calibrated Default Hazard Classifier predicting the probability
    of an account rolling into severe delinquency / default in the next 90 days.
    
    Splits data: 80% training / data context, 20% out-of-sample testing.
    Verifies that test prediction accuracy meets or exceeds the required threshold (>= 81%).
    """
    print("\n----------------------------------------------------------------------")
    print("--- Training Model 1: 90-Day Delinquency Migration & Default Hazard ---")
    print("----------------------------------------------------------------------")

    numeric_features = [
        'term_progress_ratio',
        'dti_ratio',
        'rebate_streak',
        'late_penalty_streak',
        'overdue_count',
        'partial_payment_ratio',
        'on_time_reliability',
        'ci_negative_flags',
        'has_phone_bounce',
        'length_of_stay_years'
    ]
    categorical_features = ['employment_type', 'residential_ownership']
    all_feature_cols = numeric_features + categorical_features

    X = df[all_feature_cols]
    y = df['will_default_90d']

    # 80% Training Context / 20% Out-of-sample Testing Partition
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    print(f" [Partition] 80/20 Data Split: {len(X_train):,} training records (80.0%) | {len(X_test):,} testing records (20.0%)")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )

    base_estimator = HistGradientBoostingClassifier(
        max_iter=250,
        max_depth=6,
        learning_rate=0.06,
        min_samples_leaf=15,
        l2_regularization=0.5,
        random_state=random_state
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', CalibratedClassifierCV(estimator=base_estimator, method='sigmoid', cv=5))
    ])

    print(" [Fitting] Training and calibrating GBDT pipeline with 5-fold cross-validation...")
    pipeline.fit(X_train, y_train)

    # Out-of-sample evaluation on the 20% testing dataset
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n>>> Model 1 Out-of-Sample Test Evaluation (20% Holdout Data):")
    print(f"  Test Accuracy:     {acc * 100:.2f}%  [Target Threshold: >= {target_accuracy * 100:.1f}%]")
    print(f"  Balanced Accuracy: {bal_acc * 100:.2f}%")
    print(f"  ROC-AUC Score:     {auc:.4f}")
    print(f"  F1-Score:          {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    if acc < target_accuracy:
        raise ValueError(
            f"Model 1 accuracy {acc * 100:.2f}% did not meet target threshold of {target_accuracy * 100:.1f}%!"
        )

    model_path = os.path.join(MODELS_DIR, 'default_hazard_model.joblib')
    joblib.dump(pipeline, model_path)
    print(f" [SUCCESS] Serialized Default Hazard Model to: {model_path}")

    return pipeline


def train_early_settlement_model(df, target_accuracy=0.81, random_state=42):
    """
    Trains Model 3: Option Contract Early Settlement Propensity Classifier.
    
    Splits data: 80% training / data context, 20% out-of-sample testing.
    Verifies that test prediction accuracy meets or exceeds the required threshold (>= 81%).
    """
    print("\n----------------------------------------------------------------------")
    print("--- Training Model 3: Option Contract Early Settlement Propensity ---")
    print("----------------------------------------------------------------------")

    features = [
        'term_progress_ratio',
        'on_time_reliability',
        'dti_ratio',
        'rebate_streak',
        'overdue_count',
        'monthly_income'
    ]

    X = df[features]
    y = df['early_settlement_target']

    # 80% Training Context / 20% Out-of-sample Testing Partition
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    print(f" [Partition] 80/20 Data Split: {len(X_train):,} training records (80.0%) | {len(X_test):,} testing records (20.0%)")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', features)
        ],
        remainder='drop'
    )

    base_estimator = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=15,
        random_state=random_state
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', base_estimator)
    ])

    print(" [Fitting] Training HistGradientBoosting Early Settlement Classifier...")
    pipeline.fit(X_train, y_train)

    # Out-of-sample evaluation on the 20% testing dataset
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n>>> Model 3 Out-of-Sample Test Evaluation (20% Holdout Data):")
    print(f"  Test Accuracy:     {acc * 100:.2f}%  [Target Threshold: >= {target_accuracy * 100:.1f}%]")
    print(f"  Balanced Accuracy: {bal_acc * 100:.2f}%")
    print(f"  ROC-AUC Score:     {auc:.4f}")
    print(f"  F1-Score:          {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    if acc < target_accuracy:
        raise ValueError(
            f"Model 3 accuracy {acc * 100:.2f}% did not meet target threshold of {target_accuracy * 100:.1f}%!"
        )

    model_path = os.path.join(MODELS_DIR, 'early_settlement_model.joblib')
    joblib.dump(pipeline, model_path)
    print(f" [SUCCESS] Serialized Early Settlement Model to: {model_path}")

    return pipeline


def compute_markov_roll_rates(df):
    """
    Computes Model 5: Delinquency Roll-Rate Markov Transition Matrix across loan statuses.
    """
    print("\n----------------------------------------------------------------------")
    print("--- Computing Model 5: Portfolio Roll-Rate Markov Transition Matrix ---")
    print("----------------------------------------------------------------------")

    # In simulated cohort, calculate empirical transition likelihoods:
    # 1. P(Active -> Delinquent)
    active_cohort = df[df['current_status'] == 'active']
    a_to_d_prob = float(round(len(active_cohort[active_cohort['will_default_90d'] == 1]) / max(len(active_cohort), 1), 3))
    a_to_d_prob = max(0.06, min(0.18, a_to_d_prob if a_to_d_prob > 0 else 0.095))

    # 2. P(Delinquent -> Default)
    delinq_cohort = df[df['current_status'] == 'delinquent']
    d_to_def_prob = float(round(len(delinq_cohort[delinq_cohort['overdue_count'] >= 2]) / max(len(delinq_cohort), 1), 3))
    d_to_def_prob = max(0.15, min(0.35, d_to_def_prob if d_to_def_prob > 0 else 0.225))

    # 3. P(Delinquent -> Cured/Active)
    cure_prob = float(round(len(delinq_cohort[delinq_cohort['on_time_reliability'] >= 0.70]) / max(len(delinq_cohort), 1), 3))
    cure_prob = max(0.40, min(0.70, cure_prob if cure_prob > 0 else 0.540))

    markov_data = {
        'active_to_delinquent_prob': a_to_d_prob,
        'delinquent_to_default_prob': d_to_def_prob,
        'cure_to_active_prob': cure_prob,
        'transition_matrix': {
            'ACTIVE': {'ACTIVE': round(1.0 - a_to_d_prob, 3), 'DELINQUENT': a_to_d_prob, 'DEFAULTED': 0.0},
            'DELINQUENT': {'ACTIVE': cure_prob, 'DELINQUENT': round(1.0 - cure_prob - d_to_def_prob, 3), 'DEFAULTED': d_to_def_prob},
            'DEFAULTED': {'DELINQUENT': 0.15, 'DEFAULTED': 0.55, 'REPOSSESSED': 0.30}
        }
    }

    print("Markov Migration Probabilities:")
    print(f"  P(Active -> Delinquent):   {a_to_d_prob * 100:.1f}%")
    print(f"  P(Delinquent -> Default):  {d_to_def_prob * 100:.1f}%")
    print(f"  P(Cure -> Active):         {cure_prob * 100:.1f}%")

    out_path = os.path.join(MODELS_DIR, 'markov_matrix.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(markov_data, f, indent=2)
    print(f" [SUCCESS] Saved Markov Transition Matrix to: {out_path}")

    return markov_data


def main():
    parser = argparse.ArgumentParser(description="Re-learn and optimize Bomeli ML models with 80/20 train/test evaluation.")
    parser.add_argument('--untrain-only', action='store_true', help="Un-train / wipe models without retraining.")
    parser.add_argument('--target-acc', type=float, default=0.81, help="Minimum acceptable test accuracy (default: 0.81).")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for data partitioning.")
    args = parser.parse_args()

    # Step 1: Always un-train previous models first for a clean learning run
    untrain_models()

    if args.untrain_only:
        print("[train] Models un-trained. Exiting as requested.")
        return

    # Step 2: Load simulated portfolio
    df = load_simulated_dataset()

    # Step 3: Train and optimize models with 80/20 split and accuracy validation
    print("\n=======================================================")
    print(f">>> RE-LEARNING & OPTIMIZING MODELS (TARGET: >= {args.target_acc * 100:.1f}%) <<<")
    print("=======================================================")
    
    m1 = train_default_hazard_model(df, target_accuracy=args.target_acc, random_state=args.seed)
    m3 = train_early_settlement_model(df, target_accuracy=args.target_acc, random_state=args.seed)
    m5 = compute_markov_roll_rates(df)

    print("\n=======================================================")
    print(">>> ALL MODELS RE-LEARNED, OPTIMIZED & SERIALIZED <<<")
    print("=======================================================")


if __name__ == '__main__':
    main()
