from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


RANDOM_STATE = 42

# Single source of truth for cost assumptions.
# Change here and both compute_cost_table() and compute_expected_cost_from_cm()
# pick it up automatically.
DEFAULT_FN_COST: int = 50_000   # cost of a missed fraud (false negative)
DEFAULT_FP_COST: int = 500      # cost of a false alarm (false positive)


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that encode fraud-specific balance patterns.

    All four features are computed from columns already present in PaySim.
    Operates on a copy — the caller's DataFrame is never mutated.
    Call on every split independently before selecting feature_cols.

    Feature                  Signal
    ─────────────────────────────────────────────────────────────────────────
    balance_delta_orig       Should equal −amount for legitimate transfers.
                             Fraud often zeroes the account; any mismatch is
                             a strong anomaly signal.
    balance_delta_dest       Should equal +amount for legitimate transfers.
                             Fraud destinations frequently show no increase
                             because the money moves on immediately.
    amount_to_balance_ratio  Fraud tends to transfer the entire balance in
                             one transaction (ratio ≈ 1.0). +1 in the
                             denominator guards against divide-by-zero.
    orig_balance_zeroed      Hard binary flag: origin account fully drained.
                             Very high-precision fraud signal in PaySim.
    ─────────────────────────────────────────────────────────────────────────
    """
    df = df.copy()
    df["balance_delta_orig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["balance_delta_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["orig_balance_zeroed"] = (
        (df["newbalanceOrig"] == 0) & (df["amount"] > 0)
    ).astype(int)
    return df


# ── Feature config ────────────────────────────────────────────────────────────

def get_feature_config(
    use_feature_engineering: bool = True,
) -> Tuple[str, list[str], list[str], list[str]]:
    target_col = "isFraud"
    categorical_features = ["type"]

    numeric_features = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFlaggedFraud",
    ]

    if use_feature_engineering:
        numeric_features += [
            "balance_delta_orig",
            "balance_delta_dest",
            "amount_to_balance_ratio",
            "orig_balance_zeroed",
        ]

    feature_cols = categorical_features + numeric_features
    return target_col, categorical_features, numeric_features, feature_cols


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_dataframes(
    paysim_train: pd.DataFrame,
    paysim_val: pd.DataFrame,
    paysim_test: pd.DataFrame,
    use_feature_engineering: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Prepare train/val/test splits with optional feature engineering.

    Feature engineering is applied to each split independently — no statistics
    are fitted on training data, so there is no leakage risk. The derived
    columns are purely arithmetic transformations of raw PaySim columns.

    Parameters
    ----------
    use_feature_engineering : bool
        If True (default), engineer_features() is called on every split and
        the four derived balance columns are added to the feature set.
        If False, only the seven raw PaySim features are used — matching the
        tabular baseline setup for a clean ablation comparison.
    """
    if use_feature_engineering:
        paysim_train = engineer_features(paysim_train)
        paysim_val   = engineer_features(paysim_val)
        paysim_test  = engineer_features(paysim_test)

    target_col, categorical_features, numeric_features, feature_cols = (
        get_feature_config(use_feature_engineering)
    )

    X_train = paysim_train[feature_cols].copy()
    y_train = paysim_train[target_col].copy()

    X_val = paysim_val[feature_cols].copy()
    y_val = paysim_val[target_col].copy()

    X_test = paysim_test[feature_cols].copy()
    y_test = paysim_test[target_col].copy()

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "use_feature_engineering": use_feature_engineering,
    }


# ── Preprocessor ─────────────────────────────────────────────────────────────

def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Return an unfitted ColumnTransformer.

    Safe to reuse across multiple build_xgb_pipeline() calls: sklearn clones
    each step before fitting, so each pipeline gets its own independent copy.
    """
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


# ── Imbalance helpers ─────────────────────────────────────────────────────────

def compute_scale_pos_weight(y_train: pd.Series) -> float:
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    if pos == 0:
        raise ValueError("No positive class found in y_train.")
    return neg / pos


# ── Model builder ─────────────────────────────────────────────────────────────

def build_xgb_pipeline(
    preprocessor: ColumnTransformer,
    imbalance_method: str = "none",
    random_state: int = RANDOM_STATE,
    scale_pos_weight: float = 1.0,
    smote_sampling_strategy: float = 0.1,
    early_stopping_rounds: Optional[int] = 20,
) -> Pipeline:
    """Build an XGBoost sklearn Pipeline.

    Parameters
    ----------
    imbalance_method : {'none', 'scale_pos_weight', 'smote'}
    early_stopping_rounds : int or None
        When set, XGBoost stops adding trees once validation logloss has not
        improved for this many consecutive rounds. n_estimators becomes a
        ceiling. Pass None to disable and always train all 300 trees.
    smote_sampling_strategy : float
        Target ratio of minority to majority class after SMOTE resampling.
        Default 0.1 (minority becomes 10% of majority) avoids creating an
        artificially balanced distribution on highly skewed data like PaySim.
    """
    if imbalance_method not in {"none", "scale_pos_weight", "smote"}:
        raise ValueError(
            "imbalance_method must be one of: 'none', 'scale_pos_weight', 'smote'"
        )

    common_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": random_state,
        "n_jobs": -1,
        **({} if early_stopping_rounds is None
           else {"early_stopping_rounds": early_stopping_rounds}),
    }

    if imbalance_method == "none":
        return Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(**common_params, scale_pos_weight=1.0)),
        ])

    elif imbalance_method == "scale_pos_weight":
        return Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(**common_params, scale_pos_weight=scale_pos_weight)),
        ])

    else:  # smote
        return ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(sampling_strategy=smote_sampling_strategy, random_state=random_state)),
            ("classifier", XGBClassifier(**common_params, scale_pos_weight=1.0)),
        ])


# ── Training ──────────────────────────────────────────────────────────────────

def fit_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    verbose: bool = False,
) -> Pipeline:
    """Fit the pipeline with optional early stopping via eval_set.

    When eval_set=(X_val, y_val) is supplied, X_val is transformed through
    the preprocessor step before being passed to XGBoost, so it matches the
    feature representation of the training data. No manual preprocessing
    needed from the caller.
    """
    if eval_set is not None:
        X_val_raw, y_val_es = eval_set
        preprocessor_step = model.named_steps["preprocessor"]
        preprocessor_step.fit(X_train, y_train)
        X_val_transformed = preprocessor_step.transform(X_val_raw)
        model.fit(
            X_train,
            y_train,
            classifier__eval_set=[(X_val_transformed, y_val_es)],
            classifier__verbose=verbose,
        )
    else:
        model.fit(X_train, y_train)

    clf = model.named_steps["classifier"]
    if hasattr(clf, "best_iteration") and clf.best_iteration is not None:
        print(f"  Early stopping: used {clf.best_iteration + 1} / {clf.n_estimators} trees.")

    return model


# ── Leakage investigation ─────────────────────────────────────────────────────

def investigate_flagged_fraud(df: pd.DataFrame, split_name: str = "train") -> Dict:
    """Analyse isFlaggedFraud to determine whether it is a near-leak.

    PaySim sets isFlaggedFraud=1 when a TRANSFER exceeds 200,000 units.
    Low coverage (<5%) → not a meaningful shortcut, safe to keep.
    High coverage (>50%) → potential shortcut, run ablation without it.
    """
    required = {"isFlaggedFraud", "isFraud", "type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    total_fraud       = int(df["isFraud"].sum())
    flagged           = df["isFlaggedFraud"] == 1
    flagged_count     = int(flagged.sum())
    flagged_and_fraud = int((flagged & (df["isFraud"] == 1)).sum())
    coverage          = flagged_and_fraud / total_fraud if total_fraud > 0 else 0.0
    precision_of_flag = flagged_and_fraud / flagged_count if flagged_count > 0 else 0.0

    type_breakdown = (
        df[flagged].groupby("type")["isFraud"]
        .agg(flagged_n="count", fraud_n="sum")
        .assign(fraud_pct=lambda x: (x["fraud_n"] / x["flagged_n"] * 100).round(1))
        .reset_index()
        .to_dict(orient="records")
    )

    if coverage < 0.05:
        verdict = "LOW COVERAGE — not a meaningful shortcut. Safe to keep."
    elif coverage > 0.50:
        verdict = "HIGH COVERAGE — potential shortcut. Run ablation (remove feature, compare PR-AUC)."
    else:
        verdict = "MODERATE COVERAGE — borderline. Document in thesis."

    print(f"\n=== isFlaggedFraud analysis — {split_name.upper()} ===")
    print(f"Total fraud cases       : {total_fraud:,}")
    print(f"Flagged transactions    : {flagged_count:,}")
    print(f"Flagged AND fraud       : {flagged_and_fraud:,}")
    print(f"Coverage  (flag recall) : {coverage:.2%}")
    print(f"Precision (flag)        : {precision_of_flag:.2%}")
    print(f"Verdict                 : {verdict}")
    print("\nType breakdown of flagged transactions:")
    for row in type_breakdown:
        print(f"  {str(row['type']):12s}  flagged={row['flagged_n']:,}  "
              f"fraud={row['fraud_n']:,}  ({row['fraud_pct']}% fraud)")

    return {
        "split": split_name,
        "total_fraud": total_fraud,
        "flagged_count": flagged_count,
        "flagged_and_fraud": flagged_and_fraud,
        "coverage": round(coverage, 6),
        "precision_of_flag": round(precision_of_flag, 6),
        "verdict": verdict,
        "type_breakdown": type_breakdown,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    model, X, y, split_name: str = "split"
) -> Tuple[Dict[str, float], np.ndarray]:
    y_pred  = model.predict(X)
    y_score = model.predict_proba(X)[:, 1]

    metrics = {
        "split":     split_name,
        "accuracy":  float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall":    float(recall_score(y, y_pred, zero_division=0)),
        "f1":        float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y, y_score)),
        "pr_auc":    float(average_precision_score(y, y_score)),
    }
    cm = confusion_matrix(y, y_pred)

    print(f"\n=== {split_name.upper()} ===")
    for k, v in metrics.items():
        if k != "split":
            print(f"  {k}: {v:.6f}")
    print("\nConfusion matrix:\n", cm)
    print("\nClassification report:")
    print(classification_report(y, y_pred, digits=4, zero_division=0))

    return metrics, cm


def search_thresholds(model, X_val, y_val) -> pd.DataFrame:
    val_scores = model.predict_proba(X_val)[:, 1]
    rows = []
    for thr in np.arange(0.10, 1.00, 0.05):
        y_pred_thr = (val_scores >= thr).astype(int)
        rows.append({
            "threshold":       float(thr),
            "precision":       float(precision_score(y_val, y_pred_thr, zero_division=0)),
            "recall":          float(recall_score(y_val, y_pred_thr, zero_division=0)),
            "f1":              float(f1_score(y_val, y_pred_thr, zero_division=0)),
            "predicted_fraud": int(y_pred_thr.sum()),
        })
    return pd.DataFrame(rows)


def compute_cost_table(
    threshold_results: pd.DataFrame,
    y_val: pd.Series,
    fn_cost: int = DEFAULT_FN_COST,
    fp_cost: int = DEFAULT_FP_COST,
) -> pd.DataFrame:
    support_pos = int(y_val.sum())
    support_neg = int((y_val == 0).sum())
    cost_rows = []

    for _, row in threshold_results.iterrows():
        recall          = row["recall"]
        predicted_fraud = int(row["predicted_fraud"])

        # Use int() + epsilon instead of round() to avoid off-by-one that
        # can produce negative fp counts in edge cases.
        tp = int(recall * support_pos + 1e-9)
        fn = support_pos - tp
        fp = max(predicted_fraud - tp, 0)
        tn = support_neg - fp

        cost_rows.append({
            "threshold":       row["threshold"],
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision":       row["precision"],
            "recall":          recall,
            "f1":              row["f1"],
            "predicted_fraud": predicted_fraud,
            "expected_cost":   fn * fn_cost + fp * fp_cost,
        })

    return pd.DataFrame(cost_rows).sort_values("expected_cost").reset_index(drop=True)


def evaluate_with_threshold(
    model, X_test, y_test, threshold: float
) -> Tuple[Dict[str, float], np.ndarray]:
    test_scores    = model.predict_proba(X_test)[:, 1]
    test_pred      = (test_scores >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, test_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, test_pred, zero_division=0)),
    }
    cm = confusion_matrix(y_test, test_pred)

    print(f"\nTuned threshold : {threshold}")
    print(f"  Precision     : {metrics['precision']:.6f}")
    print(f"  Recall        : {metrics['recall']:.6f}")
    print(f"  F1            : {metrics['f1']:.6f}")
    print("\nConfusion matrix:\n", cm)
    print("\nClassification report:")
    print(classification_report(y_test, test_pred, digits=4, zero_division=0))

    return metrics, cm


def compute_expected_cost_from_cm(
    cm: np.ndarray,
    fn_cost: int = DEFAULT_FN_COST,
    fp_cost: int = DEFAULT_FP_COST,
) -> Dict[str, int]:
    tp = int(cm[1, 1])
    fn = int(cm[1, 0])
    fp = int(cm[0, 1])
    tn = int(cm[0, 0])
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "expected_cost": int(fn * fn_cost + fp * fp_cost),
    }


# ── Persistence ───────────────────────────────────────────────────────────────

def save_experiment_outputs(
    results_dir: Path,
    model,
    val_metrics: Dict,
    test_metrics: Dict,
    val_cm: np.ndarray,
    test_cm: np.ndarray,
    threshold_results: Optional[pd.DataFrame] = None,
    cost_df: Optional[pd.DataFrame] = None,
    tuned_metrics: Optional[Dict] = None,
    tuned_cm: Optional[np.ndarray] = None,
    prefix: str = "",
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    p = f"{prefix}_" if prefix else ""

    joblib.dump(model, results_dir / f"{p}model.joblib")

    with open(results_dir / f"{p}validation_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=4)
    with open(results_dir / f"{p}test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=4)

    pd.DataFrame(val_cm,  index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_csv(
        results_dir / f"{p}validation_confusion_matrix.csv")
    pd.DataFrame(test_cm, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_csv(
        results_dir / f"{p}test_confusion_matrix.csv")
    pd.DataFrame([val_metrics, test_metrics]).to_csv(
        results_dir / f"{p}metrics_summary.csv", index=False)

    if threshold_results is not None:
        threshold_results.to_csv(results_dir / f"{p}threshold_results.csv", index=False)
    if cost_df is not None:
        cost_df.to_csv(results_dir / f"{p}cost_results.csv", index=False)
    if tuned_metrics is not None:
        with open(results_dir / f"{p}tuned_test_metrics.json", "w") as f:
            json.dump(tuned_metrics, f, indent=4)
    if tuned_cm is not None:
        pd.DataFrame(tuned_cm, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_csv(
            results_dir / f"{p}tuned_test_confusion_matrix.csv")