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


def get_feature_config() -> Tuple[str, list[str], list[str], list[str]]:
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

    feature_cols = categorical_features + numeric_features
    return target_col, categorical_features, numeric_features, feature_cols


def prepare_dataframes(
    paysim_train: pd.DataFrame,
    paysim_val: pd.DataFrame,
    paysim_test: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    target_col, categorical_features, numeric_features, feature_cols = get_feature_config()

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
    }


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())

    if pos == 0:
        raise ValueError("No positive class found in y_train.")

    return neg / pos


def build_xgb_pipeline(
    preprocessor: ColumnTransformer,
    imbalance_method: str = "none",
    random_state: int = RANDOM_STATE,
    scale_pos_weight: float = 1.0,
    smote_sampling_strategy: float = 1.0,
) -> Pipeline:
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
    }

    if imbalance_method == "none":
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    XGBClassifier(
                        **common_params,
                        scale_pos_weight=1.0,
                    ),
                ),
            ]
        )

    elif imbalance_method == "scale_pos_weight":
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    XGBClassifier(
                        **common_params,
                        scale_pos_weight=scale_pos_weight,
                    ),
                ),
            ]
        )

    else:  # smote
        model = ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "smote",
                    SMOTE(
                        sampling_strategy=smote_sampling_strategy,
                        random_state=random_state,
                    ),
                ),
                (
                    "classifier",
                    XGBClassifier(
                        **common_params,
                        scale_pos_weight=1.0,
                    ),
                ),
            ]
        )

    return model


def fit_model(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y, split_name: str = "split") -> Tuple[Dict[str, float], np.ndarray]:
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:, 1]

    metrics = {
        "split": split_name,
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_score)),
        "pr_auc": float(average_precision_score(y, y_score)),
    }

    cm = confusion_matrix(y, y_pred)

    print(f"\n=== {split_name.upper()} ===")
    for k, v in metrics.items():
        if k != "split":
            print(f"{k}: {v:.6f}")

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(classification_report(y, y_pred, digits=4, zero_division=0))

    return metrics, cm


def search_thresholds(model, X_val, y_val) -> pd.DataFrame:
    val_scores = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.10, 1.00, 0.05)

    rows = []
    for thr in thresholds:
        y_pred_thr = (val_scores >= thr).astype(int)

        rows.append(
            {
                "threshold": float(thr),
                "precision": float(precision_score(y_val, y_pred_thr, zero_division=0)),
                "recall": float(recall_score(y_val, y_pred_thr, zero_division=0)),
                "f1": float(f1_score(y_val, y_pred_thr, zero_division=0)),
                "predicted_fraud": int(y_pred_thr.sum()),
            }
        )

    return pd.DataFrame(rows)


def compute_cost_table(
    threshold_results: pd.DataFrame,
    y_val: pd.Series,
    fn_cost: int = 50000,
    fp_cost: int = 500,
) -> pd.DataFrame:
    support_pos = int(y_val.sum())
    support_neg = int((y_val == 0).sum())

    cost_rows = []

    for _, row in threshold_results.iterrows():
        thr = row["threshold"]
        recall = row["recall"]
        predicted_fraud = int(row["predicted_fraud"])

        tp = round(recall * support_pos)
        fn = support_pos - tp
        fp = predicted_fraud - tp
        tn = support_neg - fp

        expected_cost = fn * fn_cost + fp * fp_cost

        cost_rows.append(
            {
                "threshold": thr,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": row["precision"],
                "recall": recall,
                "f1": row["f1"],
                "predicted_fraud": predicted_fraud,
                "expected_cost": expected_cost,
            }
        )

    return pd.DataFrame(cost_rows).sort_values("expected_cost").reset_index(drop=True)


def evaluate_with_threshold(model, X_test, y_test, threshold: float) -> Tuple[Dict[str, float], np.ndarray]:
    test_scores = model.predict_proba(X_test)[:, 1]
    test_pred_tuned = (test_scores >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_test, test_pred_tuned, zero_division=0)),
        "recall": float(recall_score(y_test, test_pred_tuned, zero_division=0)),
        "f1": float(f1_score(y_test, test_pred_tuned, zero_division=0)),
    }

    cm = confusion_matrix(y_test, test_pred_tuned)

    print("\nTuned threshold:", threshold)
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1:", metrics["f1"])

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(classification_report(y_test, test_pred_tuned, digits=4, zero_division=0))

    return metrics, cm


def compute_expected_cost_from_cm(
    cm: np.ndarray,
    fn_cost: int = 50000,
    fp_cost: int = 500,
) -> Dict[str, int]:
    tp = int(cm[1, 1])
    fn = int(cm[1, 0])
    fp = int(cm[0, 1])
    tn = int(cm[0, 0])

    expected_cost = fn * fn_cost + fp * fp_cost

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "expected_cost": int(expected_cost),
    }


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

    prefix_str = f"{prefix}_" if prefix else ""

    joblib.dump(model, results_dir / f"{prefix_str}model.joblib")

    with open(results_dir / f"{prefix_str}validation_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=4)

    with open(results_dir / f"{prefix_str}test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=4)

    pd.DataFrame(val_cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        results_dir / f"{prefix_str}validation_confusion_matrix.csv"
    )
    pd.DataFrame(test_cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        results_dir / f"{prefix_str}test_confusion_matrix.csv"
    )

    pd.DataFrame([val_metrics, test_metrics]).to_csv(
        results_dir / f"{prefix_str}metrics_summary.csv", index=False
    )

    if threshold_results is not None:
        threshold_results.to_csv(results_dir / f"{prefix_str}threshold_results.csv", index=False)

    if cost_df is not None:
        cost_df.to_csv(results_dir / f"{prefix_str}cost_results.csv", index=False)

    if tuned_metrics is not None:
        with open(results_dir / f"{prefix_str}tuned_test_metrics.json", "w") as f:
            json.dump(tuned_metrics, f, indent=4)

    if tuned_cm is not None:
        pd.DataFrame(tuned_cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
            results_dir / f"{prefix_str}tuned_test_confusion_matrix.csv"
        )