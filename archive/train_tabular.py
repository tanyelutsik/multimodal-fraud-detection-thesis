from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"


TARGET_COL = "isFraud"
DROP_COLS = ["isFraud", "nameOrig", "nameDest", "sample_id"]


def load_splits():
    train_df = pd.read_csv(PROCESSED_DIR / "paysim_train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "paysim_val.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "paysim_test.csv")
    return train_df, val_df, test_df


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=DROP_COLS, errors="ignore")
    y = df[TARGET_COL]
    return X, y


def build_model():
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ])
    return model


def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.10, 0.91, 0.05)
    best_threshold = 0.5
    best_f1 = -1

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1


def evaluate_split(y_true, y_prob, threshold, split_name):
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "split": split_name,
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob)
    }

    print("\n" + "=" * 60)
    print(f"{split_name.upper()} METRICS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return metrics


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load splits
    train_df, val_df, test_df = load_splits()

    # 2. Split X and y
    X_train, y_train = split_features_target(train_df)
    X_val, y_val = split_features_target(val_df)
    X_test, y_test = split_features_target(test_df)

    print("Train:", X_train.shape, y_train.shape)
    print("Val:  ", X_val.shape, y_val.shape)
    print("Test: ", X_test.shape, y_test.shape)

    # 3. Train model
    model = build_model()
    model.fit(X_train, y_train)

    # 4. Validation probabilities
    val_prob = model.predict_proba(X_val)[:, 1]

    # 5. Find best threshold on validation set
    best_threshold, best_val_f1 = find_best_threshold(y_val, val_prob)
    print(f"\nBest validation threshold: {best_threshold}")
    print(f"Best validation F1: {best_val_f1:.4f}")

    # 6. Evaluate on validation and test
    val_metrics = evaluate_split(y_val, val_prob, best_threshold, "validation")

    test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_split(y_test, test_prob, best_threshold, "test")

    # 7. Save metrics
    metrics_df = pd.DataFrame([val_metrics, test_metrics])
    metrics_df.to_csv(RESULTS_DIR / "paysim_logreg_metrics.csv", index=False)
    print(f"\nSaved metrics to: {RESULTS_DIR / 'paysim_logreg_metrics.csv'}")