from __future__ import annotations

"""
mm_tabular_utils.py
-------------------
Shared tabular feature definitions and engineering for the mixed-all
multimodal dataset experiments.

Used by:
    17_mm_tabular_logreg_baseline.ipynb
    18_mm_tabular_xgboost_baseline.ipynb
    19_mm_deep_multimodal.ipynb  (via multimodal_utils.py)

Feature selection rules
-----------------------
INCLUDED — transaction variables (same as notebooks 11 and 12):
    tab_type, tab_amount, tab_oldbalanceOrg, tab_newbalanceOrig,
    tab_oldbalanceDest, tab_newbalanceDest, tab_isFlaggedFraud

EXCLUDED — label or identifier columns:
    tab_isFraud        ← final_label is derived from this (leakage)
    tab_final_label    ← IS the label
    img_final_label    ← IS the label
    img_image_class    ← encodes the label
    combo_type         ← derived from labels
    img_source_dataset ← confirmed shortcut in image baseline analysis
    mm_id, split       ← identifiers

tab_isFlaggedFraud is INCLUDED — it is a PaySim rule-based flag that
fires when a single transfer exceeds 200,000. It is NOT the fraud label.

Engineered features
-------------------
Same balance-delta and ratio features as notebook 12 (full PaySim XGBoost).
Applied consistently across LR, XGBoost, and deep multimodal models so
the FE comparison is valid across all model types.

Feature sets
------------
RAW_COLS    : 8 features  (7 numeric + tab_type)
FE_COLS     : 13 features (RAW_COLS + 5 engineered)
"""

from typing import List, Set
import pandas as pd


# ---------------------------------------------------------------------------
# Target column
# ---------------------------------------------------------------------------

TARGET_COL = "final_label"


# ---------------------------------------------------------------------------
# Forbidden columns — must never be used as features
# ---------------------------------------------------------------------------

FORBIDDEN_COLS: Set[str] = {
    "tab_isFraud",          # leaks final_label — derived from it
    "tab_final_label",      # IS the label
    "final_label",          # IS the label
    "img_final_label",      # IS the label
    "img_image_class",      # encodes the label
    "img_original_label",   # encodes the label
    "combo_type",           # derived from labels
    "img_source_dataset",   # confirmed shortcut
    "mm_id",                # identifier
    "split",                # split marker
    "img_split",            # split marker
    "img_image_path",       # path, not a feature
}


# ---------------------------------------------------------------------------
# Raw feature columns
# ---------------------------------------------------------------------------

NUMERIC_RAW: List[str] = [
    "tab_step",
    "tab_amount",
    "tab_oldbalanceOrg",
    "tab_newbalanceOrig",
    "tab_oldbalanceDest",
    "tab_newbalanceDest",
    "tab_isFlaggedFraud",   # rule-based PaySim flag, NOT the label
]

CAT_COLS: List[str] = [
    "tab_type",             # transaction type, encoded 0-4
]

RAW_COLS: List[str] = NUMERIC_RAW + CAT_COLS


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add balance-delta and ratio features.

    Identical to notebook 12 (full PaySim XGBoost) and multimodal_utils.py
    so results are comparable across all model types.

    Features added
    --------------
    tab_balance_delta_orig  : newbalanceOrig - oldbalanceOrg
        Negative when money left the origin account as expected in fraud.

    tab_balance_delta_dest  : newbalanceDest - oldbalanceDest
        Positive when money arrived in destination account.

    tab_amount_ratio_orig   : amount / (oldbalanceOrg + 1)
        How large the transaction is relative to origin balance.
        High ratio = emptying the account.

    tab_amount_ratio_dest   : amount / (oldbalanceDest + 1)
        How large the transaction is relative to destination balance.

    tab_orig_balance_zeroed : 1 if newbalanceOrig == 0 else 0
        Flag for accounts emptied to zero — strong fraud signal in PaySim.
    """
    d = df.copy()
    d["tab_balance_delta_orig"]  = (d["tab_newbalanceOrig"]
                                    - d["tab_oldbalanceOrg"])
    d["tab_balance_delta_dest"]  = (d["tab_newbalanceDest"]
                                    - d["tab_oldbalanceDest"])
    d["tab_amount_ratio_orig"]   = (d["tab_amount"]
                                    / (d["tab_oldbalanceOrg"] + 1))
    d["tab_amount_ratio_dest"]   = (d["tab_amount"]
                                    / (d["tab_oldbalanceDest"] + 1))
    d["tab_orig_balance_zeroed"] = (d["tab_newbalanceOrig"] == 0).astype(int)
    return d


NUMERIC_FE: List[str] = NUMERIC_RAW + [
    "tab_balance_delta_orig",
    "tab_balance_delta_dest",
    "tab_amount_ratio_orig",
    "tab_amount_ratio_dest",
    "tab_orig_balance_zeroed",
]

FE_COLS: List[str] = NUMERIC_FE + CAT_COLS


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_features(cols: List[str]) -> None:
    """Raise ValueError if any forbidden column is in cols."""
    bad = set(cols) & FORBIDDEN_COLS
    if bad:
        raise ValueError(
            f"Forbidden columns in feature list — would cause data leakage:\n"
            f"  {sorted(bad)}\n"
            f"Remove them before training."
        )
    print(f"Feature validation passed — {len(cols)} features, no leakage.")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_feature_summary() -> None:
    print(f"Raw features  ({len(RAW_COLS)}): {RAW_COLS}")
    print(f"FE  features  ({len(FE_COLS)}): {FE_COLS}")
    print(f"\nForbidden (excluded): {sorted(FORBIDDEN_COLS)}")
