"""
cost_config.py
--------------
Central cost weight configuration for all fraud detection experiments.

Weights are RELATIVE — not monetary values.
cost_score = FN_WEIGHT * FN + FP_WEIGHT * FP

Following Elkan (2001): only the relative cost matrix determines
optimal classification decisions, not absolute monetary values.
Using normalized weights avoids unverifiable monetary assumptions.

Active scenario : 10:1
Meaning         : one missed fraud counts as 10 penalty units,
                  one false alarm counts as 1 penalty unit.

All scenarios tested in sensitivity analysis:
     5:1  — moderate asymmetry
    10:1  — main scenario (asymmetric but operationally feasible)
    20:1  — higher asymmetry
    50:1  — aggressive
   100:1  — very aggressive

References:
    Elkan, C. (2001). The foundations of cost-sensitive learning.
    Bahnsen, A.C. et al. (2013). Example-dependent cost-sensitive
        logistic regression for credit scoring.
    Hoppner, S. et al. (2022). Instance-dependent cost-sensitive
        learning for detecting transfer fraud.

Usage:
    from cost_config import FN_WEIGHT, FP_WEIGHT, COST_SCENARIO
"""

# ── Active scenario ────────────────────────────────────────────────────────────
COST_SCENARIO: str = "10:1"

FN_WEIGHT: int = 10   # penalty weight for false negatives (missed fraud)
FP_WEIGHT: int = 1    # penalty weight for false positives (false alarms)

# Legacy aliases so existing notebooks work without renaming every variable
FN_COST = FN_WEIGHT
FP_COST = FP_WEIGHT

# Cost ratio for logging
COST_RATIO: int = FN_WEIGHT // FP_WEIGHT   # 10

# ── All scenarios for sensitivity analysis ─────────────────────────────────────
ALL_SCENARIOS: dict = {
    " 5:1": ( 5,  1),
    "10:1": (10,  1),
    "20:1": (20,  1),
    "50:1": (50,  1),
    "100:1":(100, 1),
}

# ── Labels for plots and tables ────────────────────────────────────────────────
COST_LABEL      = "Weighted misclassification cost"
COST_AXIS_LABEL = "Weighted cost  (FN x w_FN + FP x w_FP)"
COST_FORMULA    = f"cost = FN x {FN_WEIGHT} + FP x {FP_WEIGHT}"

if __name__ == "__main__":
    print(f"Active scenario : {COST_SCENARIO}")
    print(f"FN_WEIGHT       : {FN_WEIGHT}")
    print(f"FP_WEIGHT       : {FP_WEIGHT}")
    print(f"Formula         : {COST_FORMULA}")
    print()
    print("All scenarios:")
    for label, (fn, fp) in ALL_SCENARIOS.items():
        print(f"  {label}   cost = FN x {fn:>3} + FP x {fp}")