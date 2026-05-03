from pathlib import Path
import pandas as pd
import numpy as np

"""
build_merged_dataset_mixed_balanced.py
---------------------------------------
Creates the BALANCED MIXED multimodal dataset.

Design principles (from supervisor feedback)
--------------------------------------------

Rule 1 — Equal combo counts, not equal final labels
    Each combo type gets the same number of rows:
        tab0_img0  n rows  (label=0)
        tab1_img1  n rows  (label=1)
        tab1_img0  n rows  (label=1)
        tab0_img1  n rows  (label=1)

    Final label distribution is naturally 1:3 (label-0 : label-1).
    This is handled during training with class_weight or pos_weight.
    It is better than having balanced labels but missing tab1_img0.

Rule 2 — Split genuine/forged pools between combo types
    Genuine images split between: tab0_img0 AND tab1_img0
    Forged  images split between: tab1_img1 AND tab0_img1
    So tab1_img0 is never starved of genuine images.

Rule 3 — Source-balanced image sampling
    Sample equally from each source_dataset within genuine/forged pools.
    Prevents MIDV (4000 genuine) dominating tab0_img0 and tab1_img0.

Budget calculation
------------------
n_per_combo = min(
    min_genuine_per_source * n_genuine_sources // 2,   # split between tab0_img0 + tab1_img0
    min_forged_per_source  * n_forged_sources  // 2,   # split between tab1_img1 + tab0_img1
    tab_genuine_count // 2,
    tab_fraud_count   // 2,
)

Outputs
-------
    mm_train_mixed_balanced_group_test.csv
    mm_val_mixed_balanced_group_test.csv
    mm_test_mixed_balanced_group_test.csv
    mm_full_mixed_balanced_group_test.csv
"""

print("build_merged_dataset_mixed_balanced.py started")

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RANDOM_STATE  = 42

LABEL_MAP = {
    "bona_fide": 0, "genuine": 0, "normal": 0, "clean": 0,
    "fake": 1,      "fraud":   1, "forged": 1,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tabular_splits() -> pd.DataFrame:
    dfs = []
    for split_name, fname in [
        ("train", "paysim_train.csv"),
        ("val",   "paysim_val.csv"),
        ("test",  "paysim_test.csv"),
    ]:
        df             = pd.read_csv(PROCESSED_DIR / fname)
        df["split"]      = split_name
        df["tabular_id"] = df["sample_id"]
        dfs.append(df)

    df              = pd.concat(dfs, ignore_index=True)
    df["final_label"] = df["isFraud"].astype(int)
    print(f"Loaded tabular: {df.shape}")
    print(df["final_label"].value_counts())
    return df


def load_image_splits() -> pd.DataFrame:
    dfs = []
    for split_name, fname in [
        ("train", "image_train_group_test.csv"),
        ("val",   "image_val_group_test.csv"),
        ("test",  "image_test_group_test.csv"),
    ]:
        df         = pd.read_csv(PROCESSED_DIR / fname)
        df["split"]  = split_name
        dfs.append(df)

    df              = pd.concat(dfs, ignore_index=True)
    df["image_class"] = df["image_class"].astype(str).str.strip().str.lower()
    df["final_label"] = df["image_class"].map(LABEL_MAP)

    if df["final_label"].isna().any():
        unknown = sorted(df.loc[df["final_label"].isna(), "image_class"].unique().tolist())
        raise ValueError(f"Unknown image labels: {unknown}")

    print(f"Loaded image splits: {df.shape}")
    print(df["final_label"].value_counts())
    return df


# ---------------------------------------------------------------------------
# Source-balanced sampling
# ---------------------------------------------------------------------------

def sample_balanced_by_source(
    df:           pd.DataFrame,
    label:        int,
    n_total:      int,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Sample n_total rows of a given label drawing equally from each
    source_dataset. Prevents MIDV dominating the genuine image pool.

    For genuine images: MIDV=2788, FantasyID=650, FMIDV=0
    Without balancing: draws 2788+650 but mostly MIDV
    With balancing:    draws 650 from MIDV + 650 from FantasyID = 1300
    """
    pool    = df[df["final_label"] == label].copy()
    sources = sorted(pool["source_dataset"].unique())

    if len(sources) == 0 or n_total == 0:
        return pd.DataFrame()

    n_per_source = max(1, n_total // len(sources))
    parts        = []
    used_indices = set()

    for source in sources:
        src_rows = pool[pool["source_dataset"] == source]
        n        = min(len(src_rows), n_per_source)
        if n > 0:
            sampled = src_rows.sample(n=n, random_state=random_state)
            parts.append(sampled)
            used_indices.update(sampled.index)

    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    # Top up if short due to small sources
    if len(result) < n_total:
        remainder = pool[~pool.index.isin(used_indices)]
        shortfall = n_total - len(result)
        if len(remainder) > 0:
            extra  = remainder.sample(
                n=min(shortfall, len(remainder)), random_state=random_state
            )
            result = pd.concat([result, extra], ignore_index=True)

    return result.sample(frac=1, random_state=random_state).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Budget calculation
# ---------------------------------------------------------------------------

def compute_budget(
    tab_df: pd.DataFrame,
    img_df: pd.DataFrame,
    split_name: str,
) -> int:
    """
    Compute n_per_combo — the number of rows for EACH combo type.

    Limited by:
    - Genuine images: split between tab0_img0 and tab1_img0
      → need 2 * n_per_combo genuine images total
      → budget = min_genuine_per_source * n_genuine_sources // 2

    - Forged images: split between tab1_img1 and tab0_img1
      → need 2 * n_per_combo forged images total
      → budget = min_forged_per_source * n_forged_sources // 2

    - Tabular genuine: split between tab0_img0 and tab0_img1
      → need 2 * n_per_combo genuine tab rows

    - Tabular fraud: split between tab1_img1 and tab1_img0
      → need 2 * n_per_combo fraud tab rows
    """
    sources = sorted(img_df["source_dataset"].unique())

    genuine_per_src = [
        int((img_df[(img_df["source_dataset"] == s) &
                    (img_df["final_label"] == 0)].shape[0]))
        for s in sources
    ]
    genuine_nonzero = [c for c in genuine_per_src if c > 0]
    n_src_genuine   = len(genuine_nonzero)
    min_genuine     = min(genuine_nonzero) if genuine_nonzero else 0

    forged_per_src = [
        int((img_df[(img_df["source_dataset"] == s) &
                    (img_df["final_label"] == 1)].shape[0]))
        for s in sources
    ]
    forged_nonzero = [c for c in forged_per_src if c > 0]
    n_src_forged   = len(forged_nonzero)
    min_forged     = min(forged_nonzero) if forged_nonzero else 0

    total_genuine  = min_genuine * n_src_genuine
    total_forged   = min_forged  * n_src_forged

    n_tab_genuine  = int((tab_df["final_label"] == 0).sum())
    n_tab_fraud    = int((tab_df["final_label"] == 1).sum())

    budget = min(
        total_genuine  // 2,    # genuine split between 2 combos
        total_forged   // 2,    # forged  split between 2 combos
        n_tab_genuine  // 2,    # tab genuine split between 2 combos
        n_tab_fraud    // 2,    # tab fraud    split between 2 combos
    )
    budget = max(1, budget)

    print(f"\n  {split_name} budget:")
    print(f"    img genuine available (balanced): {total_genuine}  "
          f"(min/src={min_genuine}, n_src={n_src_genuine})")
    print(f"    img forged  available (balanced): {total_forged}   "
          f"(min/src={min_forged}, n_src={n_src_forged})")
    print(f"    tab genuine: {n_tab_genuine}  tab fraud: {n_tab_fraud}")
    print(f"    => n_per_combo = {budget}")

    return budget


# ---------------------------------------------------------------------------
# Mixed pairing
# ---------------------------------------------------------------------------

def pair_mixed_split(
    tab_df:       pd.DataFrame,
    img_df:       pd.DataFrame,
    split_name:   str,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Creates balanced mixed pairs with equal combo type counts.

    Each combo gets n_per_combo rows.
    Final label distribution: label-0=n, label-1=3n (1:3 ratio).
    Handle imbalance during training with pos_weight or class_weight.
    """
    n_per_combo = compute_budget(tab_df, img_df, split_name)

    # Tabular pools — shuffled
    tab_genuine = (tab_df[tab_df["final_label"] == 0]
                   .sample(frac=1, random_state=random_state)
                   .reset_index(drop=True))
    tab_fraud   = (tab_df[tab_df["final_label"] == 1]
                   .sample(frac=1, random_state=random_state)
                   .reset_index(drop=True))

    # Image pools — source-balanced
    # Genuine: need n_per_combo for tab0_img0 + n_per_combo for tab1_img0
    img_genuine_pool = sample_balanced_by_source(
        img_df, label=0, n_total=n_per_combo * 2, random_state=random_state
    )
    # Forged: need n_per_combo for tab1_img1 + n_per_combo for tab0_img1
    img_forged_pool  = sample_balanced_by_source(
        img_df, label=1, n_total=n_per_combo * 2, random_state=random_state
    )

    # Slice into non-overlapping halves
    img_genuine_00 = img_genuine_pool.iloc[:n_per_combo].reset_index(drop=True)
    img_genuine_10 = img_genuine_pool.iloc[n_per_combo:n_per_combo * 2].reset_index(drop=True)
    img_forged_11  = img_forged_pool.iloc[:n_per_combo].reset_index(drop=True)
    img_forged_01  = img_forged_pool.iloc[n_per_combo:n_per_combo * 2].reset_index(drop=True)

    # Log source balance of each slice
    print(f"\n  {split_name} image source balance:")
    for slice_name, pool in [
        ("img_genuine_00 (tab0_img0)", img_genuine_00),
        ("img_genuine_10 (tab1_img0)", img_genuine_10),
        ("img_forged_11  (tab1_img1)", img_forged_11),
        ("img_forged_01  (tab0_img1)", img_forged_01),
    ]:
        if len(pool) > 0 and "source_dataset" in pool.columns:
            src_dist = pool["source_dataset"].value_counts().to_dict()
            print(f"    {slice_name}: {src_dist}")

    def make_pairs(tab_part, img_part, n, combo_label, combo_code):
        n = min(n, len(tab_part), len(img_part))
        if n == 0:
            return pd.DataFrame()
        tab_s  = tab_part.iloc[:n].reset_index(drop=True).add_prefix("tab_")
        img_s  = img_part.iloc[:n].reset_index(drop=True).add_prefix("img_")
        assert len(tab_s) == len(img_s), \
            f"Mismatch {combo_code}: tab={len(tab_s)} img={len(img_s)}"
        merged = pd.concat([tab_s, img_s], axis=1)
        merged["final_label"]     = combo_label
        merged["tab_fraud_label"] = int(combo_code[0])
        merged["img_fraud_label"] = int(combo_code[1])
        merged["combo_type"]      = f"tab{combo_code[0]}_img{combo_code[1]}"
        merged["split"]           = split_name
        merged["mm_id"]           = [
            f"mm_{split_name}_{combo_code}_{i:06d}" for i in range(n)
        ]
        return merged

    parts = [
        make_pairs(tab_genuine, img_genuine_00, n_per_combo, 0, "00"),
        make_pairs(tab_fraud,   img_forged_11,  n_per_combo, 1, "11"),
        make_pairs(tab_fraud,   img_genuine_10, n_per_combo, 1, "10"),
        make_pairs(tab_genuine, img_forged_01,  n_per_combo, 1, "01"),
    ]
    parts  = [p for p in parts if len(p) > 0]
    result = pd.concat(parts, ignore_index=True)
    return result.sample(frac=1, random_state=random_state).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def save_split(df: pd.DataFrame, filename: str) -> None:
    out_path = PROCESSED_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def print_info(name: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print("Shape:", df.shape)
    if df.empty:
        print("EMPTY"); return

    print("\nFinal label distribution:")
    vc = df["final_label"].value_counts()
    print(vc)
    print(f"  Ratio label-0:label-1 = 1:{vc.get(1,0)/max(vc.get(0,1),1):.1f}")

    if "combo_type" in df.columns:
        print("\nCombo type breakdown:")
        print(df["combo_type"].value_counts().sort_index())
        print("\nCombo type x final_label:")
        print(pd.crosstab(df["combo_type"], df["final_label"]))

    if "img_source_dataset" in df.columns and "combo_type" in df.columns:
        print("\nSource x combo_type:")
        print(pd.crosstab(df["img_source_dataset"], df["combo_type"]))

    if "img_image_class" in df.columns and "combo_type" in df.columns:
        print("\nImage class x combo_type:")
        print(pd.crosstab(df["img_image_class"], df["combo_type"]))

    null_paths = df["img_image_path"].isna().sum()
    print(f"\nNull img_image_path: {null_paths}")


def check_leakage(mm_train, mm_val, mm_test) -> bool:
    train_imgs = set(mm_train["img_image_path"].dropna())
    val_imgs   = set(mm_val["img_image_path"].dropna())
    test_imgs  = set(mm_test["img_image_path"].dropna())

    tv = len(train_imgs & val_imgs)
    tt = len(train_imgs & test_imgs)
    vt = len(val_imgs   & test_imgs)

    print(f"\nLeakage check:")
    print(f"  Train/Val:  {tv}  {'LEAK' if tv>0 else 'clean'}")
    print(f"  Train/Test: {tt}  {'LEAK' if tt>0 else 'clean'}")
    print(f"  Val/Test:   {vt}  {'LEAK' if vt>0 else 'clean'}")
    return (tv + tt + vt) == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Main block running")

    tabular_df = load_tabular_splits()
    image_df   = load_image_splits()

    mm_train = pair_mixed_split(
        tabular_df[tabular_df["split"] == "train"],
        image_df[image_df["split"]     == "train"],
        "train",
    )
    mm_val = pair_mixed_split(
        tabular_df[tabular_df["split"] == "val"],
        image_df[image_df["split"]     == "val"],
        "val",
    )
    mm_test = pair_mixed_split(
        tabular_df[tabular_df["split"] == "test"],
        image_df[image_df["split"]     == "test"],
        "test",
    )

    # Drop null image paths
    for label, df in [("train", mm_train), ("val", mm_val), ("test", mm_test)]:
        n_null = df["img_image_path"].isna().sum()
        if n_null > 0:
            print(f"Warning: dropping {n_null} null image rows from {label}")
    mm_train = mm_train[mm_train["img_image_path"].notna()].reset_index(drop=True)
    mm_val   = mm_val[mm_val["img_image_path"].notna()].reset_index(drop=True)
    mm_test  = mm_test[mm_test["img_image_path"].notna()].reset_index(drop=True)

    # Cross-split deduplication
    train_imgs = set(mm_train["img_image_path"].dropna())
    val_imgs   = set(mm_val["img_image_path"].dropna())
    test_imgs  = set(mm_test["img_image_path"].dropna())
    leaked     = (train_imgs & val_imgs) | (train_imgs & test_imgs) | (val_imgs & test_imgs)
    if leaked:
        print(f"\nRemoving {len(leaked)} leaked path(s).")
        mm_val  = mm_val[~mm_val["img_image_path"].isin(leaked)].reset_index(drop=True)
        mm_test = mm_test[~mm_test["img_image_path"].isin(leaked)].reset_index(drop=True)

    mm_full = pd.concat([mm_train, mm_val, mm_test], ignore_index=True)

    # Save — note: _balanced_ prefix, does NOT overwrite old mixed files
    save_split(mm_train, "mm_train_mixed_balanced_group_test.csv")
    save_split(mm_val,   "mm_val_mixed_balanced_group_test.csv")
    save_split(mm_test,  "mm_test_mixed_balanced_group_test.csv")
    save_split(mm_full,  "mm_full_mixed_balanced_group_test.csv")

    print_info("MM TRAIN BALANCED", mm_train)
    print_info("MM VAL BALANCED",   mm_val)
    print_info("MM TEST BALANCED",  mm_test)

    clean = check_leakage(mm_train, mm_val, mm_test)
    print(f"\nAll clean: {clean}")
    print("\nSuccess")
