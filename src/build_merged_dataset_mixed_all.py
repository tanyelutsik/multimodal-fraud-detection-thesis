from pathlib import Path
import pandas as pd
import numpy as np

"""
build_merged_dataset_mixed_all.py
----------------------------------
Creates the MIXED-ALL multimodal dataset.

Key design principle
--------------------
Each image is used exactly once. The number of multimodal rows in each
split equals the number of images available in that split.

    train rows  ≈ number of images in image_train_group_test.csv
    val rows    ≈ number of images in image_val_group_test.csv
    test rows   ≈ number of images in image_test_group_test.csv

Pairing rule
------------
Genuine (bona fide) images:
    75% paired with non-fraud transaction  →  tab0_img0  →  label 0
    25% paired with fraud transaction      →  tab1_img0  →  label 1

Forged images:
    50% paired with non-fraud transaction  →  tab0_img1  →  label 1
    50% paired with fraud transaction      →  tab1_img1  →  label 1

The image label is never changed. Only the transaction it is paired with
changes. This means:
    label 0 = bona fide image + non-fraud transaction
    label 1 = forged image (any transaction) OR bona fide + fraud transaction

Source imbalance note
---------------------
Since all images are used, MIDV (genuine-only) will dominate the genuine
image pool. This is addressed during training via WeightedRandomSampler
(see notebooks 15/16), NOT by removing images here.

Outputs
-------
    mm_train_mixed_all_group_test.csv
    mm_val_mixed_all_group_test.csv
    mm_test_mixed_all_group_test.csv
    mm_full_mixed_all_group_test.csv
"""

print("build_merged_dataset_mixed_all.py started")

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RANDOM_STATE  = 42

LABEL_MAP = {
    "bona_fide": 0, "genuine": 0, "normal": 0, "clean": 0,
    "fake": 1,      "fraud":   1, "forged": 1,
}

# Pairing percentages
GENUINE_NONFR_PCT = 0.75   # 75% bona fide -> non-fraud tab -> tab0_img0 -> label 0
GENUINE_FR_PCT    = 0.25   # 25% bona fide -> fraud tab    -> tab1_img0 -> label 1
FORGED_NONFR_PCT  = 0.50   # 50% forged    -> non-fraud tab -> tab0_img1 -> label 1
FORGED_FR_PCT     = 0.50   # 50% forged    -> fraud tab    -> tab1_img1 -> label 1


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
# Pairing logic
# ---------------------------------------------------------------------------

def pair_split(
    tab_df:       pd.DataFrame,
    img_df:       pd.DataFrame,
    split_name:   str,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Pairs ALL images with transactions using percentage-based rules.

    Each image is used exactly once.
    The number of output rows equals the number of input images.

    Genuine images:
        GENUINE_NONFR_PCT -> paired with non-fraud transaction (tab0_img0, label=0)
        GENUINE_FR_PCT    -> paired with fraud transaction     (tab1_img0, label=1)

    Forged images:
        FORGED_NONFR_PCT  -> paired with non-fraud transaction (tab0_img1, label=1)
        FORGED_FR_PCT     -> paired with fraud transaction     (tab1_img1, label=1)
    """
    rng = np.random.RandomState(random_state)

    # Separate image pools
    img_genuine = (img_df[img_df["final_label"] == 0]
                   .sample(frac=1, random_state=random_state)
                   .reset_index(drop=True))
    img_forged  = (img_df[img_df["final_label"] == 1]
                   .sample(frac=1, random_state=random_state)
                   .reset_index(drop=True))

    # Separate tabular pools
    tab_nonfr = (tab_df[tab_df["final_label"] == 0]
                 .sample(frac=1, random_state=random_state)
                 .reset_index(drop=True))
    tab_fr    = (tab_df[tab_df["final_label"] == 1]
                 .sample(frac=1, random_state=random_state)
                 .reset_index(drop=True))

    # Calculate split sizes for genuine images
    n_genuine      = len(img_genuine)
    n_genuine_nonfr = int(round(n_genuine * GENUINE_NONFR_PCT))
    n_genuine_fr    = n_genuine - n_genuine_nonfr   # remainder to avoid rounding loss

    # Calculate split sizes for forged images
    n_forged       = len(img_forged)
    n_forged_nonfr  = int(round(n_forged * FORGED_NONFR_PCT))
    n_forged_fr     = n_forged - n_forged_nonfr

    print(f"\n  {split_name} image pools:")
    print(f"    genuine: {n_genuine}  →  tab0_img0={n_genuine_nonfr}  tab1_img0={n_genuine_fr}")
    print(f"    forged:  {n_forged}   →  tab0_img1={n_forged_nonfr}  tab1_img1={n_forged_fr}")

    # Check tabular pool sizes
    if len(tab_nonfr) < n_genuine_nonfr + n_forged_nonfr:
        print(f"  Warning: need {n_genuine_nonfr + n_forged_nonfr} non-fraud tab rows, "
              f"have {len(tab_nonfr)} — sampling with replacement")
    if len(tab_fr) < n_genuine_fr + n_forged_fr:
        print(f"  Warning: need {n_genuine_fr + n_forged_fr} fraud tab rows, "
              f"have {len(tab_fr)} — sampling with replacement")

    def make_pairs(img_part, tab_pool, n_needed, combo_label, combo_code, tab_offset=0):
        """Pair img_part rows with n_needed rows from tab_pool."""
        n = len(img_part)
        if n == 0:
            return pd.DataFrame()

        # Sample with replacement if needed
        if len(tab_pool) >= n_needed + tab_offset:
            tab_part = tab_pool.iloc[tab_offset:tab_offset + n].reset_index(drop=True)
        else:
            tab_part = tab_pool.sample(n=n, replace=True,
                                       random_state=random_state).reset_index(drop=True)

        img_s  = img_part.reset_index(drop=True).add_prefix("img_")
        tab_s  = tab_part.iloc[:n].reset_index(drop=True).add_prefix("tab_")

        assert len(img_s) == len(tab_s), \
            f"Mismatch {combo_code}: img={len(img_s)} tab={len(tab_s)}"

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

    # Slice image pools
    img_genuine_nonfr = img_genuine.iloc[:n_genuine_nonfr]
    img_genuine_fr    = img_genuine.iloc[n_genuine_nonfr:]
    img_forged_nonfr  = img_forged.iloc[:n_forged_nonfr]
    img_forged_fr     = img_forged.iloc[n_forged_nonfr:]

    # Non-fraud tab rows: used by tab0_img0 then tab0_img1
    parts = [
        make_pairs(img_genuine_nonfr, tab_nonfr, n_genuine_nonfr, 0, "00", tab_offset=0),
        make_pairs(img_forged_nonfr,  tab_nonfr, n_forged_nonfr,  1, "01", tab_offset=n_genuine_nonfr),
        make_pairs(img_genuine_fr,    tab_fr,    n_genuine_fr,    1, "10", tab_offset=0),
        make_pairs(img_forged_fr,     tab_fr,    n_forged_fr,     1, "11", tab_offset=n_genuine_fr),
    ]

    parts  = [p for p in parts if len(p) > 0]
    result = pd.concat(parts, ignore_index=True)
    result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"  Output rows: {len(result)}  (images in: {len(img_df)})")
    assert len(result) == len(img_df), \
        f"Row count mismatch: output={len(result)}, images={len(img_df)}"

    return result


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
    ratio = vc.get(1, 0) / max(vc.get(0, 1), 1)
    print(f"  Ratio label-0:label-1 = 1:{ratio:.2f}")

    if "combo_type" in df.columns:
        print("\nCombo type breakdown:")
        print(df["combo_type"].value_counts().sort_index())

    if "img_source_dataset" in df.columns and "combo_type" in df.columns:
        print("\nSource x combo_type:")
        print(pd.crosstab(df["img_source_dataset"], df["combo_type"]))

    null_paths = df["img_image_path"].isna().sum()
    if null_paths > 0:
        print(f"\nWARNING: {null_paths} null img_image_path rows")


def check_leakage(mm_train, mm_val, mm_test) -> bool:
    train_imgs = set(mm_train["img_image_path"].dropna())
    val_imgs   = set(mm_val["img_image_path"].dropna())
    test_imgs  = set(mm_test["img_image_path"].dropna())

    tv = len(train_imgs & val_imgs)
    tt = len(train_imgs & test_imgs)
    vt = len(val_imgs   & test_imgs)

    print(f"\nLeakage check (img_image_path):")
    print(f"  Train ∩ Val:  {tv}  {'⚠️  LEAK' if tv>0 else '✅  clean'}")
    print(f"  Train ∩ Test: {tt}  {'⚠️  LEAK' if tt>0 else '✅  clean'}")
    print(f"  Val   ∩ Test: {vt}  {'⚠️  LEAK' if vt>0 else '✅  clean'}")
    return (tv + tt + vt) == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Main block running")
    print(f"Pairing percentages:")
    print(f"  Genuine: {GENUINE_NONFR_PCT*100:.0f}% non-fraud (tab0_img0) + "
          f"{GENUINE_FR_PCT*100:.0f}% fraud (tab1_img0)")
    print(f"  Forged:  {FORGED_NONFR_PCT*100:.0f}% non-fraud (tab0_img1) + "
          f"{FORGED_FR_PCT*100:.0f}% fraud (tab1_img1)")

    tabular_df = load_tabular_splits()
    image_df   = load_image_splits()

    mm_train = pair_split(
        tabular_df[tabular_df["split"] == "train"],
        image_df[image_df["split"]     == "train"],
        "train",
    )
    mm_val = pair_split(
        tabular_df[tabular_df["split"] == "val"],
        image_df[image_df["split"]     == "val"],
        "val",
    )
    mm_test = pair_split(
        tabular_df[tabular_df["split"] == "test"],
        image_df[image_df["split"]     == "test"],
        "test",
    )

    # Drop any null image paths (safety)
    for label, df in [("train", mm_train), ("val", mm_val), ("test", mm_test)]:
        n_null = df["img_image_path"].isna().sum()
        if n_null > 0:
            print(f"Warning: dropping {n_null} null rows from {label}")
    mm_train = mm_train[mm_train["img_image_path"].notna()].reset_index(drop=True)
    mm_val   = mm_val[mm_val["img_image_path"].notna()].reset_index(drop=True)
    mm_test  = mm_test[mm_test["img_image_path"].notna()].reset_index(drop=True)

    mm_full = pd.concat([mm_train, mm_val, mm_test], ignore_index=True)

    save_split(mm_train, "mm_train_mixed_all_group_test.csv")
    save_split(mm_val,   "mm_val_mixed_all_group_test.csv")
    save_split(mm_test,  "mm_test_mixed_all_group_test.csv")
    save_split(mm_full,  "mm_full_mixed_all_group_test.csv")

    print_info("MM TRAIN MIXED ALL", mm_train)
    print_info("MM VAL MIXED ALL",   mm_val)
    print_info("MM TEST MIXED ALL",  mm_test)

    clean = check_leakage(mm_train, mm_val, mm_test)
    print(f"\nLeakage clean: {clean}")
    print("\nSuccess")
