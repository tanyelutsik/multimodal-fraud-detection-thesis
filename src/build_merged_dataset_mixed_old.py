from pathlib import Path
import pandas as pd
import numpy as np

"""
build_merged_dataset_mixed.py
------------------------------
Creates the MIXED multimodal dataset where fraud can come from either
the transaction side, the image side, or both.

Why this is stronger than the simple pairing
---------------------------------------------
Simple pairing (build_merged_dataset_group_test.py):
    tab_fraud=1  + img_forged=1  → label 1
    tab_fraud=0  + img_genuine=0 → label 0

The weakness: the image alone almost solves the task. A model that
ignores tabular features and just classifies the image will perform
near-perfectly. True multimodal fusion is not needed.

Mixed pairing (this script):
    tab_fraud=1  + img_forged=1  → label 1  (both suspicious)
    tab_fraud=1  + img_genuine=0 → label 1  (transaction suspicious only)
    tab_fraud=0  + img_forged=1  → label 1  (image suspicious only)
    tab_fraud=0  + img_genuine=0 → label 0  (fully legitimate only)

Final label = 1 if EITHER tab_isFraud=1 OR img_final_label=1
Final label = 0 ONLY if tab_isFraud=0 AND img_final_label=0

The model now genuinely needs both modalities to perform well:
- Image alone misses transaction-only fraud (1,0) pairs
- Tabular alone misses image-only fraud (0,1) pairs

Pairing strategy
----------------
Within each split:
    Combination (0,0): pair genuine-tab rows with genuine-img rows
    Combination (1,1): pair fraud-tab rows with forged-img rows
    Combination (1,0): pair fraud-tab rows with genuine-img rows
    Combination (0,1): pair genuine-tab rows with forged-img rows

To keep the final dataset balanced (50% label 0, 50% label 1) and each
fraud combination equally represented:
    n_neg = available (0,0) pairs
    n_pos = n_neg total, split equally across (1,1), (1,0), (0,1)

Outputs
-------
    mm_train_mixed_group_test.csv
    mm_val_mixed_group_test.csv
    mm_test_mixed_group_test.csv
    mm_full_mixed_group_test.csv
"""

print("build_merged_dataset_mixed.py started")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RANDOM_STATE = 42

LABEL_MAP = {
    "bona_fide": 0,
    "genuine":   0,
    "normal":    0,
    "clean":     0,
    "fake":      1,
    "fraud":     1,
    "forged":    1,
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
        df = pd.read_csv(PROCESSED_DIR / fname)
        df["split"]      = split_name
        df["tabular_id"] = df["sample_id"]
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df["final_label"] = df["isFraud"].astype(int)
    print(f"Loaded tabular splits: {df.shape}")
    print("Tabular label distribution:")
    print(df["final_label"].value_counts())
    return df


def load_image_splits() -> pd.DataFrame:
    dfs = []
    for split_name, fname in [
        ("train", "image_train_group_test.csv"),
        ("val",   "image_val_group_test.csv"),
        ("test",  "image_test_group_test.csv"),
    ]:
        df = pd.read_csv(PROCESSED_DIR / fname)
        df["split"] = split_name
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df["image_class"] = df["image_class"].astype(str).str.strip().str.lower()
    df["final_label"] = df["image_class"].map(LABEL_MAP)

    if df["final_label"].isna().any():
        unknown = sorted(df.loc[df["final_label"].isna(), "image_class"].unique().tolist())
        raise ValueError(f"Unknown image labels: {unknown}")

    print(f"Loaded image splits: {df.shape}")
    print("Image label distribution:")
    print(df["final_label"].value_counts())
    return df


# ---------------------------------------------------------------------------
# Mixed pairing
# ---------------------------------------------------------------------------

def pair_mixed_split(
    tab_df:     pd.DataFrame,
    img_df:     pd.DataFrame,
    split_name: str,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Creates mixed pairs with OR-logic labelling.

    Combination codes:
        (0,0) → final_label=0  fully legitimate
        (1,1) → final_label=1  both suspicious
        (1,0) → final_label=1  transaction fraud only
        (0,1) → final_label=1  image fraud only
    """
    rng = np.random.RandomState(random_state)

    # Separate by label
    tab_genuine = tab_df[tab_df["final_label"] == 0].sample(frac=1, random_state=random_state).reset_index(drop=True)
    tab_fraud   = tab_df[tab_df["final_label"] == 1].sample(frac=1, random_state=random_state).reset_index(drop=True)
    img_genuine = img_df[img_df["final_label"] == 0].sample(frac=1, random_state=random_state).reset_index(drop=True)
    img_forged  = img_df[img_df["final_label"] == 1].sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"\n  {split_name} pool sizes:")
    print(f"    tab genuine={len(tab_genuine)}, tab fraud={len(tab_fraud)}")
    print(f"    img genuine={len(img_genuine)}, img forged={len(img_forged)}")

    # How many (0,0) pairs can we make?
    n_neg_max = min(len(tab_genuine), len(img_genuine))

    # Divide positive budget equally across 3 fraud combinations
    # Each combination limited by its own pool sizes AND remaining pool
    # after tab0_img0 takes its share
    budget     = n_neg_max // 3

    n_11 = min(len(tab_fraud),   len(img_forged),                         budget)
    # tab1_img0 uses img_genuine AFTER n_neg rows are used for tab0_img0
    # remaining genuine = len(img_genuine) - n_neg_max
    remaining_genuine = max(0, len(img_genuine) - n_neg_max)
    n_10 = min(len(tab_fraud),   remaining_genuine,                        budget)
    # tab0_img1 uses img_forged AFTER n_11 rows are used for tab1_img1
    remaining_forged  = max(0, len(img_forged) - n_11)
    n_01 = min(len(tab_genuine), remaining_forged,                         budget)

    n_pos = n_11 + n_10 + n_01

    # Rebalance: use n_pos negatives so dataset is balanced
    n_neg = n_pos

    print(f"  Pairing plan: n_neg={n_neg} | n_11={n_11} n_10={n_10} n_01={n_01}")

    parts = []

    def make_pairs(tab_part, img_part, n, combo_label, combo_code):
        # Safety: never request more rows than available
        n = min(n, len(tab_part), len(img_part))
        if n == 0:
            return pd.DataFrame()
        tab_s = tab_part.iloc[:n].reset_index(drop=True).add_prefix("tab_")
        img_s = img_part.iloc[:n].reset_index(drop=True).add_prefix("img_")
        assert len(tab_s) == len(img_s), f"Size mismatch: tab={len(tab_s)} img={len(img_s)}"
        merged = pd.concat([tab_s, img_s], axis=1)
        merged["final_label"]      = combo_label
        merged["tab_fraud_label"]  = int(combo_code[0])
        merged["img_fraud_label"]  = int(combo_code[1])
        merged["combo_type"]       = f"tab{combo_code[0]}_img{combo_code[1]}"
        merged["split"]            = split_name
        merged["mm_id"] = [
            f"mm_{split_name}_{combo_code}_{i:06d}" for i in range(len(merged))
        ]
        return merged

    # (0,0) — fully legitimate
    parts.append(make_pairs(tab_genuine,           img_genuine, n_neg, 0, "00"))
    # (1,1) — both fraud
    parts.append(make_pairs(tab_fraud,             img_forged,  n_11,  1, "11"))
    # (1,0) — transaction fraud only
    parts.append(make_pairs(
        tab_fraud,
        img_genuine.iloc[n_neg:],                              # avoid reuse
        n_10, 1, "10"
    ))
    # (0,1) — image fraud only
    parts.append(make_pairs(
        tab_genuine.iloc[n_neg:],                              # avoid reuse
        img_forged.iloc[n_11:],                                # avoid reuse
        n_01, 1, "01"
    ))

    result = pd.concat(parts, ignore_index=True)
    result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)

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
        print("EMPTY")
        return

    print("\nFinal label distribution:")
    print(df["final_label"].value_counts())
    print((df["final_label"].value_counts(normalize=True) * 100).round(2))

    if "combo_type" in df.columns:
        print("\nCombo type breakdown:")
        print(df["combo_type"].value_counts())
        print("\nCombo type x final_label:")
        print(pd.crosstab(df["combo_type"], df["final_label"]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Main block running")

    tabular_df = load_tabular_splits()
    image_df   = load_image_splits()

    mm_train = pair_mixed_split(
        tabular_df[tabular_df["split"] == "train"],
        image_df[image_df["split"]   == "train"],
        "train",
    )
    mm_val = pair_mixed_split(
        tabular_df[tabular_df["split"] == "val"],
        image_df[image_df["split"]   == "val"],
        "val",
    )
    mm_test = pair_mixed_split(
        tabular_df[tabular_df["split"] == "test"],
        image_df[image_df["split"]   == "test"],
        "test",
    )

    mm_full = pd.concat([mm_train, mm_val, mm_test], ignore_index=True)

    # ── Cross-split deduplication ──────────────────────────────────────────
    # Remove any img_image_path that appears in more than one split.
    # This can happen at pool boundaries when genuine/forged image pools
    # are small in val/test splits.
    train_imgs = set(mm_train["img_image_path"])
    val_imgs   = set(mm_val["img_image_path"])
    test_imgs  = set(mm_test["img_image_path"])

    leaked = (train_imgs & val_imgs) | (train_imgs & test_imgs) | (val_imgs & test_imgs)

    if leaked:
        print(f"\n⚠️  Found {len(leaked)} img_image_path(s) across splits — removing from val/test.")
        print(f"  Leaked paths: {list(leaked)[:3]}")
        mm_val  = mm_val[~mm_val["img_image_path"].isin(leaked)].reset_index(drop=True)
        mm_test = mm_test[~mm_test["img_image_path"].isin(leaked)].reset_index(drop=True)
        mm_full = pd.concat([mm_train, mm_val, mm_test], ignore_index=True)
        print(f"  After fix — val: {len(mm_val)}, test: {len(mm_test)}")
    else:
        print("\n✅  No cross-split image leakage detected.")

    save_split(mm_train, "mm_train_mixed_group_test.csv")
    save_split(mm_val,   "mm_val_mixed_group_test.csv")
    save_split(mm_test,  "mm_test_mixed_group_test.csv")
    save_split(mm_full,  "mm_full_mixed_group_test.csv")

    print_info("MM TRAIN MIXED", mm_train)
    print_info("MM VAL MIXED",   mm_val)
    print_info("MM TEST MIXED",  mm_test)
    print_info("MM FULL MIXED",  mm_full)

    print("\nSuccess")