from pathlib import Path
import pandas as pd
from pandas.util import hash_pandas_object


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Updated to check _group_test.csv versions (fixed group-based splits)
DATASETS = {
    "PaySim": {
        "files": {
            "train": "paysim_train.csv",
            "val":   "paysim_val.csv",
            "test":  "paysim_test.csv",
        },
        "label_candidates": ["isFraud", "label", "target"],
        "key_candidates":   ["sample_id", "row_id", "tab_id", "transaction_id"],
        "extra_count_cols": ["isFraud"],
    },
    "Image (group_test)": {
        "files": {
            "train": "image_train_group_test.csv",
            "val":   "image_val_group_test.csv",
            "test":  "image_test_group_test.csv",
        },
        "label_candidates": ["final_label", "image_class", "original_label"],
        "key_candidates":   ["image_path", "img_image_path"],
        "extra_count_cols": ["final_label", "image_class", "original_label", "source_dataset"],
    },
    "Multimodal simple (group_test)": {
        "files": {
            "train": "mm_train_group_test.csv",
            "val":   "mm_val_group_test.csv",
            "test":  "mm_test_group_test.csv",
        },
        "label_candidates": ["final_label", "label"],
        "key_candidates":   ["img_image_path", "image_path", "mm_id"],
        "extra_count_cols": ["final_label", "img_source_dataset"],
    },
    "Multimodal mixed (group_test)": {
        "files": {
            "train": "mm_train_mixed_group_test.csv",
            "val":   "mm_val_mixed_group_test.csv",
            "test":  "mm_test_mixed_group_test.csv",
        },
        "label_candidates": ["final_label", "label"],
        "key_candidates":   ["mm_id", "img_image_path"],
        "extra_count_cols": ["final_label", "combo_type", "img_source_dataset"],
    },
    "Multimodal mixed balanced (group_test)": {
        "files": {
            "train": "mm_train_mixed_balanced_group_test.csv",
            "val":   "mm_val_mixed_balanced_group_test.csv",
            "test":  "mm_test_mixed_balanced_group_test.csv",
        },
        "label_candidates": ["final_label", "label"],
        "key_candidates":   ["mm_id", "img_image_path"],
        "extra_count_cols": ["final_label", "combo_type", "img_source_dataset"],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def build_key_series(
    df:             pd.DataFrame,
    key_candidates: list[str],
    exclude_cols:   list[str] | None = None,
) -> tuple[pd.Series, str]:
    key_col = first_existing_column(df, key_candidates)
    if key_col is not None:
        return df[key_col].astype(str), key_col

    exclude_cols = exclude_cols or []
    cols     = [c for c in df.columns if c not in exclude_cols]
    row_hash = hash_pandas_object(
        df[cols].fillna("<NA>"), index=False
    ).astype(str)
    return row_hash, "ROW_HASH_FALLBACK"


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_value_counts(df: pd.DataFrame, col: str, top_n: int = 20) -> None:
    print(f"\nValue counts for '{col}':")
    counts = df[col].value_counts(dropna=False)
    print(counts.head(top_n))
    pct = (df[col].value_counts(dropna=False, normalize=True) * 100).round(2)
    print(pct.head(top_n))


def inspect_split(
    dataset_name:   str,
    split_name:     str,
    df:             pd.DataFrame,
    label_candidates: list[str],
    key_candidates:   list[str],
    extra_count_cols: list[str],
) -> dict:
    print_header(f"{dataset_name} | {split_name}")
    print(f"Shape: {df.shape}")

    label_col = first_existing_column(df, label_candidates)
    if label_col:
        print_value_counts(df, label_col)
    else:
        print("\nNo label column found from:", label_candidates)

    for col in extra_count_cols:
        if col in df.columns and col != label_col:
            print_value_counts(df, col)

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print("\nMissing values:")
    if len(missing) == 0:
        print("  None")
    else:
        print(missing.head(20))

    key_series, key_name = build_key_series(
        df, key_candidates=key_candidates, exclude_cols=["split"]
    )
    dup_count    = int(key_series.duplicated().sum())
    unique_count = int(key_series.nunique())

    print(f"\nDuplicate check using key: {key_name}")
    print(f"  Unique keys:    {unique_count}")
    print(f"  Duplicate rows: {dup_count}")

    return {
        "df":        df,
        "label_col": label_col,
        "key_name":  key_name,
        "keys":      set(key_series.astype(str)),
    }


def check_overlap(dataset_name: str, split_results: dict) -> None:
    print_header(f"{dataset_name} | Split overlap check")

    train_keys = split_results["train"]["keys"]
    val_keys   = split_results["val"]["keys"]
    test_keys  = split_results["test"]["keys"]

    train_val  = train_keys & val_keys
    train_test = train_keys & test_keys
    val_test   = val_keys   & test_keys

    flag = lambda n: "  ⚠️  LEAKAGE" if n > 0 else "  ✅  clean"

    print(f"Train ∩ Val  overlap: {len(train_val)}{flag(len(train_val))}")
    print(f"Train ∩ Test overlap: {len(train_test)}{flag(len(train_test))}")
    print(f"Val   ∩ Test overlap: {len(val_test)}{flag(len(val_test))}")

    if len(train_val) > 0:
        print("  Example keys:", list(train_val)[:3])
    if len(train_test) > 0:
        print("  Example keys:", list(train_test)[:3])
    if len(val_test) > 0:
        print("  Example keys:", list(val_test)[:3])


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  ⚠️  File not found — skipping: {path.name}")
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    for dataset_name, cfg in DATASETS.items():
        split_results = {}
        all_loaded    = True

        for split_name, filename in cfg["files"].items():
            path = PROCESSED_DIR / filename
            df   = load_csv(path)
            if df is None:
                all_loaded = False
                break

            split_results[split_name] = inspect_split(
                dataset_name     = dataset_name,
                split_name       = split_name,
                df               = df,
                label_candidates = cfg["label_candidates"],
                key_candidates   = cfg["key_candidates"],
                extra_count_cols = cfg["extra_count_cols"],
            )

        if all_loaded:
            check_overlap(dataset_name, split_results)
        else:
            print(f"\n  Skipping overlap check for '{dataset_name}' — not all files found.")

    print_header("DONE — all checks finished")


if __name__ == "__main__":
    main()
