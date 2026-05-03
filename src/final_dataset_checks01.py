from pathlib import Path
import pandas as pd
from pandas.util import hash_pandas_object


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# This assumes your files are directly inside data/processed/
DATASETS = {
    "PaySim": {
        "files": {
            "train": "paysim_train.csv",
            "val": "paysim_val.csv",
            "test": "paysim_test.csv",
        },
        "label_candidates": ["isFraud", "label", "target"],
        # put your real ID column first if you have one
        "key_candidates": ["sample_id", "row_id", "tab_id", "transaction_id"],
        "extra_count_cols": ["isFraud"],
    },
    "Image": {
        "files": {
            "train": "image_train.csv",
            "val": "image_val.csv",
            "test": "image_test.csv",
        },
        "label_candidates": ["final_label", "image_class", "original_label"],
        "key_candidates": ["image_path", "img_image_path"],
        "extra_count_cols": ["final_label", "image_class", "original_label", "source_dataset"],
    },
    "Multimodal": {
        "files": {
            "train": "mm_train.csv",
            "val": "mm_val.csv",
            "test": "mm_test.csv",
        },
        "label_candidates": ["final_label", "label"],
        # for split overlap, image path is often the most useful key here
        "key_candidates": ["img_image_path", "image_path", "mm_id"],
        "extra_count_cols": ["final_label", "img_source_dataset", "source_dataset"],
    },
}


# -----------------------------
# Helpers
# -----------------------------
def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def build_key_series(
    df: pd.DataFrame,
    key_candidates: list[str],
    exclude_cols: list[str] | None = None
) -> tuple[pd.Series, str]:
    """
    Returns:
        key_series: a series used for duplicate/overlap checks
        key_name: name of the key used
    """
    key_col = first_existing_column(df, key_candidates)
    if key_col is not None:
        return df[key_col].astype(str), key_col

    # fallback: hash the whole row except excluded columns
    exclude_cols = exclude_cols or []
    cols = [c for c in df.columns if c not in exclude_cols]
    row_hash = hash_pandas_object(
        df[cols].fillna("<NA>"),
        index=False
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

    print(f"\nPercentages for '{col}':")
    percentages = (df[col].value_counts(dropna=False, normalize=True) * 100).round(2)
    print(percentages.head(top_n))


def inspect_split(
    dataset_name: str,
    split_name: str,
    df: pd.DataFrame,
    label_candidates: list[str],
    key_candidates: list[str],
    extra_count_cols: list[str]
) -> dict:
    print_header(f"{dataset_name} | {split_name}")

    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Label column
    label_col = first_existing_column(df, label_candidates)
    if label_col:
        print_value_counts(df, label_col)
    else:
        print("\nNo label column found from:", label_candidates)

    # Extra count columns
    for col in extra_count_cols:
        if col in df.columns and col != label_col:
            print_value_counts(df, col)

    # Missing values
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print("\nMissing values:")
    if len(missing) == 0:
        print("No missing values.")
    else:
        print(missing.head(20))

    # Duplicate check
    key_series, key_name = build_key_series(
        df,
        key_candidates=key_candidates,
        exclude_cols=["split"]
    )
    duplicate_count = int(key_series.duplicated().sum())
    unique_count = int(key_series.nunique())

    print(f"\nDuplicate check using key: {key_name}")
    print(f"Unique keys: {unique_count}")
    print(f"Duplicate rows by key: {duplicate_count}")

    return {
        "df": df,
        "label_col": label_col,
        "key_name": key_name,
        "keys": set(key_series.astype(str)),
    }


def check_overlap(dataset_name: str, split_results: dict) -> None:
    print_header(f"{dataset_name} | Split overlap")

    train_keys = split_results["train"]["keys"]
    val_keys = split_results["val"]["keys"]
    test_keys = split_results["test"]["keys"]

    train_val = train_keys & val_keys
    train_test = train_keys & test_keys
    val_test = val_keys & test_keys

    print(f"Train ∩ Val overlap: {len(train_val)}")
    print(f"Train ∩ Test overlap: {len(train_test)}")
    print(f"Val ∩ Test overlap: {len(val_test)}")

    if len(train_val) > 0:
        print("Example Train ∩ Val keys:", list(train_val)[:5])
    if len(train_test) > 0:
        print("Example Train ∩ Test keys:", list(train_test)[:5])
    if len(val_test) > 0:
        print("Example Val ∩ Test keys:", list(val_test)[:5])


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def main() -> None:
    all_results = {}

    for dataset_name, cfg in DATASETS.items():
        split_results = {}

        for split_name, filename in cfg["files"].items():
            path = PROCESSED_DIR / filename
            df = load_csv(path)

            split_results[split_name] = inspect_split(
                dataset_name=dataset_name,
                split_name=split_name,
                df=df,
                label_candidates=cfg["label_candidates"],
                key_candidates=cfg["key_candidates"],
                extra_count_cols=cfg["extra_count_cols"],
            )

        check_overlap(dataset_name, split_results)
        all_results[dataset_name] = split_results

    print_header("DONE")
    print("All checks finished.")


if __name__ == "__main__":
    main()