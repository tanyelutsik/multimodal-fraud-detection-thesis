from pathlib import Path
import pandas as pd

"""
check_split_mismatch.py
-----------------------
Checks FantasyID split_source mismatches in the group-test image splits.

Updated to use _group_test.csv files instead of the old image_train.csv.
"""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def inspect_file(filename: str) -> None:
    path = PROCESSED_DIR / filename

    if not path.exists():
        print(f"\n⚠️  File not found: {filename} — skipping")
        return

    df = pd.read_csv(path)

    print(f"\n=== {filename} ===")
    print("Shape:", df.shape)

    if "source_dataset" in df.columns:
        print("\nBy source_dataset:")
        print(df["source_dataset"].value_counts(dropna=False))

    if "image_class" in df.columns:
        print("\nBy image_class:")
        print(df["image_class"].value_counts(dropna=False))

    if "split_source" in df.columns:
        print("\nBy split_source:")
        print(df["split_source"].value_counts(dropna=False))

    if "source_dataset" in df.columns and "split_source" in df.columns:
        print("\nFantasyID rows only:")
        fantasy = df[df["source_dataset"] == "fantasyid"].copy()
        print("  Fantasy shape:", fantasy.shape)
        print(fantasy["split_source"].value_counts(dropna=False))

        for split_val in ["train", "val", "test"]:
            rows = fantasy[fantasy["split_source"] == split_val]
            print(f"\n  Fantasy rows with split_source={split_val}: {len(rows)}")
            if len(rows) > 0 and len(rows) <= 10:
                print(
                    rows[["image_path", "source_dataset", "image_class",
                           "split_source"]]
                    .head(5)
                    .to_string(index=False)
                )

    # Group key overlap check
    if "group_key" in df.columns:
        print(f"\n  Unique group_keys: {df['group_key'].nunique()}")
        print(f"  Duplicate group_keys: {df['group_key'].duplicated().sum()}")


# ---------------------------------------------------------------------------
# Run checks on group_test splits  ← updated from old image_train.csv
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Checking _group_test image splits for split_source mismatches")
    print("=" * 60)

    inspect_file("image_train_group_test.csv")
    inspect_file("image_val_group_test.csv")
    inspect_file("image_test_group_test.csv")

    print("\n" + "=" * 60)
    print("Done.")
