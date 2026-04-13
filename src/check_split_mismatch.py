from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def inspect_file(filename: str):
    df = pd.read_csv(PROCESSED_DIR / filename)

    print(f"\n=== {filename} ===")
    print("Shape:", df.shape)

    if "source_dataset" in df.columns:
        print("\nBy source_dataset:")
        print(df["source_dataset"].value_counts(dropna=False))

    if "split_source" in df.columns:
        print("\nBy split_source:")
        print(df["split_source"].value_counts(dropna=False))

    if "source_dataset" in df.columns and "split_source" in df.columns:
        print("\nFantasyID rows only:")
        fantasy = df[df["source_dataset"] == "fantasyid"].copy()
        print("Fantasy shape:", fantasy.shape)
        print(fantasy["split_source"].value_counts(dropna=False))

        bad_train = fantasy[fantasy["split_source"] == "train"]
        bad_val = fantasy[fantasy["split_source"] == "val"]
        bad_test = fantasy[fantasy["split_source"] == "test"]

        print("\nFantasy rows with split_source=train:", len(bad_train))
        print("Fantasy rows with split_source=val:", len(bad_val))
        print("Fantasy rows with split_source=test:", len(bad_test))

        if len(bad_train) > 0:
            print("\nExamples with split_source=train:")
            print(
                bad_train[["image_path", "source_dataset", "image_class", "split", "split_source"]]
                .head(10)
                .to_string(index=False)
            )

inspect_file("image_train.csv")
inspect_file("image_val.csv")
inspect_file("image_test.csv")