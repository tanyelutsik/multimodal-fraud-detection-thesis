from pathlib import Path
import pandas as pd

print("build_merged_dataset.py started")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RANDOM_STATE = 42

LABEL_MAP = {
    "bona_fide": 0,
    "genuine": 0,
    "normal": 0,
    "clean": 0,
    "fake": 1,
    "fraud": 1,
    "forged": 1
}


def load_tabular_splits():
    train_path = PROCESSED_DIR / "paysim_train.csv"
    val_path = PROCESSED_DIR / "paysim_val.csv"
    test_path = PROCESSED_DIR / "paysim_test.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    train_df["tabular_id"] = train_df["sample_id"]
    val_df["tabular_id"] = val_df["sample_id"]
    test_df["tabular_id"] = test_df["sample_id"]

    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print("Loaded tabular splits:", df.shape)
    return df


def map_tabular_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["final_label"] = df["isFraud"].astype(int)
    return df


def load_image_splits():
    train_path = PROCESSED_DIR / "image_train.csv"
    val_path = PROCESSED_DIR / "image_val.csv"
    test_path = PROCESSED_DIR / "image_test.csv"

    print("Looking for:")
    print(train_path)
    print(val_path)
    print(test_path)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    if "split" not in train_df.columns:
        train_df["split"] = "train"
    if "split" not in val_df.columns:
        val_df["split"] = "val"
    if "split" not in test_df.columns:
        test_df["split"] = "test"

    df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    print("Loaded image splits:", df.shape)
    print(df.head())
    return df


def map_image_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["image_class"] = df["image_class"].astype(str).str.strip().str.lower()
    df["final_label"] = df["image_class"].map(LABEL_MAP)

    if df["final_label"].isna().any():
        unknown = sorted(df.loc[df["final_label"].isna(), "image_class"].unique().tolist())
        raise ValueError(f"Unknown image labels found: {unknown}")

    print("Mapped labels successfully")
    print(df[["image_class", "final_label"]].head())
    return df


def pair_one_split(tab_df: pd.DataFrame, img_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    merged_parts = []

    common_labels = sorted(set(tab_df["final_label"]).intersection(set(img_df["final_label"])))

    for label in common_labels:
        tab_part = (
            tab_df[tab_df["final_label"] == label]
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )

        img_part = (
            img_df[img_df["final_label"] == label]
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )

        n = min(len(tab_part), len(img_part))

        if n == 0:
            continue

        tab_part = tab_part.iloc[:n].reset_index(drop=True).add_prefix("tab_")
        img_part = img_part.iloc[:n].reset_index(drop=True).add_prefix("img_")

        merged = pd.concat([tab_part, img_part], axis=1)
        merged["final_label"] = label
        merged["split"] = split_name
        merged["mm_id"] = [f"mm_{split_name}_{label}_{i:06d}" for i in range(len(merged))]
        merged_parts.append(merged)

    if not merged_parts:
        return pd.DataFrame()

    merged_df = pd.concat(merged_parts, ignore_index=True)
    merged_df = merged_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return merged_df


def save_split(df: pd.DataFrame, filename: str):
    out_path = PROCESSED_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def print_info(name: str, df: pd.DataFrame):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print("Shape:", df.shape)

    if "final_label" in df.columns and not df.empty:
        print("\nLabel distribution:")
        print(df["final_label"].value_counts())
        print(df["final_label"].value_counts(normalize=True) * 100)


if __name__ == "__main__":
    print("Main block is running")

    tabular_df = load_tabular_splits()
    tabular_df = map_tabular_labels(tabular_df)

    image_df = load_image_splits()
    image_df = map_image_labels(image_df)

    mm_train = pair_one_split(
        tabular_df[tabular_df["split"] == "train"],
        image_df[image_df["split"] == "train"],
        "train"
    )

    mm_val = pair_one_split(
        tabular_df[tabular_df["split"] == "val"],
        image_df[image_df["split"] == "val"],
        "val"
    )

    mm_test = pair_one_split(
        tabular_df[tabular_df["split"] == "test"],
        image_df[image_df["split"] == "test"],
        "test"
    )

    mm_full = pd.concat([mm_train, mm_val, mm_test], ignore_index=True)

    save_split(mm_train, "mm_train.csv")
    save_split(mm_val, "mm_val.csv")
    save_split(mm_test, "mm_test.csv")
    save_split(mm_full, "mm_full.csv")

    print_info("MM TRAIN", mm_train)
    print_info("MM VAL", mm_val)
    print_info("MM TEST", mm_test)
    print_info("MM FULL", mm_full)

    print("\nSuccess")