from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "PS_20174392719_1491204439457_log.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_paysim_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def inspect_data(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("DATASET SHAPE")
    print("=" * 60)
    print(df.shape)

    print("\n" + "=" * 60)
    print("COLUMNS")
    print("=" * 60)
    print(df.columns.tolist())

    print("\n" + "=" * 60)
    print("FIRST 5 ROWS")
    print("=" * 60)
    print(df.head())

    print("\n" + "=" * 60)
    print("DATA TYPES")
    print("=" * 60)
    print(df.dtypes)

    print("\n" + "=" * 60)
    print("TARGET DISTRIBUTION: isFraud")
    print("=" * 60)
    print(df["isFraud"].value_counts())

    print("\nPercentages:")
    print(df["isFraud"].value_counts(normalize=True) * 100)

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    print(df.isnull().sum())

    print("\n" + "=" * 60)
    print("DUPLICATE ROWS")
    print("=" * 60)
    print(df.duplicated().sum())

    print("\n" + "=" * 60)
    print("TRANSACTION TYPES")
    print("=" * 60)
    print(df["type"].value_counts())


def clean_and_prepare(df: pd.DataFrame):
    df = df.copy()

    # Stable ID for later multimodal matching
    df["sample_id"] = range(len(df))

    # Encode transaction type
    le = LabelEncoder()
    df["type"] = le.fit_transform(df["type"])

    type_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    return df, type_mapping


def create_train_val_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
):
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["isFraud"]
    )

    relative_val_size = val_size / (1 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=train_val_df["isFraud"]
    )

    return train_df, val_df, test_df


def print_split_info(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("SPLIT SHAPES")
    print("=" * 60)
    print("Train:", train_df.shape)
    print("Val:  ", val_df.shape)
    print("Test: ", test_df.shape)

    print("\n" + "=" * 60)
    print("TARGET DISTRIBUTION IN SPLITS")
    print("=" * 60)

    for name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        print(f"\n{name}:")
        print(split_df["isFraud"].value_counts())
        print(split_df["isFraud"].value_counts(normalize=True) * 100)


def save_full_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved full cleaned dataframe to: {output_path}")


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "paysim_train.csv", index=False)
    val_df.to_csv(output_dir / "paysim_val.csv", index=False)
    test_df.to_csv(output_dir / "paysim_test.csv", index=False)

    print(f"Saved train/val/test to: {output_dir}")


def save_type_mapping(type_mapping: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(type_mapping, f, indent=4)
    print(f"Saved type mapping to: {output_path}")


if __name__ == "__main__":
    # 1. Load raw data
    df = load_paysim_data(RAW_DATA_PATH)

    # 2. Inspect dataset
    inspect_data(df)

    # 3. Clean and prepare
    df_clean, type_mapping = clean_and_prepare(df)

    print("\n" + "=" * 60)
    print("TYPE ENCODING MAP")
    print("=" * 60)
    print(type_mapping)

    # 4. Create splits
    train_df, val_df, test_df = create_train_val_test_split(df_clean)

    # 5. Print split information
    print_split_info(train_df, val_df, test_df)

    # 6. Save processed data
    save_full_dataframe(df_clean, PROCESSED_DIR / "paysim_clean.csv")
    save_splits(train_df, val_df, test_df, PROCESSED_DIR)
    save_type_mapping(type_mapping, PROCESSED_DIR / "paysim_type_mapping.json")