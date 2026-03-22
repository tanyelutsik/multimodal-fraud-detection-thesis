import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import pandas as pd


def load_paysim_data(path: str) -> pd.DataFrame:
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




def clean_and_select_features(df: pd.DataFrame):
    df = df.copy()

    # create stable linking ID for later multimodal merge
    df["sample_id"] = range(len(df))

    # encode categorical column
    le = LabelEncoder()
    df["type"] = le.fit_transform(df["type"])

    # columns to exclude from model features
    # keep sample_id separately for future linking, not for training
    columns_to_drop_from_X = ["isFraud", "nameOrig", "nameDest", "sample_id"]

    X = df.drop(columns=columns_to_drop_from_X)
    y = df["isFraud"]

    # keep this full version for later multimodal linkage if needed
    df_link = df.copy()

    return X, y, df_link


def create_train_val_test_split(X, y, test_size=0.2, val_size=0.1, random_state=42):
    # first split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # second split: train vs val
    # val_size here is relative to the original full dataset
    relative_val_size = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def print_split_info(X_train, X_val, X_test, y_train, y_val, y_test):
    print("\n" + "=" * 60)
    print("SPLIT SHAPES")
    print("=" * 60)
    print("X_train:", X_train.shape)
    print("X_val:  ", X_val.shape)
    print("X_test: ", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_val:  ", y_val.shape)
    print("y_test: ", y_test.shape)

    print("\n" + "=" * 60)
    print("TARGET DISTRIBUTION IN SPLITS")
    print("=" * 60)

    print("\nTrain:")
    print(y_train.value_counts())
    print(y_train.value_counts(normalize=True) * 100)

    print("\nValidation:")
    print(y_val.value_counts())
    print(y_val.value_counts(normalize=True) * 100)

    print("\nTest:")
    print(y_test.value_counts())
    print(y_test.value_counts(normalize=True) * 100)


if __name__ == "__main__":
    path = r"C:\Users\Admin\Desktop\thesis\multimodal-fraud-detection-thesis\data\PS_20174392719_1491204439457_log.csv"

    # 1. Load data
    df = load_paysim_data(path)

    # 2. Inspect data
    inspect_data(df)

    # 3. Clean and select features
    X, y, df_link = clean_and_select_features(df)

    print("\n" + "=" * 60)
    print("SELECTED FEATURES FOR MODEL")
    print("=" * 60)
    print(X.columns.tolist())

    # 4. Create train / validation / test split
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(X, y)

    # 5. Print split info
    print_split_info(X_train, X_val, X_test, y_train, y_val, y_test)

    print("Missing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nTransaction types:")
print(df["type"].value_counts())