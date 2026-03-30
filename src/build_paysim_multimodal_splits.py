from pathlib import Path
import pandas as pd


def build_matching_subset(paysim_df: pd.DataFrame, n_nonfraud: int, n_fraud: int, random_state: int = 42) -> pd.DataFrame:
    nonfraud_df = paysim_df[paysim_df["isFraud"] == 0].copy()
    fraud_df = paysim_df[paysim_df["isFraud"] == 1].copy()

    if len(nonfraud_df) < n_nonfraud:
        raise ValueError(f"Not enough non-fraud rows: need {n_nonfraud}, found {len(nonfraud_df)}")
    if len(fraud_df) < n_fraud:
        raise ValueError(f"Not enough fraud rows: need {n_fraud}, found {len(fraud_df)}")

    nonfraud_sample = nonfraud_df.sample(n=n_nonfraud, random_state=random_state)
    fraud_sample = fraud_df.sample(n=n_fraud, random_state=random_state)

    merged = pd.concat([nonfraud_sample, fraud_sample], ignore_index=True)
    merged = merged.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return merged


def main():
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"

    paysim_train = pd.read_csv(processed_dir / "paysim_train.csv")
    paysim_val = pd.read_csv(processed_dir / "paysim_val.csv")
    paysim_test = pd.read_csv(processed_dir / "paysim_test.csv")

    image_train = pd.read_csv(processed_dir / "image_train.csv")
    image_val = pd.read_csv(processed_dir / "image_val.csv")
    image_test = pd.read_csv(processed_dir / "image_test.csv")

    train_counts = image_train["image_class"].value_counts()
    val_counts = image_val["image_class"].value_counts()
    test_counts = image_test["image_class"].value_counts()

    mm_train = build_matching_subset(
        paysim_train,
        n_nonfraud=int(train_counts.get("bona_fide", 0)),
        n_fraud=int(train_counts.get("forged", 0)),
        random_state=42
    )

    mm_val = build_matching_subset(
        paysim_val,
        n_nonfraud=int(val_counts.get("bona_fide", 0)),
        n_fraud=int(val_counts.get("forged", 0)),
        random_state=42
    )

    mm_test = build_matching_subset(
        paysim_test,
        n_nonfraud=int(test_counts.get("bona_fide", 0)),
        n_fraud=int(test_counts.get("forged", 0)),
        random_state=42
    )

    mm_train.to_csv(processed_dir / "paysim_mm_train.csv", index=False)
    mm_val.to_csv(processed_dir / "paysim_mm_val.csv", index=False)
    mm_test.to_csv(processed_dir / "paysim_mm_test.csv", index=False)

    print("MM TRAIN:", mm_train.shape)
    print(mm_train["isFraud"].value_counts())

    print("\nMM VAL:", mm_val.shape)
    print(mm_val["isFraud"].value_counts())

    print("\nMM TEST:", mm_test.shape)
    print(mm_test["isFraud"].value_counts())


if __name__ == "__main__":
    main()