from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def split_groups(group_df, stratify_col=None, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    if abs(train_size + val_size + test_size - 1.0) > 1e-9:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    if stratify_col is not None:
        train_groups, temp_groups = train_test_split(
            group_df,
            test_size=(1 - train_size),
            random_state=random_state,
            stratify=group_df[stratify_col],
        )

        val_groups, test_groups = train_test_split(
            temp_groups,
            test_size=(test_size / (val_size + test_size)),
            random_state=random_state,
            stratify=temp_groups[stratify_col],
        )
    else:
        train_groups, temp_groups = train_test_split(
            group_df,
            test_size=(1 - train_size),
            random_state=random_state,
        )

        val_groups, test_groups = train_test_split(
            temp_groups,
            test_size=(test_size / (val_size + test_size)),
            random_state=random_state,
        )

    return train_groups, val_groups, test_groups


def main():
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"

    input_csv = processed_dir / "image_dataset_final.csv"
    df = pd.read_csv(input_csv)

    # -----------------------------
    # MIDV: grouped split
    # -----------------------------
    midv = df[df["source_dataset"] == "midv"].copy()
    midv_groups = midv[["group_key", "doc_type"]].drop_duplicates().reset_index(drop=True)

    midv_train_g, midv_val_g, midv_test_g = split_groups(
        midv_groups,
        stratify_col="doc_type",
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_state=42,
    )

    midv_train = midv[midv["group_key"].isin(midv_train_g["group_key"])].copy()
    midv_val = midv[midv["group_key"].isin(midv_val_g["group_key"])].copy()
    midv_test = midv[midv["group_key"].isin(midv_test_g["group_key"])].copy()

    midv_train["split"] = "train"
    midv_val["split"] = "val"
    midv_test["split"] = "test"

    # -----------------------------
    # FMIDV: grouped split
    # -----------------------------
    fmidv = df[df["source_dataset"] == "fmidv"].copy()
    fmidv_groups = fmidv[["group_key", "doc_type"]].drop_duplicates().reset_index(drop=True)

    fmidv_train_g, fmidv_val_g, fmidv_test_g = split_groups(
        fmidv_groups,
        stratify_col="doc_type",
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_state=42,
    )

    fmidv_train = fmidv[fmidv["group_key"].isin(fmidv_train_g["group_key"])].copy()
    fmidv_val = fmidv[fmidv["group_key"].isin(fmidv_val_g["group_key"])].copy()
    fmidv_test = fmidv[fmidv["group_key"].isin(fmidv_test_g["group_key"])].copy()

    fmidv_train["split"] = "train"
    fmidv_val["split"] = "val"
    fmidv_test["split"] = "test"

    # -----------------------------
    # FantasyID: keep original test, split original train into train/val
    # -----------------------------
    fantasy = df[df["source_dataset"] == "fantasyid"].copy()

    fantasy["split_source"] = (
        fantasy["split_source"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "valid": "val",
            "validation": "val",
    })
)

    fantasy_train_source = fantasy[fantasy["split_source"] == "train"].copy()
    fantasy_test = fantasy[fantasy["split_source"] == "test"].copy()

    # split by unique group_key only
    fantasy_train_groups = (
        fantasy_train_source[["group_key"]]
        .drop_duplicates()
        .reset_index(drop=True)
)

    fantasy_train_g, fantasy_val_g = train_test_split(
        fantasy_train_groups,
        test_size=0.2,
        random_state=42
)

    fantasy_train = fantasy_train_source[
        fantasy_train_source["group_key"].isin(fantasy_train_g["group_key"])
    ].copy()

    fantasy_val = fantasy_train_source[
        fantasy_train_source["group_key"].isin(fantasy_val_g["group_key"])
    ].copy()

    fantasy_train["split"] = "train"
    fantasy_val["split"] = "val"
    fantasy_test["split"] = "test"

    # safety check
    assert len(set(fantasy_train["group_key"]) & set(fantasy_val["group_key"])) == 0, \
        "FantasyID group leakage between train and val"

    # -----------------------------
    # Merge final splits
    # -----------------------------
    image_train = pd.concat([midv_train, fantasy_train, fmidv_train], ignore_index=True)
    image_val = pd.concat([midv_val, fantasy_val, fmidv_val], ignore_index=True)
    image_test = pd.concat([midv_test, fantasy_test, fmidv_test], ignore_index=True)
    
    assert len(set(image_train["image_path"]) & set(image_val["image_path"])) == 0, \
        "Image train/val overlap detected"
    assert len(set(image_train["image_path"]) & set(image_test["image_path"])) == 0, \
        "Image train/test overlap detected"
    assert len(set(image_val["image_path"]) & set(image_test["image_path"])) == 0, \
        "Image val/test overlap detected"

    # Save
    train_path = processed_dir / "image_train.csv"
    val_path = processed_dir / "image_val.csv"
    test_path = processed_dir / "image_test.csv"

    image_train.to_csv(train_path, index=False)
    image_val.to_csv(val_path, index=False)
    image_test.to_csv(test_path, index=False)

    # Checks
    print("TRAIN:", image_train.shape)
    print(image_train["image_class"].value_counts(dropna=False))
    print(pd.crosstab(image_train["source_dataset"], image_train["image_class"]))

    print("\nVAL:", image_val.shape)
    print(image_val["image_class"].value_counts(dropna=False))
    print(pd.crosstab(image_val["source_dataset"], image_val["image_class"]))

    print("\nTEST:", image_test.shape)
    print(image_test["image_class"].value_counts(dropna=False))
    print(pd.crosstab(image_test["source_dataset"], image_test["image_class"]))

    print("\nSaved:")
    print(train_path)
    print(val_path)
    print(test_path)


if __name__ == "__main__":
    main()