from pathlib import Path
import pandas as pd

COMMON_COLUMNS = [
    "image_path",
    "source_dataset",
    "image_class",
    "group_key",
    "doc_type",
    "source_type",
    "file_name",
    "split_source",
    "original_label",
]


def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")
    return pd.read_csv(csv_path)


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df[columns]


def merge_image_metadata(midv_csv: str | Path,
                         fantasyid_csv: str | Path,
                         fmidv_csv: str | Path) -> pd.DataFrame:
    df_midv = load_metadata(midv_csv)
    df_fantasy = load_metadata(fantasyid_csv)
    df_fmidv = load_metadata(fmidv_csv)

    df_midv = ensure_columns(df_midv, COMMON_COLUMNS)
    df_fantasy = ensure_columns(df_fantasy, COMMON_COLUMNS)
    df_fmidv = ensure_columns(df_fmidv, COMMON_COLUMNS)

    df_merged = pd.concat([df_midv, df_fantasy, df_fmidv], ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset=["image_path"]).reset_index(drop=True)

    return df_merged


def save_metadata(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved merged metadata to: {output_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"

    midv_csv = processed_dir / "midv_image_manifest.csv"
    fantasyid_csv = processed_dir / "fantasyid_image_manifest.csv"
    fmidv_csv = processed_dir / "fmidv_image_manifest.csv"
    output_csv = processed_dir / "image_dataset_final.csv"

    df_merged = merge_image_metadata(midv_csv, fantasyid_csv, fmidv_csv)

    print("Final shape:", df_merged.shape)
    print("\nClass counts:")
    print(df_merged["image_class"].value_counts(dropna=False))

    print("\nSource x class:")
    print(pd.crosstab(df_merged["source_dataset"], df_merged["image_class"]))

    print("\nUnique groups by source:")
    print(df_merged.groupby("source_dataset")["group_key"].nunique())

    print("\nDuplicate image paths:")
    print(df_merged["image_path"].duplicated().sum())

    save_metadata(df_merged, output_csv)