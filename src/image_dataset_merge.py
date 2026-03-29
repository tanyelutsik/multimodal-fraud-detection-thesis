from pathlib import Path
import pandas as pd

COMMON_COLUMNS = [
    "sample_id",
    "source_dataset",
    "image_path",
    "file_name",
    "file_stem",
    "final_label",
    "is_synthetic",
    "width",
    "height",
    "relative_path",
    "split_source",
    "original_folder",
    "image_class",
    "capture_type",
    "attack_type",
    "source_type",
    "doc_type",
    "annotation_path"
]


def load_metadata(csv_path: str) -> pd.DataFrame:
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


def merge_image_metadata(midv_csv: str, fantasyid_csv: str) -> pd.DataFrame:
    df_midv = load_metadata(midv_csv)
    df_fantasy = load_metadata(fantasyid_csv)

    df_midv = ensure_columns(df_midv, COMMON_COLUMNS)
    df_fantasy = ensure_columns(df_fantasy, COMMON_COLUMNS)

    df_merged = pd.concat([df_midv, df_fantasy], ignore_index=True)
    return df_merged


def save_metadata(df: pd.DataFrame, output_path: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved merged metadata to: {output_path}")