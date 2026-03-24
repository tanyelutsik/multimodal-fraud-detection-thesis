from pathlib import Path
import json
import pandas as pd


def collect_image_files(midv_root: str) -> list:
    root = Path(midv_root)
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(root.rglob(ext))
    return image_files


def collect_annotation_files(midv_root: str) -> list:
    root = Path(midv_root)
    return list(root.rglob("*.json"))


def load_json(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
    

def create_metadata_from_matching_stems(midv_root: str) -> pd.DataFrame:
    image_files = collect_image_files(midv_root)
    ann_files = collect_annotation_files(midv_root)

    image_map = {Path(p).stem: str(p) for p in image_files}
    ann_map = {Path(p).stem: str(p) for p in ann_files}

    common_ids = sorted(set(image_map.keys()) & set(ann_map.keys()))

    rows = []
    for file_id in common_ids:
        rows.append({
            "file_id": file_id,
            "image_path": image_map[file_id],
            "annotation_path": ann_map[file_id],
            "image_label": 0
        })

    return pd.DataFrame(rows)


def save_metadata(df: pd.DataFrame, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)