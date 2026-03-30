from pathlib import Path
import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def get_all_image_paths(root_path):
    root_path = Path(root_path)
    image_paths = []

    for path in root_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(path)

    return sorted(image_paths)


def infer_source_type(parts):
    parts_lower = [p.lower() for p in parts]

    if "templates" in parts_lower:
        return "templates"
    if "photo" in parts_lower:
        return "photo"
    if "scan_rotated" in parts_lower:
        return "scan_rotated"
    if "scan_upright" in parts_lower:
        return "scan_upright"
    if "scan" in parts_lower:
        return "scan"

    return None


def infer_doc_type(parts):
    valid_doc_types = {
        "alb_id",
        "aze_passport",
        "esp_id",
        "est_id",
        "fin_id",
        "grc_passport",
        "lva_passport",
        "rus_internalpassport",
        "srb_passport",
        "svk_id",
    }

    for part in parts:
        lower = part.lower()
        if lower in valid_doc_types:
            return lower
    return None


def build_midv_genuine_dataframe(root_path):
    root_path = Path(root_path)

    if not root_path.exists():
        raise FileNotFoundError(f"MIDV root folder does not exist: {root_path}")

    valid_source_types = {"photo", "templates", "scan_rotated", "scan_upright"}

    image_paths = get_all_image_paths(root_path)
    rows = []

    for img_path in image_paths:
        rel_path = img_path.relative_to(root_path)
        parts = rel_path.parts

        doc_type = infer_doc_type(parts)
        source_type = infer_source_type(parts)
        file_stem = img_path.stem

        # keep only the 4 clean MIDV image groups
        if doc_type is None or source_type not in valid_source_types:
            continue

        try:
            file_stem_str = str(int(file_stem)).zfill(2)
        except ValueError:
            file_stem_str = file_stem

        group_key = f"{doc_type}__{file_stem_str}"

        rows.append({
            "image_path": str(img_path.resolve()),
            "source_dataset": "midv",
            "image_class": "bona_fide",
            "group_key": group_key,
            "doc_type": doc_type,
            "source_type": source_type,
            "file_name": img_path.name,
            "split_source": None,
            "original_label": 0,
            "relative_path": str(rel_path),
            "file_stem": file_stem,
            "suffix": img_path.suffix.lower(),
        })

    df = pd.DataFrame(rows)
    return df.reset_index(drop=True)


def save_metadata(df: pd.DataFrame, output_path: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved metadata to: {output_path}")