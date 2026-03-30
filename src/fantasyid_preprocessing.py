from pathlib import Path
import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def get_all_image_paths(root_path):
    root_path = Path(root_path)
    image_paths = []

    for path in root_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(path)

    return sorted(image_paths)


def infer_split_from_parts(parts):
    parts_lower = [p.lower() for p in parts]

    if "train" in parts_lower:
        return "train"
    if "test" in parts_lower:
        return "test"
    if "valid" in parts_lower or "val" in parts_lower or "validation" in parts_lower:
        return "val"
    return None


def infer_original_folder(parts):
    parts_lower = [p.lower() for p in parts]

    if "bonafide" in parts_lower:
        return "bonafide"
    if "attack" in parts_lower:
        return "attack"
    return None


def infer_image_class(parts):
    parts_lower = [p.lower() for p in parts]

    if "bonafide" in parts_lower:
        return "bona_fide"
    if "attack" in parts_lower:
        return "forged"
    return None


def infer_capture_type(parts):
    parts_lower = [p.lower() for p in parts]

    # bonafide/<device>/image.jpg
    if "bonafide" in parts_lower:
        idx = parts_lower.index("bonafide")
        if idx + 1 < len(parts):
            return parts[idx + 1]

    # attack/<attack_type>/<device>/image.jpg
    if "attack" in parts_lower:
        idx = parts_lower.index("attack")
        if idx + 2 < len(parts):
            return parts[idx + 2]

    return None


def infer_attack_type(parts):
    parts_lower = [p.lower() for p in parts]

    if "attack" in parts_lower:
        idx = parts_lower.index("attack")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def get_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception:
        return (None, None)


def build_fantasyid_dataframe(root_path):
    root_path = Path(root_path)

    if not root_path.exists():
        raise FileNotFoundError(f"FantasyID root folder does not exist: {root_path}")

    image_paths = get_all_image_paths(root_path)
    rows = []

    for img_path in image_paths:
        rel_path = img_path.relative_to(root_path)
        parts = rel_path.parts

        split_source = infer_split_from_parts(parts)
        original_folder = infer_original_folder(parts)
        image_class = infer_image_class(parts)
        capture_type = infer_capture_type(parts)
        attack_type = infer_attack_type(parts)
        width, height = get_image_size(img_path)

        final_label = image_class

        rows.append({
            # canonical columns for final merge
            "image_path": str(img_path.resolve()),
            "source_dataset": "fantasyid",
            "image_class": image_class,
            "group_key": img_path.stem,
            "doc_type": None,
            "source_type": capture_type,
            "file_name": img_path.name,
            "split_source": split_source,
            "original_label": original_folder,

            # extra metadata
            "relative_path": str(rel_path),
            "file_stem": img_path.stem,
            "suffix": img_path.suffix.lower(),
            "capture_type": capture_type,
            "attack_type": attack_type,
            "width": width,
            "height": height,
            "is_synthetic": False,
            "final_label": final_label,
        })

    df = pd.DataFrame(rows)

    # keep only rows that map to the final thesis label schema
    df = df[df["final_label"].isin(["bona_fide", "forged"])].reset_index(drop=True)

    return df


def save_metadata(df: pd.DataFrame, output_path: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved metadata to: {output_path}")