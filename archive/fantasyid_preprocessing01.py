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
    if "train" in parts:
        return "train"
    elif "test" in parts:
        return "test"
    return None


def infer_label_from_parts(parts):
    if "bonafide" in parts:
        return 0
    elif "attack" in parts:
        return 1
    return None


def infer_class_name_from_parts(parts):
    if "bonafide" in parts:
        return "bonafide"
    elif "attack" in parts:
        return "attack"
    return None


def infer_capture_type(parts):
    """
    For bonafide images, the folder after 'bonafide' is the capture type:
    e.g. bonafide/huawei/, bonafide/scan/
    """
    if "bonafide" in parts:
        idx = parts.index("bonafide")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def infer_attack_type(parts):
    """
    For attack images, the folder after 'attack' is the attack subtype:
    e.g. attack/digital_1/, attack/facedancer/
    """
    if "attack" in parts:
        idx = parts.index("attack")
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
    image_paths = get_all_image_paths(root_path)

    rows = []

    for img_path in image_paths:
        rel_path = img_path.relative_to(root_path)
        parts = rel_path.parts

        split = infer_split_from_parts(parts)
        image_label = infer_label_from_parts(parts)
        class_name = infer_class_name_from_parts(parts)
        capture_type = infer_capture_type(parts)
        attack_type = infer_attack_type(parts)
        width, height = get_image_size(img_path)

        rows.append({
            "image_id": img_path.stem,
            "image_path": str(img_path),
            "relative_path": str(rel_path),
            "filename": img_path.name,
            "suffix": img_path.suffix.lower(),
            "split": split,
            "class_name": class_name,
            "image_label": image_label,
            "capture_type": capture_type,
            "attack_type": attack_type,
            "width": width,
            "height": height
        })

    df = pd.DataFrame(rows)
    return df
