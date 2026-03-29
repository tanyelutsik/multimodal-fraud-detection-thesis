from pathlib import Path
import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def get_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return (None, None)


def collect_midv_metadata(midv_root: str) -> pd.DataFrame:
    """
    Parse MIDV folder structure where:
    - each source folder contains:
        - annotations/<doc_type>.json
        - images/<doc_type>/*.(jpg/png/tif)
    - one annotation JSON corresponds to all images in the matching doc_type folder
    """
    midv_root = Path(midv_root)
    rows = []

    for source_folder in midv_root.iterdir():
        if not source_folder.is_dir():
            continue

        source_type = source_folder.name
        annotations_root = source_folder / "annotations"
        images_root = source_folder / "images"

        if not annotations_root.exists() or not images_root.exists():
            continue

        for doc_folder in images_root.iterdir():
            if not doc_folder.is_dir():
                continue

            doc_type = doc_folder.name
            annotation_path = annotations_root / f"{doc_type}.json"

            if not annotation_path.exists():
                annotation_path = None
            else:
                annotation_path = str(annotation_path)

            for img_path in doc_folder.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    width, height = get_image_size(img_path)

                    rows.append({
                        "source_dataset": "midv",
                        "source_type": source_type,
                        "doc_type": doc_type,
                        "file_name": img_path.name,
                        "file_stem": img_path.stem,
                        "image_path": str(img_path),
                        "annotation_path": annotation_path,
                        "final_label": "bona_fide",
                        "is_synthetic": False,
                        "width": width,
                        "height": height
                    })

    df = pd.DataFrame(rows)
    df["sample_id"] = [f"midv_{i:06d}" for i in range(len(df))]

    columns = ["sample_id"] + [col for col in df.columns if col != "sample_id"]
    df = df[columns]

    return df


def save_metadata(df: pd.DataFrame, output_path: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)