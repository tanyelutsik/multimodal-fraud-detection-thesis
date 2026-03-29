from pathlib import Path
import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


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
                    rows.append({
                        "source_type": source_type,
                        "doc_type": doc_type,
                        "file_name": img_path.name,
                        "file_stem": img_path.stem,
                        "image_path": str(img_path),
                        "annotation_path": annotation_path
                    })

    return pd.DataFrame(rows)


def save_metadata(df: pd.DataFrame, output_path: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import pandas as pd
import random


def generate_fake_images(metadata_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fake_rows = []

    for _, row in metadata_df.iterrows():
        img_path = Path(row["image_path"])

        try:
            img = Image.open(img_path).convert("RGB")

            # 1. stronger blur
            img = img.filter(ImageFilter.GaussianBlur(radius=3))

            # 2. stronger brightness change
            img = ImageEnhance.Brightness(img).enhance(0.65)

            # 3. stronger contrast
            img = ImageEnhance.Contrast(img).enhance(1.4)

            # 4. slight rotation
            img = img.rotate(3, expand=False)

            # 5. add small black rectangle as occlusion
            draw = ImageDraw.Draw(img)
            w, h = img.size
            rect_w = int(w * 0.18)
            rect_h = int(h * 0.08)
            x1 = int(w * 0.65)
            y1 = int(h * 0.78)
            x2 = x1 + rect_w
            y2 = y1 + rect_h
            draw.rectangle([x1, y1, x2, y2], fill="black")

            fake_subdir = output_dir / row["source_type"] / row["doc_type"]
            fake_subdir.mkdir(parents=True, exist_ok=True)

            fake_filename = f"{img_path.stem}_fake.jpg"
            fake_full_path = fake_subdir / fake_filename
            img.save(fake_full_path)

            fake_rows.append({
                "source_type": row["source_type"],
                "doc_type": row["doc_type"],
                "file_name": fake_filename,
                "file_stem": f"{row['file_stem']}_fake",
                "image_path": str(fake_full_path),
                "annotation_path": row["annotation_path"],
                "image_label": 1
            })

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    return pd.DataFrame(fake_rows)

{'shape_attributes': {'name': 'rect', 'x': 771, 'y': 1928, 'width': 204, 'height': 199}, 'region_attributes': {'field_name': 'face', 'value': '', 'features': {}}}
{'shape_attributes': {'name': 'polygon', 'all_points_x': [613, 2007, 1769, 581], 'all_points_y': [1767, 1674, 2260, 2395]}, 'region_attributes': {'field_name': 'doc_quad'}}

from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import pandas as pd
import json


def load_annotation(annotation_path: str) -> dict:
    with open(annotation_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_doc_quad_for_filename(annotation_data: dict, filename: str):
    img_metadata = annotation_data.get("_via_img_metadata", {})

    for item in img_metadata.values():
        if item.get("filename") == filename:
            for region in item.get("regions", []):
                region_attrs = region.get("region_attributes", {})
                shape_attrs = region.get("shape_attributes", {})

                if region_attrs.get("field_name") == "doc_quad":
                    x_points = shape_attrs.get("all_points_x", [])
                    y_points = shape_attrs.get("all_points_y", [])
                    return x_points, y_points

    return None, None


def generate_fake_images_on_document(metadata_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fake_rows = []

    for _, row in metadata_df.iterrows():
        img_path = Path(row["image_path"])
        annotation_path = row["annotation_path"]
        filename = row["file_name"]

        print(f"\nProcessing: {filename}")
        print(f"Image path: {img_path}")
        print(f"Annotation path: {annotation_path}")

        try:
            img = Image.open(img_path).convert("RGB")
            print("Image opened successfully. Size:", img.size)

            annotation_data = load_annotation(annotation_path)
            x_points, y_points = get_doc_quad_for_filename(annotation_data, filename)

            print("doc_quad x_points:", x_points)
            print("doc_quad y_points:", y_points)

            if x_points is None or y_points is None:
                print(f"doc_quad not found for {filename}")
                continue

            x_min, x_max = min(x_points), max(x_points)
            y_min, y_max = min(y_points), max(y_points)

            print("Bounding box:", x_min, y_min, x_max, y_max)

            doc_crop = img.crop((x_min, y_min, x_max, y_max))
            print("Crop size:", doc_crop.size)

            doc_crop = doc_crop.filter(ImageFilter.GaussianBlur(radius=2.5))
            doc_crop = ImageEnhance.Brightness(doc_crop).enhance(0.75)
            doc_crop = ImageEnhance.Contrast(doc_crop).enhance(1.3)

            draw = ImageDraw.Draw(doc_crop)
            w, h = doc_crop.size
            rect_x1 = int(w * 0.55)
            rect_y1 = int(h * 0.60)
            rect_x2 = int(w * 0.90)
            rect_y2 = int(h * 0.78)
            draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], fill="black")

            fake_img = img.copy()
            fake_img.paste(doc_crop, (x_min, y_min))

            fake_subdir = output_dir / row["source_type"] / row["doc_type"]
            fake_subdir.mkdir(parents=True, exist_ok=True)

            fake_filename = f"{img_path.stem}_fake.jpg"
            fake_full_path = fake_subdir / fake_filename
            fake_img.save(fake_full_path)

            print("Saved fake image:", fake_full_path)

            fake_rows.append({
                "source_type": row["source_type"],
                "doc_type": row["doc_type"],
                "file_name": fake_filename,
                "file_stem": f"{row['file_stem']}_fake",
                "image_path": str(fake_full_path),
                "annotation_path": row["annotation_path"],
                "image_label": 1
            })

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    return pd.DataFrame(fake_rows)