from pathlib import Path
import json
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw


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


def generate_suspicious_images_on_document(
    metadata_df: pd.DataFrame,
    output_dir: str
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suspicious_rows = []

    for _, row in metadata_df.iterrows():
        img_path = Path(row["image_path"])
        annotation_path = row["annotation_path"]
        filename = row["file_name"]

        try:
            if annotation_path is None:
                continue

            img = Image.open(img_path).convert("RGB")
            annotation_data = load_annotation(annotation_path)
            x_points, y_points = get_doc_quad_for_filename(annotation_data, filename)

            if x_points is None or y_points is None:
                continue

            x_min, x_max = min(x_points), max(x_points)
            y_min, y_max = min(y_points), max(y_points)

            doc_crop = img.crop((x_min, y_min, x_max, y_max))

            # suspicious transformations
            blur_radius = 2.5
            brightness_factor = 0.75
            contrast_factor = 1.3
            rotation_angle = 0  # keep 0 for now if rotating crop is messy

            doc_crop = doc_crop.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            doc_crop = ImageEnhance.Brightness(doc_crop).enhance(brightness_factor)
            doc_crop = ImageEnhance.Contrast(doc_crop).enhance(contrast_factor)

            draw = ImageDraw.Draw(doc_crop)
            w, h = doc_crop.size
            rect_x1 = int(w * 0.55)
            rect_y1 = int(h * 0.60)
            rect_x2 = int(w * 0.90)
            rect_y2 = int(h * 0.78)
            draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], fill="black")

            suspicious_img = img.copy()
            suspicious_img.paste(doc_crop, (x_min, y_min))

            suspicious_subdir = output_dir / row["source_type"] / row["doc_type"]
            suspicious_subdir.mkdir(parents=True, exist_ok=True)

            suspicious_filename = f"{img_path.stem}_suspicious.jpg"
            suspicious_full_path = suspicious_subdir / suspicious_filename
            suspicious_img.save(suspicious_full_path)

            suspicious_rows.append({
                "source_dataset": "midv",
                "source_type": row["source_type"],
                "doc_type": row["doc_type"],
                "file_name": suspicious_filename,
                "file_stem": f"{row['file_stem']}_suspicious",
                "image_path": str(suspicious_full_path),
                "original_image_path": str(img_path),
                "annotation_path": row["annotation_path"],
                "final_label": "suspicious",
                "is_synthetic": True,
                "blur_radius": blur_radius,
                "brightness_factor": brightness_factor,
                "contrast_factor": contrast_factor,
                "occlusion_applied": True
            })

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    return pd.DataFrame(suspicious_rows)


def save_metadata(df: pd.DataFrame, output_path: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)