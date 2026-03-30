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


def get_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return (None, None)


def build_fmidv_dataframe(root_path):
    root_path = Path(root_path)

    if not root_path.exists():
        raise FileNotFoundError(f"FMIDV root folder does not exist: {root_path}")

    image_paths = get_all_image_paths(root_path)
    rows = []

    for img_path in image_paths:
        rel_path = img_path.relative_to(root_path)
        parts = rel_path.parts

        if len(parts) != 3:
            continue

        top_folder = parts[0].strip()
        doc_type = parts[1].strip()
        file_name = img_path.name
        file_stem = img_path.stem

        stem_parts = file_stem.split("_")
        if len(stem_parts) < 4:
            continue

        sample_id = stem_parts[0]
        patch_size = stem_parts[-2]
        zone_count = stem_parts[-1]
        source_type = "_".join(stem_parts[1:-2])

        if patch_size not in {"P16", "P32", "P64"}:
            continue
        if zone_count not in {"Z2", "Z4", "Z6"}:
            continue

        variant = f"{patch_size}_{zone_count}"
        group_key = f"{doc_type}__{sample_id}"

        width, height = get_image_size(img_path)

        rows.append({
            "image_path": str(img_path.resolve()),
            "source_dataset": "fmidv",
            "image_class": "forged",
            "group_key": group_key,
            "doc_type": doc_type,
            "source_type": source_type,
            "file_name": file_name,
            "split_source": None,
            "original_label": "forged",
            "relative_path": str(rel_path),
            "top_folder": top_folder,
            "sample_id": sample_id,
            "patch_size": patch_size,
            "zone_count": zone_count,
            "variant": variant,
            "width": width,
            "height": height,
        })

    df = pd.DataFrame(rows)
    return df.reset_index(drop=True)


def build_fmidv_subset(df, groups_per_doc_type=72):
    required_source_types = {"templates", "photo", "scan_rotated", "scan_upright"}
    variant_order = ["P16_Z2", "P16_Z4", "P16_Z6", "P32_Z2", "P32_Z4", "P32_Z6", "P64_Z2"]

    group_source_summary = (
        df.groupby(["group_key", "doc_type"])["source_type"]
        .agg(lambda x: set(x))
        .reset_index(name="source_set")
    )

    valid_groups = group_source_summary[
        group_source_summary["source_set"].apply(lambda x: required_source_types.issubset(x))
    ].copy()

    sampled_groups = (
        valid_groups.groupby("doc_type", group_keys=False)
        .sample(n=groups_per_doc_type, random_state=42)
        .reset_index(drop=True)
    )

    selected_group_keys = set(sampled_groups["group_key"])
    df_selected = df[df["group_key"].isin(selected_group_keys)].copy()

    selected_rows = []

    for source_type in sorted(required_source_types):
        source_df = df_selected[df_selected["source_type"] == source_type].copy()
        group_list = sorted(source_df["group_key"].unique())

        for i, group_key in enumerate(group_list):
            target_variant = variant_order[i % len(variant_order)]
            group_rows = source_df[source_df["group_key"] == group_key]

            chosen = group_rows[group_rows["variant"] == target_variant]

            if chosen.empty:
                chosen = group_rows.iloc[[0]]
            else:
                chosen = chosen.iloc[[0]]

            selected_rows.append(chosen)

    df_subset = pd.concat(selected_rows, ignore_index=True)
    df_subset = df_subset.sort_values(
        ["doc_type", "group_key", "source_type"]
    ).reset_index(drop=True)

    return df_subset


def save_metadata(df: pd.DataFrame, output_path: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved metadata to: {output_path}")