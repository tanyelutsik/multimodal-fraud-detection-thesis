from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

train_df = pd.read_csv(PROCESSED_DIR / "image_train.csv")
val_df = pd.read_csv(PROCESSED_DIR / "image_val.csv")

overlap = set(train_df["image_path"]) & set(val_df["image_path"])
print("Overlap count:", len(overlap))

overlap_list = sorted(list(overlap))
overlap_df = pd.DataFrame({"image_path": overlap_list})

train_overlap = train_df[train_df["image_path"].isin(overlap)].copy()
train_overlap["where_found"] = "train"

val_overlap = val_df[val_df["image_path"].isin(overlap)].copy()
val_overlap["where_found"] = "val"

combined = pd.concat([train_overlap, val_overlap], ignore_index=True)

print("\nSample overlapping rows:")
print(combined[[
    "image_path",
    "source_dataset",
    "image_class",
    "original_label",
    "split",
    "split_source"
]].head(20))

out_dir = PROJECT_ROOT / "results" / "tables"
out_dir.mkdir(parents=True, exist_ok=True)
combined.to_csv(out_dir / "image_train_val_overlap.csv", index=False)

print("\nSaved overlap file to results/tables/image_train_val_overlap.csv")