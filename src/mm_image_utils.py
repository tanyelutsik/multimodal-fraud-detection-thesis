from __future__ import annotations

"""
mm_image_utils.py
-----------------
Shared utilities for image-only baselines on the mixed-all multimodal
dataset (mm_*_mixed_all_group_test.csv).

Used by:
    15_mm_image_resnet18_baseline.ipynb
    16_mm_image_efficientnet_baseline.ipynb

pos_weight explained
--------------------
make_loss_fn computes pos_weight = n_neg / n_pos automatically.
For the mixed-all dataset (label ratio ~1:1.8):
    n_neg ~ 2588,  n_pos ~ 4594  =>  pos_weight ~ 0.56

pos_weight < 1 means fraud loss is DOWN-weighted. This is correct
because fraud is the majority class — without it the model would
predict fraud for everything and ignore legitimate cases.

WeightedRandomSampler
---------------------
Used in the weighted/debiased training variant.
Balances training batches by combo_type × img_source_dataset so
MIDV-dominant groups do not dominate every batch.
Only applied to training — val/test are unchanged.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import warnings
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MMImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None) -> None:
        self.df        = df.reset_index(drop=True).copy()
        self.transform = transform
        if "img_image_path" not in df.columns:
            raise ValueError("DataFrame must contain 'img_image_path'")
        if "final_label" not in df.columns:
            raise ValueError("DataFrame must contain 'final_label'")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        img   = Image.open(row["img_image_path"]).convert("RGB")
        label = float(row["final_label"])
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mm_splits(
    train_csv: str | Path,
    val_csv:   str | Path,
    test_csv:  str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_csv)
    val   = pd.read_csv(val_csv)
    test  = pd.read_csv(test_csv)

    for name, df in [("train", train), ("val", val), ("test", test)]:
        n0    = int((df["final_label"] == 0).sum())
        n1    = int((df["final_label"] == 1).sum())
        ratio = n1 / max(n0, 1)
        print(f"{name}: {len(df):,} rows | label 0:{n0:,} label 1:{n1:,} "
              f"ratio 1:{ratio:.2f}")
        if "combo_type" in df.columns:
            print(f"  combos: {df['combo_type'].value_counts().sort_index().to_dict()}")

    return train, val, test


# ---------------------------------------------------------------------------
# WeightedRandomSampler
# ---------------------------------------------------------------------------

def build_weighted_sampler(
    train_df:  pd.DataFrame,
    weight_by: List[str] = None,
) -> WeightedRandomSampler:
    """
    Balances training batches by combo_type × img_source_dataset.

    Prevents MIDV-dominant groups from dominating every batch and
    ensures all four combo types are seen equally during training.

    weight_by: columns to group by. Default: combo_type + source_dataset.
    """
    if weight_by is None:
        weight_by = [c for c in ["combo_type", "img_source_dataset"]
                     if c in train_df.columns]
        if not weight_by:
            weight_by = ["final_label"]

    group_key    = train_df[weight_by].astype(str).agg("_".join, axis=1)
    group_counts = group_key.value_counts()
    weights      = group_key.map(lambda x: 1.0 / group_counts[x]).values

    print(f"WeightedRandomSampler — weighting by: {weight_by}")
    print(f"  {len(group_counts)} groups:")
    for grp, cnt in group_counts.sort_index().items():
        print(f"    {grp}: n={cnt}  w={1/cnt:.6f}")

    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float32),
        num_samples=len(weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def build_mm_dataloaders(
    train_df:      pd.DataFrame,
    val_df:        pd.DataFrame,
    test_df:       pd.DataFrame,
    train_transform,
    val_transform,
    batch_size:    int = 32,
    num_workers:   int = 0,
    use_sampler:   bool = False,
    sampler_cols:  List[str] = None,
):
    """
    use_sampler=False (default): standard shuffle=True — baseline model
    use_sampler=True:            WeightedRandomSampler — weighted/debiased model

    Val and test always use shuffle=False regardless of use_sampler.
    """
    train_ds = MMImageDataset(train_df, train_transform)
    val_ds   = MMImageDataset(val_df,   val_transform)
    test_ds  = MMImageDataset(test_df,  val_transform)
    pin      = torch.cuda.is_available()

    if use_sampler:
        sampler      = build_weighted_sampler(train_df, weight_by=sampler_cols)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=pin,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin,
        )

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader


def make_loss_fn(train_df: pd.DataFrame, device: str) -> nn.BCEWithLogitsLoss:
    n_pos = int(train_df["final_label"].sum())
    n_neg = int((train_df["final_label"] == 0).sum())
    pw    = n_neg / n_pos
    direction = "DOWN-weighting fraud (majority)" if pw < 1 else "UP-weighting fraud (minority)"
    print(f"  pos_weight = {pw:.3f}  (n_neg={n_neg}, n_pos={n_pos})  [{direction}]")
    return nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pw], dtype=torch.float32).to(device)
    )


# ---------------------------------------------------------------------------
# Subgroup analysis
# ---------------------------------------------------------------------------

def subgroup_by_combo(
    pred_df:    pd.DataFrame,
    threshold:  float,
    model_name: str = "",
) -> pd.DataFrame:
    rows = []
    for combo, g in pred_df.groupby("combo_type"):
        y_true = g["y_true"].values
        y_prob = g["y_prob"].values
        y_pred = (y_prob >= threshold).astype(int)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                roc = float(roc_auc_score(y_true, y_prob)) \
                      if len(np.unique(y_true)) > 1 else float("nan")
            except Exception:
                roc = float("nan")
            f1   = float(f1_score(y_true,   y_pred, zero_division=0))
            rec  = float(recall_score(y_true, y_pred, zero_division=0))
            prec = float(precision_score(y_true, y_pred, zero_division=0))

        rows.append({
            "combo_type": combo, "n_rows": len(g),
            "label": int(y_true[0]) if len(np.unique(y_true))==1 else "mixed",
            "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc,
        })

    df = pd.DataFrame(rows).sort_values("combo_type").reset_index(drop=True)
    if model_name:
        print(f"\n── {model_name} subgroup by combo_type ──")
    return df