from __future__ import annotations

"""
multimodal_utils.py
-------------------
Deep multimodal fraud detection model combining:
  - Image branch:    ResNet-18 or EfficientNet-B0 backbone
  - Tabular branch:  MLP on transaction features
  - Fusion:          Late fusion — concatenate projections → classifier

IMPORTANT: Feature selection
-----------------------------
The tabular branch uses ONLY transaction features — never label columns.

EXCLUDED (would cause data leakage):
    tab_isFraud      ← final_label is partly derived from this
    tab_final_label  ← IS the label
    img_final_label  ← IS the label
    img_image_class  ← encodes the label
    combo_type       ← derived from labels
    img_source_dataset ← confirmed shortcut from baseline analysis

INCLUDED (raw transaction features):
    tab_type, tab_amount, tab_oldbalanceOrg, tab_newbalanceOrig,
    tab_oldbalanceDest, tab_newbalanceDest, tab_isFlaggedFraud

Optional engineered features (same as notebook 18 XGBoost):
    tab_balance_delta_orig  = newbalanceOrig  - oldbalanceOrg
    tab_balance_delta_dest  = newbalanceDest  - oldbalanceDest
    tab_amount_ratio_orig   = amount / (oldbalanceOrg + 1)
    tab_amount_ratio_dest   = amount / (oldbalanceDest + 1)
    tab_orig_balance_zeroed = 1 if newbalanceOrig == 0 else 0

Architecture
------------
Image  → backbone → flatten → projector (→ fusion_dim)
                                              ↓
                                         concat → fusion_head → logit
                                              ↑
Tab    → tab_encoder (→ fusion_dim)

Training strategy
-----------------
Two-stage (same as image baselines):
    Stage 1: freeze backbone, train tabular branch + fusion head
    Stage 2: unfreeze backbone with low LR (differential learning rates)

Imbalance handling
------------------
WeightedRandomSampler by combo_type × img_source_dataset — ensures each
combination group is equally represented per batch. No SMOTE (cannot
create synthetic image-transaction pairs in a meaningful way).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms as T

import warnings
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    precision_score, roc_auc_score, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

TARGET_COL = "final_label"

# Raw transaction features — same as notebooks 11, 12, 17
# tab_isFraud EXCLUDED — final_label is derived from it (leakage)
# tab_isFlaggedFraud INCLUDED — PaySim rule-based flag, not the label
TAB_RAW = [
    "tab_type",
    "tab_amount",
    "tab_oldbalanceOrg",
    "tab_newbalanceOrig",
    "tab_oldbalanceDest",
    "tab_newbalanceDest",
    "tab_isFlaggedFraud",
]

TAB_ENGINEERED = [
    "tab_balance_delta_orig",
    "tab_balance_delta_dest",
    "tab_amount_ratio_orig",
    "tab_amount_ratio_dest",
    "tab_orig_balance_zeroed",
]

TAB_ALL_FE = TAB_RAW + TAB_ENGINEERED

FORBIDDEN_COLS = {
    "tab_isFraud", "tab_final_label", "final_label",
    "img_final_label", "img_image_class", "img_original_label",
    "combo_type", "img_source_dataset",
    "mm_id", "split", "img_split", "img_image_path",
}


def validate_features(cols):
    bad = set(cols) & FORBIDDEN_COLS
    if bad:
        raise ValueError(f"Forbidden columns (leakage): {sorted(bad)}")
    print(f"Feature validation passed — {len(cols)} features, no leakage.")


def engineer_features(df):
    """Add balance-delta and ratio features — same as notebook 18."""
    d = df.copy()
    d["tab_balance_delta_orig"]  = d["tab_newbalanceOrig"] - d["tab_oldbalanceOrg"]
    d["tab_balance_delta_dest"]  = d["tab_newbalanceDest"] - d["tab_oldbalanceDest"]
    d["tab_amount_ratio_orig"]   = d["tab_amount"] / (d["tab_oldbalanceOrg"] + 1)
    d["tab_amount_ratio_dest"]   = d["tab_amount"] / (d["tab_oldbalanceDest"] + 1)
    d["tab_orig_balance_zeroed"] = (d["tab_newbalanceOrig"] == 0).astype(int)
    return d


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------

def fit_scaler(
    train_df:  pd.DataFrame,
    tab_cols:  List[str],
) -> StandardScaler:
    validate_features(tab_cols)
    scaler = StandardScaler()
    scaler.fit(train_df[tab_cols].values.astype(np.float32))
    print(f"Scaler fitted on {len(tab_cols)} features: {tab_cols}")
    return scaler


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def build_train_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize(int(image_size * 1.15)),
        T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        T.RandomGrayscale(p=0.1),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(mean=_MEAN, std=_STD),
        T.RandomErasing(p=0.3, scale=(0.02, 0.20)),
    ])


def build_val_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize(int(image_size * 1.14)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=_MEAN, std=_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultimodalDataset(Dataset):
    """
    Returns (image_tensor, tab_tensor, label) per sample.

    tab_tensor contains ONLY transaction features (TAB_RAW or TAB_ALL_FE),
    never label columns.
    """

    def __init__(
        self,
        df:        pd.DataFrame,
        scaler:    StandardScaler,
        tab_cols:  List[str],
        transform=None,
    ) -> None:
        validate_features(tab_cols)
        self.df        = df.reset_index(drop=True).copy()
        self.transform = transform
        self.tab_cols  = tab_cols

        self.tab_data = torch.tensor(
            scaler.transform(df[tab_cols].values.astype(np.float32)),
            dtype=torch.float32,
        )
        self.labels = torch.tensor(
            df[TARGET_COL].values.astype(np.float32),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["img_image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.tab_data[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# WeightedRandomSampler
# ---------------------------------------------------------------------------

def build_weighted_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Balance batches by combo_type × img_source_dataset.

    Ensures all four combo types and all image sources are seen equally
    during training, preventing MIDV-dominant groups from dominating.
    """
    cols = [c for c in ["combo_type", "img_source_dataset"]
            if c in train_df.columns]
    if not cols:
        raise ValueError("train_df must contain combo_type and/or img_source_dataset")

    group_key    = train_df[cols].astype(str).agg("_".join, axis=1)
    group_counts = group_key.value_counts()
    weights      = group_key.map(lambda x: 1.0 / group_counts[x]).values

    print(f"WeightedRandomSampler — balancing by: {cols}")
    print(f"  {len(group_counts)} groups:")
    for grp, cnt in group_counts.sort_index().items():
        print(f"    {grp}: n={cnt}  w={1/cnt:.5f}")

    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float32),
        num_samples=len(weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    train_df:    pd.DataFrame,
    val_df:      pd.DataFrame,
    test_df:     pd.DataFrame,
    scaler:      StandardScaler,
    tab_cols:    List[str],
    batch_size:  int  = 32,
    num_workers: int  = 0,
    use_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    pin = torch.cuda.is_available()

    train_ds = MultimodalDataset(train_df, scaler, tab_cols, build_train_transform())
    val_ds   = MultimodalDataset(val_df,   scaler, tab_cols, build_val_transform())
    test_ds  = MultimodalDataset(test_df,  scaler, tab_cols, build_val_transform())

    if use_sampler:
        sampler      = build_weighted_sampler(train_df)
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  sampler=sampler, num_workers=num_workers,
                                  pin_memory=pin)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  pin_memory=pin)

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def make_loss_fn(
    train_df:             pd.DataFrame,
    device:               str,
    pos_weight_override:  Optional[float] = None,
) -> nn.BCEWithLogitsLoss:
    n_pos = int(train_df[TARGET_COL].sum())
    n_neg = int((train_df[TARGET_COL] == 0).sum())
    pw    = pos_weight_override if pos_weight_override is not None else n_neg / n_pos
    dirn  = "DOWN-weights fraud (majority)" if pw < 1 else "UP-weights fraud (minority)"
    print(f"  pos_weight={pw:.3f}  (n_neg={n_neg}, n_pos={n_pos})  [{dirn}]")
    return nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pw], dtype=torch.float32).to(device)
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MultimodalFraudDetector(nn.Module):
    """
    Late-fusion multimodal model.

    Image branch:   CNN backbone → flatten → projector (→ fusion_dim)
    Tabular branch: MLP encoder (→ fusion_dim)
    Fusion:         concat → MLP head → logit

    Parameters
    ----------
    backbone      : "resnet18" (embed=512) or "efficientnet_b0" (embed=1280)
    tab_input_dim : number of tabular features
    tab_hidden    : hidden size of tabular MLP
    fusion_dim    : projected size from each branch
    fusion_hidden : hidden size of fusion MLP
    dropout       : dropout throughout
    pretrained    : use ImageNet weights for backbone
    """

    BACKBONE_DIMS = {"resnet18": 512, "efficientnet_b0": 1280}

    def __init__(
        self,
        backbone:      str   = "resnet18",
        tab_input_dim: int   = len(TAB_RAW),
        tab_hidden:    int   = 128,
        fusion_dim:    int   = 256,
        fusion_hidden: int   = 128,
        dropout:       float = 0.3,
        pretrained:    bool  = True,
    ) -> None:
        super().__init__()

        if backbone not in self.BACKBONE_DIMS:
            raise ValueError(f"backbone must be one of {list(self.BACKBONE_DIMS)}")

        self.backbone_name = backbone
        img_dim = self.BACKBONE_DIMS[backbone]

        # Image backbone
        if backbone == "resnet18":
            w = models.ResNet18_Weights.DEFAULT if pretrained else None
            b = models.resnet18(weights=w)
            self.image_backbone = nn.Sequential(*list(b.children())[:-1])
        else:
            w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            b = models.efficientnet_b0(weights=w)
            self.image_backbone = nn.Sequential(b.features, b.avgpool)

        # Image projector: img_dim → fusion_dim
        self.image_projector = nn.Sequential(
            nn.Linear(img_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # Tabular encoder: tab_input_dim → fusion_dim
        self.tab_encoder = nn.Sequential(
            nn.Linear(tab_input_dim, tab_hidden),
            nn.BatchNorm1d(tab_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(tab_hidden, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # Fusion head: 2*fusion_dim → 1
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_hidden, 1),
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"MultimodalFraudDetector | backbone={backbone} "
              f"| tab_dim={tab_input_dim} | fusion_dim={fusion_dim} "
              f"| params={n_params:,}")

    def freeze_backbone(self) -> None:
        for p in self.image_backbone.parameters():
            p.requires_grad = False
        print("Backbone frozen.")

    def unfreeze_backbone(self) -> None:
        for p in self.image_backbone.parameters():
            p.requires_grad = True
        print("Backbone unfrozen — all parameters trainable.")

    def forward(self, images: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        img_embed = self.image_backbone(images).flatten(1)
        img_proj  = self.image_projector(img_embed)
        tab_proj  = self.tab_encoder(tab)
        fused     = torch.cat([img_proj, tab_proj], dim=1)
        return self.fusion_head(fused).squeeze(1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     MultimodalFraudDetector,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn:   nn.Module,
    device:    str,
) -> float:
    model.train()
    total_loss, total_n = 0.0, 0

    for images, tab, labels in loader:
        images = images.to(device)
        tab    = tab.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss   = loss_fn(model(images, tab), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_n    += images.size(0)

    return total_loss / max(total_n, 1)


@torch.no_grad()
def predict_probs(
    model:  MultimodalFraudDetector,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []

    for images, tab, labels in loader:
        probs = torch.sigmoid(model(images.to(device), tab.to(device))).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    return (np.concatenate(all_labels).astype(int),
            np.concatenate(all_probs))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)
    m = {
        "threshold":        float(threshold),
        "accuracy":         float(accuracy_score(y_true, y_pred)),
        "precision":        float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":           float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":               float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    try:
        m["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        m["roc_auc"] = float("nan")
    return m


def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, pd.DataFrame]:
    rows = []
    for t in np.arange(0.10, 0.91, 0.05):
        m = compute_metrics(y_true, y_prob, threshold=t)
        rows.append({"threshold": round(float(t), 2), "accuracy": m["accuracy"],
                     "precision": m["precision"], "recall": m["recall"],
                     "f1": m["f1"], "roc_auc": m["roc_auc"]})
    sweep_df = pd.DataFrame(rows)
    best_t   = float(sweep_df.loc[sweep_df[metric].idxmax(), "threshold"])
    print(f"Best threshold ({metric}): {best_t:.2f}")
    return best_t, sweep_df


@torch.no_grad()
def evaluate(
    model:     MultimodalFraudDetector,
    loader:    DataLoader,
    loss_fn:   nn.Module,
    device:    str,
    threshold: float = 0.5,
) -> Dict:
    model.eval()
    total_loss, total_n = 0.0, 0
    all_probs, all_labels = [], []

    for images, tab, labels in loader:
        images   = images.to(device); tab = tab.to(device)
        logits   = model(images, tab)
        loss     = loss_fn(logits, labels.to(device))
        probs    = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs); all_labels.append(labels.numpy())
        total_loss += loss.item() * images.size(0)
        total_n    += images.size(0)

    y_true  = np.concatenate(all_labels).astype(int)
    y_prob  = np.concatenate(all_probs)
    metrics = compute_metrics(y_true, y_prob, threshold)
    metrics["loss"] = float(total_loss / max(total_n, 1))
    return metrics


def fit_model(
    model:                   MultimodalFraudDetector,
    train_loader:            DataLoader,
    val_loader:              DataLoader,
    loss_fn:                 nn.Module,
    optimizer:               torch.optim.Optimizer,
    device:                  str,
    epochs:                  int   = 15,
    threshold:               float = 0.5,
    early_stopping_patience: int   = 6,
    monitor_metric:          str   = "val_roc_auc",
    scheduler=None,
    model_save_path:         Optional[Path] = None,
) -> Tuple[MultimodalFraudDetector, pd.DataFrame]:
    best_val, best_state, patience_counter = -1.0, None, 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss  = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device, threshold)
        lr          = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch, "lr": lr,
            "train_loss": train_loss, "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
        })

        print(f"Epoch {epoch:02d} | lr={lr:.2e} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | "
              f"val_f1={val_metrics['f1']:.4f} | "
              f"val_roc_auc={val_metrics['roc_auc']:.4f}")

        if scheduler is not None:
            scheduler.step() if not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ) else scheduler.step(val_metrics[monitor_metric.replace("val_", "")])

        current = val_metrics[monitor_metric.replace("val_", "")]
        if current > best_val:
            best_val         = current
            best_state       = {k: v.detach().cpu().clone()
                                for k, v in model.state_dict().items()}
            patience_counter = 0
            if model_save_path is not None:
                Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, model_save_path)
                print(f"  ✓ Saved best model "
                      f"(epoch {epoch}, {monitor_metric}={best_val:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best weights ({monitor_metric}={best_val:.4f}).")

    return model, pd.DataFrame(history)


def run_test_evaluation(
    model:       MultimodalFraudDetector,
    test_loader: DataLoader,
    loss_fn:     nn.Module,
    device:      str,
    threshold:   float = 0.5,
) -> Dict:
    m = evaluate(model, test_loader, loss_fn, device, threshold)
    print("\n── TEST METRICS ──────────────────────────────")
    for k in ["loss", "accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"  {k:12s}: {m[k]:.4f}")
    print("  confusion_matrix:")
    print(np.array(m["confusion_matrix"]))
    print("──────────────────────────────────────────────")
    return m


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
        rows.append({"combo_type": combo, "n_rows": len(g),
                     "label": int(y_true[0]) if len(np.unique(y_true))==1 else "mixed",
                     "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc})
    df = pd.DataFrame(rows).sort_values("combo_type").reset_index(drop=True)
    if model_name:
        print(f"\n── {model_name} subgroup by combo_type ──")
    return df
