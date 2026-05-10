from __future__ import annotations

"""
debiased_utils.py
-----------------
Utilities for the debiased image baseline experiment.

What is different from image_baseline_utils.py:
  1. build_train_transform()  — stronger augmentation to reduce source-level
                                colour/resolution shortcuts
  2. build_dataloaders()      — WeightedRandomSampler so each source_dataset
                                contributes equally per batch regardless of size
  3. Everything else          — identical to image_baseline_utils.py so results
                                are directly comparable

Why this matters:
  The baseline subgroup analysis showed roc_auc=0.644 on FantasyID (the only
  mixed dataset) and perfect 1.000 on MIDV/FMIDV (single-class datasets).
  This means the model learned "MIDV=genuine, FMIDV=forged" from low-level
  image artefacts rather than genuine fraud cues.

  Two complementary fixes:
    - Stronger augmentation: makes colour/resolution less reliable as signals
    - Source balancing:      prevents MIDV (4000 images) from dominating batches
                             and ensures the model sees equal fraud signal from
                             all three sources per training step
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms as T

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

LABEL_MAP = {
    "bona_fide": 0,
    "bonafide":  0,
    "genuine":   0,
    "clean":     0,
    "forged":    1,
    "fraud":     1,
    "fake":      1,
    "attack":    1,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SplitBundle:
    train_df: pd.DataFrame
    val_df:   pd.DataFrame
    test_df:  pd.DataFrame


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ImageCSVDataset(Dataset):
    def __init__(
        self,
        df:        pd.DataFrame,
        image_col: str = "image_path",
        label_col: str = "label_int",
        transform=None,
    ) -> None:
        self.df        = df.reset_index(drop=True).copy()
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row        = self.df.iloc[idx]
        image_path = row[self.image_col]
        label      = float(row[self.label_col])

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def label_to_int(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.lower()
    mapped  = cleaned.map(LABEL_MAP)

    if mapped.isna().any():
        unknown = sorted(cleaned[mapped.isna()].unique().tolist())
        raise ValueError(f"Unknown labels found: {unknown}")

    return mapped.astype(int)


# ---------------------------------------------------------------------------
# Data loading / validation
# ---------------------------------------------------------------------------

def validate_split_df(
    df:         pd.DataFrame,
    split_name: str,
    image_col:  str = "image_path",
    label_col:  str = "image_class",
    group_col:  str = "group_key",
) -> pd.DataFrame:
    required = {image_col, label_col, group_col}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"{split_name} missing required columns: {missing}")

    out = df.copy()
    out["label_int"] = label_to_int(out[label_col])

    missing_paths = out.loc[
        ~out[image_col].apply(lambda p: Path(p).exists()), image_col
    ]
    if len(missing_paths) > 0:
        raise FileNotFoundError(
            f"{split_name} contains missing image files. "
            f"Examples: {missing_paths.head(5).tolist()}"
        )
    return out


def load_image_splits(
    train_csv: str | Path,
    val_csv:   str | Path,
    test_csv:  str | Path,
    image_col: str = "image_path",
    label_col: str = "image_class",
    group_col: str = "group_key",
) -> SplitBundle:
    train_df = validate_split_df(pd.read_csv(train_csv), "train", image_col, label_col, group_col)
    val_df   = validate_split_df(pd.read_csv(val_csv),   "val",   image_col, label_col, group_col)
    test_df  = validate_split_df(pd.read_csv(test_csv),  "test",  image_col, label_col, group_col)
    return SplitBundle(train_df=train_df, val_df=val_df, test_df=test_df)


def check_group_overlap(
    train_df:  pd.DataFrame,
    val_df:    pd.DataFrame,
    test_df:   pd.DataFrame,
    group_col: str = "group_key",
) -> Dict[str, int]:
    tg = set(train_df[group_col])
    vg = set(val_df[group_col])
    eg = set(test_df[group_col])
    return {
        "train_val_overlap":  len(tg & vg),
        "train_test_overlap": len(tg & eg),
        "val_test_overlap":   len(vg & eg),
    }


def print_split_summary(
    train_df:  pd.DataFrame,
    val_df:    pd.DataFrame,
    test_df:   pd.DataFrame,
    label_col: str = "image_class",
    group_col: str = "group_key",
) -> None:
    for name, df in [("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)]:
        print(f"\n{name}")
        print("  Rows:  ", len(df))
        print("  Groups:", df[group_col].nunique())
        print(df[label_col].value_counts(dropna=False).to_string())
        print((df[label_col].value_counts(normalize=True) * 100).round(2).to_string())

    overlap = check_group_overlap(train_df, val_df, test_df, group_col)
    print("\nGROUP OVERLAP")
    for k, v in overlap.items():
        flag = "  ⚠️  LEAKAGE" if v > 0 else ""
        print(f"  {k}: {v}{flag}")


# ---------------------------------------------------------------------------
# Transforms
#
# KEY CHANGE vs baseline:
#   - ColorJitter is stronger (brightness/contrast 0.5 vs 0.3)
#   - RandomGrayscale added: occasionally strips colour entirely so the model
#     cannot rely on colour differences between datasets
#   - GaussianBlur added: simulates different camera/scanner quality
#   - RandomErasing scale increased slightly
#
# Val/test transform is IDENTICAL to the baseline so evaluation is fair.
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_train_transform(image_size: int = 224) -> T.Compose:
    """
    Stronger augmentation pipeline to reduce dataset-level shortcuts.

    Compared to the baseline:
      - ColorJitter brightness/contrast raised from 0.3 → 0.5
      - RandomGrayscale(p=0.1): occasionally strips colour so the model
        cannot rely on colour differences between MIDV/FMIDV/FantasyID
      - GaussianBlur: simulates different scanner/camera resolution quality
      - Slightly larger random erasing scale
    """
    return T.Compose([
        T.Resize(int(image_size * 1.15)),
        T.RandomResizedCrop(image_size, scale=(0.65, 1.0)),  # more aggressive
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=15),                         # wider than baseline
        T.ColorJitter(brightness=0.5, contrast=0.5,           # stronger than baseline
                      saturation=0.4, hue=0.1),
        T.RandomGrayscale(p=0.1),                             # NEW
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),      # NEW
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        T.RandomErasing(p=0.3, scale=(0.02, 0.20)),           # slightly more aggressive
    ])


def build_val_transform(image_size: int = 224) -> T.Compose:
    """Deterministic — identical to baseline for fair comparison."""
    return T.Compose([
        T.Resize(int(image_size * 1.14)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Source-balanced sampler
#
# KEY CHANGE vs baseline:
#   MIDV has 4000 images, FMIDV 2880, FantasyID ~933 in training.
#   Without balancing, MIDV dominates every batch and the model learns
#   "MIDV patterns = genuine" early and never unlearns it.
#
#   WeightedRandomSampler assigns each image a weight = 1 / source_count
#   so each source contributes equally in expectation per batch.
# ---------------------------------------------------------------------------

def build_source_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Returns a WeightedRandomSampler that gives each source_dataset equal
    representation per batch.

    Weight per image = 1 / number_of_images_in_that_source
    """
    source_counts = train_df["source_dataset"].value_counts()
    weights = train_df["source_dataset"].map(
        lambda s: 1.0 / source_counts[s]
    ).values

    print("Source balancing weights:")
    for src, count in source_counts.items():
        w = 1.0 / count
        print(f"  {src}: {count} images → weight {w:.6f}")

    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float32),
        num_samples=len(weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    splits:          SplitBundle,
    batch_size:      int  = 32,
    num_workers:     int  = 0,
    image_col:       str  = "image_path",
    label_col:       str  = "label_int",
    image_size:      int  = 224,
    balance_sources: bool = True,
):
    """
    balance_sources=True (default): use WeightedRandomSampler so each
    source_dataset contributes equally per batch.
    Set to False to ablate and check how much balancing alone helps.
    """
    train_transform = build_train_transform(image_size)
    val_transform   = build_val_transform(image_size)

    train_ds = ImageCSVDataset(splits.train_df, image_col, label_col, train_transform)
    val_ds   = ImageCSVDataset(splits.val_df,   image_col, label_col, val_transform)
    test_ds  = ImageCSVDataset(splits.test_df,  image_col, label_col, val_transform)

    pin_memory = torch.cuda.is_available()

    if balance_sources and "source_dataset" in splits.train_df.columns:
        sampler      = build_source_sampler(splits.train_df)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            sampler=sampler,        # replaces shuffle=True
            num_workers=num_workers, pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Model  (identical to baseline)
# ---------------------------------------------------------------------------

def build_model(
    pretrained:      bool  = True,
    freeze_backbone: bool  = False,
    dropout:         float = 0.3,
) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model   = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc    = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 1),
    )
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def unfreeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True
    print("Backbone unfrozen — all parameters now trainable.")


# ---------------------------------------------------------------------------
# Loss  (identical to baseline)
# ---------------------------------------------------------------------------

def compute_pos_weight(
    train_df:  pd.DataFrame,
    label_col: str = "label_int",
) -> torch.Tensor:
    y     = train_df[label_col].to_numpy()
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        raise ValueError("Training split has no positive samples.")
    pos_weight = n_neg / n_pos
    print(f"  pos_weight = {pos_weight:.3f}  (n_neg={n_neg}, n_pos={n_pos})")
    return torch.tensor([pos_weight], dtype=torch.float32)


def make_loss_fn(
    train_df:  pd.DataFrame,
    device:    str,
    label_col: str = "label_int",
) -> nn.BCEWithLogitsLoss:
    pos_weight = compute_pos_weight(train_df, label_col=label_col).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# ---------------------------------------------------------------------------
# Training loop  (identical to baseline)
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn:   nn.Module,
    device:    str,
    scaler:    Optional[object] = None,
) -> float:
    model.train()
    running_loss = 0.0
    total        = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss   = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss   = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        total        += images.size(0)

    return running_loss / max(total, 1)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_probs(
    model:  nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs:  List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device)
        probs  = torch.sigmoid(model(images)).squeeze(1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    return np.concatenate(all_labels).astype(int), np.concatenate(all_probs)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, object]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics: Dict[str, object] = {
        "threshold":        float(threshold),
        "accuracy":         float(accuracy_score(y_true, y_pred)),
        "precision":        float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":           float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":               float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def tune_threshold(
    y_true:     np.ndarray,
    y_prob:     np.ndarray,
    thresholds: Optional[List[float]] = None,
    metric:     str = "f1",
) -> Tuple[float, pd.DataFrame]:
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05).tolist()
    rows = []
    for t in thresholds:
        m = compute_metrics(y_true, y_prob, threshold=t)
        rows.append({"threshold": round(t, 2), "accuracy": m["accuracy"],
                     "precision": m["precision"], "recall": m["recall"],
                     "f1": m["f1"], "roc_auc": m["roc_auc"]})
    sweep_df       = pd.DataFrame(rows)
    best_threshold = float(sweep_df.loc[sweep_df[metric].idxmax(), "threshold"])
    print(f"\nBest threshold ({metric}): {best_threshold:.2f}")
    return best_threshold, sweep_df


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    loss_fn:   nn.Module,
    device:    str,
    threshold: float = 0.5,
) -> Dict[str, object]:
    model.eval()
    running_loss = 0.0
    total        = 0
    all_probs:  List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for images, labels in loader:
        images   = images.to(device)
        labels_t = labels.to(device).unsqueeze(1)
        logits   = model(images)
        loss     = loss_fn(logits, labels_t)
        probs    = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
        running_loss += loss.item() * images.size(0)
        total        += images.size(0)

    y_true  = np.concatenate(all_labels).astype(int)
    y_prob  = np.concatenate(all_probs)
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = float(running_loss / max(total, 1))
    return metrics


def fit_model(
    model:                   nn.Module,
    train_loader:            DataLoader,
    val_loader:              DataLoader,
    loss_fn:                 nn.Module,
    optimizer:               torch.optim.Optimizer,
    device:                  str,
    epochs:                  int   = 15,
    threshold:               float = 0.5,
    early_stopping_patience: int   = 5,
    monitor_metric:          str   = "val_roc_auc",
    scheduler:               Optional[object] = None,
    model_save_path:         Optional[str | Path] = None,
    use_amp:                 bool  = False,
) -> Tuple[nn.Module, pd.DataFrame]:
    scaler           = torch.cuda.amp.GradScaler() if (use_amp and torch.cuda.is_available()) else None
    best_val         = -1.0
    best_state       = None
    patience_counter = 0
    history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        train_loss  = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        val_metrics = evaluate(model, val_loader, loss_fn, device, threshold)
        current_lr  = optimizer.param_groups[0]["lr"]

        row = {"epoch": epoch, "lr": current_lr,
               "train_loss": train_loss, "val_loss": val_metrics["loss"],
               "val_accuracy": val_metrics["accuracy"],
               "val_precision": val_metrics["precision"],
               "val_recall": val_metrics["recall"],
               "val_f1": val_metrics["f1"],
               "val_roc_auc": val_metrics["roc_auc"]}
        history.append(row)

        print(f"Epoch {epoch:02d} | lr={current_lr:.2e} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | "
              f"val_f1={val_metrics['f1']:.4f} | "
              f"val_roc_auc={val_metrics['roc_auc']:.4f}")

        if scheduler is not None:
            key = monitor_metric.replace("val_", "")
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics[key])
            else:
                scheduler.step()

        current_monitored = float(val_metrics[monitor_metric.replace("val_", "")])
        if current_monitored > best_val:
            best_val         = current_monitored
            best_state       = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            if model_save_path is not None:
                p = Path(model_save_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, p)
                print(f"  ✓ Saved best model "
                      f"(epoch {epoch}, {monitor_metric}={best_val:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(patience={early_stopping_patience}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best weights ({monitor_metric}={best_val:.4f}).")

    return model, pd.DataFrame(history)


def run_test_evaluation(
    model:       nn.Module,
    test_loader: DataLoader,
    loss_fn:     nn.Module,
    device:      str,
    threshold:   float = 0.5,
) -> Dict[str, object]:
    test_metrics = evaluate(model, test_loader, loss_fn, device, threshold)
    print("\n── TEST METRICS ──────────────────────────────")
    for key in ["loss", "accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"  {key:12s}: {test_metrics[key]:.4f}")
    print("  confusion_matrix:")
    print(np.array(test_metrics["confusion_matrix"]))
    print("──────────────────────────────────────────────")
    return test_metrics


# ---------------------------------------------------------------------------
# Subgroup metrics  (identical to baseline)
# ---------------------------------------------------------------------------

def subgroup_metrics(
    df:        pd.DataFrame,
    group_col: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    work = df.copy()
    work[group_col] = work[group_col].fillna("missing").astype(str)
    rows = []
    for group_value, g in work.groupby(group_col, dropna=False):
        y_true = g["y_true"].to_numpy()
        y_prob = g["y_prob"].to_numpy()
        m      = compute_metrics(y_true, y_prob, threshold=threshold)
        rows.append({group_col: group_value, "n_rows": len(g),
                     "n_neg": int((y_true==0).sum()), "n_pos": int((y_true==1).sum()),
                     "accuracy": m["accuracy"], "precision": m["precision"],
                     "recall": m["recall"], "f1": m["f1"], "roc_auc": m["roc_auc"]})
    return pd.DataFrame(rows).sort_values("n_rows", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# GradCAM  (identical to baseline)
# ---------------------------------------------------------------------------

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model         = model
        self.target_layer  = target_layer
        self._gradients:   Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def fwd(_, __, output): self._activations = output.detach()
        def bwd(_, __, grad_output): self._gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(fwd)
        self.target_layer.register_full_backward_hook(bwd)

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        self.model.eval()
        x      = x.requires_grad_(True)
        logits = self.model(x)
        self.model.zero_grad()
        logits.sum().backward()
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam     = torch.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam     = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    @staticmethod
    def overlay(pil_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
        import matplotlib.cm as cm
        w, h     = pil_image.size
        resized  = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        coloured = Image.fromarray((cm.jet(np.array(resized))[:, :, :3] * 255).astype(np.uint8))
        return Image.blend(pil_image.convert("RGB"), coloured, alpha=alpha)
