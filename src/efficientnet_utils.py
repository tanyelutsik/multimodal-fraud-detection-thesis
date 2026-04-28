from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
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
# EfficientNet variant registry
#
# B0 is the default — fastest, lightest, good baseline.
# B3 is the recommended step-up if B0 plateaus: better accuracy at moderate
# extra cost. B5+ is only worth trying if you have a large dataset (10k+).
#
# image_size must match the variant — using the wrong size degrades accuracy.
# ---------------------------------------------------------------------------

EFFICIENTNET_VARIANTS = {
    "b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, 224),
    "b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT, 240),
    "b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT, 288),
    "b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT, 300),
    "b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT, 380),
}


# ---------------------------------------------------------------------------
# Label mapping  (identical to ResNet file — shared vocabulary)
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
# EfficientNet is more sensitive to input resolution than ResNet.
# The image_size MUST match the variant (see EFFICIENTNET_VARIANTS above).
# Using 224 for a B3 model (which expects 300) noticeably hurts accuracy.
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_train_transform(image_size: int = 224) -> T.Compose:
    """
    Augmented training pipeline.

    EfficientNet-specific notes:
      - RandomResizedCrop scale=(0.75, 1.0) is slightly more conservative
        than for ResNet because EfficientNet compound-scales the resolution,
        so aggressive crops risk losing too much spatial detail.
      - ColorJitter is kept to prevent the model relying on document colour
        as a fraud signal (scanner colour drift, print quality differences).
      - RandomErasing simulates stamps, redactions, or missing regions.
    """
    return T.Compose([
        T.Resize(int(image_size * 1.15)),
        T.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])


def build_val_transform(image_size: int = 224) -> T.Compose:
    """Deterministic pipeline for validation and test."""
    return T.Compose([
        T.Resize(int(image_size * 1.14)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    splits:      SplitBundle,
    variant:     str  = "b0",
    batch_size:  int  = 32,
    num_workers: int  = 0,
    image_col:   str  = "image_path",
    label_col:   str  = "label_int",
):
    _, _, image_size = EFFICIENTNET_VARIANTS[variant]

    train_transform = build_train_transform(image_size)
    val_transform   = build_val_transform(image_size)

    train_ds = ImageCSVDataset(splits.train_df, image_col, label_col, train_transform)
    val_ds   = ImageCSVDataset(splits.val_df,   image_col, label_col, val_transform)
    test_ds  = ImageCSVDataset(splits.test_df,  image_col, label_col, val_transform)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Model
#
# EfficientNet vs ResNet head differences:
#   - ResNet:        model.fc          = nn.Linear(512, 1)
#   - EfficientNet:  model.classifier  = nn.Sequential(Dropout, Linear(in, 1))
#
# The classifier in torchvision's EfficientNet is already a Sequential, so
# we read the in_features from model.classifier[1] (the existing Linear layer)
# rather than model.fc.in_features.
#
# BatchNorm handling:
#   EfficientNet uses many more BatchNorm layers than ResNet. During Stage 2
#   fine-tuning, BN layers are kept in EVAL mode (running stats frozen) to
#   avoid corrupting the well-calibrated ImageNet statistics with your small
#   dataset. This is handled in train_one_epoch via _set_bn_eval().
# ---------------------------------------------------------------------------

def build_model(
    variant:         str   = "b0",
    pretrained:      bool  = True,
    freeze_backbone: bool  = False,
    dropout:         float = 0.4,
) -> nn.Module:
    """
    EfficientNet-B{0..4} with custom Dropout + Linear head.

    variant:  "b0" (default, fastest) through "b4" (most accurate but slower)
    dropout:  0.4 is slightly higher than ResNet default (0.3) because
              EfficientNet is a wider network and more prone to overfitting
              on small fraud datasets.

    Two-stage training strategy (same as ResNet baseline):
      Stage 1 — freeze_backbone=True : only head trains
      Stage 2 — call unfreeze_backbone() then fine-tune with lower LR,
                 BN layers stay in eval mode
    """
    model_fn, weights_cls, image_size = EFFICIENTNET_VARIANTS[variant]
    weights  = weights_cls if pretrained else None
    model    = model_fn(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 1),
    )

    for param in model.classifier.parameters():
        param.requires_grad = True

    print(f"EfficientNet-{variant.upper()} | image_size={image_size} | "
          f"in_features={in_features} | dropout={dropout}")
    return model


def unfreeze_backbone(model: nn.Module) -> None:
    """
    Unfreeze all parameters for Stage 2.
    BatchNorm layers will be switched to eval mode during training
    by train_one_epoch — do not call model.train() on the whole model
    without also calling _set_bn_eval() afterwards.
    """
    for param in model.parameters():
        param.requires_grad = True
    print("Backbone unfrozen — all parameters now trainable. "
          "BN layers will be frozen in eval mode during training.")


def _set_bn_eval(model: nn.Module) -> None:
    """Keep all BatchNorm layers in eval mode (frozen running stats)."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


# ---------------------------------------------------------------------------
# Loss
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
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:           nn.Module,
    loader:          DataLoader,
    optimizer:       torch.optim.Optimizer,
    loss_fn:         nn.Module,
    device:          str,
    freeze_bn:       bool = False,
    scaler:          Optional[object] = None,
) -> float:
    """
    freeze_bn=True : keeps BatchNorm in eval mode even during training.
    Set this to True for Stage 2 (backbone unfrozen) to preserve
    ImageNet BN statistics. Leave False for Stage 1 (BN is already
    frozen because the whole backbone is frozen).
    """
    model.train()
    if freeze_bn:
        _set_bn_eval(model)

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
        rows.append({
            "threshold": round(t, 2),
            "accuracy":  m["accuracy"],
            "precision": m["precision"],
            "recall":    m["recall"],
            "f1":        m["f1"],
            "roc_auc":   m["roc_auc"],
        })

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


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

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
    freeze_bn:               bool  = False,
    use_amp:                 bool  = False,
) -> Tuple[nn.Module, pd.DataFrame]:
    """
    freeze_bn : set True for Stage 2 to keep BatchNorm running stats frozen.
                Should be False for Stage 1 (backbone already frozen).
    """
    scaler           = torch.cuda.amp.GradScaler() if (use_amp and torch.cuda.is_available()) else None
    best_val         = -1.0
    best_state       = None
    patience_counter = 0
    history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        train_loss  = train_one_epoch(model, train_loader, optimizer, loss_fn,
                                       device, freeze_bn=freeze_bn, scaler=scaler)
        val_metrics = evaluate(model, val_loader, loss_fn, device, threshold)
        current_lr  = optimizer.param_groups[0]["lr"]

        row = {
            "epoch":         epoch,
            "lr":            current_lr,
            "train_loss":    train_loss,
            "val_loss":      val_metrics["loss"],
            "val_accuracy":  val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall":    val_metrics["recall"],
            "val_f1":        val_metrics["f1"],
            "val_roc_auc":   val_metrics["roc_auc"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | lr={current_lr:.2e} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_roc_auc={val_metrics['roc_auc']:.4f}"
        )

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


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------

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
# Subgroup metrics
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
        rows.append({
            group_col:   group_value,
            "n_rows":    len(g),
            "n_neg":     int((y_true == 0).sum()),
            "n_pos":     int((y_true == 1).sum()),
            "accuracy":  m["accuracy"],
            "precision": m["precision"],
            "recall":    m["recall"],
            "f1":        m["f1"],
            "roc_auc":   m["roc_auc"],
        })

    return (
        pd.DataFrame(rows)
        .sort_values("n_rows", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# GradCAM
#
# EfficientNet target layer: model.features[-1]
# (the last MBConv block, equivalent to ResNet's layer4[-1])
# ---------------------------------------------------------------------------

class GradCAM:
    """
    GradCAM for EfficientNet.

    Recommended target layer:
        cam = GradCAM(model, target_layer=model.features[-1])

    This is EfficientNet's equivalent of ResNet's layer4[-1] — the deepest
    convolutional block before the classifier head.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model         = model
        self.target_layer  = target_layer
        self._gradients:   Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def fwd(_, __, output):
            self._activations = output.detach()

        def bwd(_, __, grad_output):
            self._gradients = grad_output[0].detach()

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
    def overlay(
        pil_image: Image.Image,
        heatmap:   np.ndarray,
        alpha:     float = 0.5,
    ) -> Image.Image:
        import matplotlib.cm as cm

        w, h     = pil_image.size
        resized  = Image.fromarray(
            (heatmap * 255).astype(np.uint8)
        ).resize((w, h), Image.BILINEAR)
        coloured = Image.fromarray(
            (cm.jet(np.array(resized))[:, :, :3] * 255).astype(np.uint8)
        )
        return Image.blend(pil_image.convert("RGB"), coloured, alpha=alpha)
