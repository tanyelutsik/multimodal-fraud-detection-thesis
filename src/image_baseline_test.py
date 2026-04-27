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
# Label mapping
# ---------------------------------------------------------------------------

LABEL_MAP = {
    "bona_fide": 0,
    "bonafide": 0,
    "genuine": 0,
    "clean": 0,
    "forged": 1,
    "fraud": 1,
    "fake": 1,
    "attack": 1,
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
        df: pd.DataFrame,
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
    df: pd.DataFrame,
    split_name: str,
    image_col: str = "image_path",
    label_col: str = "image_class",
    group_col: str = "group_key",
) -> pd.DataFrame:
    required = {image_col, label_col, group_col}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"{split_name} missing required columns: {missing}")

    out = df.copy()
    out["label_int"] = label_to_int(out[label_col])

    missing_paths = out.loc[~out[image_col].apply(lambda p: Path(p).exists()), image_col]
    if len(missing_paths) > 0:
        examples = missing_paths.head(5).tolist()
        raise FileNotFoundError(
            f"{split_name} contains missing image files. Examples: {examples}"
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
    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)
    test_df  = pd.read_csv(test_csv)

    train_df = validate_split_df(train_df, "train", image_col, label_col, group_col)
    val_df   = validate_split_df(val_df,   "val",   image_col, label_col, group_col)
    test_df  = validate_split_df(test_df,  "test",  image_col, label_col, group_col)

    return SplitBundle(train_df=train_df, val_df=val_df, test_df=test_df)


def check_group_overlap(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    group_col: str = "group_key",
) -> Dict[str, int]:
    train_groups = set(train_df[group_col])
    val_groups   = set(val_df[group_col])
    test_groups  = set(test_df[group_col])

    return {
        "train_val_overlap":  len(train_groups & val_groups),
        "train_test_overlap": len(train_groups & test_groups),
        "val_test_overlap":   len(val_groups   & test_groups),
    }


def print_split_summary(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    label_col: str = "image_class",
    group_col: str = "group_key",
) -> None:
    for name, df in [("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)]:
        print(f"\n{name}")
        print("Rows:",   len(df))
        print("Groups:", df[group_col].nunique())
        print(df[label_col].value_counts(dropna=False))
        print((df[label_col].value_counts(normalize=True) * 100).round(2))

    overlap = check_group_overlap(train_df, val_df, test_df, group_col)
    print("\nGROUP OVERLAP")
    for k, v in overlap.items():
        print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Transforms  ← KEY FIX: separate train vs val/test pipelines
# ---------------------------------------------------------------------------

# ImageNet stats used by pretrained ResNet
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_train_transform(
    image_size: int = 224,
    color_jitter: bool = True,
    random_erasing: bool = True,
) -> T.Compose:
    """
    Augmented pipeline for training only.

    Augmentations are chosen to mimic realistic document/image variability:
      - Random resized crop: handles slight framing differences.
      - Horizontal/vertical flips: documents can be scanned either way.
      - ColorJitter: counters scanner/camera colour variations so the model
        can't just memorise "genuine docs are brighter".
      - RandomRotation: small tilts from scanning/photographing.
      - RandomErasing: simulates stamps, stickers, or redacted regions that
        appear on some documents but not others; prevents the network from
        relying on a specific region being always clean.
    """
    ops = [
        T.Resize(int(image_size * 1.15)),        # slightly oversized then crop
        T.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=10),
    ]

    if color_jitter:
        ops.append(
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
        )

    ops += [
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]

    if random_erasing:
        # applied after ToTensor; erases random rectangles with noise
        ops.append(T.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)))

    return T.Compose(ops)


def build_val_transform(image_size: int = 224) -> T.Compose:
    """Deterministic pipeline for validation and test — no randomness."""
    return T.Compose([
        T.Resize(int(image_size * 1.14)),   # 256 for 224
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    splits: SplitBundle,
    batch_size:  int = 32,
    num_workers: int = 0,
    image_col:   str = "image_path",
    label_col:   str = "label_int",
    image_size:  int = 224,
):
    # ← Augmented for train, clean for val/test
    train_transform = build_train_transform(image_size=image_size)
    val_transform   = build_val_transform(image_size=image_size)

    train_ds = ImageCSVDataset(splits.train_df, image_col, label_col, train_transform)
    val_ds   = ImageCSVDataset(splits.val_df,   image_col, label_col, val_transform)
    test_ds  = ImageCSVDataset(splits.test_df,  image_col, label_col, val_transform)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(
    pretrained:       bool = True,
    freeze_backbone:  bool = False,
    dropout:          float = 0.3,
) -> nn.Module:
    """
    ResNet-18 with a Dropout + Linear head for binary fraud detection.

    Two-stage fine-tuning strategy (recommended):
      1. freeze_backbone=True for the first N epochs → only train the head,
         this prevents the noisy gradients from an untrained head destroying
         the pretrained backbone weights early on.
      2. Call unfreeze_backbone(model) after the head has stabilised, then
         continue with a lower LR for the backbone.

    dropout: regularises the head; 0.3 is a good default for fine-tuning.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model   = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 1),
    )

    # Final head is always trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all backbone parameters after initial head-only warmup."""
    for param in model.parameters():
        param.requires_grad = True
    print("Backbone unfrozen — all parameters are now trainable.")


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_pos_weight(
    train_df: pd.DataFrame,
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
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn:   nn.Module,
    device:    str,
    scaler:    Optional[torch.cuda.amp.GradScaler] = None,
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

        batch_size    = images.size(0)
        running_loss += loss.item() * batch_size
        total        += batch_size

    return running_loss / max(total, 1)


# ---------------------------------------------------------------------------
# Inference helpers
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
        logits = model(images)
        probs  = torch.sigmoid(logits).squeeze(1).cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels.numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels).astype(int)

    return y_true, y_prob


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
    """
    Sweep decision thresholds on the validation set and return the best one.

    For fraud detection you usually care more about recall (catching all fraud)
    than precision, so consider passing metric='recall' if false negatives are
    more costly than false positives.

    Returns
    -------
    best_threshold : float
    sweep_df       : DataFrame with all threshold results
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.05).tolist()

    rows = []
    for t in thresholds:
        m = compute_metrics(y_true, y_prob, threshold=t)
        rows.append({
            "threshold": t,
            "accuracy":  m["accuracy"],
            "precision": m["precision"],
            "recall":    m["recall"],
            "f1":        m["f1"],
            "roc_auc":   m["roc_auc"],
        })

    sweep_df      = pd.DataFrame(rows)
    best_idx      = sweep_df[metric].idxmax()
    best_threshold = float(sweep_df.loc[best_idx, "threshold"])

    print(f"\nBest threshold ({metric}): {best_threshold:.2f}")
    print(sweep_df.to_string(index=False))

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

        logits = model(images)
        loss   = loss_fn(logits, labels_t)
        probs  = torch.sigmoid(logits).squeeze(1).cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels.numpy())

        batch_size    = images.size(0)
        running_loss += loss.item() * batch_size
        total        += batch_size

    avg_loss = running_loss / max(total, 1)
    y_true   = np.concatenate(all_labels).astype(int)
    y_prob   = np.concatenate(all_probs)

    metrics         = compute_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = float(avg_loss)
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
    epochs:                  int = 15,
    threshold:               float = 0.5,
    early_stopping_patience: int = 5,
    monitor_metric:          str = "val_roc_auc",   # smoother than val_f1
    scheduler:               Optional[object] = None,
    model_save_path:         str | Path | None = None,
    use_amp:                 bool = False,
) -> Tuple[nn.Module, pd.DataFrame]:
    """
    Full training loop with:
      - Optional mixed-precision (use_amp)
      - LR scheduler step per epoch
      - Early stopping on monitor_metric (default: val_roc_auc — smoother
        than val_f1, less prone to firing early due to threshold noise)
      - Best-model checkpointing
    """
    scaler      = torch.cuda.amp.GradScaler() if (use_amp and torch.cuda.is_available()) else None
    best_val    = -1.0
    best_state  = None
    patience_counter = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model, loader=train_loader,
            optimizer=optimizer, loss_fn=loss_fn,
            device=device, scaler=scaler,
        )

        val_metrics = evaluate(
            model=model, loader=val_loader,
            loss_fn=loss_fn, device=device, threshold=threshold,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch":        epoch,
            "lr":           current_lr,
            "train_loss":   train_loss,
            "val_loss":     val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision":val_metrics["precision"],
            "val_recall":   val_metrics["recall"],
            "val_f1":       val_metrics["f1"],
            "val_roc_auc":  val_metrics["roc_auc"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | lr={current_lr:.2e} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_roc_auc={val_metrics['roc_auc']:.4f}"
        )

        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics[monitor_metric.replace("val_", "")])
            else:
                scheduler.step()

        # Early stopping & checkpointing
        current_monitored = float(val_metrics[monitor_metric.replace("val_", "")])
        if current_monitored > best_val:
            best_val   = current_monitored
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0

            if model_save_path is not None:
                model_save_path = Path(model_save_path)
                model_save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, model_save_path)
                print(f"  ✓ Saved best model (epoch {epoch}, {monitor_metric}={best_val:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch} (patience={early_stopping_patience}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best weights ({monitor_metric}={best_val:.4f}).")

    return model, pd.DataFrame(history)


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------

def run_test_evaluation(
    model:     nn.Module,
    test_loader: DataLoader,
    loss_fn:   nn.Module,
    device:    str,
    threshold: float = 0.5,
) -> Dict[str, object]:
    test_metrics = evaluate(
        model=model, loader=test_loader,
        loss_fn=loss_fn, device=device, threshold=threshold,
    )

    print("\n── TEST METRICS ──────────────────────────")
    for key in ["loss", "accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"  {key}: {test_metrics[key]:.4f}")
    print("  confusion_matrix:")
    print(np.array(test_metrics["confusion_matrix"]))
    print("──────────────────────────────────────────")

    return test_metrics


# ---------------------------------------------------------------------------
# Subgroup metrics
# ---------------------------------------------------------------------------

def subgroup_metrics(
    df:         pd.DataFrame,
    group_col:  str,
    threshold:  float = 0.5,
) -> pd.DataFrame:
    """
    Compute per-group classification metrics.
    df must contain columns: y_true, y_prob.

    This is the primary diagnostic for catching dataset-level shortcuts.
    If the model has near-perfect recall on one source_dataset and poor
    recall on another, it has learned dataset-specific artefacts rather
    than fraud patterns.
    """
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
# GradCAM — verify the model is attending to fraud regions, not backgrounds
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Lightweight GradCAM for a ResNet-style model.

    Usage
    -----
    cam     = GradCAM(model, target_layer=model.layer4[-1])
    heatmap = cam(image_tensor.unsqueeze(0).to(device))   # (H, W) numpy array
    overlay = cam.overlay(pil_image, heatmap)

    Why this matters
    ----------------
    If the heatmap consistently lights up the document border or background
    rather than the signature/stamp/seal area, the model is relying on a
    dataset artefact (e.g., fake images always have white borders) instead
    of genuine fraud cues. Catching this early saves a lot of retraining.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model        = model
        self.target_layer = target_layer
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_, __, output):
            self._activations = output.detach()

        def backward_hook(_, __, grad_output):
            self._gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        self.model.eval()
        x = x.requires_grad_(True)

        logits = self.model(x)
        self.model.zero_grad()
        logits.sum().backward()

        grads  = self._gradients          # (B, C, H, W)
        acts   = self._activations        # (B, C, H, W)
        weights = grads.mean(dim=(2, 3), keepdim=True)   # global average pool

        cam = (weights * acts).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam

    @staticmethod
    def overlay(pil_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
        """Blend the heatmap onto the original PIL image for visual inspection."""
        import matplotlib.cm as cm

        h, w   = pil_image.size[1], pil_image.size[0]
        resized = Image.fromarray(
            (cm.jet(np.array(
                Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                    (w, h), Image.BILINEAR
                )
            ))[:, :, :3] * 255).astype(np.uint8)
        )
        return Image.blend(pil_image.convert("RGB"), resized, alpha=alpha)