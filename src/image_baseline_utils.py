from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


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


@dataclass
class SplitBundle:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


class ImageCSVDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_col: str = "image_path",
        label_col: str = "label_int",
        transform=None,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = row[self.image_col]
        label = float(row[self.label_col])

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


def label_to_int(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.lower()
    mapped = cleaned.map(LABEL_MAP)

    if mapped.isna().any():
        unknown = sorted(cleaned[mapped.isna()].unique().tolist())
        raise ValueError(f"Unknown labels found: {unknown}")

    return mapped.astype(int)


def validate_split_df(
    df: pd.DataFrame,
    split_name: str,
    image_col: str = "image_path",
    label_col: str = "image_class",
    group_col: str = "group_key",
) -> pd.DataFrame:
    required = {image_col, label_col, group_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{split_name} missing required columns: {missing}")

    out = df.copy()
    out["label_int"] = label_to_int(out[label_col])

    missing_paths = out.loc[~out[image_col].apply(lambda p: Path(p).exists()), image_col]
    if len(missing_paths) > 0:
        examples = missing_paths.head(5).tolist()
        raise FileNotFoundError(f"{split_name} contains missing image files. Examples: {examples}")

    return out


def load_image_splits(
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
    image_col: str = "image_path",
    label_col: str = "image_class",
    group_col: str = "group_key",
) -> SplitBundle:
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    train_df = validate_split_df(train_df, "train", image_col, label_col, group_col)
    val_df = validate_split_df(val_df, "val", image_col, label_col, group_col)
    test_df = validate_split_df(test_df, "test", image_col, label_col, group_col)

    return SplitBundle(train_df=train_df, val_df=val_df, test_df=test_df)


def check_group_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str = "group_key",
) -> Dict[str, int]:
    train_groups = set(train_df[group_col])
    val_groups = set(val_df[group_col])
    test_groups = set(test_df[group_col])

    return {
        "train_val_overlap": len(train_groups & val_groups),
        "train_test_overlap": len(train_groups & test_groups),
        "val_test_overlap": len(val_groups & test_groups),
    }


def print_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = "image_class",
    group_col: str = "group_key",
) -> None:
    for name, df in [("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)]:
        print(f"\n{name}")
        print("Rows:", len(df))
        print("Groups:", df[group_col].nunique())
        print(df[label_col].value_counts(dropna=False))
        print((df[label_col].value_counts(normalize=True) * 100).round(2))

    overlap = check_group_overlap(train_df, val_df, test_df, group_col)
    print("\nGROUP OVERLAP")
    for k, v in overlap.items():
        print(f"{k}: {v}")


def build_transforms():
    weights = models.ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    return transform


def build_dataloaders(
    splits: SplitBundle,
    batch_size: int = 32,
    num_workers: int = 0,
    image_col: str = "image_path",
    label_col: str = "label_int",
):
    transform = build_transforms()

    train_ds = ImageCSVDataset(
        df=splits.train_df,
        image_col=image_col,
        label_col=label_col,
        transform=transform,
    )
    val_ds = ImageCSVDataset(
        df=splits.val_df,
        image_col=image_col,
        label_col=label_col,
        transform=transform,
    )
    test_ds = ImageCSVDataset(
        df=splits.test_df,
        image_col=image_col,
        label_col=label_col,
        transform=transform,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def build_model(pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)

    # Always train final layer
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def compute_pos_weight(train_df: pd.DataFrame, label_col: str = "label_int") -> torch.Tensor:
    y = train_df[label_col].to_numpy()
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    if n_pos == 0:
        raise ValueError("Training split has no positive samples.")

    pos_weight = n_neg / n_pos
    return torch.tensor([pos_weight], dtype=torch.float32)


def make_loss_fn(train_df: pd.DataFrame, device: str, label_col: str = "label_int"):
    pos_weight = compute_pos_weight(train_df, label_col=label_col).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

    return running_loss / max(total, 1)


@torch.no_grad()
def predict_probs(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()

    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels.numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels).astype(int)

    return y_true, y_prob


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, object]:
    y_pred = (y_prob >= threshold).astype(int)

    metrics: Dict[str, object] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    threshold: float = 0.5,
) -> Dict[str, object]:
    model.eval()

    running_loss = 0.0
    total = 0

    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device)
        labels_t = labels.to(device).unsqueeze(1)

        logits = model(images)
        loss = loss_fn(logits, labels_t)

        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

    avg_loss = running_loss / max(total, 1)
    y_true = np.concatenate(all_labels).astype(int)
    y_prob = np.concatenate(all_probs)

    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = float(avg_loss)
    return metrics


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epochs: int = 10,
    threshold: float = 0.5,
    early_stopping_patience: int = 3,
    model_save_path: str | Path | None = None,
) -> Tuple[nn.Module, pd.DataFrame]:
    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            threshold=threshold,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_roc_auc={val_metrics['roc_auc']:.4f}"
        )

        current_val_f1 = float(val_metrics["f1"])
        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0

            if model_save_path is not None:
                model_save_path = Path(model_save_path)
                model_save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, model_save_path)
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history_df = pd.DataFrame(history)
    return model, history_df


def run_test_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    threshold: float = 0.5,
) -> Dict[str, object]:
    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=device,
        threshold=threshold,
    )

    print("\nTEST METRICS")
    for key in ["loss", "accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"{key}: {test_metrics[key]:.4f}")

    print("confusion_matrix:")
    print(np.array(test_metrics["confusion_matrix"]))

    return test_metrics