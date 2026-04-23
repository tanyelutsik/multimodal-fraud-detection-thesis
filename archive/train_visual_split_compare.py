from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torchvision import models, transforms

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ExperimentConfig:
    old_train_csv: str = str(PROCESSED_DIR / "image_train.csv")
    old_test_csv: str = str(PROCESSED_DIR / "image_test.csv")
    new_train_csv: str = str(PROCESSED_DIR / "image_train_group_test.csv")
    new_test_csv: str = str(PROCESSED_DIR / "image_test_group_test.csv")
    image_col: str = "image_path"
    label_col: str = "image_class"
    group_col: str = "group_key"
    batch_size: int = 32
    embedding_cache_dir: str = str(PROCESSED_DIR / "embedding_cache")
    results_path: str = str(RESULTS_DIR / "visual_split_compare_results.json")


def label_to_int(series: pd.Series) -> pd.Series:
    mapping = {
        "bona_fide": 0,
        "bonafide": 0,
        "genuine": 0,
        "clean": 0,
        "forged": 1,
        "fraud": 1,
        "fake": 1,
        "attack": 1,
    }
    cleaned = series.astype(str).str.strip().str.lower()
    mapped = cleaned.map(mapping)

    if mapped.isna().any():
        unknown = sorted(cleaned[mapped.isna()].unique().tolist())
        raise ValueError(f"Unknown labels found: {unknown}")

    return mapped.astype(int)


def load_split(csv_path: Path, image_col: str, label_col: str, group_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()

    required = {image_col, label_col, group_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {missing}")

    df["label_int"] = label_to_int(df[label_col])

    if not df[image_col].apply(lambda p: Path(p).exists()).all():
        bad = df.loc[~df[image_col].apply(lambda p: Path(p).exists()), image_col].head(5).tolist()
        raise FileNotFoundError(f"Some image files do not exist. Examples: {bad}")

    return df


def build_feature_extractor() -> Tuple[nn.Module, transforms.Compose]:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(DEVICE)
    feature_extractor.eval()

    transform = weights.transforms()
    return feature_extractor, transform


def load_image(path: str, transform: transforms.Compose) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return transform(image)


@torch.no_grad()
def extract_embeddings(
    df: pd.DataFrame,
    image_col: str,
    batch_size: int,
    feature_extractor: nn.Module,
    transform: transforms.Compose,
) -> np.ndarray:
    tensors: List[torch.Tensor] = []
    embeddings: List[np.ndarray] = []

    image_paths = df[image_col].tolist()

    for idx, image_path in enumerate(image_paths, start=1):
        tensors.append(load_image(image_path, transform))

        if len(tensors) == batch_size or idx == len(image_paths):
            batch = torch.stack(tensors).to(DEVICE)
            feats = feature_extractor(batch)
            feats = feats.view(feats.size(0), -1)
            embeddings.append(feats.cpu().numpy())
            tensors = []

    return np.vstack(embeddings)


def get_or_create_embeddings(
    df: pd.DataFrame,
    split_name: str,
    cfg: ExperimentConfig,
    feature_extractor: nn.Module,
    transform: transforms.Compose,
) -> np.ndarray:
    cache_dir = Path(cfg.embedding_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"{split_name}_resnet18_embeddings_{len(df)}.npy"

    if cache_path.exists():
        print(f"Loading cached embeddings: {cache_path}")
        return np.load(cache_path)

    print(f"Extracting embeddings for {split_name}...")
    X = extract_embeddings(
        df=df,
        image_col=cfg.image_col,
        batch_size=cfg.batch_size,
        feature_extractor=feature_extractor,
        transform=transform,
    )
    np.save(cache_path, X)
    print(f"Saved embeddings to: {cache_path}")
    return X


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }


def tune_with_random_cv(X_train: np.ndarray, y_train: np.ndarray) -> GridSearchCV:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    search.fit(X_train, y_train)
    return search


def overlap_counts(train_df: pd.DataFrame, test_df: pd.DataFrame, group_col: str) -> Dict[str, int]:
    train_groups = set(train_df[group_col])
    test_groups = set(test_df[group_col])
    return {"train_test_overlap": int(len(train_groups & test_groups))}


def run_one_setup(
    name: str,
    train_csv: Path,
    test_csv: Path,
    cfg: ExperimentConfig,
    feature_extractor: nn.Module,
    transform: transforms.Compose,
) -> Dict:
    print("\n" + "=" * 70)
    print(f"RUNNING SETUP: {name}")
    print("=" * 70)

    train_df = load_split(train_csv, cfg.image_col, cfg.label_col, cfg.group_col)
    test_df = load_split(test_csv, cfg.image_col, cfg.label_col, cfg.group_col)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("Train class counts:")
    print(train_df["label_int"].value_counts())
    print("Test class counts:")
    print(test_df["label_int"].value_counts())

    overlap = overlap_counts(train_df, test_df, cfg.group_col)
    print("Train-Test overlap:", overlap["train_test_overlap"])

    X_train = get_or_create_embeddings(train_df, f"{name}_train", cfg, feature_extractor, transform)
    X_test = get_or_create_embeddings(test_df, f"{name}_test", cfg, feature_extractor, transform)

    y_train = train_df["label_int"].to_numpy()
    y_test = test_df["label_int"].to_numpy()

    print(f"\nTuning baseline for {name} with random CV...")
    search = tune_with_random_cv(X_train, y_train)

    test_metrics = evaluate_model(search.best_estimator_, X_test, y_test)

    result = {
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_groups": int(train_df[cfg.group_col].nunique()),
        "test_groups": int(test_df[cfg.group_col].nunique()),
        "train_test_overlap": overlap["train_test_overlap"],
        "best_params": search.best_params_,
        "best_cv_score_f1": float(search.best_score_),
        "test_metrics": test_metrics,
    }

    print("\nBest params:", search.best_params_)
    print("Best CV F1:", round(search.best_score_, 4))
    print("Test metrics:", test_metrics)

    return result


def main() -> None:
    cfg = ExperimentConfig()

    print("Using device:", DEVICE)
    feature_extractor, transform = build_feature_extractor()

    old_result = run_one_setup(
        name="old_row_split",
        train_csv=Path(cfg.old_train_csv),
        test_csv=Path(cfg.old_test_csv),
        cfg=cfg,
        feature_extractor=feature_extractor,
        transform=transform,
    )

    new_result = run_one_setup(
        name="new_group_split",
        train_csv=Path(cfg.new_train_csv),
        test_csv=Path(cfg.new_test_csv),
        cfg=cfg,
        feature_extractor=feature_extractor,
        transform=transform,
    )

    results = {
        "config": asdict(cfg),
        "old_row_split": old_result,
        "new_group_split": new_result,
    }

    results_path = Path(cfg.results_path)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    for setup_name, result in [("OLD ROW SPLIT", old_result), ("NEW GROUP SPLIT", new_result)]:
        print(f"\n{setup_name}")
        print("Train rows:", result["train_rows"])
        print("Test rows:", result["test_rows"])
        print("Train groups:", result["train_groups"])
        print("Test groups:", result["test_groups"])
        print("Train-Test overlap:", result["train_test_overlap"])
        print("Best params:", result["best_params"])
        print("Best CV F1:", round(result["best_cv_score_f1"], 4))
        print("Test metrics:", result["test_metrics"])

    print(f"\nSaved results to: {results_path}")


if __name__ == "__main__":
    main()