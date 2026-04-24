from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torchvision import models

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OLD_TRAIN = PROCESSED_DIR / "image_train.csv"
OLD_TEST = PROCESSED_DIR / "image_test.csv"

NEW_TRAIN = PROCESSED_DIR / "image_train_group_test.csv"
NEW_TEST = PROCESSED_DIR / "image_test_group_test.csv"

RESULTS_PATH = RESULTS_DIR / "visual_pipeline_compare_results.json"


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


def load_split(csv_path: Path, image_col="image_path", label_col="image_class", group_col="group_key") -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()

    required = {image_col, label_col, group_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {missing}")

    df["label_int"] = label_to_int(df[label_col])

    bad_paths = df.loc[~df[image_col].apply(lambda p: Path(p).exists()), image_col]
    if len(bad_paths) > 0:
        raise FileNotFoundError(f"Missing image files. Examples: {bad_paths.head(5).tolist()}")

    return df


def build_feature_extractor() -> Tuple[nn.Module, object]:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(DEVICE)
    feature_extractor.eval()

    transform = weights.transforms()
    return feature_extractor, transform


def load_image(path: str, transform) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return transform(image)


@torch.no_grad()
def extract_embeddings(
    df: pd.DataFrame,
    image_col: str,
    batch_size: int,
    feature_extractor: nn.Module,
    transform,
) -> np.ndarray:
    tensors: List[torch.Tensor] = []
    embeddings: List[np.ndarray] = []

    paths = df[image_col].tolist()

    for idx, image_path in enumerate(paths, start=1):
        tensors.append(load_image(image_path, transform))

        if len(tensors) == batch_size or idx == len(paths):
            batch = torch.stack(tensors).to(DEVICE)
            feats = feature_extractor(batch)
            feats = feats.view(feats.size(0), -1)
            embeddings.append(feats.cpu().numpy())
            tensors = []

    return np.vstack(embeddings)


def get_or_create_embeddings(
    df: pd.DataFrame,
    split_name: str,
    feature_extractor: nn.Module,
    transform,
    batch_size: int = 32,
) -> np.ndarray:
    cache_dir = PROCESSED_DIR / "embedding_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"{split_name}_resnet18_embeddings_{len(df)}.npy"

    if cache_path.exists():
        print(f"Loading cached embeddings: {cache_path}")
        return np.load(cache_path)

    print(f"Extracting embeddings for {split_name}...")
    X = extract_embeddings(
        df=df,
        image_col="image_path",
        batch_size=batch_size,
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


def tune_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    cv_type: str,
) -> GridSearchCV:
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

    if cv_type == "random":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        fit_kwargs = {}
    elif cv_type == "group":
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        fit_kwargs = {"groups": groups}
    else:
        raise ValueError("cv_type must be 'random' or 'group'")

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    search.fit(X_train, y_train, **fit_kwargs)
    return search


def train_test_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame, group_col="group_key") -> int:
    return len(set(train_df[group_col]) & set(test_df[group_col]))


def run_pipeline(
    name: str,
    train_csv: Path,
    test_csv: Path,
    cv_type: str,
    feature_extractor: nn.Module,
    transform,
) -> Dict:
    print("\n" + "=" * 70)
    print(f"RUNNING: {name}")
    print("=" * 70)

    train_df = load_split(train_csv)
    test_df = load_split(test_csv)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("Train class counts:")
    print(train_df["label_int"].value_counts())
    print("Test class counts:")
    print(test_df["label_int"].value_counts())

    overlap = train_test_overlap(train_df, test_df)
    print("Train-Test overlap:", overlap)

    X_train = get_or_create_embeddings(train_df, f"{name}_train", feature_extractor, transform)
    X_test = get_or_create_embeddings(test_df, f"{name}_test", feature_extractor, transform)

    y_train = train_df["label_int"].to_numpy()
    y_test = test_df["label_int"].to_numpy()
    groups_train = train_df["group_key"].to_numpy()

    print(f"\nTuning with {cv_type} CV...")
    search = tune_model(X_train, y_train, groups_train, cv_type=cv_type)
    test_metrics = evaluate_model(search.best_estimator_, X_test, y_test)

    result = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_groups": int(train_df["group_key"].nunique()),
        "test_groups": int(test_df["group_key"].nunique()),
        "train_test_overlap": int(overlap),
        "cv_type": cv_type,
        "best_params": search.best_params_,
        "best_cv_score_f1": float(search.best_score_),
        "test_metrics": test_metrics,
    }

    print("\nBest params:", search.best_params_)
    print("Best CV F1:", round(search.best_score_, 4))
    print("Test metrics:", test_metrics)

    return result


def main() -> None:
    print("Using device:", DEVICE)
    feature_extractor, transform = build_feature_extractor()

    results = {}

    results["old_row_split_random_cv"] = run_pipeline(
        name="old_row_split_random_cv",
        train_csv=OLD_TRAIN,
        test_csv=OLD_TEST,
        cv_type="random",
        feature_extractor=feature_extractor,
        transform=transform,
    )

    results["new_group_split_group_cv"] = run_pipeline(
        name="new_group_split_group_cv",
        train_csv=NEW_TRAIN,
        test_csv=NEW_TEST,
        cv_type="group",
        feature_extractor=feature_extractor,
        transform=transform,
    )

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    for name, result in results.items():
        print(f"\n{name}")
        print("Train rows:", result["train_rows"])
        print("Test rows:", result["test_rows"])
        print("Train groups:", result["train_groups"])
        print("Test groups:", result["test_groups"])
        print("Train-Test overlap:", result["train_test_overlap"])
        print("CV type:", result["cv_type"])
        print("Best params:", result["best_params"])
        print("Best CV F1:", round(result["best_cv_score_f1"], 4))
        print("Test metrics:", result["test_metrics"])

    print(f"\nSaved results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()