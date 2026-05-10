from __future__ import annotations

"""
focal_loss.py
-------------
Focal Loss for binary classification on imbalanced datasets.

Why focal loss for this task
-----------------------------
The mixed-all dataset has 64% fraud labels. Standard BCE and even
weighted BCE caused the model to collapse — predicting fraud for
everything (tab0_img0 F1 = 0.000).

Focal loss was introduced by Lin et al. (2017) "Focal Loss for Dense
Object Detection" (RetinaNet paper) specifically to address class
imbalance. It down-weights easy examples so the model focuses on
hard ones.

Formula
-------
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

Where:
    p_t   = sigmoid probability for the true class
    gamma = focusing parameter (0 = standard BCE, 2 = recommended)
    alpha = class weight (equivalent to pos_weight in BCE)

Intuition
---------
When the model correctly predicts a fraud sample with high confidence
(p_t close to 1), the factor (1 - p_t)^gamma is close to 0, so the
loss contribution is small. The model focuses more on uncertain/hard
examples instead of confidently-predicted easy ones.

Reference
---------
Lin et al. (2017). Focal Loss for Dense Object Detection. ICCV 2017.
https://arxiv.org/abs/1708.02002
"""

import torch
from torch import nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    Focal loss for binary classification.

    Parameters
    ----------
    gamma : float
        Focusing parameter. Higher = more focus on hard examples.
        0.0  = standard weighted BCE
        1.0  = mild focus
        2.0  = recommended default (Lin et al.)
        5.0  = strong focus

    alpha : float | None
        Weight for the positive class (fraud = label 1).
        Equivalent to pos_weight in BCEWithLogitsLoss.
        None = no class weighting, only focusing.
        0.25 = down-weight fraud if majority class
        Use n_neg/n_pos for automatic balancing (same as BCE pos_weight).

    reduction : str
        "mean" | "sum" | "none"
    """

    def __init__(
        self,
        gamma:     float = 2.0,
        alpha:     float | None = None,
        reduction: str   = "mean",
    ) -> None:
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : raw model output before sigmoid, shape (N,) or (N,1)
        targets : binary labels {0, 1}, same shape as logits
        """
        logits  = logits.view(-1)
        targets = targets.view(-1)

        # Standard BCE per sample (no reduction)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # p_t = probability of the true class
        prob = torch.sigmoid(logits)
        p_t  = prob * targets + (1 - prob) * (1 - targets)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma

        # Alpha weighting (class balance)
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def make_focal_loss(
    train_df,
    device:    str,
    gamma:     float = 2.0,
    use_alpha: bool  = True,
    label_col: str   = "final_label",
) -> BinaryFocalLoss:
    """
    Build a BinaryFocalLoss for the training set.

    alpha is auto-computed as n_neg / (n_neg + n_pos) — the proportion
    of negative samples. This down-weights the majority class (fraud)
    similar to pos_weight in BCE.

    Parameters
    ----------
    gamma      : focusing parameter (2.0 recommended)
    use_alpha  : whether to apply class-frequency alpha weighting
    """
    import numpy as np
    y     = train_df[label_col].values
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    n_tot = n_pos + n_neg

    if use_alpha:
        # alpha = proportion of negatives = weight for positive class
        alpha = n_neg / n_tot
        print(f"  FocalLoss gamma={gamma}  alpha={alpha:.3f}  "
              f"(n_neg={n_neg}, n_pos={n_pos})")
    else:
        alpha = None
        print(f"  FocalLoss gamma={gamma}  alpha=None  "
              f"(n_neg={n_neg}, n_pos={n_pos})")

    return BinaryFocalLoss(gamma=gamma, alpha=alpha).to(device)
