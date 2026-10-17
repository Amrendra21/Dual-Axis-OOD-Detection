"""
Masked multi-task loss functions and sparse label evaluation metrics for Tox21.

Handles missing labels (~16-20% NaNs) safely:
- Excludes NaNs via binary mask.
- Never fills NaNs with zeros or treats them as negative labels.
- Normalizes strictly by the number of observed labels.
- Safely handles batches with zero observed labels without throwing exceptions or NaNs.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def masked_multitask_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    """
    Compute binary cross-entropy with logits over observed labels only.

    Args:
        logits: Unnormalized model predictions [batch_size, num_tasks]
        targets: Ground truth labels [batch_size, num_tasks], with missing values as NaN

    Returns:
        loss: Scalar loss tensor (0.0 if no observed labels present)
        num_observed: Total number of valid observed labels used in loss computation
    """
    if logits.shape != targets.shape:
        raise ValueError(f"Shape mismatch: logits {logits.shape} vs targets {targets.shape}")

    # Create mask of finite, observed labels
    mask = torch.isfinite(targets)
    num_observed = int(mask.sum().item())

    if num_observed == 0:
        # Safe fallback: zero loss with gradient tracking attached to logits
        return logits.sum() * 0.0, 0

    # Compute unreduced element-wise BCE loss
    # Replace NaNs in targets with 0.0 purely for F.binary_cross_entropy_with_logits calculation,
    # as the unobserved positions are zeroed out immediately by the mask multiplication.
    clean_targets = torch.where(mask, targets, torch.zeros_like(targets))
    element_losses = F.binary_cross_entropy_with_logits(logits, clean_targets, reduction="none")

    # Apply mask and normalize strictly by the count of observed labels
    masked_losses = element_losses * mask.float()
    loss = masked_losses.sum() / float(num_observed)

    return loss, num_observed


def compute_multitask_roc_auc(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
) -> Tuple[float, Dict[int, float]]:
    """
    Compute macro-averaged ROC-AUC across tasks with observed positive and negative labels.

    Args:
        y_true: Ground truth binary labels with NaNs [N, num_tasks]
        y_pred_probs: Predicted probabilities in [0, 1] [N, num_tasks]

    Returns:
        mean_auc: Mean ROC-AUC over tasks with at least one positive and one negative sample
        task_aucs: Dictionary of individual task AUC scores
    """
    num_tasks = y_true.shape[1]
    task_aucs = {}

    for task_idx in range(num_tasks):
        true_task = y_true[:, task_idx]
        pred_task = y_pred_probs[:, task_idx]

        # Filter out NaNs
        valid_mask = np.isfinite(true_task)
        if valid_mask.sum() == 0:
            continue

        clean_true = true_task[valid_mask]
        clean_pred = pred_task[valid_mask]

        # ROC-AUC requires both positive and negative classes
        unique_classes = np.unique(clean_true)
        if len(unique_classes) > 1 and (0 in unique_classes) and (1 in unique_classes):
            try:
                auc = float(roc_auc_score(clean_true, clean_pred))
                task_aucs[task_idx] = auc
            except ValueError:
                pass

    if len(task_aucs) == 0:
        return 0.5, {}

    mean_auc = float(np.mean(list(task_aucs.values())))
    return mean_auc, task_aucs
