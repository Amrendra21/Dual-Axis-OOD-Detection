"""
Regularized Mahalanobis Distance Novelty Estimator for Dual-Axis OOD Detection.

Formulation
-----------
Given a set of in-distribution (ID) reference embeddings E_train ∈ R^{N x D} extracted
from the frozen toxicity GNN backbone, we fit:
  - mean vector  μ ∈ R^D
  - regularized covariance matrix  Σ_reg = (1 - α) * Σ + α * σ²_avg * I

The Mahalanobis distance for a new graph embedding z is:
  d(z) = sqrt( (z - μ)^T  Σ_reg^{-1}  (z - μ) )

Score direction:
  novelty_score = d(z)  (higher = more novel / more OOD)

Permitted fitting splits
------------------------
  ONLY `id_train` (5,432 valid, in-distribution graphs) is permitted for fitting.
  `id_val`, `novel_*`, `corrupt_*`, and all test splits are strictly excluded from
  fitting. They may be used only for post-hoc evaluation.

Covariance estimation regime
-----------------------------
With N=5,432 samples and D=256 dimensions (N/D ≈ 21), the sample covariance matrix
is well-defined (N >> D) but benefits from mild regularization to avoid near-singularity
and ensure numerical stability of the Cholesky / inverse.

We use Ledoit-Wolf shrinkage (via scikit-learn) to automatically select the optimal
regularization coefficient α, giving a data-driven, reproducible estimator.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.data import BatchGraphs, MolecularDataset, collate_molecular_graphs, get_dataloader
from src.toxicity_model import GINEToxicityModel


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PERMITTED_FIT_SPLITS = ("id_train",)
"""Only `id_train` is a permitted fitting split for the novelty estimator."""

PERMITTED_EVAL_SPLITS = (
    "id_val",
    "id_test",
    "novel_scaffold_val",
    "novel_scaffold_test",
    "novel_dataset_val",
    "novel_dataset_test",
)
"""Splits that may be evaluated post-hoc but never used to fit the estimator."""


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    encoder: GINEToxicityModel,
    split_name: str,
    root_dir: Union[str, Path] = "processed_v3",
    batch_size: int = 256,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract 256-dimensional graph embeddings from the frozen toxicity GNN backbone
    for all graphs in the specified split.

    The encoder is placed in eval mode and no gradients are accumulated.
    UID order is strictly preserved across batches.

    Args:
        encoder:    Trained GINEToxicityModel; its weights are not modified.
        split_name: Dataset split (e.g., 'id_train', 'id_val').
        root_dir:   Path to the processed_v3 directory or repo root.
        batch_size: Number of graphs per forward pass batch.
        device:     Device to run embedding extraction on.

    Returns:
        embeddings: NumPy array of shape [N, embed_dim], float32.
        uids:       List of N UIDs in the same order as embeddings rows.

    Raises:
        ValueError: If any embeddings contain NaN or Inf.
    """
    encoder = encoder.to(device)
    encoder.eval()

    loader = get_dataloader(
        split_name=split_name,
        root_dir=root_dir,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    all_embeddings: List[np.ndarray] = []
    all_uids: List[str] = []

    for batch in loader:
        batch = batch.to(device)
        embs = encoder.get_embeddings(batch)  # [B, embed_dim]
        all_embeddings.append(embs.cpu().numpy())
        all_uids.extend(batch.uid)

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)

    # Verify numerical sanity
    if not np.all(np.isfinite(embeddings)):
        n_bad = int(np.sum(~np.isfinite(embeddings)))
        raise ValueError(
            f"Extracted embeddings for split '{split_name}' contain {n_bad} "
            f"NaN or Inf values. Check the encoder checkpoint for gradient issues."
        )

    logger.info(
        "Extracted %d embeddings (dim=%d) from split '%s'.",
        embeddings.shape[0], embeddings.shape[1], split_name,
    )
    return embeddings, all_uids


# ---------------------------------------------------------------------------
# Mahalanobis Novelty Estimator
# ---------------------------------------------------------------------------

class MahalanobisNoveltyEstimator:
    """
    Regularized Mahalanobis distance novelty estimator.

    Fits a Gaussian reference distribution on in-distribution (ID) embeddings
    using Ledoit-Wolf shrinkage-regularized covariance estimation.

    Usage::

        estimator = MahalanobisNoveltyEstimator()
        estimator.fit(train_embeddings)          # np.ndarray [N, D]
        scores = estimator.score(test_embeddings) # np.ndarray [M]
        estimator.save("checkpoints/novelty_mahal/estimator.npz")

        # Reload
        estimator2 = MahalanobisNoveltyEstimator.load("checkpoints/novelty_mahal/estimator.npz")
        scores2 = estimator2.score(test_embeddings)
        assert np.allclose(scores, scores2)

    Score direction:
        Higher score => more novel (farther from ID training distribution).
    """

    def __init__(self) -> None:
        self._mean: Optional[np.ndarray] = None          # [D]
        self._precision: Optional[np.ndarray] = None     # [D, D], inverse of Σ_reg
        self._shrinkage: Optional[float] = None
        self._n_train: Optional[int] = None
        self._embed_dim: Optional[int] = None
        self._fit_split: Optional[str] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        embeddings: np.ndarray,
        fit_split: str = "id_train",
    ) -> "MahalanobisNoveltyEstimator":
        """
        Fit the Mahalanobis estimator on in-distribution embeddings.

        Uses Ledoit-Wolf shrinkage to obtain a well-conditioned covariance
        estimate, then computes and stores the precision matrix (inverse).

        Args:
            embeddings: ID training embeddings, shape [N, D]. Must be finite.
            fit_split:  Name of the split used for fitting (logged only).

        Returns:
            self (for chaining).

        Raises:
            ValueError: If embeddings are not 2-D, contain non-finite values,
                        or if the precision matrix computation fails.
        """
        if fit_split not in PERMITTED_FIT_SPLITS:
            raise ValueError(
                f"Attempting to fit novelty estimator on '{fit_split}', which is not "
                f"in the permitted fitting splits: {PERMITTED_FIT_SPLITS}. "
                f"Only id_train is allowed!"
            )

        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D embedding array [N, D], got shape {embeddings.shape}."
            )
        if not np.all(np.isfinite(embeddings)):
            n_bad = int(np.sum(~np.isfinite(embeddings)))
            raise ValueError(
                f"Embedding array contains {n_bad} non-finite values (NaN/Inf). "
                f"Check the encoder output before fitting."
            )

        N, D = embeddings.shape
        logger.info("Fitting MahalanobisNoveltyEstimator on %d x %d embeddings.", N, D)

        # Compute mean
        mean = embeddings.mean(axis=0)  # [D]

        # Ledoit-Wolf shrinkage: use sklearn to select the optimal alpha, but
        # compute the precision ourselves for robust numerical invertibility.
        from sklearn.covariance import LedoitWolf
        from scipy.linalg import pinvh

        lw = LedoitWolf(assume_centered=False, store_precision=False)
        lw.fit(embeddings)

        # lw.covariance_ is already the Ledoit-Wolf regularized covariance
        covariance = lw.covariance_.astype(np.float64)   # [D, D]
        shrinkage = float(lw.shrinkage_)

        # Numerical stabilization on the COVARIANCE before inversion:
        # Add a small relative jitter (1e-6 * trace/D) to the diagonal.
        # This is provably PD and safe — it shrinks the condition number
        # while contributing < 1ppm of the total signal variance.
        trace_avg = float(np.trace(covariance)) / D
        jitter = max(1e-6 * trace_avg, 1e-10)
        covariance_reg = covariance + jitter * np.eye(D, dtype=np.float64)

        # Compute precision via pinvh (guaranteed symmetric PD output via
        # eigendecomposition, avoiding Cholesky failure on near-singular matrices)
        precision = pinvh(covariance_reg, lower=True)

        # Final sanity check: all eigenvalues must be strictly positive
        min_eigval = float(np.linalg.eigvalsh(precision).min())
        if min_eigval <= 0.0:
            raise ValueError(
                f"Precision matrix has non-positive minimum eigenvalue "
                f"({min_eigval:.6e}) after jitter stabilization "
                f"(shrinkage={shrinkage:.6f}, jitter={jitter:.2e}). "
                f"This is unexpected -- please report this with the dataset."
            )

        self._mean = mean.astype(np.float64)
        self._precision = precision
        self._shrinkage = shrinkage
        self._n_train = N
        self._embed_dim = D
        self._fit_split = fit_split
        self._is_fitted = True

        logger.info(
            "Fit complete. Ledoit-Wolf shrinkage=%.6f, embed_dim=%d, n_train=%d.",
            shrinkage, D, N,
        )
        return self

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute Mahalanobis novelty scores for query embeddings.

        Args:
            embeddings: Query embeddings, shape [M, D]. Must match embed_dim.

        Returns:
            scores: 1D array of shape [M], dtype float32.
                    Higher values indicate greater novelty (farther from ID distribution).

        Raises:
            RuntimeError: If estimator has not been fitted yet.
            ValueError: If embedding dimension does not match fitted dimension.
        """
        self._check_is_fitted()

        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D embedding array [M, D], got shape {embeddings.shape}."
            )
        if embeddings.shape[1] != self._embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch: estimator fitted on D={self._embed_dim}, "
                f"but input has D={embeddings.shape[1]}."
            )
        if not np.all(np.isfinite(embeddings)):
            n_bad = int(np.sum(~np.isfinite(embeddings)))
            raise ValueError(
                f"Query embeddings contain {n_bad} non-finite values (NaN/Inf)."
            )

        # Center embeddings
        centered = embeddings.astype(np.float64) - self._mean[np.newaxis, :]  # [M, D]

        # Mahalanobis distance:  sqrt( z^T Σ^{-1} z )  for each row z
        # Efficient: compute  Λz = Σ^{-1} z  via matmul, then dot with z
        prec_centered = centered @ self._precision          # [M, D]
        sq_distances = np.sum(centered * prec_centered, axis=1)  # [M]

        # Guard against numerical near-zero negative values
        sq_distances = np.clip(sq_distances, a_min=0.0, a_max=None)
        distances = np.sqrt(sq_distances).astype(np.float32)

        return distances

    def score_single(self, embedding: np.ndarray) -> float:
        """Score a single embedding vector [D] -> scalar float."""
        return float(self.score(embedding[np.newaxis, :])[0])

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """
        Save fitted estimator parameters to a compressed NumPy archive.

        Args:
            path: Output path (extension .npz recommended).

        Raises:
            RuntimeError: If estimator has not been fitted yet.
        """
        self._check_is_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            mean=self._mean,
            precision=self._precision,
            shrinkage=np.array([self._shrinkage], dtype=np.float64),
            n_train=np.array([self._n_train], dtype=np.int64),
            embed_dim=np.array([self._embed_dim], dtype=np.int64),
        )
        logger.info("MahalanobisNoveltyEstimator saved to '%s'.", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MahalanobisNoveltyEstimator":
        """
        Load a previously saved estimator from an .npz file.

        Args:
            path: Path to the saved .npz file.

        Returns:
            A fitted MahalanobisNoveltyEstimator instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Novelty estimator file not found: {path}")

        data = np.load(str(path), allow_pickle=False)
        estimator = cls()
        estimator._mean = data["mean"]
        estimator._precision = data["precision"]
        estimator._shrinkage = float(data["shrinkage"][0])
        estimator._n_train = int(data["n_train"][0])
        estimator._embed_dim = int(data["embed_dim"][0])
        estimator._fit_split = "id_train"  # Always fitted on id_train
        estimator._is_fitted = True

        logger.info(
            "Loaded MahalanobisNoveltyEstimator from '%s' "
            "(n_train=%d, embed_dim=%d, shrinkage=%.6f).",
            path, estimator._n_train, estimator._embed_dim, estimator._shrinkage,
        )
        return estimator

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "MahalanobisNoveltyEstimator is not fitted. "
                "Call .fit(embeddings) before scoring."
            )

    @property
    def is_fitted(self) -> bool:
        """Whether the estimator has been fitted."""
        return self._is_fitted

    @property
    def embed_dim(self) -> Optional[int]:
        """Embedding dimension expected by the fitted estimator."""
        return self._embed_dim

    @property
    def n_train(self) -> Optional[int]:
        """Number of training samples used to fit the estimator."""
        return self._n_train

    @property
    def shrinkage(self) -> Optional[float]:
        """Ledoit-Wolf shrinkage coefficient selected during fitting."""
        return self._shrinkage

    def summary(self) -> Dict:
        """Return a serializable summary dictionary of the fitted estimator."""
        self._check_is_fitted()
        return {
            "estimator": "MahalanobisNoveltyEstimator",
            "covariance_regularization": "Ledoit-Wolf shrinkage",
            "fit_split": self._fit_split,
            "n_train": self._n_train,
            "embed_dim": self._embed_dim,
            "shrinkage_coefficient": self._shrinkage,
            "score_direction": "higher = more novel (greater distance from ID distribution)",
        }

    def __repr__(self) -> str:
        if not self._is_fitted:
            return "MahalanobisNoveltyEstimator(not fitted)"
        return (
            f"MahalanobisNoveltyEstimator("
            f"fitted=True, n_train={self._n_train}, "
            f"embed_dim={self._embed_dim}, shrinkage={self._shrinkage:.6f})"
        )
