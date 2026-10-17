"""
Step 7: ECFP4 / Maximum Tanimoto Similarity Novelty Estimator.

Standard implementation:
- 2048-bit Morgan Fingerprint (radius=2, corresponding to ECFP4) via RDKit.
- Fitted reference database: SMILES from id_train only.
- Novelty score formulation:
    novelty_score = 1.0 - max_{m in id_train} Tanimoto(query_mol, m)
- Direction:
    Higher score in [0.0, 1.0] indicates greater novelty (lower structural similarity to training library).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator

logger = logging.getLogger(__name__)

# Permitted fitting splits: strictly id_train
PERMITTED_FIT_SPLITS = ("id_train",)


class ECFP4TanimotoNoveltyEstimator:
    """
    ECFP4 Maximum Tanimoto Distance Novelty Estimator.

    Extracts 2048-bit Morgan fingerprints (radius=2, equivalent to ECFP4)
    and evaluates the minimum Tanimoto distance (1 - maximum similarity)
    against an in-distribution reference database.
    """

    def __init__(
        self,
        radius: int = 2,
        n_bits: int = 2048,
    ) -> None:
        self.radius = radius
        self.n_bits = n_bits
        self._generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )
        self._ref_uids: List[str] = []
        self._ref_fps: List[Any] = []
        self._fit_split: Optional[str] = None
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def n_ref(self) -> int:
        return len(self._ref_fps)

    def _smiles_to_fp(self, smiles: str) -> Any:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Failed to parse SMILES: '{smiles}'")
        return self._generator.GetFingerprint(mol)

    def fit(
        self,
        smiles_list: Sequence[str],
        uids: Sequence[str],
        fit_split: str = "id_train",
    ) -> "ECFP4TanimotoNoveltyEstimator":
        """
        Build the reference fingerprint database from in-distribution molecules.

        Args:
            smiles_list: List of valid SMILES strings.
            uids: List of unique molecular graph IDs.
            fit_split: Split name (strictly 'id_train').
        """
        if fit_split not in PERMITTED_FIT_SPLITS:
            raise ValueError(
                f"Attempting to fit ECFP4 estimator on '{fit_split}', which is forbidden! "
                f"Only id_train is permitted: {PERMITTED_FIT_SPLITS}."
            )

        if len(smiles_list) != len(uids):
            raise ValueError(
                f"Length mismatch: {len(smiles_list)} SMILES vs {len(uids)} UIDs."
            )

        logger.info(
            "Fitting ECFP4TanimotoNoveltyEstimator on %d reference molecules (radius=%d, n_bits=%d)...",
            len(smiles_list),
            self.radius,
            self.n_bits,
        )

        fps = [self._smiles_to_fp(s) for s in smiles_list]
        self._ref_uids = list(uids)
        self._ref_fps = fps
        self._fit_split = fit_split
        self._is_fitted = True

        return self

    def score(self, query_smiles: Sequence[str]) -> np.ndarray:
        """
        Compute Tanimoto novelty score:
            score = 1.0 - max_{ref in id_train} TanimotoSimilarity(query, ref)

        Higher score = more novel (farther from training set).
        """
        if not self._is_fitted:
            raise RuntimeError("Estimator has not been fitted. Call .fit() first.")

        scores: List[float] = []
        for s in query_smiles:
            q_fp = self._smiles_to_fp(s)
            sims = DataStructs.BulkTanimotoSimilarity(q_fp, self._ref_fps)
            max_sim = max(sims) if len(sims) > 0 else 0.0
            scores.append(1.0 - max_sim)

        scores_arr = np.array(scores, dtype=np.float32)
        if not np.all(np.isfinite(scores_arr)):
            raise ValueError("Computed scores contain non-finite values.")
        return scores_arr

    def score_split_from_table(
        self,
        split_name: str,
        master_table_path: Union[str, Path] = "processed_v3/tables/master_table.csv",
    ) -> Tuple[List[str], np.ndarray]:
        """
        Extract molecules for a specific split from master_table.csv, preserving UID order.
        """
        master_df = pd.read_csv(master_table_path)
        sub = master_df[master_df["set_name"] == split_name].copy()
        if len(sub) == 0:
            raise ValueError(f"No records found for split '{split_name}'.")

        uids = sub["uid"].tolist()
        smiles_list = sub["smiles"].tolist()
        scores = self.score(smiles_list)
        return uids, scores
