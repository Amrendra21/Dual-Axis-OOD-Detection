"""
Validity detector architecture and dataset utilities for Dual-Axis OOD Detection.

Task formulation:
- Negative class (0.0): Valid in-distribution graphs (id_train / id_val).
- Positive class (1.0): Corrupted molecular graphs (corrupt_train / corrupt_val).
- Output score: validity_score in [0.0, 1.0], where HIGHER means more corrupted / OOD.
- Family policy:
    - Trained on: id_train + corrupt_train (Families A & B only).
    - Validated on: id_val + corrupt_val (Families A & B reported separately).
    - Held-out: Family C exists ONLY in corrupt_test and is strictly EXCLUDED from all training and validation.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.data import BatchGraphs, MolecularDataset, collate_molecular_graphs
from src.toxicity_model import GINEToxicityModel


class ValidityDataset(Dataset):
    """
    Combines valid and corrupted splits for validity classification.

    Preserves:
    - Graph UID
    - Target label: 0.0 for valid, 1.0 for corrupted
    - Family tag: 'valid' for ID graphs, 'A' or 'B' for corrupted graphs
    """

    def __init__(
        self,
        valid_split: str,
        corrupt_split: str,
        root_dir: str = "processed_v3",
    ):
        self.valid_dataset = MolecularDataset(valid_split, root_dir=root_dir)
        self.corrupt_dataset = MolecularDataset(corrupt_split, root_dir=root_dir)

        # Enforce strict holdout of Family C
        if "test" in corrupt_split.lower():
            raise ValueError(
                f"Attempted to initialize ValidityDataset with test split '{corrupt_split}'. "
                f"Test splits and Family C are strictly forbidden during model fitting and validation!"
            )

        self.graphs = []
        self.labels = []
        self.uids = []

        # Valid graphs -> label 0.0
        for g in self.valid_dataset:
            self.graphs.append(g)
            self.labels.append(0.0)
            self.uids.append(g["uid"])

        # Corrupted graphs -> label 1.0
        for g in self.corrupt_dataset:
            self.graphs.append(g)
            self.labels.append(1.0)
            self.uids.append(g["uid"])

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], float, str]:
        return self.graphs[idx], self.labels[idx], self.uids[idx]


def collate_validity_graphs(batch: List[Tuple[Dict[str, Any], float, str]]) -> Tuple[BatchGraphs, torch.Tensor]:
    """
    Collate a list of (graph_dict, label, uid) tuples into BatchGraphs and a 1D target label tensor.
    """
    graphs = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    batched_graphs = collate_molecular_graphs(graphs)
    return batched_graphs, labels


class GINEValidityDetector(nn.Module):
    """
    Edge-aware Graph Isomorphism Network for Validity Detection.

    Reuses the edge-aware GINE encoder architecture from the toxicity model to extract
    valence-aware graph representations, and adds a separate validity classification head.

    Direction:
        Returns validity_score in [0.0, 1.0].
        Higher score = higher probability of structural corruption / OOD.
    """

    def __init__(
        self,
        encoder: GINEToxicityModel,
        embed_dim: int = 256,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim
        
        # Dedicated validity detection classification head
        self.validity_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch_data: BatchGraphs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning unnormalized validity logits and latent embeddings.

        Returns:
            logits: Shape [batch_size], where positive values indicate corrupted graph.
            embeddings: Shape [batch_size, embed_dim]
        """
        embeddings = self.encoder.get_embeddings(batch_data)
        logits = self.validity_head(embeddings).squeeze(-1)
        return logits, embeddings

    def score_batch(self, batch_data: BatchGraphs) -> torch.Tensor:
        """
        Predict validity scores in [0.0, 1.0] for an input batch.

        Guarantees:
        - Output shape [batch_size].
        - Consistent direction: HIGHER = MORE CORRUPTED / OOD.
        - Preserves strict UID order of batch_data.uid.
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(batch_data)
            scores = torch.sigmoid(logits)
        return scores
