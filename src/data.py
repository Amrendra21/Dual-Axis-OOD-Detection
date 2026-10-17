"""
Dataset loader and batching utilities for the Dual-Axis OOD Molecule Dataset.

Supports PyTorch and PyTorch Geometric batching representations while strictly preserving:
- Graph UIDs
- Atom features [N, 30]
- Edge indices [2, E]
- Edge attributes [E, 6]
- Multitask toxicity labels [1, 12] or [B, 12] with NaNs preserved
- Existing split boundaries without leakage
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset, DataLoader


# Documented 10 canonical splits in processed_v3
VALID_SPLIT_NAMES = (
    "id_train",
    "id_val",
    "id_test",
    "novel_scaffold_val",
    "novel_scaffold_test",
    "novel_dataset_val",
    "novel_dataset_test",
    "corrupt_train",
    "corrupt_val",
    "corrupt_test",
)


@dataclass
class BatchGraphs:
    """
    Standard disjoint-union graph batch container (PyG-compatible convention).

    Attributes:
        uid: List of graph identifiers in order of graphs in batch.
        x: Concatenated node features [total_nodes, 30].
        edge_index: Shifted edge indices [2, total_edges].
        edge_attr: Concatenated edge attributes [total_edges, 6].
        y: Stacked graph-level labels [batch_size, 12] (NaNs preserved).
        batch: Node-to-graph membership tensor [total_nodes], values in [0, batch_size - 1].
        ptr: Prefix sum of node offsets [batch_size + 1].
        num_graphs: Number of graphs in the batch.
    """
    uid: List[str]
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    y: torch.Tensor
    batch: torch.Tensor
    ptr: torch.Tensor
    num_graphs: int

    def to(self, device: Union[str, torch.device]) -> "BatchGraphs":
        """Move all tensors in the batch to the target device."""
        return BatchGraphs(
            uid=self.uid,
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device),
            y=self.y.to(device),
            batch=self.batch.to(device),
            ptr=self.ptr.to(device),
            num_graphs=self.num_graphs,
        )

    def __len__(self) -> int:
        return self.num_graphs

    def __getitem__(self, item: str) -> Any:
        """Allow dictionary-style access for PyG / generic loader compatibility."""
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(f"Key '{item}' not found in BatchGraphs")


def collate_molecular_graphs(graph_list: Sequence[Dict[str, Any]]) -> BatchGraphs:
    """
    Collate a sequence of individual molecular graph dictionaries into a BatchGraphs object.

    Handles disjoint union graph batching:
    - Node attributes concatenated along dimension 0.
    - Edge indices shifted by cumulative node count offsets.
    - Edge attributes concatenated along dimension 0.
    - Labels stacked along dimension 0 to [batch_size, 12] with NaNs preserved.
    - Preserves individual UIDs in order.
    """
    if not graph_list:
        raise ValueError("Cannot collate an empty list of graphs.")

    uids: List[str] = []
    x_list: List[torch.Tensor] = []
    edge_index_list: List[torch.Tensor] = []
    edge_attr_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []
    batch_list: List[torch.Tensor] = []

    node_offset = 0
    ptr = [0]

    for i, g in enumerate(graph_list):
        uid = g.get("uid")
        x = g.get("x")
        edge_index = g.get("edge_index")
        edge_attr = g.get("edge_attr")
        y = g.get("y")

        if uid is None or x is None or edge_index is None or edge_attr is None or y is None:
            raise ValueError(f"Graph at index {i} is missing one or more required keys.")

        num_nodes = x.size(0)
        num_edges = edge_attr.size(0)

        uids.append(uid)
        x_list.append(x)
        # Shift edge indices by the current cumulative node offset
        edge_index_list.append(edge_index + node_offset)
        edge_attr_list.append(edge_attr)
        y_list.append(y.view(1, -1) if y.ndim == 1 else y)

        batch_list.append(torch.full((num_nodes,), i, dtype=torch.int64))
        node_offset += num_nodes
        ptr.append(node_offset)

    return BatchGraphs(
        uid=uids,
        x=torch.cat(x_list, dim=0),
        edge_index=torch.cat(edge_index_list, dim=1) if x_list else torch.empty((2, 0), dtype=torch.int64),
        edge_attr=torch.cat(edge_attr_list, dim=0),
        y=torch.cat(y_list, dim=0),
        batch=torch.cat(batch_list, dim=0),
        ptr=torch.tensor(ptr, dtype=torch.int64),
        num_graphs=len(graph_list),
    )


class MolecularDataset(Dataset):
    """
    In-memory Dataset for loading preprocessed molecular graph tensors (.pt) from processed_v3.

    Args:
        split_name: Name of the split, e.g. 'id_train', 'id_val', 'corrupt_test', etc.
        root_dir: Path to repository root or processed_v3 directory.
        transform: Optional callable transform to apply to each graph dictionary.
    """

    def __init__(
        self,
        split_name: str,
        root_dir: Union[str, Path] = "processed_v3",
        transform: Optional[Any] = None,
    ):
        if split_name not in VALID_SPLIT_NAMES:
            raise ValueError(
                f"Invalid split_name '{split_name}'. Must be one of {VALID_SPLIT_NAMES}"
            )

        self.split_name = split_name
        self.transform = transform

        root_path = Path(root_dir)
        # Handle cases where root_dir points to project root vs processed_v3 directly
        if (root_path / "processed_v3").exists():
            root_path = root_path / "processed_v3"

        graphs_dir = root_path / "graphs"
        file_path = graphs_dir / f"{split_name}.pt"

        if not file_path.exists():
            raise FileNotFoundError(f"Graph file not found: {file_path.resolve()}")

        # Load graphs safely using PyTorch weights_only
        self.graphs: List[Dict[str, Any]] = torch.load(file_path, weights_only=True)
        self.file_path = file_path

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.graphs[idx]
        if self.transform is not None:
            item = self.transform(item)
        return item


def get_dataloader(
    split_name: str,
    root_dir: Union[str, Path] = "processed_v3",
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Convenience factory to create a DataLoader for any split in processed_v3.

    Args:
        split_name: Name of the split (e.g., 'id_train', 'id_val', etc.)
        root_dir: Directory containing processed_v3 or project root
        batch_size: Number of graphs per batch
        shuffle: Whether to shuffle graphs
        num_workers: Worker processes for data loading
        pin_memory: Pin memory for GPU transfer if applicable

    Returns:
        torch.utils.data.DataLoader yielding BatchGraphs objects
    """
    dataset = MolecularDataset(split_name=split_name, root_dir=root_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_molecular_graphs,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
