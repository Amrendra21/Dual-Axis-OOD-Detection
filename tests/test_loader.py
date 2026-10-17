"""
Smoke test and validation script for the Dual-Axis OOD dataset loader.

Checks:
1. Loading an individual graph from an existing split.
2. Checking UID and tensor shapes of x, edge_index, edge_attr, and y.
3. Loading a small batch via DataLoader and BatchGraphs.
4. Confirming UIDs can be recovered for every graph in the batch.
5. Verifying node-feature, edge-feature, and label dimensions.
6. Verifying missing labels remain NaN.
7. Confirming loader preserves split assignments and counts without alteration.
8. Exits with 0 on complete pass, or nonzero on any failure.
"""

import sys
from pathlib import Path
import torch

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data import MolecularDataset, get_dataloader, VALID_SPLIT_NAMES


def run_smoke_test() -> None:
    print("=" * 60)
    print("RUNNING DATASET LOADER SMOKE TEST")
    print("=" * 60)

    # 1. Single Graph Test
    print("\n--- Test 1: Load single graph from id_train ---")
    ds_train = MolecularDataset(split_name="id_train", root_dir="processed_v3")
    print(f"Dataset id_train size: {len(ds_train)} graphs")
    assert len(ds_train) == 5432, f"Expected 5432 graphs in id_train, got {len(ds_train)}"

    g0 = ds_train[0]
    print(f"Graph 0 UID        : {g0['uid']}")
    print(f"Graph 0 x shape    : {g0['x'].shape} (dtype: {g0['x'].dtype})")
    print(f"Graph 0 edge_index : {g0['edge_index'].shape} (dtype: {g0['edge_index'].dtype})")
    print(f"Graph 0 edge_attr  : {g0['edge_attr'].shape} (dtype: {g0['edge_attr'].dtype})")
    print(f"Graph 0 y shape    : {g0['y'].shape} (dtype: {g0['y'].dtype})")

    assert g0["uid"] == "T21-00000", f"Expected UID T21-00000, got {g0['uid']}"
    assert g0["x"].ndim == 2 and g0["x"].shape[1] == 30, f"Expected x shape [N, 30], got {g0['x'].shape}"
    assert g0["edge_index"].ndim == 2 and g0["edge_index"].shape[0] == 2, f"Expected edge_index shape [2, E], got {g0['edge_index'].shape}"
    assert g0["edge_attr"].ndim == 2 and g0["edge_attr"].shape[1] == 6, f"Expected edge_attr shape [E, 6], got {g0['edge_attr'].shape}"
    assert g0["y"].ndim == 2 and g0["y"].shape[1] == 12, f"Expected y shape [1, 12], got {g0['y'].shape}"
    print("PASSED: Single graph schema verified.")

    # 2. Batch Loading Test
    print("\n--- Test 2: Batch loading via DataLoader ---")
    batch_size = 5
    loader = get_dataloader(split_name="id_train", root_dir="processed_v3", batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))

    print(f"Batch num_graphs   : {batch.num_graphs}")
    print(f"Batch UIDs         : {batch.uid}")
    print(f"Batch x shape      : {batch.x.shape}")
    print(f"Batch edge_index   : {batch.edge_index.shape}")
    print(f"Batch edge_attr    : {batch.edge_attr.shape}")
    print(f"Batch y shape      : {batch.y.shape}")
    print(f"Batch vector shape : {batch.batch.shape}")
    print(f"Batch ptr vector   : {batch.ptr.tolist()}")

    assert batch.num_graphs == batch_size, f"Expected {batch_size} graphs, got {batch.num_graphs}"
    assert len(batch.uid) == batch_size, f"Expected {batch_size} UIDs, got {len(batch.uid)}"
    assert batch.x.shape[1] == 30, f"Expected x node dim 30, got {batch.x.shape[1]}"
    assert batch.edge_index.shape[0] == 2, f"Expected edge_index shape [2, E], got {batch.edge_index.shape}"
    assert batch.edge_attr.shape[1] == 6, f"Expected edge_attr dim 6, got {batch.edge_attr.shape[1]}"
    assert batch.edge_index.shape[1] == batch.edge_attr.shape[0], "Edge index count != Edge attr count"
    assert batch.y.shape == (batch_size, 12), f"Expected y shape ({batch_size}, 12), got {batch.y.shape}"
    assert batch.batch.shape[0] == batch.x.shape[0], "Batch tensor length != total nodes"
    assert len(batch.ptr) == batch_size + 1, "Ptr length != batch_size + 1"
    print("PASSED: Batch collation and UID recovery verified.")

    # 3. Label Missingness Test
    print("\n--- Test 3: Missing labels remain NaN ---")
    # Verify NaNs in batch.y
    nan_count = torch.isnan(batch.y).sum().item()
    print(f"NaN count in batch labels: {nan_count} / {batch.y.numel()}")
    assert nan_count > 0, "Expected missing labels (NaN) in Tox21 training batch!"

    # Corrupted and SIDER graphs must have 100% NaN in y
    loader_corrupt = get_dataloader(split_name="corrupt_val", root_dir="processed_v3", batch_size=4, shuffle=False)
    batch_corrupt = next(iter(loader_corrupt))
    assert torch.isnan(batch_corrupt.y).all(), "Corrupted graphs must have 100% NaN in y labels!"
    print(f"Corrupted graphs labels NaN status: 100% NaN verified ({batch_corrupt.y.numel()} NaNs)")

    loader_sider = get_dataloader(split_name="novel_dataset_val", root_dir="processed_v3", batch_size=4, shuffle=False)
    batch_sider = next(iter(loader_sider))
    assert torch.isnan(batch_sider.y).all(), "SIDER novel_dataset graphs must have 100% NaN in y labels!"
    print(f"SIDER graphs labels NaN status: 100% NaN verified ({batch_sider.y.numel()} NaNs)")
    print("PASSED: Missing label semantics strictly preserved without zero-filling.")

    # 4. Multi-Split Coverage Test
    print("\n--- Test 4: Verify loader on all 10 splits without leakage ---")
    expected_counts = {
        "id_train": 5432,
        "id_val": 679,
        "id_test": 679,
        "novel_scaffold_val": 408,
        "novel_scaffold_test": 347,
        "novel_dataset_val": 225,
        "novel_dataset_test": 198,
        "corrupt_train": 1086,
        "corrupt_val": 543,
        "corrupt_test": 543,
    }
    for s_name, exp_n in expected_counts.items():
        ds = MolecularDataset(split_name=s_name, root_dir="processed_v3")
        assert len(ds) == exp_n, f"Split {s_name} expected {exp_n}, got {len(ds)}"
        print(f"  Checked split {s_name:<25}: {len(ds)} graphs OK")
    print("PASSED: All 10 splits loaded with exact expected counts.")

    print("\n" + "=" * 60)
    print("ALL SMOKE CHECKS PASSED SUCCESSFULLY (Exit Code 0)")
    print("=" * 60)


if __name__ == "__main__":
    try:
        run_smoke_test()
        sys.exit(0)
    except Exception as e:
        print(f"\nSMOKE TEST FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
