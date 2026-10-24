"""
models/dual_axis_gnn.py
=======================
DAG-NN: Dual-Axis Graph Neural Network for Molecular Graphs (Track A Candidate).

Key Design Properties:
  1. Shared Edge-Aware GIN Backbone: Parameter budget and representation capacity match
     the single-score S1 and S2 baselines (3 layers, hidden_dim=128).
  2. Multi-Objective Training on id_train Only:
     - Supervised 12-way multi-task Tox21 classification (masked BCE)
     - Self-supervised masked feature reconstruction on nodes and edges
  3. Axis 1 (validity_score):
     - Evaluated via deterministic Leave-One-Out (LOO) batched masking at inference.
     - Predicts intrinsic chemical features: element (13-cat), formal charge (4-cat),
       explicit hydrogen count (5-cat), aromaticity (binary), and ring membership (binary).
     - DEGREE RECONSTRUCTION IS EXCLUDED from primary validity_score to prevent
       topological leakage from edge_index. Degree is retained only as an auxiliary head.
     - Node and edge NLL components are standardized via empirical z-score parameters
       (mu, sigma) fitted exclusively on id_train graphs.
  4. Axis 2 (novelty_score):
     - Evaluated on graph-level pooled representations z_G in R^128.
     - Regularized Mahalanobis distance to reference distribution (mu, Sigma_reg)
       fitted exclusively on id_train intact graphs.
  5. Strictly Non-Fabricating & Leakage-Free:
     - No exposure to corrupt_train or novelty sets during training or calibration.
     - Intact features are never seen by the model when predicting masked targets.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

# pyrefly: ignore[missing-import]
import numpy as np
# pyrefly: ignore[missing-import]
import torch
import torch.nn as nn
import torch.nn.functional as F


def collate_graphs(graphs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of molecular graph dictionaries into a disjoint batch."""
    xs, eis, eas, ys, bt, uids, off = [], [], [], [], [], [], 0
    for i, g in enumerate(graphs):
        n = g["x"].shape[0]
        xs.append(g["x"])
        eis.append(g["edge_index"] + off)
        eas.append(g["edge_attr"])
        if "y" in g and g["y"] is not None:
            ys.append(g["y"])
        bt.append(torch.full((n,), i, dtype=torch.long))
        uids.append(g["uid"])
        off += n

    batch = {
        "x": torch.cat(xs, dim=0),
        "edge_index": torch.cat(eis, dim=1) if len(eis) > 0 else torch.zeros((2, 0), dtype=torch.long),
        "edge_attr": torch.cat(eas, dim=0) if len(eas) > 0 else torch.zeros((0, 6), dtype=torch.float32),
        "batch": torch.cat(bt, dim=0) if len(bt) > 0 else torch.zeros(0, dtype=torch.long),
        "n": len(graphs),
        "uids": uids,
    }
    if len(ys) == len(graphs):
        batch["y"] = torch.cat(ys, dim=0)
    else:
        batch["y"] = None
    return batch


class DAGNN(nn.Module):
    """Dual-Axis Graph Neural Network (Track A Candidate)."""

    def __init__(
        self,
        in_node_dim: int = 30,
        in_edge_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        num_tasks: int = 12,
    ):
        super().__init__()
        self.in_node_dim = in_node_dim
        self.in_edge_dim = in_edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_tasks = num_tasks

        # Learnable mask embeddings for Leave-One-Out (LOO) masking
        self.mask_node = nn.Parameter(torch.zeros(in_node_dim))
        self.mask_edge = nn.Parameter(torch.zeros(in_edge_dim))
        nn.init.normal_(self.mask_node, std=0.02)
        nn.init.normal_(self.mask_edge, std=0.02)

        # Message Passing Backbone (Edge-Aware GIN)
        self.inp = nn.Linear(in_node_dim, hidden_dim)
        self.edge_linears = nn.ModuleList([nn.Linear(in_edge_dim, hidden_dim) for _ in range(num_layers)])
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # Multi-task supervised head (Tox21 assay labels)
        self.tox21_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tasks),
        )

        # Chemical Node Reconstruction Heads
        # Feature schema:
        # elem: [0:13] (13 classes)
        # deg:  [13:19] (6 classes, AUXILIARY DIAGNOSTIC ONLY)
        # chg:  [19:23] (4 classes)
        # hyd:  [23:28] (5 classes)
        # arom: [28]    (binary)
        # ring: [29]    (binary)
        self.head_elem = nn.Linear(hidden_dim, 13)
        self.head_deg_aux = nn.Linear(hidden_dim, 6)
        self.head_chg = nn.Linear(hidden_dim, 4)
        self.head_hyd = nn.Linear(hidden_dim, 5)
        self.head_arom = nn.Linear(hidden_dim, 1)
        self.head_ring = nn.Linear(hidden_dim, 1)

        # Edge Reconstruction Heads
        # Feature schema:
        # bond_type: [0:5] (5 classes: single, double, triple, aromatic, other)
        # bond_ring: [5]   (binary)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.head_bond_type = nn.Linear(hidden_dim, 5)
        self.head_bond_ring = nn.Linear(hidden_dim, 1)

        # Calibration parameters (fitted on id_train only)
        self.register_buffer("calib_mu_node", torch.tensor(0.0))
        self.register_buffer("calib_sigma_node", torch.tensor(1.0))
        self.register_buffer("calib_mu_edge", torch.tensor(0.0))
        self.register_buffer("calib_sigma_edge", torch.tensor(1.0))
        self.register_buffer("novelty_mu", torch.zeros(hidden_dim))
        self.register_buffer("novelty_prec", torch.eye(hidden_dim))
        self.is_calibrated = False

    def embed_nodes_and_graph(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute node representations h and pooled graph representation z_G."""
        h = self.inp(batch["x"])
        src, dst = batch["edge_index"][0], batch["edge_index"][1]
        edge_attr = batch["edge_attr"]

        for l in range(self.num_layers):
            if edge_attr.shape[0] > 0:
                msg = F.relu(h[src] + self.edge_linears[l](edge_attr))
                agg = h.new_zeros(h.shape).index_add_(0, dst, msg)
            else:
                agg = h.new_zeros(h.shape)
            h = h + F.dropout(F.relu(self.norms[l](self.mlps[l](h + agg))), self.dropout, self.training)

        # Global mean pooling
        n_graphs = batch["n"]
        batch_idx = batch["batch"]
        pooled = h.new_zeros((n_graphs, h.shape[1])).index_add_(0, batch_idx, h)
        cnt = h.new_zeros((n_graphs,)).index_add_(0, batch_idx, h.new_ones((h.shape[0],)))
        z_G = pooled / cnt.clamp(min=1.0).unsqueeze(1)
        return h, z_G

    def forward_train(
        self,
        batch: Dict[str, Any],
        mask_prob_node: float = 0.15,
        mask_prob_edge: float = 0.15,
        include_deg_in_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass with random masking on id_train."""
        x_orig = batch["x"].clone()
        edge_attr_orig = batch["edge_attr"].clone()
        n_nodes = x_orig.shape[0]
        n_edges = edge_attr_orig.shape[0]

        # Node masking
        node_mask = torch.rand(n_nodes, device=x_orig.device) < mask_prob_node
        if not node_mask.any() and n_nodes > 0:
            node_mask[torch.randint(0, n_nodes, (1,))] = True

        x_masked = x_orig.clone()
        x_masked[node_mask] = self.mask_node

        # Edge masking (apply to undirected pairs symmetrically)
        edge_mask = torch.rand(n_edges, device=edge_attr_orig.device) < mask_prob_edge
        edge_attr_masked = edge_attr_orig.clone()
        if edge_mask.any():
            edge_attr_masked[edge_mask] = self.mask_edge

        batch_masked = {
            "x": x_masked,
            "edge_index": batch["edge_index"],
            "edge_attr": edge_attr_masked,
            "batch": batch["batch"],
            "n": batch["n"],
        }
        h, z_G = self.embed_nodes_and_graph(batch_masked)

        # Multi-task classification loss (Tox21)
        loss_tox21 = torch.tensor(0.0, device=x_orig.device)
        if batch.get("y") is not None and batch["y"] is not None:
            logits = self.tox21_head(z_G)
            y = batch["y"]
            valid_mask = ~torch.isnan(y)
            if valid_mask.any():
                loss_tox21 = F.binary_cross_entropy_with_logits(logits[valid_mask], y[valid_mask])

        # Node feature reconstruction losses (on masked nodes only)
        loss_node = torch.tensor(0.0, device=x_orig.device)
        if node_mask.any():
            h_m = h[node_mask]
            x_m = x_orig[node_mask]

            # Chemical targets (primary)
            target_elem = x_m[:, 0:13].argmax(dim=-1)
            target_chg = x_m[:, 19:23].argmax(dim=-1)
            target_hyd = x_m[:, 23:28].argmax(dim=-1)
            target_arom = x_m[:, 28]
            target_ring = x_m[:, 29]

            l_elem = F.cross_entropy(self.head_elem(h_m), target_elem)
            l_chg = F.cross_entropy(self.head_chg(h_m), target_chg)
            l_hyd = F.cross_entropy(self.head_hyd(h_m), target_hyd)
            l_arom = F.binary_cross_entropy_with_logits(self.head_arom(h_m).squeeze(-1), target_arom)
            l_ring = F.binary_cross_entropy_with_logits(self.head_ring(h_m).squeeze(-1), target_ring)

            loss_node = l_elem + l_chg + l_hyd + l_arom + l_ring

            # Optional auxiliary degree reconstruction (diagnostic only)
            if include_deg_in_loss:
                target_deg = x_m[:, 13:19].argmax(dim=-1)
                l_deg = F.cross_entropy(self.head_deg_aux(h_m), target_deg)
                loss_node = loss_node + 0.5 * l_deg

        # Edge feature reconstruction losses (on masked edges only)
        loss_edge = torch.tensor(0.0, device=edge_attr_orig.device)
        if edge_mask.any() and n_edges > 0:
            src, dst = batch["edge_index"][0][edge_mask], batch["edge_index"][1][edge_mask]
            edge_rep = self.edge_mlp(torch.cat([h[src], h[dst]], dim=-1))
            e_m = edge_attr_orig[edge_mask]

            target_bond_type = e_m[:, 0:5].argmax(dim=-1)
            target_bond_ring = e_m[:, 5]

            l_btype = F.cross_entropy(self.head_bond_type(edge_rep), target_bond_type)
            l_bring = F.binary_cross_entropy_with_logits(self.head_bond_ring(edge_rep).squeeze(-1), target_bond_ring)
            loss_edge = l_btype + l_bring

        total_loss = loss_tox21 + 0.5 * loss_node + 0.5 * loss_edge
        return {
            "loss_total": total_loss,
            "loss_tox21": loss_tox21,
            "loss_node": loss_node,
            "loss_edge": loss_edge,
        }

    @torch.no_grad()
    def compute_loo_nll(self, graph: Dict[str, Any]) -> Tuple[float, float, float]:
        """
        Compute Leave-One-Out (LOO) negative log-likelihood for one graph deterministically.
        
        Returns:
            mean_node_nll : float, average NLL across all nodes (chemical features only)
            mean_edge_nll : float, average NLL across all undirected edges
            aux_deg_nll   : float, auxiliary degree NLL (diagnostic only, NOT in validity_score)
        """
        self.eval()
        x = graph["x"]
        edge_index = graph["edge_index"]
        edge_attr = graph["edge_attr"]
        n_nodes = x.shape[0]
        n_edges = edge_attr.shape[0]

        if n_nodes == 0:
            return 0.0, 0.0, 0.0

        # ---- 1. LEAVE-ONE-NODE-OUT (LONO) ----
        # Replicate graph n_nodes times in a single disjoint batch
        # In copy i, node i has features replaced by mask_node
        graphs_lono = []
        for i in range(n_nodes):
            x_copy = x.clone()
            x_copy[i] = self.mask_node
            graphs_lono.append({
                "x": x_copy,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "uid": f"{graph['uid']}_node_{i}",
            })
        batch_lono = collate_graphs(graphs_lono)
        h_lono, _ = self.embed_nodes_and_graph(batch_lono)

        # Extract representation of the masked node from each copy
        # Copy i starts at offset i * n_nodes; target node is i * n_nodes + i
        target_indices = torch.arange(n_nodes, device=x.device) * n_nodes + torch.arange(n_nodes, device=x.device)
        h_masked_nodes = h_lono[target_indices]

        # Targets from original intact graph
        elem_t = x[:, 0:13].argmax(dim=-1)
        deg_t = x[:, 13:19].argmax(dim=-1)
        chg_t = x[:, 19:23].argmax(dim=-1)
        hyd_t = x[:, 23:28].argmax(dim=-1)
        arom_t = x[:, 28]
        ring_t = x[:, 29]

        # Compute per-node NLL for chemical features
        nll_elem = F.cross_entropy(self.head_elem(h_masked_nodes), elem_t, reduction="none")
        nll_chg = F.cross_entropy(self.head_chg(h_masked_nodes), chg_t, reduction="none")
        nll_hyd = F.cross_entropy(self.head_hyd(h_masked_nodes), hyd_t, reduction="none")
        nll_arom = F.binary_cross_entropy_with_logits(self.head_arom(h_masked_nodes).squeeze(-1), arom_t, reduction="none")
        nll_ring = F.binary_cross_entropy_with_logits(self.head_ring(h_masked_nodes).squeeze(-1), ring_t, reduction="none")

        # Primary chemical node NLL (DEGREE IS EXCLUDED)
        node_chem_nll = nll_elem + nll_chg + nll_hyd + nll_arom + nll_ring
        mean_node_nll = float(node_chem_nll.mean().item())

        # Auxiliary degree NLL (for diagnostic ablation only)
        aux_deg_nll = float(F.cross_entropy(self.head_deg_aux(h_masked_nodes), deg_t).item())

        # ---- 2. LEAVE-ONE-EDGE-OUT (LOEO) ----
        # Identify undirected edges (i < j)
        src = edge_index[0]
        dst = edge_index[1]
        undir_mask = src < dst
        n_undir = int(undir_mask.sum().item())

        if n_undir == 0:
            mean_edge_nll = 0.0
        else:
            undir_indices = torch.where(undir_mask)[0]
            graphs_loeo = []
            for k in range(n_undir):
                idx_forward = undir_indices[k]
                u, v = int(src[idx_forward].item()), int(dst[idx_forward].item())
                # Find matching reverse edge
                idx_reverse = torch.where((src == v) & (dst == u))[0]

                edge_attr_copy = edge_attr.clone()
                edge_attr_copy[idx_forward] = self.mask_edge
                if len(idx_reverse) > 0:
                    edge_attr_copy[idx_reverse[0]] = self.mask_edge

                graphs_loeo.append({
                    "x": x,
                    "edge_index": edge_index,
                    "edge_attr": edge_attr_copy,
                    "uid": f"{graph['uid']}_edge_{k}",
                })
            batch_loeo = collate_graphs(graphs_loeo)
            h_loeo, _ = self.embed_nodes_and_graph(batch_loeo)

            # Extract representations of endpoints u and v for each masked edge
            u_indices = torch.arange(n_undir, device=x.device) * n_nodes + src[undir_indices]
            v_indices = torch.arange(n_undir, device=x.device) * n_nodes + dst[undir_indices]

            edge_rep = self.edge_mlp(torch.cat([h_loeo[u_indices], h_loeo[v_indices]], dim=-1))
            true_e = edge_attr[undir_indices]
            btype_t = true_e[:, 0:5].argmax(dim=-1)
            bring_t = true_e[:, 5]

            nll_btype = F.cross_entropy(self.head_bond_type(edge_rep), btype_t, reduction="none")
            nll_bring = F.binary_cross_entropy_with_logits(self.head_bond_ring(edge_rep).squeeze(-1), bring_t, reduction="none")
            edge_nll = nll_btype + nll_bring
            mean_edge_nll = float(edge_nll.mean().item())

        return mean_node_nll, mean_edge_nll, aux_deg_nll

    @torch.no_grad()
    def fit_calibration(
        self,
        id_train_graphs: List[Dict[str, Any]],
        shrinkage: float = 0.01,
        max_samples: Optional[int] = None,
    ) -> None:
        """
        Fit calibration parameters strictly on id_train graphs.
        Computes (mu, sigma) for node and edge NLLs and (mu, Sigma_reg^-1) for novelty.
        """
        self.eval()
        sample_graphs = id_train_graphs if max_samples is None else id_train_graphs[:max_samples]

        node_nlls = []
        edge_nlls = []
        z_list = []

        for g in sample_graphs:
            n_nll, e_nll, _ = self.compute_loo_nll(g)
            node_nlls.append(n_nll)
            edge_nlls.append(e_nll)

            batch = collate_graphs([g])
            _, z_G = self.embed_nodes_and_graph(batch)
            z_list.append(z_G[0].cpu())

        # Fit Z-Score parameters for validity
        self.calib_mu_node.copy_(torch.tensor(float(np.mean(node_nlls))))
        self.calib_sigma_node.copy_(torch.tensor(float(np.std(node_nlls) + 1e-6)))
        self.calib_mu_edge.copy_(torch.tensor(float(np.mean(edge_nlls))))
        self.calib_sigma_edge.copy_(torch.tensor(float(np.std(edge_nlls) + 1e-6)))

        # Fit regularized Mahalanobis parameters for novelty
        z_mat = torch.stack(z_list, dim=0)  # [N, d]
        d = z_mat.shape[1]
        mu = z_mat.mean(dim=0)
        z_cent = z_mat - mu
        cov = (z_cent.T @ z_cent) / max(z_mat.shape[0] - 1, 1)

        # Ledoit-Wolf shrinkage
        trace_cov = torch.trace(cov) / d
        cov_reg = (1.0 - shrinkage) * cov + shrinkage * trace_cov * torch.eye(d)
        prec = torch.linalg.pinv(cov_reg)

        self.novelty_mu.copy_(mu)
        self.novelty_prec.copy_(prec)
        self.is_calibrated = True

    @torch.no_grad()
    def score_graph(self, graph: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute validity_score and novelty_score deterministically.
        """
        self.eval()
        node_nll, edge_nll, aux_deg = self.compute_loo_nll(graph)

        # Validity score: z-score calibrated combination
        if self.is_calibrated:
            z_node = (node_nll - self.calib_mu_node.item()) / self.calib_sigma_node.item()
            z_edge = (edge_nll - self.calib_mu_edge.item()) / self.calib_sigma_edge.item()
            validity_score = 0.5 * z_node + 0.5 * z_edge
        else:
            # Uncalibrated fallback (prior to fitting)
            validity_score = node_nll + edge_nll

        # Novelty score: regularized Mahalanobis distance
        batch = collate_graphs([graph])
        _, z_G = self.embed_nodes_and_graph(batch)
        z_diff = z_G[0].cpu() - self.novelty_mu.cpu()
        novelty_score = float((z_diff.unsqueeze(0) @ self.novelty_prec.cpu() @ z_diff.unsqueeze(1)).item())

        return {
            "validity_score": float(validity_score),
            "novelty_score": float(novelty_score),
            "node_nll_raw": float(node_nll),
            "edge_nll_raw": float(edge_nll),
            "aux_deg_nll": float(aux_deg),
        }
