# GNN Backbone & Toxicity Multi-Task Model Documentation (Step 2)

**Model Component:** Edge-Aware Graph Isomorphism Network (GINE) for Multi-Task Toxicity Prediction  
**Dataset Target:** `processed_v3` (In-distribution Tox21 splits: `id_train`, `id_val`)  
**Implementation Modules:**
- Model Architecture: `src/toxicity_model.py`
- Masked Loss & Evaluation Metrics: `src/loss.py`
- Training & Validation Pipeline: `src/train_toxicity.py`
- Unit & Integration Test Suite: `tests/test_model.py`
- Data Loader & Batching: `src/data.py`

---

## 1. Architecture Specification

Molecules are topological graphs where edge (bond) features carry essential chemical valence information. To enable effective chemical representation learning, the backbone adopts an **edge-aware Graph Isomorphism Network (GINE)** architecture:

1. **Input Dimensions:**
   - Node features $x \in \mathbb{R}^{N \times 30}$ (13 element indicators, 6 degree indicators, 4 formal charge indicators, 5 hydrogen count indicators, `is_aromatic`, `in_ring`).
   - Edge index $\mathcal{E} \in \mathbb{N}^{2 \times E}$ (bidirectional).
   - Edge attributes $e \in \mathbb{R}^{E \times 6}$ (`bond_single`, `bond_double`, `bond_triple`, `bond_aromatic`, `bond_other`, `bond_in_ring`).
2. **Node & Edge Encoding:**
   - Input node projection: $\text{MLP}_{\text{in}}(x): \mathbb{R}^{30} \to \mathbb{R}^{D}$ ($D=128$), followed by BatchNorm1d and ReLU.
   - Linear edge projection: $W_{\text{edge}} e_{uv} \in \mathbb{R}^{D}$.
3. **Message Passing Layer (GINELayer):**
   $$m_v = \sum_{u \in \mathcal{N}(v)} \text{ReLU}(h_u + W_{\text{edge}} e_{uv})$$
   $$h'_v = \text{MLP}\left((1 + \epsilon) h_v + m_v\right) + h_v \quad \text{(with residual connection)}$$
4. **Global Graph Pooling:**
   To accommodate variable-sized graphs in disjoint-union batches without distortion:
   $$h_{\text{mean}} = \frac{1}{|V_g|} \sum_{v \in V_g} h_v, \quad h_{\text{sum}} = \sum_{v \in V_g} h_v$$
   $$z_g = [h_{\text{mean}} \,\|\, h_{\text{sum}}] \in \mathbb{R}^{2D} = \mathbb{R}^{256}$$
5. **Multi-Task Prediction Head:**
   Linear classification head mapping $z_g \to 12$ unnormalized logits for the 12 Tox21 assays.

---

## 2. Masked Multi-Task BCE Loss & Missing Labels

Tox21 training labels contain ~15.9% missing assay results, represented as `torch.nan`.

### Formulation
Missing entries are excluded via an indicator mask $M \in \{0, 1\}^{B \times 12}$ where $M_{ij} = \mathbb{I}(y_{ij} \ne \text{NaN})$:
$$\mathcal{L} = \frac{1}{\sum_{i,j} M_{ij}} \sum_{i=1}^B \sum_{j=1}^{12} M_{ij} \cdot \ell_{\text{BCE}}(\hat{y}_{ij}, y_{ij})$$

### Guarantees
- **No Zero-Filling:** Unassayed tasks are **never** treated as negative examples.
- **Observed Normalization:** Loss is divided strictly by the number of observed labels ($\sum M_{ij}$), preventing dilution by missing entries.
- **Safe Fallback:** If a batch has 0 observed labels (such as corrupted graphs or SIDER batches during testing), the loss returns `0.0 * logits.sum()`, preserving autograd connectivity without raising exceptions or producing `NaN`.

---

## 3. Strict Split Policy

- **`id_train` (5,432 graphs):** Exclusively used for gradient updates.
- **`id_val` (679 graphs):** Exclusively used for validation loss tracking, learning rate decay (`ReduceLROnPlateau`), and model checkpoint selection.
- **Test Splits (`id_test`, `novel_scaffold_test`, `novel_dataset_test`, `corrupt_test`):** **Strictly untouched.** Zero test data is used for early stopping, architecture choices, or hyperparameter tuning.

---

## 4. Graph Embedding Extraction for Downstream OOD Scoring

The model exposes a dedicated, deterministic embedding extraction API:

```python
from src.toxicity_model import GINEToxicityModel
from src.data import get_dataloader

model = GINEToxicityModel(...)
# Load best trained checkpoint
checkpoint = torch.load("checkpoints/toxicity_gine/best_model.pt", weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

loader = get_dataloader(split_name="id_val", batch_size=64, shuffle=False)
batch = next(iter(loader))

# Extract graph embeddings [B, 256]
with torch.no_grad():
    embeddings = model.get_embeddings(batch)

# Each row embeddings[i] strictly corresponds to batch.uid[i]
assert embeddings.size(0) == len(batch.uid)
```

These graph embeddings will serve directly in future steps as the input to:
- **Novelty Scorers:** Mahalanobis distance / k-NN density in latent space.
- **Validity Detectors:** Energy scores or auxiliary valence classification.

---

## 5. Execution Commands & Verification

### Running Unit Tests
```bash
python tests/test_model.py
```
*Result:* Passed (exit code 0). Validates loss masking, standard vs. sparse targets, all-NaN safe fallback, GINE forward pass, embedding extraction, and seed reproducibility.

### Running a Fast Training Smoke Test (CPU)
```bash
python src/train_toxicity.py --smoke_test
```
*Result:* Runs 2 epochs on CPU, validates gradient updates, records history, and saves `best_model.pt` in `checkpoints/smoke_toxicity_gine/`.

### Running Full Model Training
```bash
python src/train_toxicity.py --epochs 25 --batch_size 64 --lr 0.001 --hidden_dim 128 --num_layers 4 --output_dir checkpoints/toxicity_gine
```
Runs full training on `id_train` and checkpoint selection on `id_val`.
