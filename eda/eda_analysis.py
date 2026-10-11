# -*- coding: utf-8 -*-
"""
Dual-Axis OOD Detection - EDA & Experimental Validation
=========================================================
Produces:
  plots/01_molecule_sizes.png
  plots/02_atom_type_distribution.png
  plots/03_bond_type_distribution.png
  plots/04_scaffold_distribution.png
  plots/05_corruption_rates.png
  plots/06_shortcut_analysis.png
  plots/07_leakage_audit.png
  plots/08_label_distribution.png
  leakage_audit.csv
  data_quality_report.md
"""

import os
import sys

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

import json
import csv
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── RDKit ──────────────────────────────────────────────────────────────────────
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR.parent / "Dataset"
PLOT_DIR   = SCRIPT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Aesthetics ─────────────────────────────────────────────────────────────────
PALETTE = {
    "id_train":  "#4C72B0",
    "id_val":    "#55A868",
    "id_test":   "#C44E52",
    "ood_valid": "#DD8452",
    "invalid":   "#8172B2",
}
SPLIT_ORDER = ["id_train", "id_val", "id_test", "ood_valid"]
SPLIT_LABELS = {
    "id_train":  "ID Train",
    "id_val":    "ID Val",
    "id_test":   "ID Test",
    "ood_valid": "OOD Valid",
}
sns.set_theme(style="whitegrid", font_scale=1.1)

LABEL_COLS = [
    "NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase",
    "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
    "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53",
]

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading datasets …")

id_train = pd.read_csv(DATA_DIR / "id_train.csv")
id_val   = pd.read_csv(DATA_DIR / "id_val.csv")
id_test  = pd.read_csv(DATA_DIR / "id_test.csv")
ood      = pd.read_csv(DATA_DIR / "ood_valid.csv")
split_ids = pd.read_csv(DATA_DIR / "split_ids.csv")

audit_train = pd.read_csv(DATA_DIR / "corruption_audit_train.csv")
audit_val   = pd.read_csv(DATA_DIR / "corruption_audit_val.csv")
audit_test  = pd.read_csv(DATA_DIR / "corruption_audit_test.csv")

inv_train = pd.read_csv(DATA_DIR / "invalid_graphs_train.csv")
inv_val   = pd.read_csv(DATA_DIR / "invalid_graphs_val.csv")
inv_test  = pd.read_csv(DATA_DIR / "invalid_graphs_test.csv")

rdkit_val = pd.read_csv(DATA_DIR / "rdkit_reconstruction_validation.csv")
rdkit_sum = pd.read_csv(DATA_DIR / "rdkit_validation_summary.csv")
source_inv = pd.read_csv(DATA_DIR / "source_invalid_audit.csv")

# Combine all valid-molecule splits
valid_all = pd.concat([id_train, id_val, id_test, ood], ignore_index=True)

# ── Compute RDKit features for all valid molecules ─────────────────────────────
print("Computing molecular features …")

def mol_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    atom_counts = Counter(a.GetSymbol() for a in mol.GetAtoms())
    bond_counts = Counter(str(b.GetBondTypeAsDouble()) for b in mol.GetBonds())
    ring_info = mol.GetRingInfo()
    return {
        "n_atoms":    mol.GetNumAtoms(),
        "n_bonds":    mol.GetNumBonds(),
        "n_rings":    ring_info.NumRings(),
        "n_heavy":    mol.GetNumHeavyAtoms(),
        "mol_wt":     Descriptors.MolWt(mol),
        "n_aromatic": sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()),
        "atom_counts": atom_counts,
        "bond_counts": bond_counts,
        "has_cycle":  ring_info.NumRings() > 0,
        "n_hbd":      rdMolDescriptors.CalcNumHBD(mol),
        "n_hba":      rdMolDescriptors.CalcNumHBA(mol),
        "tpsa":       Descriptors.TPSA(mol),
        "logp":       Descriptors.MolLogP(mol),
    }

feat_rows = []
for _, row in valid_all.iterrows():
    f = mol_features(row["canonical_smiles"])
    f["mol_id"] = row["mol_id"]
    f["split"]  = row["split"]
    feat_rows.append(f)

feat_df = pd.DataFrame(feat_rows)

# ── Invalid graph features (from JSON) ─────────────────────────────────────────
print("Computing invalid-graph features …")

def inv_features(row):
    try:
        g = json.loads(row["graph_json"])
        n_atoms = len(g["atoms"])
        n_bonds = len(g["bonds"])
        atom_counts = Counter(a["atomic_num"] for a in g["atoms"])
        bond_counts = Counter(b["type"] for b in g["bonds"])
        return {
            "n_atoms": n_atoms,
            "n_bonds": n_bonds,
            "atom_counts": atom_counts,
            "bond_counts": bond_counts,
            "atom_i": int(row["atom_i"]),
            "explicit_valence_i": float(row["explicit_valence_i"]),
        }
    except Exception:
        return {}

inv_all = pd.concat([
    inv_train.assign(split_label="invalid_train"),
    inv_val.assign(split_label="invalid_val"),
    inv_test.assign(split_label="invalid_test"),
], ignore_index=True)

inv_feat_rows = []
for _, row in inv_all.iterrows():
    f = inv_features(row)
    f["source_mol_id"] = row["source_mol_id"]
    f["split_label"] = row["split_label"]
    inv_feat_rows.append(f)

inv_feat_df = pd.DataFrame(inv_feat_rows)

# ══════════════════════════════════════════════════════════════════════════════
# 2. LEAKAGE AUDIT
# ══════════════════════════════════════════════════════════════════════════════
print("Running leakage audit …")

audit_results = {}

# a) Mol-ID uniqueness across valid splits
mol_split_map = defaultdict(list)
for _, row in valid_all.iterrows():
    mol_split_map[row["mol_id"]].append(row["split"])
mol_cross = {k: v for k, v in mol_split_map.items() if len(v) > 1}
audit_results["mol_id_cross_split_leakage"] = len(mol_cross)

# b) Scaffold leakage: OOD scaffolds appearing in any ID split
ood_scaffolds = set(ood["scaffold_key"])
id_scaffolds  = set(pd.concat([id_train, id_val, id_test])["scaffold_key"])
scaffold_overlap = ood_scaffolds & id_scaffolds
audit_results["scaffold_overlap_ood_vs_id"] = len(scaffold_overlap)

# c) Canonical SMILES leakage
ood_smiles = set(ood["canonical_smiles"])
id_smiles  = set(pd.concat([id_train, id_val, id_test])["canonical_smiles"])
smiles_overlap = ood_smiles & id_smiles
audit_results["canonical_smiles_overlap_ood_vs_id"] = len(smiles_overlap)

# d) Duplicates within each split
for name, df in [("id_train", id_train), ("id_val", id_val),
                  ("id_test", id_test), ("ood_valid", ood)]:
    dups = df["canonical_smiles"].duplicated().sum()
    audit_results[f"within_split_dup_{name}"] = int(dups)

# e) Invalid graph duplicates
for name, df in [("inv_train", inv_train), ("inv_val", inv_val), ("inv_test", inv_test)]:
    dups = df["source_mol_id"].duplicated().sum()
    audit_results[f"dup_{name}"] = int(dups)

# f) Invalid source mol IDs crossing valid splits
inv_train_ids = set(inv_train["source_mol_id"])
inv_val_ids   = set(inv_val["source_mol_id"])
inv_test_ids  = set(inv_test["source_mol_id"])
inv_cross = (inv_train_ids & inv_val_ids) | (inv_train_ids & inv_test_ids) | (inv_val_ids & inv_test_ids)
audit_results["inv_cross_split_leakage"] = len(inv_cross)

# g) Corruption success rates
for name, audit in [("train", audit_train), ("val", audit_val), ("test", audit_test)]:
    total = len(audit)
    success = audit["corruption_success"].astype(str).str.lower().eq("true").sum()
    audit_results[f"corruption_success_{name}"] = int(success)
    audit_results[f"corruption_total_{name}"] = int(total)
    audit_results[f"corruption_rate_{name}"] = round(100 * success / total, 2)

# Save audit CSV
audit_rows = [{"check": k, "result": v} for k, v in audit_results.items()]
pd.DataFrame(audit_rows).to_csv(SCRIPT_DIR / "leakage_audit.csv", index=False)
print("  → leakage_audit.csv saved")

# ══════════════════════════════════════════════════════════════════════════════
# 3. PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def save(fig, name, dpi=150):
    path = PLOT_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name} saved")

# ── PLOT 1: Molecule Size Distributions ───────────────────────────────────────
print("Plot 1: Molecule sizes …")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Molecule Size Distributions by Split", fontsize=14, fontweight="bold")

metrics = [("n_atoms", "Number of Atoms"), ("n_bonds", "Number of Bonds"), ("mol_wt", "Molecular Weight (Da)")]
for ax, (col, label) in zip(axes, metrics):
    for split in SPLIT_ORDER:
        data = feat_df[feat_df["split"] == split][col].dropna()
        ax.hist(data, bins=30, alpha=0.55, color=PALETTE[split],
                label=SPLIT_LABELS[split], density=True, edgecolor="none")
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(label)

handles = [mpatches.Patch(color=PALETTE[s], label=SPLIT_LABELS[s]) for s in SPLIT_ORDER]
fig.legend(handles=handles, loc="upper right", ncol=2, framealpha=0.9)
plt.tight_layout()
save(fig, "01_molecule_sizes.png")

# ── PLOT 2: Atom Type Distribution ────────────────────────────────────────────
print("Plot 2: Atom types …")

def get_atom_counts(df_subset, feat_col="atom_counts"):
    combined = Counter()
    for val in feat_df[feat_df["split"].isin(df_subset)]["atom_counts"]:
        if isinstance(val, dict):
            combined += Counter(val)
    return combined

top_atoms = ["C", "N", "O", "S", "F", "Cl", "Br", "P", "Si", "I"]
atom_data = {}
for split in SPLIT_ORDER:
    counts = Counter()
    for val in feat_df[feat_df["split"] == split]["atom_counts"]:
        if isinstance(val, dict):
            counts += Counter(val)
    total = sum(counts.values())
    atom_data[split] = {a: counts.get(a, 0) / total * 100 for a in top_atoms}

df_atom = pd.DataFrame(atom_data, index=top_atoms)
fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(top_atoms))
width = 0.18
for i, split in enumerate(SPLIT_ORDER):
    ax.bar(x + i * width, df_atom[split], width, label=SPLIT_LABELS[split],
           color=PALETTE[split], alpha=0.85)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(top_atoms, fontsize=12)
ax.set_ylabel("% of all atoms", fontsize=11)
ax.set_title("Atom Type Distribution (% of total atoms per split)", fontsize=13, fontweight="bold")
ax.legend(framealpha=0.9)
plt.tight_layout()
save(fig, "02_atom_type_distribution.png")

# ── PLOT 3: Bond Type Distribution ────────────────────────────────────────────
print("Plot 3: Bond types …")
bond_map = {"1.0": "Single", "1.5": "Aromatic", "2.0": "Double", "3.0": "Triple"}
bond_types = list(bond_map.keys())

bond_data = {}
for split in SPLIT_ORDER:
    counts = Counter()
    for val in feat_df[feat_df["split"] == split]["bond_counts"]:
        if isinstance(val, dict):
            counts += Counter(val)
    total = sum(counts.values()) or 1
    bond_data[split] = {bond_map.get(k, k): counts.get(k, 0) / total * 100 for k in bond_types}

df_bond = pd.DataFrame(bond_data).T
fig, ax = plt.subplots(figsize=(9, 5))
df_bond.plot(kind="bar", ax=ax, colormap="Set2", alpha=0.9, edgecolor="white", width=0.7)
ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLIT_ORDER], rotation=0)
ax.set_ylabel("% of all bonds", fontsize=11)
ax.set_title("Bond Type Distribution per Split", fontsize=13, fontweight="bold")
ax.legend(title="Bond Type", framealpha=0.9)
plt.tight_layout()
save(fig, "03_bond_type_distribution.png")

# ── PLOT 4: Scaffold Distribution ─────────────────────────────────────────────
print("Plot 4: Scaffold distribution …")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Top-15 Scaffold Keys per Split", fontsize=14, fontweight="bold")

for ax, (split, df) in zip(axes.flat, [
        ("id_train", id_train), ("id_val", id_val),
        ("id_test", id_test),   ("ood_valid", ood)]):
    top = df["scaffold_key"].value_counts().head(15)
    short = [s.replace("SCAFFOLD::", "").replace("ACYCLIC::", "AC::") for s in top.index]
    short = [s[:30] + "…" if len(s) > 30 else s for s in short]
    ax.barh(range(len(top)), top.values, color=PALETTE[split], alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(short, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(SPLIT_LABELS[split], fontweight="bold", color=PALETTE[split])
    ax.set_xlabel("Count")

plt.tight_layout()
save(fig, "04_scaffold_distribution.png")

# ── PLOT 5: Corruption Success Rates ──────────────────────────────────────────
print("Plot 5: Corruption rates …")
corr_data = []
for split, audit in [("Train", audit_train), ("Val", audit_val), ("Test", audit_test)]:
    success = audit["corruption_success"].astype(str).str.lower().eq("true").sum()
    fail    = len(audit) - success
    corr_data.append({"Split": split, "Outcome": "Successful", "Count": int(success)})
    corr_data.append({"Split": split, "Outcome": "Failed",     "Count": int(fail)})

corr_df = pd.DataFrame(corr_data)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Corruption Pipeline Statistics", fontsize=14, fontweight="bold")

# Stacked bar
pivot = corr_df.pivot(index="Split", columns="Outcome", values="Count").fillna(0)
pivot[["Successful","Failed"]].plot(
    kind="bar", stacked=True, ax=axes[0],
    color=["#2ecc71","#e74c3c"], alpha=0.85, edgecolor="white"
)
axes[0].set_title("Corruption Attempts by Split", fontweight="bold")
axes[0].set_xlabel("")
axes[0].set_ylabel("# Molecules")
axes[0].set_xticklabels(["Train","Val","Test"], rotation=0)
axes[0].legend(framealpha=0.9)

# Pie for total
total_s = corr_df[corr_df["Outcome"]=="Successful"]["Count"].sum()
total_f = corr_df[corr_df["Outcome"]=="Failed"]["Count"].sum()
axes[1].pie(
    [total_s, total_f],
    labels=[f"Successful\n{total_s:,} ({100*total_s/(total_s+total_f):.1f}%)",
            f"Failed\n{total_f:,} ({100*total_f/(total_s+total_f):.1f}%)"],
    colors=["#2ecc71","#e74c3c"], startangle=90, autopct=None,
    wedgeprops=dict(edgecolor="white", linewidth=2)
)
axes[1].set_title("Total Corruption Outcome (all splits)", fontweight="bold")
plt.tight_layout()
save(fig, "05_corruption_rates.png")

# ── PLOT 6: Shortcut Analysis ──────────────────────────────────────────────────
print("Plot 6: Shortcut analysis …")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Shortcut Analysis — Are ID and OOD Molecules Systematically Different?",
             fontsize=13, fontweight="bold")

metrics_sc = [
    ("n_atoms",   "Atom Count"),
    ("mol_wt",    "Molecular Weight (Da)"),
    ("n_rings",   "Ring Count"),
    ("n_aromatic","Aromatic Atom Count"),
    ("logp",      "LogP"),
    ("tpsa",      "TPSA (Å²)"),
]

for ax, (col, label) in zip(axes.flat, metrics_sc):
    id_data  = feat_df[feat_df["split"].isin(["id_train","id_val","id_test"])][col].dropna()
    ood_data = feat_df[feat_df["split"] == "ood_valid"][col].dropna()

    # KDE plot
    id_data.plot.kde(ax=ax, color=PALETTE["id_train"], label="ID (all)", linewidth=2)
    ood_data.plot.kde(ax=ax, color=PALETTE["ood_valid"], label="OOD Valid", linewidth=2, linestyle="--")

    # Mann-Whitney U test
    stat, pval = stats.mannwhitneyu(id_data, ood_data, alternative="two-sided")
    id_med  = id_data.median()
    ood_med = ood_data.median()

    ax.axvline(id_med,  color=PALETTE["id_train"], linestyle=":", alpha=0.7, linewidth=1.5)
    ax.axvline(ood_med, color=PALETTE["ood_valid"], linestyle=":", alpha=0.7, linewidth=1.5)

    significance = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
    ax.set_title(f"{label}\nMW p={pval:.3g} {significance}  |  ID med={id_med:.1f}, OOD med={ood_med:.1f}",
                 fontsize=10)
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.9)

plt.tight_layout()
save(fig, "06_shortcut_analysis.png")

# ── PLOT 7: Leakage Audit Summary ─────────────────────────────────────────────
print("Plot 7: Leakage audit …")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Leakage Audit Summary", fontsize=14, fontweight="bold")

# Left: leakage checks (all should be 0)
checks = {
    "Mol-ID Cross-Split\n(valid mols)":      audit_results["mol_id_cross_split_leakage"],
    "Scaffold Overlap\n(OOD vs ID)":          audit_results["scaffold_overlap_ood_vs_id"],
    "Canonical SMILES\nOverlap (OOD vs ID)":  audit_results["canonical_smiles_overlap_ood_vs_id"],
    "Invalid Graph\nCross-Split":             audit_results["inv_cross_split_leakage"],
    "Within-Split Dups\n(id_train)":          audit_results["within_split_dup_id_train"],
    "Within-Split Dups\n(ood_valid)":         audit_results["within_split_dup_ood_valid"],
}
labels = list(checks.keys())
values = list(checks.values())
colors = ["#2ecc71" if v == 0 else "#e74c3c" for v in values]

axes[0].barh(labels, values, color=colors, alpha=0.85, edgecolor="white")
axes[0].set_xlabel("Count (0 = Pass ✓)", fontsize=11)
axes[0].set_title("Leakage Checks (all should be 0)", fontweight="bold")
for i, v in enumerate(values):
    axes[0].text(max(values) * 0.02 + v, i, f"  {v}", va="center",
                 color="green" if v == 0 else "red", fontweight="bold")

# Right: corruption success rate per split
splits_c = ["Train", "Val", "Test"]
rates = [audit_results[f"corruption_rate_{s.lower()}"] for s in splits_c]
bar = axes[1].bar(splits_c, rates, color=["#4C72B0","#55A868","#C44E52"], alpha=0.85, edgecolor="white", width=0.5)
axes[1].set_ylim(0, 100)
axes[1].axhline(sum(rates)/len(rates), color="gray", linestyle="--", linewidth=1.5, label=f"Mean {sum(rates)/len(rates):.1f}%")
axes[1].set_ylabel("Success Rate (%)", fontsize=11)
axes[1].set_title("Corruption Success Rate per Split", fontweight="bold")
axes[1].legend()
for rect, r in zip(bar, rates):
    axes[1].text(rect.get_x() + rect.get_width()/2, r + 1, f"{r:.1f}%",
                 ha="center", fontweight="bold")
plt.tight_layout()
save(fig, "07_leakage_audit.png")

# ── PLOT 8: Bioactivity Label Distribution ────────────────────────────────────
print("Plot 8: Label distributions …")

def label_positive_rate(df, labels):
    rates = {}
    for col in labels:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            pos  = (vals == 1).sum()
            total = vals.notna().sum()
            rates[col] = 100 * pos / total if total > 0 else 0
    return rates

label_data = {}
for name, df in [("ID Train", id_train), ("ID Val", id_val),
                  ("ID Test", id_test), ("OOD Valid", ood)]:
    label_data[name] = label_positive_rate(df, LABEL_COLS)

df_label = pd.DataFrame(label_data)
fig, ax = plt.subplots(figsize=(14, 6))
df_label.plot(kind="bar", ax=ax, colormap="tab10", alpha=0.85, edgecolor="white", width=0.75)
ax.set_xticklabels(LABEL_COLS, rotation=30, ha="right", fontsize=10)
ax.set_ylabel("Positive Rate (%)", fontsize=11)
ax.set_title("Bioactivity Label Positive Rates by Split", fontsize=13, fontweight="bold")
ax.legend(framealpha=0.9)
plt.tight_layout()
save(fig, "08_label_distribution.png")

# ── PLOT 9: ID vs Invalid graph size comparison ───────────────────────────────
print("Plot 9: Invalid graph vs valid molecule sizes …")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Valid vs Corrupted (Invalid) Graph Size Comparison", fontsize=13, fontweight="bold")

# Use id_train feat vs inv_train features
id_atoms  = feat_df[feat_df["split"]=="id_train"]["n_atoms"].dropna()
inv_atoms = inv_feat_df[inv_feat_df["split_label"]=="invalid_train"]["n_atoms"].dropna()

for ax, (d1, d2, label) in zip(axes, [
    (id_atoms, inv_atoms, "Atom Count"),
    (feat_df[feat_df["split"]=="id_train"]["n_bonds"].dropna(),
     inv_feat_df[inv_feat_df["split_label"]=="invalid_train"]["n_bonds"].dropna(),
     "Bond Count"),
]):
    ax.hist(d1, bins=30, alpha=0.55, color=PALETTE["id_train"], label="ID Train (valid)", density=True)
    ax.hist(d2, bins=30, alpha=0.55, color=PALETTE["invalid"],  label="Corrupted (invalid train)", density=True)
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"{label}: Valid vs Corrupted")
    ax.legend(framealpha=0.9)

plt.tight_layout()
save(fig, "09_valid_vs_invalid_size.png")

# ── PLOT 10: Valence distribution of corrupted atoms ─────────────────────────
print("Plot 10: Over-valent atom distribution …")
if "explicit_valence_i" in inv_feat_df.columns:
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = inv_feat_df["explicit_valence_i"].dropna()
    ax.hist(vals, bins=range(int(vals.min()), int(vals.max())+2), color=PALETTE["invalid"],
            alpha=0.85, edgecolor="white", align="left")
    ax.set_xlabel("Explicit Valence of Corrupted Carbon (atom_i)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("All Corrupted Atoms have Valence ≥ 5", fontsize=12, fontweight="bold")
    ax.axvline(5, color="red", linestyle="--", linewidth=2, label="Over-valent threshold (5)")
    ax.legend()
    plt.tight_layout()
    save(fig, "10_overvalent_distribution.png")

print("\n✅ All plots saved to:", PLOT_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# 4. DESCRIPTIVE STATISTICS TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\nComputing summary statistics …")

stat_rows = []
for split in SPLIT_ORDER:
    sub = feat_df[feat_df["split"] == split]
    stat_rows.append({
        "split": SPLIT_LABELS[split],
        "n_molecules": len(sub),
        "atom_count_mean": sub["n_atoms"].mean(),
        "atom_count_median": sub["n_atoms"].median(),
        "atom_count_std": sub["n_atoms"].std(),
        "bond_count_mean": sub["n_bonds"].mean(),
        "mol_wt_mean": sub["mol_wt"].mean(),
        "n_rings_mean": sub["n_rings"].mean(),
        "pct_cyclic": 100 * sub["has_cycle"].mean(),
        "logp_mean": sub["logp"].mean(),
        "tpsa_mean": sub["tpsa"].mean(),
    })

# Invalid graphs
for label, df in [("Invalid Train", inv_feat_df[inv_feat_df["split_label"]=="invalid_train"]),
                   ("Invalid Val",   inv_feat_df[inv_feat_df["split_label"]=="invalid_val"]),
                   ("Invalid Test",  inv_feat_df[inv_feat_df["split_label"]=="invalid_test"])]:
    stat_rows.append({
        "split": label,
        "n_molecules": len(df),
        "atom_count_mean": df["n_atoms"].mean() if "n_atoms" in df else None,
        "atom_count_median": df["n_atoms"].median() if "n_atoms" in df else None,
        "atom_count_std": df["n_atoms"].std() if "n_atoms" in df else None,
        "bond_count_mean": df["n_bonds"].mean() if "n_bonds" in df else None,
        "mol_wt_mean": None,
        "n_rings_mean": None,
        "pct_cyclic": None,
        "logp_mean": None,
        "tpsa_mean": None,
    })

stat_df = pd.DataFrame(stat_rows).round(2)
stat_df.to_csv(SCRIPT_DIR / "descriptive_statistics.csv", index=False)
print("  → descriptive_statistics.csv saved")

# ══════════════════════════════════════════════════════════════════════════════
# 5. SHORTCUT ANALYSIS — STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════════════════
print("\nRunning statistical shortcut tests …")

id_all  = feat_df[feat_df["split"].isin(["id_train","id_val","id_test"])]
ood_all = feat_df[feat_df["split"] == "ood_valid"]

shortcut_rows = []
for col, label in metrics_sc:
    id_d  = id_all[col].dropna()
    ood_d = ood_all[col].dropna()
    stat, pval = stats.mannwhitneyu(id_d, ood_d, alternative="two-sided")
    cohens_d = (id_d.mean() - ood_d.mean()) / np.sqrt((id_d.std()**2 + ood_d.std()**2) / 2)
    shortcut_rows.append({
        "metric": label,
        "id_mean": round(id_d.mean(), 3),
        "ood_mean": round(ood_d.mean(), 3),
        "id_median": round(id_d.median(), 3),
        "ood_median": round(ood_d.median(), 3),
        "mann_whitney_p": round(pval, 6),
        "cohens_d": round(cohens_d, 4),
        "significant_p005": pval < 0.05,
        "effect_size": "large" if abs(cohens_d) > 0.8 else ("medium" if abs(cohens_d) > 0.5 else ("small" if abs(cohens_d) > 0.2 else "negligible")),
    })

shortcut_df = pd.DataFrame(shortcut_rows)
shortcut_df.to_csv(SCRIPT_DIR / "shortcut_analysis.csv", index=False)
print("  → shortcut_analysis.csv saved")

# ══════════════════════════════════════════════════════════════════════════════
# 6. EVALUATION METRICS PROPOSAL
# ══════════════════════════════════════════════════════════════════════════════
eval_metrics = {
    "axis1_structural_validity": {
        "primary": ["AUROC", "AUPR (OOD=positive)"],
        "secondary": ["FPR@95TPR", "Detection Accuracy"],
        "threshold_free": True,
        "note": "Compare valid vs invalid graphs. Positive class = invalid."
    },
    "axis2_semantic_novelty": {
        "primary": ["AUROC", "AUPR (OOD=positive)"],
        "secondary": ["FPR@95TPR", "Detection Accuracy"],
        "threshold_free": True,
        "note": "Compare ID test vs ood_valid. Positive class = OOD."
    },
    "combined_dual_axis": {
        "primary": ["Mean AUROC (axis1 + axis2) / 2"],
        "secondary": ["Harmonic Mean AUROC"],
        "note": "Equal weighting of both axes unless domain priority is specified."
    },
    "result_format": {
        "required_columns": ["model_name","split","axis","metric","value","seed","timestamp"],
        "score_range": "[0,1] for AUROC/AUPR; [0,100] for percentage metrics",
        "reporting": "Mean ± std over ≥3 seeds"
    }
}
with open(SCRIPT_DIR / "evaluation_metrics.json", "w") as f:
    json.dump(eval_metrics, f, indent=2)
print("  → evaluation_metrics.json saved")

# ══════════════════════════════════════════════════════════════════════════════
# 7. PRINT SHORTCUT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("SHORTCUT ANALYSIS SUMMARY")
print("="*65)
print(shortcut_df[["metric","id_mean","ood_mean","mann_whitney_p","effect_size","significant_p005"]].to_string(index=False))

print("\n" + "="*65)
print("LEAKAGE AUDIT SUMMARY")
print("="*65)
for k, v in audit_results.items():
    status = "✅" if (v == 0 or "rate" in k) else "❌"
    print(f"  {status} {k}: {v}")

print("\n✅ EDA analysis complete.")
print(f"   Plots:    {PLOT_DIR}")
print(f"   Reports:  {SCRIPT_DIR}")
