# Data card: dual-axis graph OOD dataset

Build status: **PASSED**. Built with seed 42.

## What this is
Molecular graphs for separating **structural validity** (corrupted / invalid molecules)
from **semantic novelty** (valid but unfamiliar molecules).
In-distribution data is Tox21; novelty comes from SIDER (dataset shift) and from held-out
Tox21 scaffolds; corruption is synthetic and valence-only.

## Sets
| set | n | what it is |
|---|---|---|
| id_train / id_val / id_test | 5432 / 679 / 679 | Tox21, scaffold split |
| novel_scaffold_val / test | 408 / 347 | Tox21 molecules whose cyclic scaffolds never appear in any ID split |
| novel_dataset_val / test | 225 / 198 | SIDER molecules, de-overlapped with Tox21 |
| corrupt_train / val / test | 1086 / 543 / 543 | valence-corrupted copies of the same split's ID molecules |

Corruption families: A = bond-order inflation on a non-aromatic single/double bond; B = bond relocation: one end of an existing single bond is moved onto a saturated atom (bond count unchanged, graph stays connected);
C = carbon -> oxygen atom swap (test only: held-out corruption family). 1-3 edits per graph on distinct atoms. Family C exists only in
corrupt_test. Use `val` sets for tuning, `test` sets for reporting; novelty val/test are split
group-wise, so no scaffold crosses between them.

## Files
- `graphs/<set>.pt`: `torch.load(path, weights_only=True)` -> list of dicts
  `{uid, x[N,30], edge_index[2,E] (both directions), edge_attr[E,6], y[1,12]}`
- `tables/master_table.csv`: one row per graph (metadata + 12 labels). Join on `uid`.
- `tables/eligible_pairs.csv`: corrupted graph -> valid twin (same split).
- `tables/matched_controls.csv`: each novelty graph -> a size-matched ID graph of the same split
  (|d atoms|, |d bonds| <= caliper). Use it to compare novelty vs ID without graph size doing the work.
- `tables/dropped_rows.csv`: every row removed anywhere, with stage and reason.
- `splits/split_ids.json`: uid lists per set.
- `feature_schema.json`: column names of x / edge_attr / y.
- `audits/audit_results.json`, `BUILD_REPORT.md`, `config.json` (args, versions, SHA-256 of raw and output files).

## Things the EDA owner must know
1. Labels: 12 Tox21 tasks, NaN = missing. SIDER and corrupted graphs have all-NaN `y`
   (their labels are not Tox21 tasks). For a corrupted graph use `twin_uid` to look up the valid labels.
2. Ring-containing ID molecules are scaffold-split, so their scaffolds in val/test are unseen in training. Ring-free molecules have no scaffold and are split randomly, so all three ID splits have a similar ring-free share. Novelty sets are further away than that, but the gap between ID-test and novelty is smaller than with a random-split ID set.
3. `size_matched` is False when fewer molecules were available than the ID set size. Check it before
   comparing AUROCs across sets of different size.
4. Features deliberately exclude hybridization, explicit valence, chirality and conjugation (they
   cannot be recomputed on corrupted graphs without encoding the edit).
5. Read the shortcut audit in `BUILD_REPORT.md` (count-only models: no bond types, no connectivity).
   - Family A leaves every count unchanged: it is the clean validity test.
   - Family B moves a bond (bond count unchanged, no degree-5 atoms), but degrees still shift a little.
   - Family C changes element counts by construction; this cannot be removed. Report it separately.
   - Novelty vs ID: compare against `matched_controls.csv`; the `novelty_size_matched` AUROC shows
     what remains after size is matched.
   Always report per family and include a count-only baseline.
6. Duplicate Tox21 molecules were merged: agreeing labels kept, conflicting labels set to NaN.
7. novel_scaffold sets contain ring-containing molecules only (a ring-free molecule has no scaffold to be
   novel). Compare them with `matched_controls.csv` (matched on size AND ring-free status) or with the
   ring-containing subset of the ID sets, never with the raw ID set alone.
8. Columns `degree_0`, `num_h_4plus` and `bond_other` are always 0 in this data (no isolated atoms, no
   CH4-like atoms, no exotic bonds). They are kept for schema stability.
9. SIDER is dataset-shift novelty, not scaffold novelty: some of its ring-containing molecules share
   scaffolds with Tox21-train (the count is in `audits/audit_results.json`).
