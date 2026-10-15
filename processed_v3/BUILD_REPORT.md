# Build report (PASSED)

- seed: 42
- id split: scaffold
- RDKit 2026.09.1, torch 2.11.0+cpu, pandas 2.2.3

## Cleaning

| dataset | raw_rows | kept_after_clean | dropped | dup_groups | label_conflicts |
|---|---|---|---|---|---|
| tox21 | 7831 | 7545 | 286 | 182 | 95 |
| sider | 1427 | 1303 | 124 | 40 | 0 |

### Dropped rows by stage / reason

| dataset | stage | reason | n |
|---|---|---|---|
| sider | clean | duplicate_smiles | 46 |
| sider | clean | too_large | 42 |
| sider | clean | too_small | 36 |
| sider | deoverlap | in_tox21_smiles | 880 |
| tox21 | clean | duplicate_smiles | 240 |
| tox21 | clean | parse_failed | 8 |
| tox21 | clean | too_large | 7 |
| tox21 | clean | too_small | 31 |

## Sets

| set_name | n | size_matched |
|---|---|---|
| id_train | 5432 | True |
| id_val | 679 | True |
| id_test | 679 | True |
| novel_scaffold_val | 408 | False |
| novel_scaffold_test | 347 | False |
| novel_dataset_val | 225 | False |
| novel_dataset_test | 198 | False |
| corrupt_train | 1086 | True |
| corrupt_val | 543 | True |
| corrupt_test | 543 | True |

## Corruption

| set_name | family | n | mean_edits | mean_target | mean_viol_atoms |
|---|---|---|---|---|---|
| corrupt_test | A | 181 | 1.8 | 2.02 | 2.2 |
| corrupt_test | B | 181 | 1.73 | 1.99 | 1.73 |
| corrupt_test | C | 181 | 1.84 | 1.99 | 1.84 |
| corrupt_train | A | 543 | 1.66 | 1.97 | 2.13 |
| corrupt_train | B | 543 | 1.66 | 1.97 | 1.66 |
| corrupt_val | A | 272 | 1.79 | 2.03 | 2.33 |
| corrupt_val | B | 271 | 1.78 | 1.99 | 1.78 |

Edit-count distribution:

| set_name | family | n_edits | n |
|---|---|---|---|
| corrupt_test | A | 1.0 | 76 |
| corrupt_test | A | 2.0 | 65 |
| corrupt_test | A | 3.0 | 40 |
| corrupt_test | B | 1.0 | 84 |
| corrupt_test | B | 2.0 | 61 |
| corrupt_test | B | 3.0 | 36 |
| corrupt_test | C | 1.0 | 72 |
| corrupt_test | C | 2.0 | 66 |
| corrupt_test | C | 3.0 | 43 |
| corrupt_train | A | 1.0 | 266 |
| corrupt_train | A | 2.0 | 195 |
| corrupt_train | A | 3.0 | 82 |
| corrupt_train | B | 1.0 | 271 |
| corrupt_train | B | 2.0 | 186 |
| corrupt_train | B | 3.0 | 86 |
| corrupt_val | A | 1.0 | 118 |
| corrupt_val | A | 2.0 | 94 |
| corrupt_val | A | 3.0 | 60 |
| corrupt_val | B | 1.0 | 115 |
| corrupt_val | B | 2.0 | 101 |
| corrupt_val | B | 3.0 | 55 |

## Audit

Errors: 0  |  Warnings: 16

- WARNING constant atom features across the whole dataset: ['degree_0', 'num_h_4plus']
- WARNING constant bond features across the whole dataset: ['bond_other']
- WARNING validity shortcut: count-only AUROC 0.721 for valid vs corrupted (train, family B); size/composition alone separates them
- WARNING validity shortcut: count-only AUROC 0.652 for valid vs corrupted (val, family B); size/composition alone separates them
- WARNING validity shortcut: count-only AUROC 0.652 for valid vs corrupted (test, family B); size/composition alone separates them
- WARNING validity shortcut: count-only AUROC 0.798 for valid vs corrupted (test, family C); size/composition alone separates them
- WARNING novelty shortcut: count-only AUROC 0.752 for ID vs novel_scaffold_val; size/composition alone separates them (see novelty_size_matched)
- WARNING novelty shortcut: count-only AUROC 0.728 for ID vs novel_dataset_val; size/composition alone separates them (see novelty_size_matched)
- WARNING novelty shortcut: count-only AUROC 0.755 for ID vs novel_scaffold_test; size/composition alone separates them (see novelty_size_matched)
- WARNING novelty shortcut: count-only AUROC 0.693 for ID vs novel_dataset_test; size/composition alone separates them (see novelty_size_matched)
- WARNING pair-level count cue (train, family B): in >50% of pairs the corrupted graph differs from its twin in: degree_hist
- WARNING pair-level count cue (val, family B): in >50% of pairs the corrupted graph differs from its twin in: degree_hist
- WARNING pair-level count cue (test, family B): in >50% of pairs the corrupted graph differs from its twin in: degree_hist
- WARNING pair-level count cue (test, family C): in >50% of pairs the corrupted graph differs from its twin in: element_counts
- WARNING novelty shortcut remains after size matching: AUROC 0.689 for novel_scaffold_val (composition/degree structure, not size)
- WARNING novelty shortcut remains after size matching: AUROC 0.664 for novel_dataset_val (composition/degree structure, not size)

Shortcut audit (count-only AUROC; 0.5 = no shortcut):

```
{
  "validity": {
    "train/A": 0.5184631621629685,
    "train/B": 0.7206257780779339,
    "train/ALL": 0.6248045481791282,
    "val/A": 0.49270391146149184,
    "val/B": 0.6523322228804027,
    "val/ALL": 0.5883028611569934,
    "test/A": 0.4890641909210001,
    "test/B": 0.6518319921236136,
    "test/C": 0.7978380621485935,
    "test/ALL": 0.6263354461793831
  },
  "novelty": {
    "val/novel_scaffold": 0.7517976262670016,
    "val/novel_dataset": 0.7280183276059565,
    "test/novel_scaffold": 0.7545020860478836,
    "test/novel_dataset": 0.6930683863673555
  },
  "threshold": 0.65,
  "pair_deltas": {
    "train/A": {
      "n_pairs": 543,
      "frac_edge_count_changed": 0.0,
      "frac_degree_hist_changed": 0.0,
      "frac_element_counts_changed": 0.0
    },
    "train/B": {
      "n_pairs": 543,
      "frac_edge_count_changed": 0.0,
      "frac_degree_hist_changed": 0.8121546961325967,
      "frac_element_counts_changed": 0.0
    },
    "val/A": {
      "n_pairs": 272,
      "frac_edge_count_changed": 0.0,
      "frac_degree_hist_changed": 0.0,
      "frac_element_counts_changed": 0.0
    },
    "val/B": {
      "n_pairs": 271,
      "frac_edge_count_changed": 0.0,
      "frac_degree_hist_changed": 0.7859778597785978,
      "frac_element_counts_changed": 0.0
    },
    "test/A": {
      "n_pairs": 181,
      "frac_edge_count_changed": 0.0,
      "frac_degree_hist_changed": 0.0,
      "frac_element_counts_changed": 0.0
    },
    "test/B": {
      "n_pairs": 181,
      "frac_edge_count_changed": 0.0,
      "frac_degree_hist_changed": 0.7679558011049724,
      "frac_element_counts_changed": 0.0
    },
    "test/C": {
      "n_pairs": 181,
      "frac_edge_count_changed": 0.0,
      "frac_degree_hist_changed": 0.0,
      "frac_element_counts_changed": 1.0
    }
  },
  "novelty_size_matched": {
    "val/novel_scaffold": {
      "n_targets": 408,
      "n_matched": 372,
      "auroc": 0.6885189039195283
    },
    "test/novel_scaffold": {
      "n_targets": 347,
      "n_matched": 323,
      "auroc": 0.6458750682935712
    },
    "val/novel_dataset": {
      "n_targets": 225,
      "n_matched": 211,
      "auroc": 0.6636980301430786
    },
    "test/novel_dataset": {
      "n_targets": 198,
      "n_matched": 187,
      "auroc": 0.6292001487031371
    }
  }
}
```
