# Final Delivery Notes: Dual-Axis OOD Detection Manuscript

**Date:** October 23, 2026  
**Lead:** Member 5 (Evaluation, Benchmarking & Manuscript Lead)  
**Approved Research Scope:** Option A — Diagnostic / Benchmarking Study (No Claims of Algorithmic Superiority)  
**Evidence State:** Frozen on Authoritative `processed_v3` Validation Partition (Test Holdout Quarantined)  

---

## 1. Final Deliverables

| Deliverable | Path | Format | Status |
| :--- | :--- | :---: | :---: |
| **Complete Final Manuscript** | [`paper/final_manuscript.md`](file:///c:/Users/Lenovo/Dual-Axis-OOD-Detection-member5/paper/final_manuscript.md) | Markdown | **Complete & Verified** |
| **Faculty-Ready Academic PDF** | [`paper/final_manuscript.pdf`](file:///c:/Users/Lenovo/Dual-Axis-OOD-Detection-member5/paper/final_manuscript.pdf) | PDF (6 pages) | **Compiled & Verified** |
| **Executive Delivery Summary** | [`paper/FINAL_DELIVERY_NOTES.md`](file:///c:/Users/Lenovo/Dual-Axis-OOD-Detection-member5/paper/FINAL_DELIVERY_NOTES.md) | Markdown | **Complete** |
| **PDF Compilation Script** | [`scripts/generate_paper_pdf.py`](file:///c:/Users/Lenovo/Dual-Axis-OOD-Detection-member5/scripts/generate_paper_pdf.py) | Python (ReportLab) | **Reproducible** |

---

## 2. Manuscript Title & Abstract Contribution

* **Title:**  
  *Dual-Axis Out-of-Distribution Detection on Molecular Graphs: A Diagnostic Benchmarking Study of Physical Validity and Scaffold Novelty*

* **Abstract-Level Contribution:**  
  Conventional molecular OOD benchmarks conflate physical valence violations ("Texas carbons") with synthesizable scaffold novelty into a single scalar anomaly score. This paper formalizes and benchmarks a **Dual-Axis OOD Detection framework** on 10,140 molecular graphs (`processed_v3`). Across 5 seeds, we show:
  1. **Physical Validity Detection:** Message passing reliably detects valence violations without rule engines (AUROC $0.9584 \pm 0.0119$ under Track A, which receives no corrupted graphs during training, and $0.9966$ under outlier exposure Track B), substantially outperforming the atom-count control on Family A ($0.5107$).
  2. **Failure-Mode Separation vs. Triage Limitation:** Single-score baselines conflate invalidity with novelty near chance level (Task X AUROC $\approx 0.56$), whereas dual-axis scoring cleanly separates the two failure modes (Task X AUROC $0.9073 \pm 0.0069$). Critically, joint 3-class triage (Task D) exposes an operational failure: macro recall ($0.5377 \pm 0.0182$) masks near-zero novel-chemotype detection ($2.09\% \pm 0.52\%$) because inverted novelty scores displace the decision threshold above the novel median.
  3. **Benchmarking Confounders:** Standard distance-based novelty estimators yield sub-0.50 raw AUROC on scaffold validation ($0.4539 \pm 0.0061$). The diagnostic results indicate that reference-distribution shift from scaffold-disjoint validation partitioning contributes to the observed novelty-ranking failure (0.0% Murcko scaffold overlap between train and val), dispersing in-distribution validation molecules into the embedding periphery (distance 7.63 vs. 6.75 for novel molecules). Re-centering on an ID reference with generic framework overlap restores AUROC to $0.7440$ under this exploratory single-seed condition, while separation against an acyclic reference ($1.0000$) reveals a cyclic-vs-acyclic topological confound. Apparent raw gains on cross-domain novelty (SIDER) are heavily confounded by atom count (AUROC $0.6280$ raw $\to 0.5024$ matched).

---

## 3. Disclosed Major Limitations

1. **Validation-Only Scope:** All metrics, thresholds, and diagnostic pilots are derived strictly from validation partitions. The test partition remains quarantined.
2. **Task D Novel-Class Under-Detection:** While invalid graph detection is robust ($67.62\% \pm 5.66\%$), novel chemotype recall is $2.09\% \pm 0.52\%$. Dual-axis triage requires calibrated, decoupled thresholding.
3. **Compound Anomaly Scope:** `processed_v3` evaluates invalid graphs and novel graphs separately; graphs that are simultaneously invalid and novel do not exist in the dataset.
4. **Member 4 Checkpoint Boundaries:** Model weights for Member 4 were unbundled in the repository handoff. Member 4 results are prediction-level verified only; cross-axis prediction scores were absent, preventing Task D evaluation on Member 4.
5. **Topological / Confounding Sensitivity:** Distance-based density estimators are vulnerable to reference-scaffold shift, molecular size, and cyclic-versus-acyclic graph topology.

---

## 4. Unresolved Blockers Affecting Faculty Review

* **Zero material blockers remain.** All evidence, tables, citations, and figures agree with authoritative validation records and the approved Option A framing.
