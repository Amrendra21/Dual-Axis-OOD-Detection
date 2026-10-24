"""
Generate publication-grade academic PDF for Dual-Axis OOD Detection Manuscript.
Uses ReportLab to produce publication-grade typography, tables, figures, and page layout.
"""
import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    KeepTogether, HRFlowable, Image
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to add 'Page X of Y' footer and running header."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4A5568"))

        # Running header (pages 2+)
        if self._pageNumber > 1:
            header_text = "Dual-Axis OOD Detection on Molecular Graphs: A Diagnostic Benchmarking Study"
            self.drawString(54, letter[1] - 36, header_text)
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 40, letter[0] - 54, letter[1] - 40)

        # Running footer (all pages)
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 30, footer_text)
        self.drawString(54, 30, "Faculty Review Draft | Evidence-Frozen Pre-Submission | Option A Study")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 40, letter[0] - 54, 40)
        
        self.restoreState()


def build_pdf(output_pdf_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)
    
    # 0.75 in (54 pt) margins
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        alignment=1, # Center
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=8
    )

    meta_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        alignment=1,
        textColor=colors.HexColor("#334155"),
        spaceAfter=10
    )

    scope_banner_style = ParagraphStyle(
        "ScopeBanner",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        alignment=1,
        textColor=colors.HexColor("#1E3A8A"),
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=12,
        spaceAfter=5,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=9,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=6,
        alignment=4 # Justified
    )

    abstract_style = ParagraphStyle(
        "Abstract_Text",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.2,
        leading=11.5,
        textColor=colors.HexColor("#1E293B"),
        alignment=4
    )

    caption_style = ParagraphStyle(
        "Caption_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=4,
        spaceBefore=5,
        alignment=0
    )

    fig_caption_style = ParagraphStyle(
        "FigCaption_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.8,
        leading=10,
        textColor=colors.HexColor("#475569"),
        spaceAfter=8,
        spaceBefore=4,
        alignment=0
    )

    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=9.2,
        textColor=colors.HexColor("#0F172A")
    )

    table_cell_compact = ParagraphStyle(
        "TableCellCompact",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.5,
        leading=8.2,
        textColor=colors.HexColor("#0F172A")
    )

    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=9.2,
        textColor=colors.white
    )

    table_header_compact = ParagraphStyle(
        "TableHeaderCompact",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=6.5,
        leading=8.2,
        textColor=colors.white
    )

    ref_style = ParagraphStyle(
        "Ref_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=3.5,
        leftIndent=16,
        firstLineIndent=-16
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("Dual-Axis Out-of-Distribution Detection on Molecular Graphs:<br/>A Diagnostic Benchmarking Study of Physical Validity and Scaffold Novelty", title_style))
    story.append(Paragraph("<b>Member 5</b> (Evaluation, Benchmarking & Manuscript Lead) on behalf of the Dual-Axis OOD Detection Team<br/><i>Pre-Submission Faculty Review Draft — Evidence-Frozen Edition (October 2026)</i>", meta_style))
    
    # Scope Banner
    banner_data = [[
        Paragraph("<b>APPROVED RESEARCH SCOPE — OPTION A: DIAGNOSTIC & BENCHMARKING STUDY</b><br/>"
                  "This work formalizes dual-axis evaluation and forensic confound analysis on the authoritative <code>processed_v3</code> dataset. "
                  "No claims of general algorithmic superiority, universal novelty detection, or state-of-the-art performance are made.", scope_banner_style)
    ]]
    banner_table = Table(banner_data, colWidths=[504])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EFF6FF")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#93C5FD")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 8))

    # Abstract Box
    abstract_content = (
        "<b>Abstract—</b>Graph Neural Networks (GNNs) deployed for molecular property prediction and virtual screening "
        "fail unpredictably when evaluated on out-of-distribution (OOD) molecular graphs. Conventional OOD detection benchmarks "
        "collapse all distributional shifts into a single scalar anomaly score, conflating two fundamentally distinct failure modes: "
        "<b>Physical / Structural Chemical Validity</b> (violations of covalent valence constraints, such as pentavalent 'Texas carbons') "
        "and <b>Semantic Scaffold Novelty</b> (synthesizable, chemically valid molecules featuring core ring scaffolds unseen during training).<br/><br/>"
        "In this work, we formalize and benchmark a <b>Dual-Axis OOD Detection framework</b> on the curated <code>processed_v3</code> "
        "benchmark (10,140 graphs spanning Tox21 toxicology and SIDER clinical databases). Axis 1 evaluates structural validity across "
        "controlled graph corruption families, while Axis 2 evaluates semantic novelty under a hybrid split (ring-containing molecules split "
        "by Bemis-Murcko scaffold; ring-free molecules split randomly) alongside cross-domain dataset shift (SIDER). We present a "
        "comprehensive multi-seed validation benchmark comparing a unified multi-task architecture (DAG-NN, Track A: zero corruption exposure), "
        "decoupled validity and novelty models (Member 4, Track B: outlier exposure), multi-seed single-score baselines (S1 Energy, S2 Mahalanobis), "
        "and non-learning size controls.<br/><br/>"
        "<b>Core Findings:</b> (1) <i>Physical Valence Detection:</i> GNN message passing reliably detects physical valence violations without "
        "explicit chemical rule engines, achieving validation AUROC 0.9584 ± 0.0119 under Track A, which receives no corrupted graphs during training, and 0.9966 under outlier exposure Track B, "
        "substantially outperforming the atom-count control on Family A (AUROC 0.5107). (2) <i>Failure-Mode Conflation:</i> Single-score OOD detectors conflate validity "
        "violations with scaffold novelty near chance level (Task X cause-separation AUROC 0.5655 ± 0.0104 for Energy, 0.5677 ± 0.0101 for Mahalanobis), "
        "whereas dual-axis scoring cleanly separates the two failure modes (Task X AUROC 0.9073 ± 0.0069). Critically, however, joint 3-class triage (Task D) "
        "reveals that the macro recall (0.5377 ± 0.0182) masks a severe per-class imbalance: invalid detection is robust (recall 67.62% ± 5.66%) while "
        "novel-chemotype detection is near-zero (recall 2.09% ± 0.52%), because inverted novelty scores displace the operational threshold above the median "
        "novel molecule score. (3) <i>Confounders in Molecular OOD Benchmarks:</i> Standard distance-based novelty estimators yield sub-0.50 raw AUROC on "
        "scaffold validation (0.4539 ± 0.0061). The diagnostic results indicate that reference-distribution shift contributes to the observed novelty-ranking failure: "
        "in-distribution validation molecules are themselves scaffold-disjoint from training (0.0% Murcko scaffold overlap), dispersing them into the "
        "representation periphery (mean distance 7.63 vs. 6.75 for novel molecules). Re-centering the density estimator on an in-distribution reference sharing "
        "generic Murcko framework topology elevates AUROC to 0.7440, while separation against an acyclic reference (1.0000) exposes a dominant "
        "cyclic-vs-acyclic topological confound. Furthermore, apparent raw gains on cross-domain novelty (SIDER) are heavily confounded by heavy atom count "
        "disparities (atom count alone reaches AUROC 0.6280), collapsing to chance level (0.5024) under size- and ring-matched controls."
    )
    abs_table = Table([[Paragraph(abstract_content, abstract_style)]], colWidths=[504])
    abs_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(abs_table)
    story.append(Spacer(1, 10))

    # Section 1: Introduction
    story.append(Paragraph("1. Introduction", h1_style))
    story.append(Paragraph(
        "Deep learning on molecular graphs has become a cornerstone of computational chemistry, enabling property prediction, "
        "bioactivity profiling, and virtual screening directly from molecular topologies G = (V, E). Despite impressive benchmark accuracy on "
        "in-distribution test sets, graph neural networks (GNNs) are notoriously brittle under distributional shift: when queried on inputs outside "
        "their training support, standard architectures frequently emit overconfident, erroneous predictions.", body_style))
    story.append(Paragraph(
        "In computer vision and natural language processing, out-of-distribution (OOD) detection typically addresses semantic class novelty or "
        "domain style shift. In molecular chemistry, however, distributional shift exhibits a unique dual nature: "
        "<br/><b>1. Axis 1 — Structural / Syntactic Validity:</b> Generative algorithms, de novo molecular design pipelines, and graph perturbation "
        "techniques can produce graphs that violate fundamental physical valence constraints (e.g., carbon possessing 5 covalent bonds, colloquially "
        "termed a 'Texas carbon'). A robust molecular model must recognize non-physical graphs as chemically invalid. "
        "<br/><b>2. Axis 2 — Semantic / Scaffold Novelty:</b> Medicinal chemistry frequently explores uncharted chemical space, where synthesized molecules "
        "are physically valid and synthesizable, but possess core ring scaffolds never encountered during training.", body_style))
    story.append(Paragraph(
        "Treating these distinct axes as a single scalar OOD problem obscures whether a detector fails due to physical topological blind spots or lack "
        "of chemical semantics. A single high anomaly score cannot inform a medicinal chemist whether a candidate molecule represents a viable novel "
        "therapeutic chemotype or an unphysical graph artifact. To address this challenge, we present a <b>Diagnostic Benchmarking Study (Option A)</b> "
        "on the curated <code>processed_v3</code> dataset. We investigate whether unified graph architectures can distinguish unphysical graphs from novel "
        "chemotypes without explicit rule engines, and whether conventional molecular OOD benchmarks suffer from hidden topological and size confounders.", body_style))

    # Section 2: Related Work
    story.append(Paragraph("2. Related Work", h1_style))
    story.append(Paragraph(
        "<b>2.1 Graph Neural Networks for Molecular Representation:</b> Message passing neural networks (MPNNs) iteratively aggregate neighborhood embeddings: "
        "<i>h_v^(k) = UPDATE(h_v^(k-1), AGGREGATE({h_u^(k-1) : u in N(v)}))</i>. Prominent architectures include Graph Convolutional Networks (GCN) [8], "
        "Graph Isomorphism Networks (GIN) [13], and edge-featured variants like GINE [5]. Standard message passing is permutation-invariant but does not "
        "inherently guarantee physical valence conservation.", body_style))
    story.append(Paragraph(
        "<b>2.2 Out-of-Distribution Detection in Machine Learning:</b> Baseline post-hoc OOD detection relies on Maximum Softmax Probability (MSP) [3]. "
        "Subsequent developments introduce energy-based scoring [11], feature-space Mahalanobis distances [10], and outlier exposure (OE) [4]. In molecular "
        "graphs, benchmarks such as GOOD [2] and DrugOOD [7] examine covariate and scaffold shifts, but collapse physical validity and semantic novelty "
        "into a single evaluation axis.", body_style))
    story.append(Paragraph(
        "<b>2.3 Molecular Splitting and Chemical Validity:</b> Random splitting in molecular benchmarks causes severe analog leakage. Bemis-Murcko scaffold "
        "clustering [1] separates distinct chemical series. However, when evaluation sets are made scaffold-disjoint from training, the reference in-distribution "
        "validation set is itself composed of novel scaffolds, creating subtle distribution shifts for distance-based density estimators [12].", body_style))

    # Section 3: Problem Formulation & Tasks
    story.append(Paragraph("3. Problem Formulation, Tasks & Training Regimes", h1_style))
    story.append(Paragraph(
        "Let a molecular graph be G = (V, E), where v in V represents an atom with feature vector x_v in R^30, and e_uv in E represents a chemical bond with "
        "feature vector e_uv in R^6. Let M denote the set of all physically valid chemical molecules conforming to covalent valence rules.", body_style))
    story.append(Paragraph(
        "<b>3.1 Taxonomy of Graph Partitions:</b><br/>"
        "• <i>In-Distribution (D_in):</i> Chemically valid graphs (G in M) drawn from the Tox21 training distribution whose Bemis-Murcko scaffold belongs to S_train.<br/>"
        "• <i>Axis 1 (Structural OOD, D_syn):</i> Graphs G not in M violating covalent valence constraints (specifically carbon valence > 4 or oxygen valence > 2).<br/>"
        "• <i>Axis 2 (Semantic OOD, D_sem):</i> Chemically valid molecules (G in M) whose core scaffold belongs to a disjoint held-out set S_ood.<br/>"
        "• <i>Axis 2 (Cross-Domain Shift, D_far):</i> Chemically valid approved pharmaceutical molecules drawn from SIDER.<br/>"
        "• <i>Scope Boundary (Compound Anomalies):</i> Graphs that are simultaneously invalid and novel (D_syn intersect D_sem) do not exist in processed_v3. "
        "Task D evaluates triage between distinct failure modes, not compound anomalies.", body_style))
    story.append(Paragraph(
        "<b>3.2 Training Regimes (Track Disclosures):</b><br/>"
        "• <b>Track A (Strict Zero-Exposure):</b> Models are trained strictly on <code>id_train</code> with multi-task assay labels and masked feature reconstruction. "
        "Models receive zero exposure to corrupted graphs during training.<br/>"
        "• <b>Track B (Outlier Exposure):</b> Models are trained on <code>id_train</code> plus <code>corrupt_train</code> (Families A & B labeled as invalid). "
        "Track B models must not be characterized as unsupervised.", body_style))
    story.append(Paragraph(
        "<b>3.3 Evaluation Tasks Formalized:</b><br/>"
        "• <i>Task V1 (Validity vs. ID):</i> Validity score evaluated on invalid graphs vs. <code>id_val</code> (0 = ID, 1 = Invalid).<br/>"
        "• <i>Task V2 (Validity vs. All-Valid):</i> Validity score evaluated on invalid graphs vs. <code>id_val</code> + novel graphs (0 = Valid, 1 = Invalid).<br/>"
        "• <i>Task N (Raw Novelty):</i> Novelty score evaluated on novel graphs vs. <code>id_val</code> (0 = ID, 1 = Novel).<br/>"
        "• <i>Task NM (Matched Controls):</i> Novelty score evaluated on novel graphs vs. size- and ring-matched ID controls.<br/>"
        "• <i>Task S (Single-Score Detection):</i> Single scalar score evaluated on all OOD inputs vs. <code>id_val</code>.<br/>"
        "• <i>Task X (Cause Separation):</i> Score evaluated on corrupted graphs vs. novel graphs (0 = Novel, 1 = Invalid).<br/>"
        "• <i>Task D (Joint 3-Class Triage):</i> Applies 95th-percentile thresholds (tau_v, tau_n) derived from <code>id_val</code>: "
        "Invalid if S_v >= tau_v; else Novel if S_n >= tau_n; else ID.", body_style))

    # Section 4: Dataset Construction & Integrity
    story.append(Paragraph("4. Dataset Construction, Controlled Corruption & Integrity", h1_style))
    story.append(Paragraph(
        "We utilize the authoritative <code>processed_v3</code> dataset package, comprising <b>10,140 total molecular graphs</b> curated from the "
        "Tox21 toxicology challenge [6] and SIDER clinical database [9], both widely adopted from the MoleculeNet molecular machine learning benchmark collection [12].", body_style))
    story.append(Paragraph(
        "<b>4.1 Hybrid Splitting Protocol:</b> Rather than a purely scaffold-disjoint split across all molecules, <code>processed_v3</code> enforces a "
        "hybrid splitting protocol: (1) <i>Ring-Containing Molecules (75% of ID):</i> Partitioned strictly by Bemis-Murcko scaffold clusters to prevent chemotype leakage. "
        "(2) <i>Ring-Free / Acyclic Molecules (25% of ID):</i> Assigned randomly across training, validation, and test splits. "
        "Exactly two validation molecules were excluded per policy: <code>SID-00482</code> (exact graph duplicate of training molecule) and <code>T21-02289</code> "
        "(exact graph duplicate across sets sharing identical model input with multiple non-standard element training graphs).", body_style))

    # Table 1: Dataset counts
    story.append(Paragraph("<b>Table 1:</b> Authoritative Dataset Structure and Validation Populations (<code>processed_v3</code>)", caption_style))
    t1_headers = ["Split / Set Name", "Raw", "Excl.", "Active", "Underlying Class / Axis", "Role in Benchmark"]
    t1_rows = [
        ["id_train", "5,432", "0", "5,432", "In-Distribution (Tox21)", "Training support (all tracks)"],
        ["id_val", "679", "1", "678", "In-Distribution Validation", "Validation negative reference"],
        ["corrupt_val (Family A)", "272", "0", "272", "Axis 1: Bond Order Corruptions", "Validation positive (Texas carbon)"],
        ["corrupt_val (Family B)", "271", "0", "271", "Axis 1: Bond Relocation Corruptions", "Validation positive (Relocation)"],
        ["novel_scaffold_val", "408", "0", "408", "Axis 2: Held-out Scaffold Families", "Validation positive (Tox21 novel)"],
        ["novel_dataset_val", "225", "1", "224", "Axis 2: Cross-Domain Novelty (SIDER)", "Validation positive (SIDER novel)"],
        ["Matched Scaffold Pairs", "372 pr", "0", "372 pr", "Task NM Matched Controls", "Size/ring-matched validation"],
        ["Matched Dataset Pairs", "209 pr", "0", "209 pr", "Task NM Matched Controls", "Size/ring-matched validation"],
    ]
    t1_table_data = [[Paragraph(f"<b>{h}</b>", table_header) for h in t1_headers]]
    for row in t1_rows:
        t1_table_data.append([Paragraph(cell, table_cell) for cell in row])
    
    t1_table = Table(t1_table_data, colWidths=[114, 38, 38, 44, 135, 135])
    t1_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E293B")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t1_table)
    story.append(Spacer(1, 6))

    # Figure 1: Molecular Size Distribution
    if os.path.exists("eda/plots/01_molecule_sizes.png"):
        story.append(Image("eda/plots/01_molecule_sizes.png", width=504, height=155))
        story.append(Paragraph(
            "<b>Figure 1:</b> Distribution of heavy atom counts and molecular sizes across dataset partitions in <code>processed_v3</code>. "
            "Notice the rightward shift in SIDER (novel dataset validation), which introduces heavy atom count confounding that collapses from "
            "raw AUROC 0.6280 to 0.5024 under matched controls.", fig_caption_style))
        story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>4.2 Controlled Graph Corruption Families:</b><br/>"
        "• <i>Family A (Bond Order Increment):</i> Inserts an additional covalent bond between adjacent carbon atoms, driving carbon valence to 5 ('Texas carbon') "
        "while freezing hydrogen counts. Atom composition and degree sequence are perfectly preserved.<br/>"
        "• <i>Family B (Bond Relocation):</i> Relocates an existing covalent bond onto an already saturated carbon atom, altering local degree distribution.<br/>"
        "• <i>Family C (Heteroatom Substitution — Held-Out Test Only):</i> Replaces a neutral atom with an oxygen, violating oxygen valency. "
        "Quarantined strictly in <code>corrupt_test</code> (N = 181); completely absent from validation.", body_style))

    story.append(Paragraph(
        "<b>4.3 Matched-Control Caliper (Task NM):</b> To eliminate molecular size and ring presence as trivial detection shortcuts, "
        "<code>matched_controls_filtered.csv</code> pairs each novelty graph with an ID validation graph satisfying: (1) Identical ring status "
        "(<code>is_acyclic</code> matches exactly), (2) Heavy atom count difference |Delta atoms| <= 2, and (3) Bond count difference |Delta bonds| <= 2. "
        "Greedy matching covers <b>91.2%</b> of scaffold novelty graphs (372 pairs) and <b>93.3%</b> of dataset novelty graphs (209 pairs).", body_style))

    # Section 5: Standardized Evaluation Protocol
    story.append(Paragraph("5. Standardized Evaluation Protocol & Metrics", h1_style))
    story.append(Paragraph(
        "Given ground-truth binary label y in {0, 1} (0 = ID/Negative, 1 = OOD/Positive) and predicted anomaly score S(G) in R where higher values indicate greater anomaly: "
        "<br/>1. <b>AUROC:</b> Area under the Receiver Operating Characteristic curve, measuring rank separation across all thresholds. "
        "<br/>2. <b>AUPR-Out:</b> Area under the Precision-Recall curve with OOD treated as positive (y = 1). "
        "<br/>3. <b>AUPR-In:</b> Area under the Precision-Recall curve with ID treated as positive (y = 0, evaluated via -S). "
        "<br/>4. <b>FPR@95%TPR:</b> False Positive Rate on ID negatives at the operational decision threshold tau achieving >= 95% True Positive Rate on OOD positives. "
        "<br/>All hyperparameters, early stopping checkpoints, and decision thresholds were derived strictly on validation partitions. The test partition was kept strictly quarantined.", body_style))

    # Section 6: Results & Benchmarking
    story.append(Paragraph("6. Experimental Results & Benchmarking", h1_style))
    story.append(Paragraph("All reported numerical results are reproduced directly from verified validation artifacts.", body_style))

    # Table 2: Primary Benchmark Table (Compact font to guarantee single-line fit)
    story.append(Paragraph("<b>Table 2:</b> Primary Validation Benchmark Performance Across Tasks (Validation Partitions Only)", caption_style))
    t2_headers = ["Model Architecture", "Track", "Task V1 (Fam A)", "Task V1 (Fam B)", "Task V2 (All Valid)", "Task N (Scaffold)", "Task NM (Scaf. Match)", "Task N (Dataset)", "Task NM (Data. Match)"]
    t2_rows = [
        ["DAG-NN (5 seeds)", "Track A", "0.9584 ± 0.0119", "0.8565 ± 0.0083", "0.9074 ± 0.0071", "0.4539 ± 0.0061", "0.4844 ± 0.0046", "0.4521 ± 0.0047", "0.5488 ± 0.0047"],
        ["Member 4 GINE/Mahal", "Track B", "0.9966", "0.9952", "[Unavailable]*", "0.4570", "0.4783", "0.6104", "0.5378"],
        ["C2natoms Control", "N/A", "0.5107", "0.5281", "0.4969", "0.5033", "0.5034", "0.6280", "0.5024"],
        ["C1 Histogram GBDT", "N/A", "0.8890", "0.6570", "[Unavailable]", "0.7520", "0.6890", "0.7280", "0.6640"],
    ]
    t2_table_data = [[Paragraph(f"<b>{h}</b>", table_header_compact) for h in t2_headers]]
    for row in t2_rows:
        t2_table_data.append([Paragraph(cell, table_cell_compact) for cell in row])
    
    t2_table = Table(t2_table_data, colWidths=[88, 38, 54, 54, 56, 54, 54, 53, 53])
    t2_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E293B")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 2),
        ('RIGHTPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t2_table)
    story.append(Paragraph("<font size=6.5><i>*Note: Member 4 handoff lacked cross-axis validity predictions on novel graphs, precluding Task V2 evaluation. Uncertainty is reported as ± standard deviation across 5 seeds for DAG-NN; Member 4 and non-learning baselines are single-run evaluations.</i></font>", body_style))
    story.append(Spacer(1, 4))

    # Figure 2: Valid vs Invalid Size Distribution
    if os.path.exists("eda/plots/09_valid_vs_invalid_size.png"):
        story.append(Image("eda/plots/09_valid_vs_invalid_size.png", width=504, height=155))
        story.append(Paragraph(
            "<b>Figure 2:</b> Size distributions for valid versus corrupted molecular graphs across Family A (bond order increment) and Family B (bond relocation). "
            "The distributions demonstrate that physical valence corruption does not inflate molecule size, precluding atom-count shortcuts for structural validity detection.", fig_caption_style))
        story.append(Spacer(1, 4))

    # Table 3: Single Score Conflation
    story.append(Paragraph("<b>Table 3:</b> Single-Score Conflation vs. Dual-Axis Cause Separation (Task X) Across 5 Seeds", caption_style))
    t3_headers = ["Evaluation Method", "Track", "Task S AUROC (All OOD)", "Task X AUROC (Fam A vs Nov)", "Task X AUROC (ALL vs Nov)", "Task D Macro Recall", "Task D Novel Recall"]
    t3_rows = [
        ["S1 Energy GIN", "Track A", "0.4790 ± 0.0027", "0.5344 ± 0.0133", "0.5655 ± 0.0104", "N/A (Single score)", "N/A"],
        ["S2 Mahalanobis GIN", "Track A", "0.4988 ± 0.0069", "0.5799 ± 0.0077", "0.5677 ± 0.0101", "N/A (Single score)", "N/A"],
        ["DAG-NN (Dual-Axis)", "Track A", "N/A (Dual outputs)", "0.9587 ± 0.0097", "0.9073 ± 0.0069", "0.5377 ± 0.0182", "0.0209 ± 0.0052"],
    ]
    t3_table_data = [[Paragraph(f"<b>{h}</b>", table_header) for h in t3_headers]]
    for row in t3_rows:
        t3_table_data.append([Paragraph(cell, table_cell) for cell in row])
    
    t3_table = Table(t3_table_data, colWidths=[104, 44, 76, 76, 76, 64, 64])
    t3_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E293B")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t3_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>6.1 Analysis of Physical Valence Detection (Axis 1):</b> Under Track A (zero exposure to corruptions), DAG-NN achieves AUROC "
        "<b>0.9584 ± 0.0119</b> on Family A and <b>0.8565 ± 0.0083</b> on Family B. Under Track B (outlier exposure), Member 4's GINE detector reaches "
        "AUROC > 0.995. Both learning models dramatically outperform graph-size controls (C2natoms AUROC 0.5107). However, the C1 histogram probe demonstrates "
        "that raising bond orders produces global bond-type imbalances that a boosted tree can partially detect (AUROC 0.8890) without local message passing.", body_style))

    story.append(Paragraph(
        "<b>6.2 Analysis of Failure-Mode Separation (Task X):</b> Single-score OOD baselines (S1 Energy and S2 Mahalanobis) fail to separate invalid graphs from "
        "novel graphs on the evaluated cohorts, yielding Task X AUROCs of <b>0.5655</b> and <b>0.5677</b> (near chance). Because a single scalar score treats any "
        "deviation from training as anomalous, single-score detectors cannot distinguish physical corruption from chemical novelty. In contrast, DAG-NN's dual-axis "
        "separation achieves Task X AUROC <b>0.9073 ± 0.0069</b>.", body_style))

    story.append(Paragraph(
        "<b>6.3 Analysis of 3-Class Triage (Task D):</b> Under the operational 3-class decision rule, DAG-NN achieves a macro recall of "
        "<b>0.5377 ± 0.0182</b>. While invalid recall is robust (<b>67.62% ± 5.66%</b>), novel recall is severely depressed (<b>2.09% ± 0.52%</b>). "
        "Because raw scaffold novelty scores are inverted, the 95th-percentile threshold tau_n derived from <code>id_val</code> sits above the median of the novel "
        "molecules, resulting in high false negative rates for novel chemotypes.", body_style))

    # Section 7: Discussion & Diagnostics
    story.append(Paragraph("7. Discussion & Forensic Diagnostic Analysis", h1_style))
    story.append(Paragraph(
        "<b>7.1 Mechanism of Scaffold Novelty Failure (AUROC < 0.50):</b> Both DAG-NN (AUROC 0.4539) and Member 4 (AUROC 0.4570) perform below random guessing "
        "on raw scaffold novelty. Forensic investigation confirms that the sub-0.50 AUROC does not arise from an inverted scoring convention. Instead, the diagnostic results "
        "indicate that reference-distribution shift contributes to the observed novelty-ranking failure: "
        "<br/>• <i>Scaffold-Disjoint Split Drift:</i> In <code>processed_v3</code>, <code>id_val</code> contains 509 unique cyclic scaffolds, <b>0.0% of which appear in <code>id_train</code></b>. "
        "<br/>• <i>Embedding Geometry:</i> Because the Gaussian centroid mu_train was fitted strictly on <code>id_train</code>, <code>id_val</code> molecules disperse far into "
        "the periphery (mean Euclidean distance to mu_train = 7.63, total variance 63.35), exceeding the dispersion of curated <code>novel_scaffold_val</code> molecules "
        "(mean distance 6.75, total variance 42.61). Evaluating novel molecules against <code>id_val</code> compares two distinct sets of out-of-training scaffolds, contributing to the inverted ranking.", body_style))

    story.append(Paragraph(
        "<b>7.2 Molecular Size Confounding & Pooling Dynamics:</b><br/>"
        "• <i>Variant A Pilot ([z_G || log(N)]):</i> Appending a linear size coordinate left scaffold AUROC at 0.4454 (Delta = -0.0037), because the 1/N variance "
        "contraction of mean pooling operates across all 128 hidden dimensions.<br/>"
        "• <i>Variant B Pilot (1/sqrt(N) sum h_v):</i> Normalized-sum pooling reversed size correlation from negative to positive (r = +0.231), lifting dataset "
        "AUROC to 0.5623 but worsening scaffold AUROC to 0.4354.<br/>"
        "• <i>Size Bias in Cross-Domain Novelty:</i> On raw SIDER, C2 atom count achieves AUROC <b>0.6280</b>, collapsing to <b>0.5024</b> under matched controls. "
        "Member 4 collapses from <b>0.6104</b> to <b>0.5378</b>, confirming that raw cross-domain gains are heavily confounded by molecular size disparities.", body_style))

    story.append(Paragraph(
        "<b>7.3 Controlled Reference Replacement Diagnostic:</b> To diagnostically evaluate whether the frozen GNN representation z_G possesses structural novelty "
        "discrimination capacity, a controlled reference replacement pilot (single seed, exploratory) evaluated three ID reference conditions:", body_style))

    # Table 4: Controlled ID Reference Replacement Diagnostic
    story.append(Paragraph("<b>Table 4:</b> Controlled ID-Reference Replacement Diagnostic Pilot (Seed 0 Frozen Weights)", caption_style))
    t4_headers = ["Reference Condition", "Neg (N)", "Target (N)", "Paradigm A AUROC", "Paradigm B AUROC", "Paradigm B AUPR", "Diagnostic Finding"]
    t4_rows = [
        ["Full Active ID (id_val)", "678", "408", "0.4491", "0.5219", "0.3894", "Eliminates train-to-val shift; centers ID distribution"],
        ["Generic Murcko Overlap", "190", "408", "0.4395", "0.7440", "0.8697", "Exploratory indication under matched framework"],
        ["Acyclic / Ring-Free Reference", "169", "408", "0.2056", "1.0000", "1.0000", "Topological cycle confounding (trees vs. rings)"],
    ]
    t4_table_data = [[Paragraph(f"<b>{h}</b>", table_header) for h in t4_headers]]
    for row in t4_rows:
        t4_table_data.append([Paragraph(cell, table_cell) for cell in row])
    
    t4_table = Table(t4_table_data, colWidths=[108, 38, 44, 76, 76, 68, 94])
    t4_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E293B")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t4_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "• <b>Paradigm B Generic Overlap (AUROC 0.7440):</b> When the density estimator is re-centered on the 190 ID validation molecules that share generic "
        "carbon frameworks with training, novel molecules are recognized as geometric outliers (mean distance 164.9 vs. 74.0). This exploratory, single-seed result "
        "suggests that the frozen representation may contain structural information useful for distinguishing the selected novel molecules under this particular "
        "reference condition. Because the reference cohort is restricted and the density estimator is re-estimated, this result does not establish general "
        "scaffold-novelty detection.<br/>"
        "• <b>Paradigm B Acyclic Reference (AUROC 1.0000):</b> Centering on ring-free ID molecules achieves perfect separation (score gap +169.5). "
        "This is explicitly disclosed as <b>topological cycle detection rather than scaffold novelty detection</b>: 100% of novel molecules contain rings, "
        "while 100% of acyclic reference molecules are trees.", body_style))

    # Section 8: Limitations
    story.append(Paragraph("8. Limitations, Threats to Validity & Reproducibility Boundaries", h1_style))
    story.append(Paragraph(
        "1. <b>Benchmark Reference Shift:</b> Scaffold-disjoint splitting in validation inherently penalizes distance-based density estimators fitted on training data.<br/>"
        "2. <b>Compound Anomaly Scope:</b> <code>processed_v3</code> contains no simultaneous corrupted-novel molecules; compound failure modes remain uncharacterized.<br/>"
        "3. <b>Member 4 Checkpoint Boundaries:</b> Model weights for Member 4 were unbundled in the repository handoff. Member 4 results are prediction-level verified only. "
        "Cross-axis prediction scores were absent, preventing Task D evaluation on Member 4.<br/>"
        "4. <b>Validation-Only Evidence Scope:</b> All reported results, threshold selections, and diagnostic pilots are strictly validation-only. "
        "The final test partition remains quarantined.", body_style))

    # Section 9: Conclusion
    story.append(Paragraph("9. Conclusion", h1_style))
    story.append(Paragraph(
        "This diagnostic benchmarking study formalizes the Dual-Axis OOD Detection paradigm for molecular graphs. By decoupling physical chemical validity from "
        "semantic novelty, we demonstrate that GNNs can reliably identify physical valence corruptions without explicit rule engines, resolving the failure-mode "
        "conflation of single-score detectors (improving cause separation from 0.56 to >0.90). However, joint 3-class triage (Task D) exposes a critical limitation: "
        "the macro recall of 0.5377 ± 0.0182 conceals near-zero novel-chemotype detection (recall 2.09% ± 0.52%), because the inverted novelty score distribution "
        "displaces the operational threshold beyond the reach of most novel molecules. Furthermore, our forensic diagnostic pilots indicate that reference-distribution shift "
        "from scaffold-disjoint validation partitioning, molecular size disparities, and topological cycle confounding contribute substantially to the observed novelty-ranking behaviors. "
        "Establishing reliable molecular AI requires moving beyond single scalar OOD metrics toward multi-axis evaluation with rigorous matched controls, operational "
        "thresholds calibrated for each axis independently, and honest per-class reporting.", body_style))

    # Section 10: References
    story.append(Paragraph("References", h1_style))
    refs = [
        "1. Bemis, G. W., & Murcko, M. A. (1996). The properties of known drugs. 1. Molecular frameworks. <i>Journal of Medicinal Chemistry</i>, 39(15), 2887-2893.",
        "2. Gui, S., Li, X., Wang, L., & Ji, S. (2022). GOOD: A graph out-of-distribution benchmark. <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, 35, 2059-2073.",
        "3. Hendrycks, D., & Gimpel, K. (2017). A baseline for detecting misclassified and out-of-distribution examples in neural networks. <i>International Conference on Learning Representations (ICLR)</i>.",
        "4. Hendrycks, D., Mazeika, M., & Dietterich, T. (2019). Deep anomaly detection with outlier exposure. <i>International Conference on Learning Representations (ICLR)</i>.",
        "5. Hu, W., Liu, B., Gomes, J., Zitnik, M., Liang, P., Pande, V., & Leskovec, J. (2020). Strategies for pre-training graph neural networks. <i>International Conference on Learning Representations (ICLR)</i>.",
        "6. Huang, W. et al. (2021). Tox21 Data Challenge: High-throughput screening for toxicology. <i>Frontiers in Environmental Science</i>, 9, 602606.",
        "7. Ji, Y., Zhang, L., Wang, J., Wu, B., Huang, L., Xu, T., Rong, Y., Li, F., & Tsung, F. (2023). DrugOOD: Out-of-distribution (OOD) dataset curator and benchmark for AI-aided drug discovery. <i>International Conference on Learning Representations (ICLR)</i>.",
        "8. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. <i>International Conference on Learning Representations (ICLR)</i>.",
        "9. Kuhn, M., Letunic, I., Jensen, L. J., & Bork, P. (2016). The SIDER database of drugs and side effects. <i>Nucleic Acids Research</i>, 44(D1), D1075-D1079.",
        "10. Lee, K., Lee, K., Lee, H., & Shin, J. (2018). A simple unified framework for detecting out-of-distribution samples and adversarial attacks. <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, 31.",
        "11. Liu, W., Wang, X., Owens, J., & Li, Y. (2020). Energy-based out-of-distribution detection. <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, 33, 21464-21475.",
        "12. Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Hao, C., Leswing, K., Xie, Z., Yao, V., & Pande, V. S. (2018). MoleculeNet: A benchmark for molecular machine learning. <i>Chemical Science</i>, 9(2), 513-530.",
        "13. Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are graph neural networks? <i>International Conference on Learning Representations (ICLR)</i>."
    ]
    for r in refs:
        story.append(Paragraph(r, ref_style))

    # Build the PDF document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated academic PDF: {output_pdf_path}")

if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "paper/final_manuscript.pdf"
    build_pdf(out_path)
