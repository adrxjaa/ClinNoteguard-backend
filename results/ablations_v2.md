# ClinNoteGuard Verification Ablation & Evaluation Report (v2 — Synthea + Gemini)

**Dataset**: 170 Synthea+Gemini patients (110 train / 26 val / 34 held-out test, N=575 test claims)  
**Ontology**: Curated concept layer (`concept_hierarchy.json`)  

---

## 1. System Architecture & Real Inference Ablations

| Config | Description | Test Accuracy | Macro-F1 | Stale-as-Supported |
| :--- | :--- | :---: | :---: | :---: |
| **A0 Majority Class** | Always predict training majority class | 0.2313 | **0.0626** | 1.0000 |
| **A1 MLP Base** | MLP on [claim emb, top-1 evidence emb, BM25 score, dose tag] | 0.9130 | **0.8890** | 0.4186 |
| **A2 MLP Recency** | A1 with recency_only evidence filter | 0.9026 | **0.8781** | 0.3721 |
| **A3 Transf (No GNN)** | Claim-Conditioned Transformer without DenseGIN, edge bias, or time bias | 0.9443 | **0.9224** | 0.0581 |
| **A6 Full (GNN+Time)** | 3-layer DenseGIN + Transformer with edge bias & per-type continuous time decay | 0.9809 | **0.9766** | 0.0698 |
| **A7 EHR-Only** | A6 with transcript evidence excluded from candidate pool | 0.9774 | **0.9728** | 0.0698 |
| **A8 No Time Feat** | A6 with log gap days, date buckets, and time bias zeroed out | 0.9339 | **0.9149** | 0.2791 |

---

## 2. Candidate Pool Size Sweep (Top-K)

| Top-K | Test Accuracy | Macro-F1 | Stale-as-Supported |
| :---: | :---: | :---: | :---: |
| **K = 3** | 0.9530 | **0.9395** | 0.2442 |
| **K = 5** | 0.9670 | **0.9580** | 0.1628 |
| **K = 10** | 0.9809 | **0.9766** | 0.0698 |
| **K = 15** | 0.9809 | **0.9766** | 0.0698 |
| **K = 20** | 0.9826 | **0.9790** | 0.0581 |

---

## 3. Calibration & Temporal Sensitivity
- **Temperature Scaling**: $T = 1.4598$ | Uncalibrated ECE = `0.0160` -> Calibrated ECE = `0.0094`
- **Date-Shuffle Sensitivity**: `HISTORY_SUPPORTED` F1 `0.9322` -> `0.7547`, `STALE` F1 `0.9581` -> `0.7831`
