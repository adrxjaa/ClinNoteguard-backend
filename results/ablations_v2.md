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
| **A3 Transf (No GNN)** | Claim-Conditioned Transformer without DenseGIN, edge bias, or time bias | 0.9687 | **0.9599** | 0.0698 |
| **A6 Full (GNN+Time)** | 3-layer DenseGIN + Transformer with edge bias & per-type continuous time decay | 0.9687 | **0.9590** | 0.1047 |
| **A7 EHR-Only** | A6 with transcript evidence excluded from candidate pool | 0.9652 | **0.9539** | 0.1047 |
| **A8 No Time Feat** | A6 with log gap days, date buckets, and time bias zeroed out | 0.9009 | **0.8094** | 0.0116 |

---

## 2. Candidate Pool Size Sweep (Top-K)

| Top-K | Test Accuracy | Macro-F1 | Stale-as-Supported |
| :---: | :---: | :---: | :---: |
| **K = 3** | 0.9565 | **0.9439** | 0.1977 |
| **K = 5** | 0.9722 | **0.9647** | 0.1163 |
| **K = 10** | 0.9652 | **0.9543** | 0.0930 |
| **K = 15** | 0.9687 | **0.9590** | 0.1047 |
| **K = 20** | 0.9722 | **0.9639** | 0.0930 |

---

## 3. Calibration & Temporal Sensitivity
- **Temperature Scaling**: $T = 1.4901$ | Uncalibrated ECE = `0.0257` -> Calibrated ECE = `0.0161`
- **Date-Shuffle Sensitivity**: `HISTORY_SUPPORTED` F1 `0.8696` -> `0.5679`, `STALE` F1 `0.9112` -> `0.7943`
