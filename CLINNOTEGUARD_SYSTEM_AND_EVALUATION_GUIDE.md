# ClinNoteGuard: Comprehensive System Architecture, AI Modules, and Evaluation Guide

> **Notice**: Synthetic, rule-labeled evaluation. Not a clinically validated diagnostic result.  
> **Concept Layer**: Curated concept layer (`verify/concepts/concept_hierarchy.json` + `CuratedDictionaryLinker`).  
> **Trained Checkpoint**: `checkpoints/verifier_demo.pt`

---

## 1. Executive Glossary & Terminology Index

| Term / Keyword | Plain English Clinical & Technical Definition |
| :--- | :--- |
| **Ablation Study** | A scientific experiment in machine learning where specific components or features of a system are systematically stripped away or modified one at a time. This isolates cause-and-effect to prove which modules (e.g., GNN, temporal rules, attention bias) actually drive accuracy rather than relying on guesswork. |
| **Atomic Claim** | A single, irreducible clinical statement extracted from a medical note (e.g., *"Patient takes Atorvastatin 20 mg oral daily"*). Decomposing notes into atomic propositions isolates errors to specific facts rather than accepting or rejecting an entire paragraph. |
| **EHR / EMR** | Electronic Health Record / Electronic Medical Record. Structured and unstructured historical patient data: past encounters, medication orders, problem lists, lab results, allergies, and progress notes with timestamps. |
| **CUI (Concept Unique Identifier)** | A universal alphanumeric code assigned to medical concepts (e.g., `C0286651` for Atorvastatin) regardless of whether a clinician writes "Lipitor", "atorvastatin", or "Atorva". |
| **Curated Concept Layer** | An offline medical knowledge graph representing concept taxonomies, brand-generic mappings, and class subsumptions used to test hierarchical clinical inference without proprietary licensing dependencies. |
| **Typed Validity** | A temporal reconciliation mechanism that applies distinct clinical persistence rules based on the attribute type: medications supersede upon dose changes or stops, whereas chronic diagnoses and allergies persist indefinitely unless explicitly resolved. |
| **EvidenceGraph** | A heterogeneous, multi-relational graph constructed for each claim, connecting the claim node to candidate EHR and transcript evidence nodes, medical concept nodes, and temporal decay edges. |
| **Dense GIN (Graph Isomorphism Network)** | A graph neural network layer that updates node embeddings by aggregating messages from immediate neighbors while preserving distinct multiset graph structures. |
| **Claim-Conditioned Transformer** | A sequence transformer verifier where the sequence starts with `[CLS]` and the claim token, followed by evidence tokens, modulated by learned edge-type and continuous time attention biases. |
| **Reciprocal Rank Fusion (RRF)** | An unsupervised rank-aggregation algorithm ($k=60$) combining disparate search rankings (lexical BM25 and dense semantic embeddings) without needing score calibration: $\text{RRF}(d) = \sum_{m} \frac{1}{60 + r_m(d)}$. |
| **Cross-Encoder Reranker** | A deep transformer that scores query-document pairs simultaneously using full cross-attention (`ms-marco-MiniLM-L-6-v2`), capturing intricate lexical nuances (e.g., dose numbers, lateralities) that bi-encoders miss. |
| **Temperature Scaling** | A post-processing calibration technique that divides raw model logits by a learned scalar temperature parameter $T > 0$. It adjusts overconfident softmax probabilities without altering class ranking or accuracy. |
| **Expected Calibration Error (ECE)** | A metric measuring the difference between predicted confidence and empirical accuracy across probability bins. An ECE of $0.0336$ means predicted probabilities match real-world accuracy within $3.4\%$. |
| **Date-Shuffle Sensitivity Test** | An evaluation where patient encounter dates are randomly permuted. If the model truly utilizes time features, performance on time-sensitive classes (`HISTORY_SUPPORTED`, `STALE`) will drop significantly. |
| **Counterfactual Date-Flip Pair** | A test setup evaluating the exact same patient claim and EHR history across two different encounter dates (e.g., before vs. after a drug titration) to verify whether the predicted label flips dynamically. |
| **Temporal Sensitivity Index (TSI)** | The empirical probability that the model's predicted label changes given that the clinical ground truth changed across a temporal boundary: $\text{TSI} = P(\hat{Y}_{t_1} \ne \hat{Y}_{t_2} \mid Y_{t_1}^* \ne Y_{t_2}^*)$. |
| **Stale-as-Supported Rate** | The percentage of expired or discontinued historical medical facts erroneously classified as active support. In clinical safety, this must be zero. |
| **Proposition Decomposition** | Splitting compound clinical sentences with multiple clauses and conjunctions into isolated atomic propositions using dependency parsing and syntax splitting. |

---

## 2. What is an Ablation Study and What Did We Evaluate?

### Why Do an Ablation Study?
In complex AI systems, researchers frequently combine multiple architectures—GNNs, cross-encoders, rule engines, temporal features, and attention biases. Without an ablation study, it is impossible to determine whether high performance is driven by the sophisticated neural architecture or simply by memorizing dataset artifacts. 

By systematically "turning off" one component at a time, the ablation study provides scientific proof of which components are essential.

```mermaid
flowchart TD
    A[Full Model: A6] -->|Disable GNN| A3[Transformer No GNN]
    A -->|Disable Typed Rules| A5[Uniform Supersede]
    A -->|Disable Dialogue| A7[EHR-Only]
    A -->|Disable Time Features| A8[No Time Feat]
    A -->|Replace with Simple MLP| A1[MLP Baseline]
    A1 -->|Apply Recency Filter| A2[MLP Recency-Only]
```

### The 9 Ablation Architectures Evaluated (A0 to A8)

1. **A0: Majority Class (Chance Baseline)**
   * *What it does*: Ignores all inputs and predicts the most frequent label in the training set.
   * *Result*: **Macro-F1: 0.0476, Accuracy: 16.67%**. Establishes the statistical random chance floor.
2. **A1: MLP Baseline**
   * *What it does*: Simple 2-layer MLP trained on concatenated vectors: `[claim text emb, top-1 evidence emb, cross-encoder score, dose flag]`.
   * *Result*: **Macro-F1: 0.4972, Accuracy: 50.83%**. Captures surface semantic similarity and dose matches, but lacks multi-hop reasoning.
3. **A2: MLP Recency Filter**
   * *What it does*: Same as A1, but strictly retains only the single most recent EHR record by timestamp, discarding older history.
   * *Result*: **Macro-F1: 0.2976, Accuracy: 31.67%**. **Drops by 20 points** because chronic conditions (e.g., Hypertension diagnosed 3 years ago) are discarded when newer unrelated records exist.
4. **A3: Sequence Transformer (No GNN, No Biases)**
   * *What it does*: Standard transformer taking flat token sequence `[CLS, claim, ev_1, ..., ev_K]` without graph convolutions or custom attention biases.
   * *Result*: **Macro-F1: 0.1024, Accuracy: 19.53%**. Suffers because flat sequence attention fails to infer structural clinical relationships.
5. **A4: Transformer + 3-Layer Dense GIN**
   * *What it does*: Adds a 3-layer Dense GIN over the EvidenceGraph before sequence projection.
   * *Result*: **Macro-F1: 0.1634, Accuracy: 25.83%**. **+0.061 F1 gain**, confirming message passing along ontology and evidence edges helps link concepts.
6. **A5: Uniform Supersede Rule**
   * *What it does*: Removes typed clinical validity rules and treats all clinical events with "latest event wins" semantics.
   * *Result*: **Macro-F1: 0.1494, Accuracy: 24.43%**. Erroneously supersedes persistent allergies and chronic illnesses.
7. **A6: Full Model (Stage 8 Checkpoint)**
   * *What it does*: 3-layer Dense GIN + Claim-Conditioned Transformer with edge-type attention bias, softplus temporal decay bias, and clinician-editable typed validity rules.
   * *Result*: **Macro-F1: 0.6098, Accuracy: 60.00%**. Outperforms baseline A1 by **11.26 points** and A2 by **31.22 points**.
8. **A7: EHR-Only (No Dialogue Evidence)**
   * *What it does*: Excludes consultation transcript evidence from candidate pools.
   * *Result*: **Macro-F1: 0.0764, Accuracy: 17.13%**. Severe collapse because `TRANSCRIPT_SUPPORTED` claims cannot be verified.
9. **A8: No Time Features**
   * *What it does*: Zeroes out date gap days and date bucket embeddings.
   * *Result*: **Macro-F1: 0.1194, Accuracy: 21.43%**. The model becomes blind to whether an event occurred yesterday or 5 years ago.

---

## 3. Comprehensive Breakdown of All Evaluations Performed

```
+----------------------------------------------------------------------------------------------------+
|                                    SUMMARY OF EVALUATION SUITE                                     |
+------------------------------------+--------------------------------+------------------------------+
| Evaluation Category                | Metric / Result                | Clinical & Scientific Value  |
+------------------------------------+--------------------------------+------------------------------+
| 1. Claim-Only Leakage Test         | Macro-F1 = 0.3802 (< 0.95 cap) | Eliminates syntax shortcuts  |
| 2. Original EHR Row Byte-Identity  | 259 canonical rows preserved   | Zero dataset corruption      |
| 3. Temporal Validity Delta (A1-A2) | +19.96 F1 advantage for typed  | Proves need for typed rules  |
| 4. Test Set Accuracy (Full Model)  | 60.00% (N=360 test claims)     | Balanced 6-class prediction  |
| 5. Test Set Macro-F1 (Full Model)  | 0.6098 (All classes non-zero)  | High holistic faithfulness   |
| 6. Stale-as-Supported Rate         | 0.0000 (Zero false positives)  | Critical patient safety      |
| 7. Date-Shuffle Sensitivity Drop   | -0.2650 F1 on HISTORY_SUPPORT  | Proves active time use       |
| 8. Calibration Error (ECE)         | ECE = 0.0336 (T = 1.2597)      | Well-calibrated probabilities|
| 9. Candidate Pool Sweep            | Peaks at K = 20 - 50           | Validates retrieval depth    |
| 10. Stages 1-4 Benchmark Guard     | 100% Top-1, 1.0000 MRR         | Zero regression on prior art |
+------------------------------------+--------------------------------+------------------------------+
```

### Visual Figures Generated (`results/plots/`)
* [`results/plots/evaluation_dashboard.png`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/results/plots/evaluation_dashboard.png): 4-panel dashboard containing the normalized confusion matrix, ablation progression, date-shuffle sensitivity, and calibration curve.
* [`results/plots/confusion_matrix.png`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/results/plots/confusion_matrix.png): Standalone 6x6 confusion matrix.
* [`results/plots/ablation_study.png`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/results/plots/ablation_study.png): Comparative bar chart across A0–A8.
* [`results/plots/date_shuffle_sensitivity.png`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/results/plots/date_shuffle_sensitivity.png): Highlighting the -51% F1 drop under date perturbation.
* [`results/plots/calibration_reliability.png`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/results/plots/calibration_reliability.png): Reliability diagram for temperature scaling.

---

## 4. Mapping Diagram Blocks to Files and Folders (`stages/`)

The system architecture diagram from `System architucture.drawio` is implemented across modular stage directories under [`stages/`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/stages/):

```mermaid
graph LR
    subgraph "Section A: Audio"
        A1[Cell Phone / Mic] --> S1[stages/stage1_audio/diarizer.py]
    end
    subgraph "Section B: Note Generation"
        S1 --> S2[stages/stage2_soap/generator.py]
    end
    subgraph "Section C: Verification Pipeline"
        S2 --> S3[stages/stage3_claims/extractor.py]
        S3 --> S4[stages/stage4_retrieval/retriever.py]
        S4 --> S5[stages/stage5_concepts/service.py]
        S5 --> S6[stages/stage6_validity/service.py]
        S6 --> S7[stages/stage7_8_verifier/service.py]
    end
    subgraph "Section D: Clinician Delivery"
        S7 --> D1[labeled_note.json / API View]
    end
```

### Detailed File-by-File Mapping

| Diagram Block | Implemented Module File | Responsibility & Implementation Details |
| :--- | :--- | :--- |
| **A. Audio Conversation & Speech to Text** | [`stages/stage1_audio/diarizer.py`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/stages/stage1_audio/diarizer.py) | Wraps `faster-whisper` (CTranslate2 CUDA float16) and `SpeechBrain ECAPA-TDNN` speaker encoder with KMeans clustering to output diarized speaker segments. |
| **B. Note Generation (Clinical LLM Decoder)** | [`stages/stage2_soap/generator.py`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/stages/stage2_soap/generator.py) | Synthesizes 4-section clinical SOAP notes from conversation transcripts with robust clinical NLP fallback. |
| **C1. Atomic Claim Extraction** | [`stages/stage3_claims/extractor.py`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/stages/stage3_claims/extractor.py) | Sentence and clause proposition decomposition extracting unique claim IDs (`CLM-001`, ...) with section mapping. |
| **C2. Hybrid Retrieval + Cross-Encoder** | [`stages/stage4_retrieval/retriever.py`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/stages/stage4_retrieval/retriever.py) | Implements BM25 + Dense MiniLM RRF ($k=60$) candidate retrieval and `ms-marco-MiniLM-L-6-v2` cross-encoder reranking with dose agreement extraction. |
| **C3. Clinical Entity Linking** | [`stages/stage5_concepts/service.py`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/stages/stage5_concepts/service.py) | Wraps `CuratedDictionaryLinker` and `ConceptHierarchy` to extract CUIs, infer attribute types, and evaluate taxonomic subsumption. |
| **C4. Temporal Validity & Graph Builder** | [`stages/stage6_validity/service.py`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/stages/stage6_validity/service.py) | Resolves temporal validity states (`CURRENT` vs `SUPERSEDED`) based on `validity_rules.yaml` and constructs multi-relational `EvidenceGraph` tensors. |
| **C5. GIN + Transformer Verifier** | [`stages/stage7_8_verifier/service.py`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/stages/stage7_8_verifier/service.py) | Loads `checkpoints/verifier_demo.pt`, executes Dense GIN + Transformer forward pass with attention biases, and outputs temperature-calibrated verdicts. |
| **Master Orchestrator Pipeline** | [`stages/pipeline.py`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/stages/pipeline.py) | End-to-end multi-stage pipeline coordinator (`ClinicalVerificationPipeline`). |

### How Django Calls These Modules
In any Django view, Celery background worker, or management command:

```python
# In api/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from stages import ClinicalVerificationPipeline

class VerifyConsultationView(APIView):
    def post(self, request):
        transcript_data = request.data.get("transcript")
        patient_ehr = request.data.get("patient_ehr_records")
        patient_id = request.data.get("patient_id")
        encounter_date = request.data.get("encounter_date", "2024-01-15")

        pipeline = ClinicalVerificationPipeline()
        result = pipeline.run_session(
            transcript_data=transcript_data,
            patient_records=patient_ehr,
            patient_id=patient_id,
            encounter_date=encounter_date,
        )
        return Response(result)
```

---

## 5. Limitations of Current AI & What Medical Ontology Hierarchy Adds

### Critical Issues with Current AI Systems
1. **Synthetic Rule-Templated Training**:
   * *Problem*: Training on templated sentences risks the model learning synthetic carrier regularities.
   * *Consequence*: Real doctor speech contains messy conversational disfluencies, stuttering, and complex physician jargon.
2. **Small Curated Ontology Vocabulary**:
   * *Problem*: The curated concept layer covers common cardiac, diabetes, respiratory, and antibiotic entities (~50–100 concepts).
   * *Consequence*: Unseen medications or rare conditions cannot have their taxonomy resolved automatically.
3. **Candidate Truncation Windows ($K=20$)**:
   * *Problem*: Long-term patient charts contain hundreds of past notes and lab results. Truncating to top-20 candidates can miss vital past records unless retrieval is tuned.
4. **Omission Blind-Spot**:
   * *Problem*: The verifier checks whether claims written in the note are true. It does not detect clinical discussion points mentioned in dialogue that the doctor *forgot* to document.

### What the Full Medical Knowledge Layer Adds
* **Automated Hierarchical Subsumption**:
  * Instead of hardcoded dictionary rules, a medical ontology knows that *Amoxicillin* is a *Penicillin-class antibiotic*, *Lisinopril* is an *ACE inhibitor*, and *Metoprolol* is a *Beta-blocker*.
* **Drug-Drug Contraindications & Interaction Checking**:
  * Cross-links known interactions (e.g., Warfarin + Ibuprofen) to automatically classify claims suggesting unsafe combinations as `CLINICAL_CONFLICT`.
* **Standardized Clinical Vocabularies**:
  * Maps interchangeably across RxNorm (drugs), SNOMED-CT (clinical findings/procedures), ICD-10-CM (diagnoses), and LOINC (laboratory tests).
* **Consumer Health Vocabulary (CHV) Lay-Language Translation**:
  * Maps patient lay expressions ("water pill", "sugar trouble", "fluid in lungs") directly to canonical medical concepts (`furosemide`, `type 2 diabetes`, `congestive heart failure`).

---

## 6. Longitudinal EHR $\rightarrow$ Transcript vs. The ACI-BENCH Paradigm

### The ACI-BENCH Approach (Single-Visit Vacuum)
* **What it does**: Takes an isolated consultation recording $\rightarrow$ generates a SOAP note $\rightarrow$ evaluates faithfulness purely against that single visit's dialogue.
* **Fundamental Flaws**:
  1. *Ignores Real Clinical Practice*: Doctors do not see established patients with a blank slate; they review the patient's existing chart first.
  2. *Cannot Detect Stale Information*: If a note says *"Patient takes 10 mg Atorvastatin"*, but in the visit the doctor said *"Let's increase your dose"*, an isolated dialogue model cannot tell whether 10 mg was the old dose without past EHR context.
  3. *False Hallucination Penalties*: If a doctor writes *"Patient has known type 2 diabetes"* based on historical chart review without re-diagnosing it during a brief visit, ACI-BENCH flags it as an unsupported hallucination.

### The Longitudinal EHR $\rightarrow$ Transcript Paradigm (Why It Is Exponentially Better)

```mermaid
graph TD
    subgraph "Longitudinal Paradigm"
        EHR[5-10 Year Longitudinal EHR Chart] -->|Contextualizes| Diag[Doctor-Patient Consultation]
        Diag -->|Generates| Note[SOAP Clinical Note]
        Note -->|Verifies against Both| Verifier[ClinNoteGuard Multi-Source Verifier]
        EHR --> Verifier
        Diag --> Verifier
    end
```

1. **True Clinical Grounding**: Models how medicine is actually practiced—synthesizing current dialogue with historical records.
2. **Accurate Temporal Reconciliation**: Distinguishes between active, current therapy and superseded, expired prescriptions (`HISTORY_SUPPORTED` vs `STALE`).
3. **Medication Titration Tracking**: Reliably verifies whether a note correctly documents dose changes, discontinuations, and new starts.
4. **Counterfactual Clinical Robustness**: Enables realistic stress-testing across temporal boundaries (e.g., evaluating notes under encounter date shifts).
