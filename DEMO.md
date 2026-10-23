# ClinNoteGuard Verification Pipeline: End-to-End Demo Guide (Stages 5-8)

> **Notice**: Synthetic, rule-labeled evaluation. Not a clinically validated diagnostic result.  
> **Concept Layer**: Curated concept layer (`verify/concepts/concept_hierarchy.json` + `CuratedDictionaryLinker`).  
> **Model Checkpoint**: `checkpoints/verifier_demo.pt`

---

## 1. Quick Start Commands

### Milestone 3: Model Training, Temperature Calibration & Evaluation
Trains the 3-layer Dense GIN + Claim-Conditioned Transformer with custom attention biases, fits temperature scaling calibration, and runs the Date-Shuffle Sensitivity Test:
```bash
./venv/bin/python scripts/train_verifier.py
```
* **Output Artifacts**: `checkpoints/verifier_demo.pt`
* **Performance Output**: Test Accuracy, Macro-F1, ECE, Per-Class F1 table, and Date-Shuffle drops.

To generate the ablation study report across configurations A0 through A8 and the candidate pool size sweep $K \in \{3, 5, 10, 20, 50\}$:
```bash
./venv/bin/python scripts/run_ablations.py
```
* **Output Artifact**: [`results/ablations.md`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/results/ablations.md)

To generate all 6 publication-quality evaluation figures:
```bash
./venv/bin/python scripts/generate_evaluation_plots.py
```
* **Output Artifacts**: [`results/plots/`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/results/plots/)

---

### Milestone 4: End-to-End Session Verification CLI
Verifies a patient consultation session using the full 5-stage pipeline:
$$\text{Transcript} \rightarrow \text{SOAP Note} \rightarrow \text{Atomic Claims} \rightarrow \text{Hybrid Retrieval (EHR + Dialogue)} \rightarrow \text{Cross-Encoder} \rightarrow \text{EvidenceGraph} \rightarrow \text{Neural Verifier}$$

```bash
./venv/bin/python manage.py verify_session \
  --transcript data/samples/sample_transcript.json \
  --patient P_D2N001 \
  --output data/samples/labeled_note.json
```
* **Console Output**: Formatted ASCII table with the banner `synthetic-trained, unvalidated`.
* **Output Artifact**: [`data/samples/labeled_note.json`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/data/samples/labeled_note.json)

---

### Milestone 5: Date-Flip Clinical Demonstration
Demonstrates the model's sensitivity to temporal state transitions (dose changes, discontinued drugs, persistent allergies) across identical patient evidence:
```bash
./venv/bin/python scripts/demo_date_flip.py
```
* Shows side-by-side verification verdicts before vs after medication titration and discontinuation.

---

### Counterfactual Pair Generation
Generates paired counterfactual evaluation instances straddling state-change events:
```bash
./venv/bin/python scripts/make_counterfactual_pairs.py
```
* **Output Artifact**: [`data/counterfactual_pairs.jsonl`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/data/counterfactual_pairs.jsonl) (flagged with `needs_doctor_review=true`).

---

## 2. Output Schema Specifications

### `labeled_note.json`
```json
{
  "patient_id": "P_D2N001",
  "encounter_date": "2024-01-15",
  "verified_at": "2024-01-15T09:00:00Z",
  "checkpoint": "checkpoints/verifier_demo.pt",
  "claims": [
    {
      "claim_id": "CLM-001",
      "section": "Plan",
      "claim_text": "Patient is prescribed Azithromycin 250 mg daily.",
      "label": "TRANSCRIPT_SUPPORTED",
      "calibrated_confidence": 0.9421,
      "top_evidence": [
        {
          "record_id": "TRANS_D2N001_U000",
          "source": "TRANSCRIPT",
          "attention_weight": 0.8124
        }
      ],
      "validity_status": "CURRENT",
      "checkpoint": "checkpoints/verifier_demo.pt"
    }
  ]
}
```

### `data/doctor_review_sheet.csv`
Contains 150 unperturbed sentences extracted from AI-synthesized SOAP notes on test encounters, strictly quarantined from training:
* `claim`: Raw claim extracted from note.
* `transcript_snippet`: Doctor-patient discussion segment.
* `ehr_facts_as_of_encounter`: Historical patient EHR context as of encounter date.
* `blank_label`: Empty column for clinical expert annotation.

---

## 3. Configurable Ablation Flags

All settings are centrally configured in [`verify/configs/verify_config.yaml`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/verify/configs/verify_config.yaml):
* `ontology_backend`: `"curated"` (curated concept layer) | `"umls"`
* `use_gnn`: `true` | `false` (ablation A3 vs A4)
* `use_edge_bias`: `true` | `false` (learned edge-type attention bias)
* `time_bias`: `"per_type"` | `"shared"` | `"off"` (softplus temporal decay)
* `validity_mode`: `"typed"` | `"uniform_supersede"` | `"recency_only"` | `"none"`
* `use_stale_class`: `true` | `false` (5 vs 6 class prediction)

---

## 4. Known Limitations & Production Roadmap

1. **Curated Concept Layer vs UMLS**:
   * Current deployment uses the offline curated concept hierarchy (`verify/concepts/concept_hierarchy.json`) and dictionary linker. Full UMLS integration (MRCONSO / MRREL) will automatically engage once licensed data is placed in the designated path.
2. **Synthetic Training Data**:
   * Model weights are trained on rule-templated multi-turn longitudinal timelines. For clinical production deployment, fine-tuning on doctor-annotated clinical notes (e.g. ACI-BENCH expert labels) is required.
3. **Absence of Omission Detection**:
   * The pipeline verifies factual claims present in the note; detection of clinical discussion points omitted from the note will be addressed in a future extension.

---

## 5. Modular AI Architecture (`stages/` Package)

Each AI component is isolated into clean, decoupled stage folders designed for seamless invocation by Django views, background tasks, or management commands:

| Stage Directory | Module Role | Primary Python Entry Point |
| :--- | :--- | :--- |
| [`stages/stage1_audio/`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/stages/stage1_audio/) | Stage 1: Audio Transcription & Speaker Diarization | `get_stage1_diarizer().process(audio_path)` |
| [`stages/stage2_soap/`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/stages/stage2_soap/) | Stage 2: SOAP Clinical Note Generation | `get_stage2_soap_generator().generate(transcript)` |
| [`stages/stage3_claims/`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/stages/stage3_claims/) | Stage 3A: Atomic Claim Extraction & Proposition Decomposition | `get_stage3_claim_extractor().extract(soap_note, transcript)` |
| [`stages/stage4_retrieval/`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/stages/stage4_retrieval/) | Stage 3B & 4: Hybrid EHR/Transcript Retrieval & Cross-Encoder | `get_stage4_retriever().search(query, records, top_k, rerank_k)` |
| [`stages/stage5_concepts/`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/stages/stage5_concepts/) | Stage 5: Concept Linking & Curated Medical Ontology Hierarchy | `get_stage5_concept_service().link(text)` |
| [`stages/stage6_validity/`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/stages/stage6_validity/) | Stage 6: Temporal Validity Engine & Multi-Relational Graph | `get_stage6_validity_service().build_evidence_graph(...)` |
| [`stages/stage7_8_verifier/`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/stages/stage7_8_verifier/) | Stage 7 & 8: Dense GIN & Claim-Conditioned Transformer Verifier | `get_stage7_8_verifier().verify_graph(graph, claim_text, candidates)` |
| [`stages/pipeline.py`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/stages/pipeline.py) | Full End-to-End Orchestrator Pipeline | `ClinicalVerificationPipeline().run_session(...)` |

### Calling from Django
```python
from stages import ClinicalVerificationPipeline

pipeline = ClinicalVerificationPipeline()
result = pipeline.run_session(
    transcript_data=transcript_segments,
    patient_records=patient_ehr_rows,
    patient_id="P_D2N001",
    encounter_date="2024-01-15",
)
# Returns: { "patient_id": ..., "soap_note": ..., "claims": [ { "claim_id": ..., "label": ..., "calibrated_confidence": ..., "top_evidence": ... } ] }
```

---

## 6. High-Resolution Evaluation Figures (`results/plots/`)

Generated publication-grade figures (300 DPI) summarizing the Stage 5-8 evaluation:

* [`results/plots/evaluation_dashboard.png`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/results/plots/evaluation_dashboard.png): Comprehensive 4-panel multi-plot dashboard (Confusion Matrix, Ablation Progression, Date-Shuffle Sensitivity, Confidence Calibration).
* [`results/plots/confusion_matrix.png`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/results/plots/confusion_matrix.png): Normalized 6x6 test confusion matrix across all clinical verification classes.
* [`results/plots/ablation_study.png`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/results/plots/ablation_study.png): Architectural comparison across configurations A0 through A8.
* [`results/plots/date_shuffle_sensitivity.png`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/results/plots/date_shuffle_sensitivity.png): F1 degradation under encounter date perturbations demonstrating temporal validity usage.
* [`results/plots/calibration_reliability.png`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/results/plots/calibration_reliability.png): Reliability diagram for temperature-scaled calibration (ECE = 0.0336).
* [`results/plots/candidate_pool_sweep.png`](file:///home/statixlo/Develop/Final_year_project/clinnote_backend/results/plots/candidate_pool_sweep.png): Performance scaling across top-K evidence candidate pools.

