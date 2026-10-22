# ClinNoteGuard Backend — Project Milestones & Future Checkpoints

## ✅ Completed Milestones
1. **Stage 1: Speech-to-Text & Diarization**
   - Live PCM audio WebSocket streaming on port 8001.
   - Faster-Whisper transcription + SpeechBrain ECAPA-TDNN speaker diarization.
   - Dynamic role mapping (`doctor` vs `patient`).

2. **Stage 2: Structured SOAP Note Generation**
   - Synthesizes `Subjective`, `Objective`, `Assessment`, and `Plan` sections.
   - LLM integration with fallback support.

3. **Stage 3A: Atomic Claim Extraction**
   - Decomposes compound clinical statements into standalone, verifiable atomic propositions (`CLM-xxx`).
   - Extracts key clinical entities and preserves section context.
   - Stored in `ConsultationSession.atomic_claims_json`.

4. **Stage 3B: Hybrid Candidate Retrieval (BM25 + Dense + RRF)**
   - Patient-scoped EHR event and claim store.
   - Global-IDF BM25 with clinical tokenizer, abbreviation expansion, and unit normalization.
   - Dense bi-encoder semantic retriever with fine-tuning pipeline.
   - Reciprocal Rank Fusion (RRF) with exact dose agreement verification & conflict tagging (`dose_flag = match | mismatch`).
   - Evaluated on 20-EHR history patient benchmark with 88.9% Top-1 and 100% Top-5 recall.

---

## 📌 Future Checkpoints & Roadmap

### Checkpoint 3C: Lay Language & Consumer Health Vocabulary (CHV)
- **Goal**: Bridge the vocabulary gap between patient lay expressions and formal clinical EHR records.
- **Components**:
  - Integration of UMLS Consumer Health Vocabulary (CHV) or SNOMED-CT lay synonyms (e.g. *"water pill"* $\rightarrow$ *diuretic/furosemide*, *"sugar trouble"* $\rightarrow$ *diabetes mellitus*, *"heart flutter"* $\rightarrow$ *palpitations/arrhythmia*).
  - Fine-tuning dense encoders with medical dialogue / consumer health QA pairs (e.g. MedQuAD, HealthQA).
  - Query expansion module before BM25 lexical search.

### Checkpoint 4: Cross-Encoder Reranker & UMLS Entity Linking
- Stage 4 in architecture diagram:
  - Takes Top-20 candidates from Hybrid RRF.
  - WordPiece tokenization with `[CLS] claim [SEP] evidence [SEP]`.
  - Deep cross-attention reranking.

### Checkpoint 5: Cascade Classifier & Semantic Entropy Confidence Calibration
- Classifies atomic claims into formal validation categories:
  - `Transcript-Supported`
  - `History-Supported`
  - `Clinical Inference`
  - `Clinical Conflict`
  - `Hallucinated (fallback)`
- Low-confidence / high semantic entropy cases routed to mandatory clinician review.

### Checkpoint 6: Fine-Tuned BioMistral Drop-In
- Replace baseline LLM generation with local fine-tuned BioMistral model for note and claim generation.
