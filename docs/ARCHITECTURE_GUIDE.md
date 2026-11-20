# ClinNoteGuard Architecture & Code Walkthrough

This guide explains what each model does, its exact layers/dimensions, and which file/function implements it.

## 1. End-to-End Pipeline
Orchestrator: `ClinicalVerificationPipeline.run_session()` in `stages/pipeline.py`.

1. **Stage 1 (Audio & Diarization)** — `api/management/commands/run_websocket_server.py`, `stages/stage1_audio/diarizer.py`
   - `faster-whisper` (`small`, CUDA FP16) + SpeechBrain `spkrec-ecapa-voxceleb` embeddings + `KMeans(n_clusters=2)`.
2. **Stage 2 (SOAP Note)** — `stages/stage2_soap/generator.py`
   - `Stage2SOAPGenerator.generate()` via `google-genai` (`gemini-3.5-flash`) with deterministic fallback.
3. **Stage 3A (Atomic Claims)** — `stages/stage3_claims/extractor.py`
   - `Stage3ClaimExtractor.extract()` splits SOAP sections into atomic claims.
4. **Stage 3B & 4 (Hybrid Retrieval + Reranking)** — `api/retrieval/`, `stages/stage4_retrieval/retriever.py`
   - **BM25**: `BM25Retriever.search()` (`api/retrieval/bm25_retriever.py`) — Okapi BM25 (`k1=1.5`, `b=0.75`, corpus-wide IDF).
   - **Dense Bi-Encoder**: `DenseRetriever.search()` (`api/retrieval/dense_retriever.py`) — `sentence-transformers/all-MiniLM-L6-v2` (384-d) or `retrieval_models/dense_finetuned`.
   - **RRF Fusion + Dose Check**: `HybridRetriever.search()` (`api/retrieval/hybrid.py`) — RRF (`k=60`) + `DOSE_MATCH_BONUS = 0.01` (`dose_agreement()` in `text_utils.py`).
   - **Cross-Encoder**: `CrossEncoderReranker.rerank()` (`api/retrieval/reranker.py`) — `cross-encoder/ms-marco-MiniLM-L-6-v2` (`max_length=512`) over `[CLS] claim [SEP] evidence [SEP]`.
5. **Stage 5 (Concept Linking)** — `verify/concepts/linker.py`, `hierarchy.py`, `attr_type.py`
   - `CuratedDictionaryLinker` maps mentions to CUIs; `ConceptHierarchy` provides `IS_A` / `SUBCLASS_OF` ancestors; `AttrType` classifies into 8 clinical attribute types.
6. **Stage 6 (Temporal Validity & EvidenceGraph)** — `verify/graph/builder.py`, `validity.py`, `transcript_store.py`
   - `resolve_state()` computes temporal status (`CURRENT`, `SUPERSEDED`, `DISCONTINUED`, `STALE`, `UNKNOWN`).
   - `EvidenceGraph`: 3 node types (`CLAIM`, `EVIDENCE`, `CONCEPT`), 9 edge types (`NUM_EDGE_TYPES = 9`), 7 date gap buckets.

## 2. GNN Design — Stage 7 (`verify/model/graph_encoder.py`)
- **Class**: `GraphEncoder` + `DenseGINELayer`
- **Layers**: `3` `DenseGINELayer`s, `hidden_dim = 256`, `dropout = 0.1`
- **Node Input**: `384`-d frozen `all-MiniLM-L6-v2` text embedding + `32`-d learned metadata (`4 x 8`-d: `node_type`, `attr_type`, `status`, `date_bucket`) = **`416`-d** -> `Linear(416, 256) -> ReLU -> LayerNorm(256)`
- **Edge Input**: `10` continuous edge attributes + `8`-d learned `edge_type_emb` = **`18`-d** (`edge_dim = 18`)
- **Edge Gate**: `Linear(18, 32) -> ReLU -> Linear(32, 1) -> Sigmoid`
- **GIN Update**: `msg = ReLU(Linear(256 + 18, 256))`, `aggr = sum(gates * msg)`, `(1 + eps) * h + aggr` -> 2-layer `node_mlp` (`256 -> 256 -> 256`) + residual + `LayerNorm(256)`

## 3. Transformer Design — Stage 8 (`verify/model/verifier.py`)
- **Class**: `ClaimConditionedVerifier`, `TransformerVerifierEncoderLayer`, `MultiHeadAttentionWithBiases`
- **Sequence**: `[CLS], claim_token, evidence_token_1, ..., evidence_token_K`
- **Claim Token**: `[text_emb (384), gnn_emb (256)]` = **`640`-d** -> `Linear(640, 256) -> ReLU -> LayerNorm(256)`
- **Evidence Token**: `[text_emb (384), gnn_emb (256), source (8), attr (8), status (8), date (8), scalars (5)]` = **`677`-d** -> `Linear(677, 256) -> ReLU -> LayerNorm(256)`
- **Transformer**: `3` layers, `d_model = 256`, `n_heads = 4` (`d_k = 64`), `dim_feedforward = 512`, `dropout = 0.1`
- **Attention Biases**:
  - Edge-type bias: learned `[9, 4]` table per `(edge_type, head)`
  - Temporal decay bias: `-softplus(a[attr_type]) * log1p(gap_days)`
- **Output Head**: `Linear(256, 6)` from `[CLS]` over `TRANSCRIPT_SUPPORTED`, `HISTORY_SUPPORTED`, `CLINICAL_INFERENCE`, `CLINICAL_CONFLICT`, `STALE`, `HALLUCINATED`
- **Calibration**: `TemperatureScaler` (`verify/model/calibrate.py`) — `softmax(logits / T)` fit via L-BFGS.
