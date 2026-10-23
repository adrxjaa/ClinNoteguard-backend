"""
Django Management Command: verify_session (Milestone 4).

Executes end-to-end verification pipeline:
  Diarized Transcript -> SOAP Note -> Atomic Claims -> Hybrid Retrieval (EHR + Transcript)
  -> Cross-Encoder Reranker -> EvidenceGraph -> Stage 7-8 Verifier.

Usage:
  python manage.py verify_session --transcript <path_to_json> [--note <path_to_soap>] --patient <id> --encounter-date YYYY-MM-DD
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import numpy as np

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

from verify.eval.metrics import VERIFICATION_CLASSES
from verify.graph.builder import GraphBuilder
from verify.graph.transcript_store import TranscriptStore
from verify.model.verifier import ClaimConditionedVerifier, Verdict
from api.services.claim_extractor import extractor as claim_extractor
from api.retrieval.pipeline import get_default_retriever, get_default_reranker
from api.retrieval.text_utils import dose_agreement

CONSOLE_BANNER = """
+-------------------------------------------------------------------------------+
|                      CLINNOTEGUARD SESSION VERIFICATION                       |
|                   BANNER: synthetic-trained, unvalidated                      |
|                   ONTOLOGY: curated concept layer                             |
+-------------------------------------------------------------------------------+
"""

class Command(BaseCommand):
    help = "Verify clinical consultation claims against transcript and EHR records (Stages 1-8)."

    def add_arguments(self, parser):
        parser.add_argument("--transcript", type=str, required=True, help="Path to transcript JSON file or JSON string.")
        parser.add_argument("--note", type=str, default=None, help="Path to existing SOAP note JSON (optional).")
        parser.add_argument("--patient", type=str, required=True, help="Patient identifier (e.g. P_D2N001).")
        parser.add_argument("--encounter-date", type=str, default=None, help="Encounter date in YYYY-MM-DD format.")
        parser.add_argument("--encounter-time", type=str, default=None, help="Explicit encounter timestamp in ISO format (e.g. 2024-01-15T14:30:00Z).")
        parser.add_argument("--output", type=str, default="labeled_note.json", help="Destination file for verification output.")

    def handle(self, *args, **options):
        self.stdout.write(CONSOLE_BANNER)

        transcript_arg = options["transcript"]
        note_arg = options["note"]
        patient_id = options["patient"]
        encounter_date = options["encounter_date"]
        encounter_time = options["encounter_time"]
        output_path = options["output"]

        now_utc = datetime.now(timezone.utc)
        if encounter_time:
            time_str = encounter_time.strip()
            encounter_time_iso = time_str if (time_str.endswith("Z") or "+" in time_str) else f"{time_str}Z"
        elif encounter_date:
            date_str = encounter_date.strip()
            if "T" in date_str or " " in date_str:
                normalized = date_str.replace(" ", "T")
                encounter_time_iso = normalized if (normalized.endswith("Z") or "+" in normalized) else f"{normalized}Z"
            else:
                current_time_str = now_utc.strftime("%H:%M:%SZ")
                encounter_time_iso = f"{date_str}T{current_time_str}"
        else:
            encounter_time_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        # 1. Load Transcript
        transcript_data = self._load_json_or_file(transcript_arg)
        if not transcript_data:
            raise CommandError(f"Could not load valid transcript data from: {transcript_arg}")

        self.stdout.write(self.style.SUCCESS(f"[+] Loaded transcript ({len(transcript_data)} segments)."))

        # 2. Obtain SOAP Note (Existing or Stage 2 Generator)
        if note_arg:
            soap_note = self._load_json_or_file(note_arg)
            self.stdout.write(self.style.SUCCESS(f"[+] Loaded existing SOAP note from {note_arg}."))
        else:
            self.stdout.write("[+] No SOAP note provided. Synthesizing from transcript (Stage 2)...")
            from api.management.commands.run_websocket_server import Command as WSCommand
            ws_cmd = WSCommand()
            soap_note = ws_cmd.generate_soap_note_from_transcript(transcript_data)
            self.stdout.write(self.style.SUCCESS("[+] Synthesized SOAP note."))

        # 3. Stage 3A: Atomic Claim Extraction
        self.stdout.write("[+] Extracting atomic claims (Stage 3A)...")
        raw_claims = claim_extractor.extract_claims(soap_note, transcript_data)
        self.stdout.write(self.style.SUCCESS(f"[+] Extracted {len(raw_claims)} atomic claims."))

        if not raw_claims:
            self.stdout.write(self.style.WARNING("[-] No atomic claims extracted. Exiting."))
            return

        # 4. Stage 3B & 4: Hybrid Retrieval over Patient EHR + TranscriptStore
        self.stdout.write(f"[+] Retrieving candidate evidence for patient {patient_id}...")
        transcript_store = TranscriptStore()
        enc_id = patient_id.replace("P_", "")
        transcript_store.ingest_dialogue(enc_id, transcript_data, encounter_time=encounter_time_iso)

        # Ingest rendered EHR
        rendered_ehr_path = Path(settings.BASE_DIR) / "verify_data" / "rendered_ehr.jsonl"
        patient_ehr_records = []
        if rendered_ehr_path.exists():
            with open(rendered_ehr_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if r.get("patient_id") == patient_id:
                            patient_ehr_records.append(r)

        builder = GraphBuilder()

        # Load Trained Verifier Model
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint_path = Path(settings.BASE_DIR) / "checkpoints" / "verifier_demo.pt"

        model = ClaimConditionedVerifier().to(device)
        checkpoint_name = "random_init"
        if checkpoint_path.exists():
            try:
                ckpt = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(ckpt["model_state_dict"])
                checkpoint_name = "checkpoints/verifier_demo.pt"
                self.stdout.write(self.style.SUCCESS(f"[+] Loaded Stage 8 model from {checkpoint_name} on {device.upper()}."))
            except Exception as e:
                self.stdout.write(self.style.WARNING(f"[-] Could not load checkpoint ({e}). Using initialized architecture."))
        else:
            self.stdout.write(self.style.WARNING(f"[-] Checkpoint not found at {checkpoint_path}. Using base model."))

        model.eval()

        # 5. Execute Verification for each claim
        labeled_claims = []
        rows_to_print = []

        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

        for c in raw_claims:
            cid = c.get("claim_id", "CLM-000")
            c_text = c.get("claim_text", "")
            section = c.get("section", "SOAP")

            # Candidate retrieval
            trans_cands = transcript_store.search(c_text, enc_id, top_k=5)
            all_cands = list(patient_ehr_records[:15]) + trans_cands

            graph = builder.build_graph(c, all_cands, encounter_time=encounter_time_iso)
            tensors = graph.to_tensors()

            # Pre-compute text embeddings for graph nodes
            node_texts = [n.text or " " for n in graph.nodes]
            embs = encoder.encode(node_texts, convert_to_numpy=True, show_progress_bar=False)
            node_text_embs = torch.tensor(embs, dtype=torch.float32, device=device)

            num_ev = len(tensors["evidence_node_indices"])
            scalars = torch.zeros((num_ev, 5), device=device)
            gap_days = torch.zeros(num_ev, device=device)

            for i, ev_idx in enumerate(tensors["evidence_node_indices"]):
                if i < len(all_cands):
                    cand = all_cands[i]
                    scalars[i, 0] = float(cand.get("score", 0.0)) / 10.0
                    scalars[i, 1] = 1.0 / (60 + (i + 1))
                    d_flag = dose_agreement(c_text, cand.get("claim_text", ""))
                    scalars[i, 2] = 1.0 if d_flag == "match" else 0.0
                    scalars[i, 3] = 1.0 if d_flag == "mismatch" else 0.0
                    scalars[i, 4] = 0.5
                gap_days[i] = float(tensors["date_buckets"][ev_idx].item() * 14)

            with torch.no_grad():
                logits, _, attn_w = model(
                    node_text_embs=node_text_embs,
                    node_types=tensors["node_types"].to(device),
                    attr_types=tensors["attr_types"].to(device),
                    statuses=tensors["statuses"].to(device),
                    date_buckets=tensors["date_buckets"].to(device),
                    edge_index=tensors["edge_index"].to(device),
                    edge_types=tensors["edge_type"].to(device),
                    edge_attrs=tensors["edge_attr"].to(device),
                    claim_idx=tensors["claim_node_idx"],
                    evidence_indices=tensors["evidence_node_indices"],
                    scalars=scalars,
                    gap_days_tensor=gap_days,
                )
                cal_logits = model.calibrator(logits)
                probs = torch.softmax(cal_logits, dim=-1).cpu().numpy()[0]

            pred_idx = int(np.argmax(probs))
            pred_label = VERIFICATION_CLASSES[pred_idx]
            conf = float(probs[pred_idx])

            # Top 3 evidence by claim-to-evidence attention
            top_ev = []
            if num_ev > 0 and attn_w.numel() > 0:
                top_ev_indices = torch.argsort(attn_w, descending=True)[:3].cpu().numpy()
                for tidx in top_ev_indices:
                    node_idx = tensors["evidence_node_indices"][tidx]
                    top_ev.append({
                        "record_id": tensors["record_ids"][node_idx],
                        "source": "EHR" if tensors["source_types"][node_idx] == 1 else "TRANSCRIPT",
                        "attention_weight": round(float(attn_w[tidx].item()), 4),
                    })

            labeled_claims.append({
                "claim_id": cid,
                "section": section,
                "claim_text": c_text,
                "label": pred_label,
                "calibrated_confidence": round(conf, 4),
                "top_evidence": top_ev,
                "validity_status": graph.nodes[1].status.value if len(graph.nodes) > 1 else graph.nodes[0].status.value,
                "checkpoint": checkpoint_name,
            })

            rows_to_print.append((cid, section, pred_label, f"{conf:.2%}", c_text[:45]))

        # 6. Save labeled_note.json
        out_file = Path(output_path)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "patient_id": patient_id,
                "encounter_date": encounter_date,
                "verified_at": datetime.now().isoformat() + "Z",
                "checkpoint": checkpoint_name,
                "claims": labeled_claims,
            }, f, indent=2)

        self.stdout.write(self.style.SUCCESS(f"\n[+] Wrote verification output to {out_file}"))

        # 7. Print Console Table
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write(f"{'CLAIM ID':<10} | {'SECTION':<11} | {'LABEL':<22} | {'CONF':<7} | {'CLAIM TEXT'}")
        self.stdout.write("-" * 80)
        for cid, sec, lbl, cnf, txt in rows_to_print:
            self.stdout.write(f"{cid:<10} | {sec:<11} | {lbl:<22} | {cnf:<7} | {txt}...")
        self.stdout.write("=" * 80 + "\n")

    def _load_json_or_file(self, arg: str) -> Any:
        # Check if file path
        if os.path.exists(arg):
            with open(arg, "r", encoding="utf-8") as f:
                return json.load(f)
        try:
            return json.loads(arg)
        except Exception:
            return None
