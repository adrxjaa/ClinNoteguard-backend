#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "clinnote_backend.settings")

import django

django.setup()

import numpy as np
import pytest
import torch
import websockets
from rest_framework.test import APIClient

from api.management.commands.run_websocket_server import Command as WebSocketCommand
from api.models import ConsultationSession
from stages import ClinicalVerificationPipeline


def test_cuda_and_checkpoints():
    assert torch.__version__ is not None
    if torch.cuda.is_available():
        x = torch.randn((32, 32), device="cuda", dtype=torch.float16)
        y = torch.matmul(x, x)
        assert y.shape == (32, 32)

    assert (BASE_DIR / "checkpoints" / "verifier_demo.pt").exists(), "Missing verifier_demo.pt"
    assert (BASE_DIR / "verify" / "concepts" / "concept_hierarchy.json").exists(), "Missing concept_hierarchy.json"
    assert (BASE_DIR / "verify" / "configs" / "verify_config.yaml").exists(), "Missing verify_config.yaml"


def test_stages_verification_pipeline():
    pipeline = ClinicalVerificationPipeline()
    transcript = [
        {
            "speaker": "doctor",
            "text": "Good morning! I see you take Lisinopril 10 mg daily and Metformin 500 mg twice daily.",
            "timestamp": "2024-01-15T09:00:00Z",
        },
        {
            "speaker": "patient",
            "text": "Yes doctor, my home blood pressure has been around 138 over 88.",
            "timestamp": "2024-01-15T09:00:15Z",
        },
        {
            "speaker": "doctor",
            "text": "Your HbA1c was 7.8 percent and we note your sulfa drug allergy. Let us add Atorvastatin 20 mg at night.",
            "timestamp": "2024-01-15T09:00:35Z",
        },
    ]
    ehr_records = [
        {"record_id": "EHR_RX_001", "text": "Lisinopril 10 mg oral daily", "date": "2023-10-01", "source": "EHR"},
        {"record_id": "EHR_RX_002", "text": "Metformin 500 mg oral twice daily", "date": "2023-10-01", "source": "EHR"},
        {"record_id": "EHR_LAB_001", "text": "HbA1c 7.8%", "date": "2024-01-05", "source": "EHR"},
        {"record_id": "EHR_ALG_001", "text": "Sulfa drug allergy", "date": "2020-05-12", "source": "EHR"},
    ]

    result = pipeline.run_session(
        transcript_data=transcript,
        patient_records=ehr_records,
        patient_id="P_SYSTEST",
        encounter_date="2024-01-15",
    )

    assert "soap_note" in result
    soap = result["soap_note"]
    for section in ("Subjective", "Objective", "Assessment", "Plan"):
        assert section in soap and isinstance(soap[section], str)

    claims = result.get("claims", [])
    assert len(claims) > 0, "Expected at least one extracted & verified claim"
    for claim in claims:
        assert "claim_id" in claim
        assert "claim_text" in claim
        assert "label" in claim
        assert 0.0 <= float(claim["calibrated_confidence"]) <= 1.0
        assert "validity_status" in claim


def test_django_rest_api_full_workflow():
    patient_client = APIClient()
    doctor_client = APIClient()
    anon_client = APIClient()

    # 1. Health check
    health = anon_client.get("/api/health/")
    assert health.status_code == 200
    assert health.json().get("status") == "ok"

    # 2. Demo Login (Patient & Doctor)
    pat_login = anon_client.post("/api/auth/demo-login/", {"role": "PATIENT"}, format="json")
    assert pat_login.status_code == 200, pat_login.content
    pat_token = pat_login.json()["token"]
    patient_client.credentials(HTTP_AUTHORIZATION=f"Token {pat_token}")

    doc_login = anon_client.post("/api/auth/demo-login/", {"role": "DOCTOR"}, format="json")
    assert doc_login.status_code == 200, doc_login.content
    doc_token = doc_login.json()["token"]
    doctor_client.credentials(HTTP_AUTHORIZATION=f"Token {doc_token}")

    # 3. Profile & Clinics
    assert patient_client.get("/api/auth/me/").status_code == 200
    assert doctor_client.get("/api/auth/me/").status_code == 200
    assert patient_client.get("/api/clinics/").status_code == 200

    # 4. Patient creates pairing code
    create_res = patient_client.post("/api/consultation/create/", {}, format="json")
    assert create_res.status_code == 201, create_res.content
    code = create_res.json()["code"]
    assert len(code) == 6

    try:
        # 5. Doctor joins session
        join_res = doctor_client.post("/api/consultation/join/", {"code": code}, format="json")
        assert join_res.status_code == 200, join_res.content
        assert join_res.json()["status"] == "CONNECTED"

        # 6. Patient status polling sees CONNECTED
        status_res = patient_client.get("/api/consultation/status/")
        assert status_res.status_code == 200
        assert status_res.json()["status"] == "CONNECTED"

        # 7. Doctor fetches patient briefing & logs an action
        briefing_res = doctor_client.get(f"/api/consultation/briefing/{code}/")
        assert briefing_res.status_code == 200, briefing_res.content

        action_res = doctor_client.post(
            f"/api/consultation/action/{code}/",
            {"action": "INSPECT_EHR", "detail": "System test inspected HbA1c lab"},
            format="json",
        )
        assert action_res.status_code == 200, action_res.content

        # 8. Doctor ends session with transcript -> runs Stages 2-8 pipeline
        transcript_payload = [
            {"speaker": "doctor", "text": "Your HbA1c is 7.8% and we will continue Metformin 500 mg twice daily."},
            {"speaker": "patient", "text": "Okay doctor, I have been taking it regularly."},
            {"speaker": "doctor", "text": "Let us also start Atorvastatin 20 mg at night."},
        ]
        end_res = doctor_client.post(
            "/api/consultation/end/",
            {"code": code, "transcript": transcript_payload},
            format="json",
        )
        assert end_res.status_code == 200, end_res.content
        end_data = end_res.json()
        assert end_data["status"] == "ENDED"
        assert isinstance(end_data.get("soap_note"), dict) and len(end_data["soap_note"]) > 0
        assert isinstance(end_data.get("atomic_claims"), list) and len(end_data["atomic_claims"]) > 0

        # 9. Re-extract / verify claims endpoint
        claims_res = doctor_client.post(
            f"/api/consultation/extract-claims/{code}/",
            {"transcript": transcript_payload, "soap_note": end_data["soap_note"]},
            format="json",
        )
        assert claims_res.status_code == 200, claims_res.content
        assert claims_res.json()["claims_count"] > 0

        # 10. Finalize prescription & QR token
        rx_res = doctor_client.post(
            f"/api/consultation/prescribe/{code}/",
            {
                "soap_note": end_data["soap_note"],
                "atomic_claims": claims_res.json()["atomic_claims"],
                "prescriptions": [
                    {"drug": "Atorvastatin 20 mg", "dosage": "1 tab OD HS", "duration": "30 days"}
                ],
            },
            format="json",
        )
        assert rx_res.status_code == 200, rx_res.content
        rx_token = rx_res.json()["qr_code_token"]
        assert rx_token.startswith("RX-")

        # 11. Pharmacy lookup
        pharm_res = anon_client.get(f"/api/pharmacy/lookup/{rx_token}/")
        assert pharm_res.status_code == 200, pharm_res.content

        # 12. Consultation detail & history lists
        assert doctor_client.get(f"/api/consultation/detail/{code}/").status_code == 200
        assert patient_client.get("/api/consultation/patient/list/").status_code == 200
        assert doctor_client.get("/api/consultation/doctor/list/").status_code == 200
    finally:
        ConsultationSession.objects.filter(code=code).delete()


def test_websocket_server_protocol():
    cmd = WebSocketCommand()

    async def _run_ws_roundtrip():
        async with websockets.serve(cmd.ws_handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            uri = f"ws://127.0.0.1:{port}"
            async with websockets.connect(uri) as ws:
                await ws.send(json.dumps({
                    "type": "start",
                    "session_id": "",
                    "sample_rate": 16000,
                    "channels": 1,
                }))
                raw_resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
                resp = json.loads(raw_resp)
                assert resp["type"] == "status"
                assert resp["status"] == "connected"

                # Stream 0.5s of 16kHz PCM16 silence
                pcm_chunk = np.zeros(8000, dtype=np.int16).tobytes()
                await ws.send(pcm_chunk)
                await ws.send(json.dumps({"type": "stop"}))

    asyncio.run(_run_ws_roundtrip())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
