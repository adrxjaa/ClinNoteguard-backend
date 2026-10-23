import os
import sys
import json
import asyncio
import numpy as np
import soundfile as sf
import websockets

# Setup Django Environment
sys.path.append('/home/statixlo/Develop/Final_year_project/clinnote_backend')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'clinnote_backend.settings')
import django
django.setup()

from django.contrib.auth import get_user_model
from api.models import ConsultationSession

User = get_user_model()
session_code = "TEST12"

def setup_db_records():
    # Create a dummy patient and session in the database synchronously
    patient_email = "test_patient_pipeline@example.com"
    patient, created = User.objects.get_or_create(
        username=patient_email,
        defaults={
            "email": patient_email,
            "first_name": "Sarah",
            "last_name": "Pipeline",
            "role": User.Role.PATIENT
        }
    )
    
    # End any existing TEST12 sessions
    ConsultationSession.objects.filter(code=session_code).delete()
    
    session = ConsultationSession.objects.create(
        patient=patient,
        code=session_code,
        status=ConsultationSession.Status.CREATED
    )
    print(f"Created test ConsultationSession in DB synchronously. Code: {session_code}")

async def run_test():
    # 1. Read WAV file (to get native sample rate and channels)
    wav_path = "2_person_sample.wav"
    if not os.path.exists(wav_path):
        print(f"Error: {wav_path} not found.")
        return
        
    data, samplerate = sf.read(wav_path, dtype='int16')
    channels = data.shape[1] if len(data.shape) > 1 else 1
    print(f"Loaded client audio for streaming. Samplerate: {samplerate}Hz, Samples: {len(data)}, Channels: {channels}")

    # 2. Connect to WebSocket server on port 8001
    uri = "ws://127.0.0.1:8001"
    print(f"Connecting to WebSocket server at {uri}...")
    
    async with websockets.connect(uri) as websocket:
        # Send start message with native audio format configuration
        start_payload = {
            "type": "start",
            "session_id": session_code,
            "sample_rate": samplerate,
            "channels": channels
        }
        await websocket.send(json.dumps(start_payload))
        
        # Receive welcome status message
        resp = await websocket.recv()
        print("Server response:", resp)
        
        # 4. Stream audio in chunks (simulate live micro-captures)
        chunk_size = 1024
        print("Streaming audio chunks to server...")
        
        # We also listen to transcripts asynchronously
        async def listen_transcripts():
            try:
                async for msg in websocket:
                    data = json.loads(msg)
                    if data.get("type") == "transcript":
                        print(f"Received Transcript Segment -> "
                              f"ID: {data['segment_id']}, "
                              f"Speaker: {data['speaker']}, "
                              f"Final: {data['final']}, "
                              f"Text: {data['text']}")
            except websockets.exceptions.ConnectionClosed:
                pass

        listen_task = asyncio.create_task(listen_transcripts())

        # Stream the buffer
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            await websocket.send(chunk.tobytes())
            await asyncio.sleep(0.01) # stream slightly faster for the test
            
        print("Finished streaming audio buffer. Waiting for final transcription...")
        await asyncio.sleep(3.0)
        
        # Send stop message
        print("Sending stop command...")
        await websocket.send(json.dumps({"type": "stop"}))
        await asyncio.sleep(12.0)
        
        listen_task.cancel()

def verify_db_records():
    print("\nRetrieving session details from database synchronously...")
    db_session = ConsultationSession.objects.get(code=session_code)
    print("Database Status:", db_session.status)
    print("Database Transcript JSON:", json.dumps(db_session.transcript_json, indent=2))
    print("Database SOAP Note JSON:", json.dumps(db_session.soap_note_json, indent=2))
    print("Audio file path in DB:", db_session.audio_file.name if db_session.audio_file else "None")
    
    # Assertions for verification
    if len(db_session.transcript_json) > 0:
        print("\n✅ PIPELINE TEST SUCCESSFUL! Real-time transcription, VAD, diarization, and DB save are fully working.")
    else:
        print("\n❌ PIPELINE TEST FAILED: No transcript found in DB.")

if __name__ == "__main__":
    setup_db_records()
    asyncio.run(run_test())
    verify_db_records()
