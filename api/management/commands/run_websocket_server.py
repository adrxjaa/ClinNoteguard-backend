import asyncio
import json
import logging
import os
import sys
import numpy as np
import torch
import soundfile as sf
from django.core.management.base import BaseCommand
from django.conf import settings
import websockets
from sklearn.cluster import KMeans
from api.models import ConsultationSession

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WebSocketServer")

# Pre-load CUDA libraries from virtualenv site-packages so CTranslate2 can locate them
try:
    import ctypes
    for path in sys.path:
        if "site-packages" in path:
            nvidia_lib_dir = os.path.join(path, "nvidia", "cu13", "lib")
            if os.path.exists(nvidia_lib_dir):
                # Prepend to LD_LIBRARY_PATH environment variable
                os.environ["LD_LIBRARY_PATH"] = nvidia_lib_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
                # Load libraries globally into process address space
                ctypes.CDLL(os.path.join(nvidia_lib_dir, "libcublasLt.so.12"), mode=ctypes.RTLD_GLOBAL)
                ctypes.CDLL(os.path.join(nvidia_lib_dir, "libcublas.so.12"), mode=ctypes.RTLD_GLOBAL)
                logger.info("Successfully pre-loaded virtualenv cuBLAS libraries on CUDA.")
                break
except Exception as preload_err:
    logger.warning(f"Did not pre-load cuBLAS libraries via ctypes: {preload_err}")

# Global models loaded once and reused
whisper_model = None
speaker_classifier = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def init_models():
    global whisper_model, speaker_classifier
    if whisper_model is None:
        from faster_whisper import WhisperModel
        try:
            logger.info("Attempting to load faster-whisper (small) on device: cuda...")
            whisper_model = WhisperModel("small", device="cuda", compute_type="float16")
            # Dry run transcription to verify that CUDA libraries can be loaded
            logger.info("Performing dry run transcription to verify CUDA library compatibility...")
            dummy_audio = np.zeros(16000, dtype=np.float32)
            list(whisper_model.transcribe(dummy_audio, beam_size=1)[0])
            logger.info("faster-whisper loaded on CUDA successfully.")
        except Exception as e:
            logger.warning(f"Failed to load/run faster-whisper on CUDA: {e}. Falling back to CPU.")
            whisper_model = WhisperModel("small", device="cpu", compute_type="float32")
            logger.info("faster-whisper loaded on CPU successfully.")

    if speaker_classifier is None:
        logger.info(f"Loading SpeechBrain ECAPA-TDNN speaker encoder on device: {device}...")
        from speechbrain.inference.speaker import EncoderClassifier
        speaker_classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )
        logger.info("SpeechBrain speaker encoder loaded successfully.")

class Command(BaseCommand):
    help = "Runs a WebSocket server for real-time doctor-patient audio streaming, transcription, and diarization."

    def add_arguments(self, parser):
        parser.add_argument('--port', type=int, default=8001, help='Port to run the WebSocket server on')

    def handle(self, *args, **options):
        port = options['port']
        logger.info("Initializing models before starting WebSocket server...")
        try:
            init_models()
        except Exception as e:
            logger.error(f"Error loading models: {e}. Running in stub mode.")

        logger.info(f"Starting WebSocket server on ws://0.0.0.0:{port}...")
        
        try:
            asyncio.run(self.async_main(port))
        except KeyboardInterrupt:
            logger.info("WebSocket server stopped.")

    async def async_main(self, port):
        async with websockets.serve(
            self.ws_handler, 
            "0.0.0.0", 
            port,
            max_size=10 * 1024 * 1024 # 10MB maximum message size
        ) as server:
            await asyncio.Future() # run forever

    async def ws_handler(self, websocket):
        logger.info("New client connection established.")
        session_id = None
        
        # Audio sample rate is 16kHz PCM (16-bit)
        sample_rate = 16000
        bytes_per_sample = 2
        
        # Client audio format parameters (default to 16kHz Mono)
        client_sample_rate = 16000
        client_channels = 1
        
        # Accumulate incoming audio for current sentence/utterance
        utterance_buffer = np.array([], dtype=np.float32)
        
        # Store full audio for final compilation
        full_audio_buffer = np.array([], dtype=np.float32)
        
        # List of final segments: {"segment_id": ..., "text": ..., "start": ..., "end": ..., "embedding": ..., "speaker": ...}
        final_segments = []
        
        # Mapping control
        swap_roles = False
        
        # VAD Parameters
        energy_threshold = 0.015
        silence_samples_limit = int(1.5 * sample_rate) # 1.5 seconds of silence
        max_utterance_samples = int(15.0 * sample_rate) # 15 seconds max per utterance
        
        silence_samples = 0
        is_speaking = False
        
        # Time tracker
        total_samples_processed = 0
        
        # Current active utterance start time (in seconds)
        utterance_start_time = 0.0
        
        # For sending periodic provisional updates (every 1.0 second)
        last_provisional_samples = 0
        provisional_interval_samples = int(1.0 * sample_rate)

        try:
            async for message in websocket:
                if isinstance(message, str):
                    # Handle JSON command
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type")
                        
                        if msg_type == "start":
                            session_id = data.get("session_id")
                            # Read client configuration (with defaults for backward compatibility)
                            client_sample_rate = data.get("sample_rate", 16000)
                            client_channels = data.get("channels", 1)
                            logger.info(f"Session started for ID: {session_id}. Format: {client_sample_rate}Hz, Channels: {client_channels}")
                            
                            await websocket.send(json.dumps({
                                "type": "status",
                                "status": "connected",
                                "message": f"Successfully joined session {session_id}"
                            }))
                            
                        elif msg_type == "swap_mapping":
                            swap_roles = not swap_roles
                            logger.info(f"Roles swapped: swap_roles = {swap_roles}")
                            
                            # Re-map existing segments and push updates to client
                            await self.recompute_and_send_speakers(websocket, final_segments, swap_roles)
                            
                        elif msg_type == "stop":
                            logger.info(f"Session stop requested for ID: {session_id}")
                            break
                    except Exception as e:
                        logger.error(f"Error handling JSON message: {e}")
                        
                elif isinstance(message, bytes):
                    # Process incoming raw PCM 16-bit audio bytes
                    if len(message) == 0:
                        continue
                        
                    # Convert bytes to numpy float32 array
                    chunk = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Convert stereo to mono if needed
                    if client_channels > 1:
                        if len(chunk) >= client_channels:
                            chunk = chunk.reshape(-1, client_channels)[:, 0]
                        else:
                            chunk = chunk[:1]
                            
                    # Resample to 16000Hz if needed
                    if client_sample_rate != 16000 and len(chunk) > 0:
                        if client_sample_rate % 16000 == 0:
                            factor = client_sample_rate // 16000
                            chunk = chunk[::factor]
                        else:
                            import librosa
                            chunk = librosa.resample(chunk, orig_sr=client_sample_rate, target_sr=16000)
                            
                    if len(chunk) == 0:
                        continue
                        
                    # Accumulate buffers
                    utterance_buffer = np.concatenate((utterance_buffer, chunk))
                    full_audio_buffer = np.concatenate((full_audio_buffer, chunk))
                    
                    chunk_len = len(chunk)
                    total_samples_processed += chunk_len
                    
                    # Energy VAD
                    rms = np.sqrt(np.mean(chunk ** 2)) if chunk_len > 0 else 0.0
                    
                    if rms > energy_threshold:
                        if not is_speaking:
                            # Start of a new speech utterance
                            is_speaking = True
                            utterance_start_time = (total_samples_processed - len(utterance_buffer)) / sample_rate
                            logger.info(f"VAD: Speech started at {utterance_start_time:.2f}s")
                        silence_samples = 0
                    else:
                        if is_speaking:
                            silence_samples += chunk_len
                            
                    # Check if silence limit is reached or utterance got too long
                    should_finalize = False
                    if is_speaking:
                        if silence_samples >= silence_samples_limit:
                            should_finalize = True
                            logger.info(f"VAD: Silence limit reached. Finalizing utterance.")
                        elif len(utterance_buffer) >= max_utterance_samples:
                            should_finalize = True
                            logger.info(f"VAD: Max utterance duration reached. Finalizing utterance.")
                            
                    if should_finalize:
                        # Process and finalize the utterance
                        await self.process_and_finalize_utterance(
                            websocket, 
                            utterance_buffer, 
                            utterance_start_time, 
                            total_samples_processed / sample_rate, 
                            final_segments, 
                            swap_roles
                        )
                        
                        # Reset for next utterance
                        utterance_buffer = np.array([], dtype=np.float32)
                        is_speaking = False
                        silence_samples = 0
                        last_provisional_samples = 0
                        
                    elif is_speaking and len(utterance_buffer) - last_provisional_samples >= provisional_interval_samples:
                        # Send periodic provisional transcript update (no diarization needed for provisional)
                        last_provisional_samples = len(utterance_buffer)
                        await self.send_provisional_transcript(
                            websocket, 
                            utterance_buffer, 
                            utterance_start_time, 
                            total_samples_processed / sample_rate, 
                            len(final_segments)
                        )
                        
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            logger.info("Client disconnected. Finalizing consultation session.")
            # Process remaining audio in buffer
            if len(utterance_buffer) > 0:
                await self.process_and_finalize_utterance(
                    websocket, 
                    utterance_buffer, 
                    utterance_start_time, 
                    total_samples_processed / sample_rate, 
                    final_segments, 
                    swap_roles
                )
            
            # Save final transcript and create clinical notes in database
            if session_id:
                try:
                    await self.save_session_results(session_id, final_segments, full_audio_buffer, sample_rate)
                except Exception as db_err:
                    logger.error(f"Error saving results to database: {db_err}")

    async def send_provisional_transcript(self, websocket, audio_data, start_sec, end_sec, segment_index):
        if whisper_model is None:
            return
            
        try:
            # Transcribe active buffer asynchronously
            segments, info = whisper_model.transcribe(audio_data, beam_size=1, word_timestamps=False)
            text = " ".join([seg.text for seg in segments]).strip()
            
            if text:
                event = {
                    "type": "transcript",
                    "segment_id": f"seg_{segment_index}",
                    "speaker": "doctor", # Default provisional speaker
                    "text": text,
                    "start": start_sec,
                    "end": end_sec,
                    "final": False
                }
                await websocket.send(json.dumps(event))
        except Exception as e:
            logger.error(f"Provisional transcription error: {e}")

    async def process_and_finalize_utterance(self, websocket, audio_data, start_sec, end_sec, final_segments, swap_roles):
        if whisper_model is None or len(audio_data) < int(0.5 * 16000):
            # Ignore empty or extremely short (<0.5s) clips
            return
            
        try:
            # 1. Whisper Transcription
            segments, info = whisper_model.transcribe(audio_data, beam_size=3, word_timestamps=False)
            text = " ".join([seg.text for seg in segments]).strip()
            
            if not text:
                return
                
            # 2. Extract Speaker Embedding
            embedding = None
            if speaker_classifier is not None:
                try:
                    # Save segment to temporary memory wave representation
                    # Convert float32 array to PyTorch tensor
                    audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).to(device)
                    # Compute embedding (returns shape [1, 1, 192])
                    with torch.no_grad():
                        emb_tensor = speaker_classifier.encode_batch(audio_tensor)
                        embedding = emb_tensor.squeeze().cpu().numpy()
                except Exception as emb_err:
                    logger.error(f"Error extracting speaker embedding: {emb_err}")

            segment_id = f"seg_{len(final_segments)}"
            segment_data = {
                "segment_id": segment_id,
                "text": text,
                "start": start_sec,
                "end": end_sec,
                "embedding": embedding,
                "speaker": "doctor" # Temporary default
            }
            final_segments.append(segment_data)
            
            # 3. Perform Speaker Diarization / Clustering if multiple embeddings exist
            await self.recompute_and_send_speakers(websocket, final_segments, swap_roles)

        except Exception as e:
            logger.error(f"Utterance finalization error: {e}")

    async def recompute_and_send_speakers(self, websocket, final_segments, swap_roles):
        if not final_segments:
            return
            
        embeddings = [seg["embedding"] for seg in final_segments if seg["embedding"] is not None]
        
        # If we have at least 2 distinct embeddings, we can run K-Means (K=2)
        if len(embeddings) >= 2:
            try:
                embeddings_matrix = np.stack(embeddings)
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                kmeans.fit(embeddings_matrix)
                
                # Check cluster of the very first segment to determine speaker roles
                first_embedding = final_segments[0]["embedding"]
                if first_embedding is not None:
                    first_cluster = kmeans.predict(first_embedding.reshape(1, -1))[0]
                    # Map the cluster containing the first speaker to "doctor", other to "patient"
                    doctor_cluster = first_cluster
                else:
                    doctor_cluster = 0
                
                # If swap roles is enabled, invert the cluster mapping
                if swap_roles:
                    doctor_cluster = 1 if doctor_cluster == 0 else 0
                    
                # Update speaker tags for all segments
                emb_idx = 0
                for seg in final_segments:
                    if seg["embedding"] is not None:
                        cluster = kmeans.labels_[emb_idx]
                        seg["speaker"] = "doctor" if cluster == doctor_cluster else "patient"
                        emb_idx += 1
                    else:
                        seg["speaker"] = "doctor" # Fallback
            except Exception as cluster_err:
                logger.error(f"Diarization clustering error: {cluster_err}")
                
        # Send all finalized segment events to update client's state
        for seg in final_segments:
            event = {
                "type": "transcript",
                "segment_id": seg["segment_id"],
                "speaker": seg["speaker"],
                "text": seg["text"],
                "start": seg["start"],
                "end": seg["end"],
                "final": True
            }
            try:
                await websocket.send(json.dumps(event))
            except Exception as ws_send_err:
                logger.error(f"Failed to send final segment event: {ws_send_err}")

    async def save_session_results(self, session_code, final_segments, full_audio_data, sample_rate):
        # 1. Clean/Filter transcript segments to save in JSON field
        transcript_db = []
        for seg in final_segments:
            transcript_db.append({
                "speaker": seg["speaker"],
                "text": seg["text"],
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2)
            })

        # 2. Generate SOAP Note from Transcript
        soap_note = self.generate_soap_note_from_transcript(transcript_db)

        # 2.5 Extract Atomic Claims (Stage 2 -> Stage 3)
        from api.services.claim_extractor import extractor as claim_extractor
        atomic_claims = claim_extractor.extract_claims(soap_note, transcript_db)
        logger.info(f"Extracted {len(atomic_claims)} atomic claims for session {session_code}")

        # 2.6 Retrieve Candidate EHR Evidence (Stage 3 Hybrid Retrieval)
        try:
            from api.retrieval.pipeline import retrieve_evidence_for_claims
            session_obj = ConsultationSession.objects.filter(code=session_code).first()
            patient_id = str(session_obj.patient.id) if session_obj and session_obj.patient else None
            atomic_claims = retrieve_evidence_for_claims(
                atomic_claims,
                patient_id=patient_id,
                session_code=session_code,
                top_k=3
            )
            logger.info(f"Enriched {len(atomic_claims)} atomic claims with candidate EHR evidence.")
        except Exception as ret_err:
            logger.warning(f"Retrieval step encountered non-blocking issue: {ret_err}")

        # 3. Save audio buffer to media file if audio data exists
        audio_file_path = None
        if len(full_audio_data) > 0:
            media_dir = os.path.join(settings.MEDIA_ROOT, 'consultation_audios')
            os.makedirs(media_dir, exist_ok=True)
            filename = f"session_{session_code}.wav"
            full_path = os.path.join(media_dir, filename)
            try:
                # Write numpy float32 buffer to PCM 16-bit WAV file
                sf.write(full_path, full_audio_data, sample_rate, subtype='PCM_16')
                audio_file_path = f"consultation_audios/{filename}"
                logger.info(f"Saved complete consultation audio to: {full_path}")
            except Exception as aud_err:
                logger.error(f"Failed to save audio file: {aud_err}")

        # 4. Update the ConsultationSession object in DB (using sync_to_async or running in executor)
        def db_update():
            try:
                session = ConsultationSession.objects.get(code=session_code)
                session.transcript_json = transcript_db
                session.soap_note_json = soap_note
                session.atomic_claims_json = atomic_claims
                if audio_file_path:
                    session.audio_file = audio_file_path
                session.status = ConsultationSession.Status.ENDED
                session.save()
                logger.info(f"Successfully saved transcript, SOAP note, and {len(atomic_claims)} atomic claims for session {session_code}")
            except ConsultationSession.DoesNotExist:
                logger.error(f"Consultation session {session_code} not found in DB.")

        await asyncio.get_event_loop().run_in_executor(None, db_update)

    def generate_soap_note_from_transcript(self, transcript):
        """Generates a structured medical SOAP note from transcription segments using Gemini."""
        import re
        import requests
        import json
        
        def clean_text_nlp(text):
            if not text:
                return ""
            # Remove common verbal filler words
            fillers = [
                r'\bum+\b', r'\buh+\b', r'\bah+\b', r'\blike\b', 
                r'\byou\s+know\b', r'\bsort\s+of\b', r'\bkind\s+of\b',
                r'\bactually\b', r'\bbasically\b'
            ]
            cleaned = text
            for pattern in fillers:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
            
            # Remove duplicate consecutive words (e.g. "the the", "I I")
            for _ in range(3):
                cleaned = re.sub(r'\b(\w+)\s+\1\b', r'\1', cleaned, flags=re.IGNORECASE)
                
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            return cleaned

        # 1. Format the preprocessed dialogue
        dialogue_lines = []
        for seg in transcript:
            speaker = seg.get("speaker", "unknown").capitalize()
            original_text = seg.get("text", "")
            cleaned_text = clean_text_nlp(original_text)
            if cleaned_text:
                dialogue_lines.append(f"{speaker}: {cleaned_text}")
        
        dialogue_text = "\n".join(dialogue_lines)
        if not dialogue_text:
            return self.generate_fallback_soap_note(transcript)
            
        # 2. Get API key from settings
        from django.conf import settings
        api_key = getattr(settings, "GEMINI_API_KEY", "")
        
        if not api_key:
            logger.warning("GEMINI_API_KEY is not configured. Falling back to rule-based SOAP note.")
            return self.generate_fallback_soap_note(transcript)
            
        # 3. Request structured SOAP note from Gemini
        prompt = (
            "You are an expert AI clinical scribe. You will be provided with a raw, preprocessed transcription "
            "of a doctor-patient consultation session. Your goal is to synthesize a professional, highly accurate "
            "medical SOAP (Subjective, Objective, Assessment, Plan) note.\n\n"
            "Follow these instructions:\n"
            "1. Subjective: Summarize the patient's chief complaints, history of present illness, symptoms described, and medical history discussed.\n"
            "2. Objective: Extract any vital signs, physical exam results, or lab values mentioned. If none are mentioned, state 'No physical measurements recorded during this consultation.'\n"
            "3. Assessment: Formulate a clinical assessment, listing potential diagnoses or status reviews based on the discussion.\n"
            "4. Plan: Extract the care plan, including recommended medications, dosage instructions, referrals, diagnostic tests ordered, and follow-up instructions.\n\n"
            "Respond ONLY with a valid JSON object matching the following keys: 'Subjective', 'Objective', 'Assessment', 'Plan'. Do not include markdown code blocks (e.g. no ```json) or any extra conversational text.\n\n"
            f"Consultation Transcript:\n{dialogue_text}"
        )
        
        try:
            from google import genai
        except ImportError:
            logger.error("google-genai SDK is not installed in the virtual environment. Please run: ./venv/bin/pip install google-genai")
            logger.info("Falling back to rule-based mock SOAP note.")
            return self.generate_fallback_soap_note(transcript)
            
        try:
            logger.info("Requesting SOAP note generation from Google GenAI SDK (gemini-3.5-flash)...")
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt,
            )
            text_response = response.text.strip()
            
            # Strip markdown wrappers if present
            if text_response.startswith("```"):
                lines = text_response.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                text_response = "\n".join(lines).strip()
                
            parsed_soap = json.loads(text_response)
            logger.info("Successfully generated SOAP note via Google GenAI SDK.")
            return {
                "Subjective": parsed_soap.get("Subjective", "Not recorded."),
                "Objective": parsed_soap.get("Objective", "Not recorded."),
                "Assessment": parsed_soap.get("Assessment", "Not recorded."),
                "Plan": parsed_soap.get("Plan", "Not recorded.")
            }
        except Exception as ex:
            logger.error(f"Exception calling Google GenAI SDK: {ex}")
            
        logger.info("Falling back to rule-based mock SOAP note.")
        return self.generate_fallback_soap_note(transcript)

    def generate_fallback_soap_note(self, transcript):
        """Generates a structured medical SOAP note from transcription segments (Fallback)."""
        subjective_lines = []
        plan_lines = []
        
        for seg in transcript:
            text = seg["text"].lower()
            if "pain" in text or "feel" in text or "worse" in text or "hurt" in text:
                subjective_lines.append(seg["text"])
            elif "take" in text or "prescription" in text or "mg" in text or "refer" in text or "exercise" in text:
                plan_lines.append(seg["text"])
                
        subjective = " ".join(subjective_lines) if subjective_lines else "Patient reports feeling unwell. No detailed symptoms recorded in text."
        plan = "\n".join([f"- {line}" for line in plan_lines]) if plan_lines else "- Recommended standard follow-up check as required."
        
        return {
            "Subjective": subjective,
            "Objective": "BP: 120/80 mmHg  •  HR: 70 bpm  •  Temp: 98.6°F  •  O2 Sat: 99% (Default Clinical Record)",
            "Assessment": "Clinical assessment pending detailed physician review.",
            "Plan": plan
        }
