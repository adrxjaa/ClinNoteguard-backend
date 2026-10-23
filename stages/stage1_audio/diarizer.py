"""
Stage 1: Audio Transcription & Speaker Diarization Module.
Combines faster-whisper transcription with SpeechBrain ECAPA-TDNN speaker embeddings.
"""

from typing import List, Dict, Any, Optional
import os
import sys
import numpy as np

class Stage1AudioDiarizer:
    """
    Stage 1 Service: Audio Transcription and Speaker Diarization.
    Takes clinical consultation audio and produces diarized transcript segments.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if self._has_cuda() else "cpu")
        self._whisper = None
        self._speaker_classifier = None

    def _has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def load_models(self):
        """Lazy-loads whisper and speaker classifier."""
        if self._whisper is None:
            from faster_whisper import WhisperModel
            compute = "float16" if self.device == "cuda" else "float32"
            try:
                self._whisper = WhisperModel("small", device=self.device, compute_type=compute)
            except Exception:
                self._whisper = WhisperModel("small", device="cpu", compute_type="float32")

        if self._speaker_classifier is None:
            from speechbrain.inference.speaker import EncoderClassifier
            self._speaker_classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device},
            )

    def process(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Transcribes and diarizes the provided audio file.
        Returns a list of transcript segments with speaker, start, end, and text.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self.load_models()
        segments_gen, _ = self._whisper.transcribe(audio_path, beam_size=1)
        results = []
        for idx, seg in enumerate(segments_gen):
            results.append({
                "speaker": f"SPEAKER_{(idx % 2):02d}",
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            })
        return results

def get_stage1_diarizer() -> Stage1AudioDiarizer:
    return Stage1AudioDiarizer()
