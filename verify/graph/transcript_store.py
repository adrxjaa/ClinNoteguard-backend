from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import re
from rank_bm25 import BM25Okapi
from api.retrieval.text_utils import tokenize_clinical

@dataclass
class TranscriptUtterance:
    utterance_id: str
    encounter_id: str
    speaker: str
    text: str
    window_text: str
    timestamp: str # encounter_time
    source: str = "TRANSCRIPT"

class TranscriptStore:
    """
    Ingests and indexes clinical dialogue transcripts (Django transcript_json & ACI-BENCH text).
    Provides windowed utterance retrieval matching the EHR candidate interface.
    """

    def __init__(self):
        self.utterances: List[TranscriptUtterance] = []
        self.encounter_to_utterances: Dict[str, List[TranscriptUtterance]] = {}
        self.bm25_indices: Dict[str, BM25Okapi] = {}
        self.tokenized_corpus: Dict[str, List[List[str]]] = {}

    def ingest_dialogue(
        self,
        encounter_id: str,
        dialogue: Any,
        encounter_time: str = "2024-01-15T09:00:00Z",
    ) -> List[TranscriptUtterance]:
        """
        Ingests dialogue in either Django JSON format or ACI-BENCH raw text format.
        Chunks turns into 2-turn windowed utterances.
        """
        turns: List[Dict[str, str]] = []

        if isinstance(dialogue, list):
            # Django transcript_json: list of dicts with 'speaker', 'text'
            for item in dialogue:
                speaker = item.get("speaker", "Speaker")
                text = item.get("text", "").strip()
                if text:
                    turns.append({"speaker": speaker, "text": text})

        elif isinstance(dialogue, str):
            # ACI-BENCH format: "[doctor] text\n[patient] text"
            # Regex to match [doctor] or [patient] or [doctor/patient]
            pattern = re.compile(r"\[(doctor|patient|speaker|clinician)\]\s*", re.IGNORECASE)
            parts = pattern.split(dialogue)
            # parts will alternate: empty, speaker, text, speaker, text
            i = 1
            while i < len(parts) - 1:
                speaker = parts[i].capitalize()
                text = parts[i + 1].strip()
                if text:
                    turns.append({"speaker": speaker, "text": text})
                i += 2

        if not turns and isinstance(dialogue, str) and dialogue.strip():
            # Fallback if no speaker tags
            turns.append({"speaker": "Dialogue", "text": dialogue.strip()})

        # Build windowed utterances (sliding window of size 2)
        windowed: List[TranscriptUtterance] = []
        for idx in range(len(turns)):
            curr = turns[idx]
            prev = turns[idx - 1] if idx > 0 else None
            if prev:
                window_text = f"[{prev['speaker']}]: {prev['text']} | [{curr['speaker']}]: {curr['text']}"
            else:
                window_text = f"[{curr['speaker']}]: {curr['text']}"

            utt_id = f"TRANS_{encounter_id}_U{idx:03d}"
            utt = TranscriptUtterance(
                utterance_id=utt_id,
                encounter_id=encounter_id,
                speaker=curr["speaker"],
                text=curr["text"],
                window_text=window_text,
                timestamp=encounter_time,
                source="TRANSCRIPT",
            )
            windowed.append(utt)

        self.encounter_to_utterances[encounter_id] = windowed
        self.utterances.extend(windowed)

        # Index with BM25
        if windowed:
            corpus = [tokenize_clinical(u.window_text) for u in windowed]
            self.tokenized_corpus[encounter_id] = corpus
            self.bm25_indices[encounter_id] = BM25Okapi(corpus)

        return windowed

    def search(
        self,
        query: str,
        encounter_id: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k relevant transcript window candidates for an encounter.
        Matches EHR candidate schema.
        """
        if encounter_id not in self.bm25_indices:
            return []

        bm25 = self.bm25_indices[encounter_id]
        tokenized_query = tokenize_clinical(query)
        if not tokenized_query:
            return []

        scores = bm25.get_scores(tokenized_query)
        utterances = self.encounter_to_utterances[encounter_id]

        ranked = sorted(
            zip(utterances, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        candidates: List[Dict[str, Any]] = []
        for rank, (utt, score) in enumerate(ranked[:top_k], 1):
            candidates.append({
                "claim_id": utt.utterance_id,
                "record_id": utt.utterance_id,
                "claim_text": utt.window_text,
                "section": "Transcript",
                "source": "TRANSCRIPT",
                "timestamp": utt.timestamp,
                "bm25_score": float(score),
                "bm25_rank": rank,
                "score": float(score),
                "rank": rank,
                "entities": [],
            })

        return candidates
