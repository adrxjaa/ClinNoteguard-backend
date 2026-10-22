"""
Fine-tune the dense bi-encoder on (claim, EHR-event) pairs.

Loss: InfoNCE / MultipleNegativesRankingLoss with
  - in-batch negatives
  - one hard negative per pair (a NON-relevant event from the SAME patient's history)

Training pairs come from:
  1. retrieval_data/train_claims.jsonl      (claims from ACI notes, silver labels)
  2. template augmentation of EHR events    (e.g. diagnosis 'GERD' -> 'Patient has a history of GERD.')
     because the demo dataset is tiny.

Usage:  ./venv/bin/python retrieval_training/train_dense.py [--epochs 5] [--base MODEL]
"""

import argparse
import json
import os
import random
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from api.retrieval.dense_retriever import DEFAULT_DENSE_MODEL, FINETUNED_DIR  # noqa: E402
from api.retrieval.ehr_store import EHRStore  # noqa: E402

DATA_DIR = os.path.join(ROOT, "retrieval_data")


def load_jsonl(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing - run retrieval_training/build_dataset.py first")
    with open(path) as f:
        return [json.loads(l) for l in f]


def template_claims(ev):
    """Cheap paraphrase-style pseudo claims generated from structured EHR fields."""
    out = []
    if ev.diagnosis:
        out += [f"The patient has a history of {ev.diagnosis}.", f"Patient was diagnosed with {ev.diagnosis}."]
    if ev.medication:
        out += [f"The patient takes {ev.medication}.", f"Patient was prescribed {ev.medication}."]
    if ev.investigation:
        out += [f"The patient had a {ev.investigation} done.", f"{ev.investigation} was performed previously."]
    return out


def build_pairs(include_aug=True, seed=13):
    rng = random.Random(seed)
    store = EHRStore.from_csv(["train"])
    by_id = {e.doc_id: e for e in store.all_events()}
    triples = []
    for r in load_jsonl("train_claims.jsonl"):
        for p in r["positives"]:
            neg = rng.choice(r["negatives"]) if r["negatives"] else None
            triples.append((r["claim_text"], by_id[p].text, by_id[neg].text if neg else None))
    n_real = len(triples)
    if include_aug:
        for pid in store.patient_ids():
            evs = store.events(pid)
            for ev in evs:
                others = [e for e in evs if e.doc_id != ev.doc_id]
                for c in template_claims(ev):
                    neg = rng.choice(others).text if others else None
                    triples.append((c, ev.text, neg))
    rng.shuffle(triples)
    return triples, n_real


def encode(model, texts, device):
    feats = model.tokenize(texts)
    feats = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in feats.items()}
    return F.normalize(model(feats)["sentence_embedding"], dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_DENSE_MODEL)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--no-aug", action="store_true")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.base, device=device)
    triples, n_real = build_pairs(include_aug=not args.no_aug)
    print(f"Training triples: {len(triples)} ({n_real} from ACI-Bench claims, {len(triples) - n_real} augmented) on {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_total = args.epochs * ((len(triples) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.1 * steps_total))))
    model.train()
    for ep in range(args.epochs):
        random.shuffle(triples)
        total = 0.0
        for i in range(0, len(triples), args.batch):
            batch = triples[i:i + args.batch]
            if len(batch) < 2:
                continue
            q = encode(model, [b[0] for b in batch], device)
            docs = [b[1] for b in batch] + [b[2] for b in batch if b[2]]
            d = encode(model, docs, device)
            loss = F.cross_entropy(q @ d.T * args.scale, torch.arange(len(batch), device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(batch)
        print(f"epoch {ep + 1}/{args.epochs}  loss={total / len(triples):.4f}")

    os.makedirs(FINETUNED_DIR, exist_ok=True)
    model.save(FINETUNED_DIR)
    print(f"Saved fine-tuned dense encoder -> {FINETUNED_DIR}")


if __name__ == "__main__":
    main()
