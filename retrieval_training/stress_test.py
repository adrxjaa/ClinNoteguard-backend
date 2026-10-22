"""
Stress test: cases where BM25 and dense encoders are expected to fail in OPPOSITE ways.

  LEXICAL  (BM25 should win)  : dose / number / laterality / look-alike drug names.
                                Dense encoders embed "aspirin 25 mg" ~ "aspirin 200 mg".
  SEMANTIC (dense should win) : synonyms, lay terms, brand/generic, abbreviations not in our list.
                                BM25 has no token overlap.

Each case = a small hand-written patient history (target event + distractors) and a claim.
A system passes if the TARGET event is ranked #1.

Usage: ./venv/bin/python retrieval_training/stress_test.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from api.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from api.retrieval.dense_retriever import DEFAULT_DENSE_MODEL, FINETUNED_DIR, DenseRetriever  # noqa: E402
from api.retrieval.ehr_store import EHREvent, EHRStore  # noqa: E402
from api.retrieval.hybrid import HybridRetriever  # noqa: E402

# (category, claim, target_description, [distractor descriptions])
CASES = [
    # ---------------- LEXICAL: should favour BM25 ----------------
    ("LEXICAL", "Patient takes aspirin 200 mg daily.",
     "Patient takes aspirin 200 mg daily.",
     ["Patient takes aspirin 25 mg daily.", "Patient takes aspirin 75 mg daily."]),
    ("LEXICAL", "Patient takes aspirin 25 mg daily.",
     "Patient takes aspirin 25 mg daily.",
     ["Patient takes aspirin 200 mg daily.", "Patient takes aspirin 325 mg daily."]),
    ("LEXICAL", "Metformin was increased to 1000 mg twice daily.",
     "Metformin dose increased to 1000 mg twice daily.",
     ["Metformin dose increased to 500 mg twice daily.", "Metformin started at 500 mg once daily."]),
    ("LEXICAL", "Patient has pain in the left knee.",
     "Patient reports chronic pain in the left knee.",
     ["Patient reports chronic pain in the right knee.", "Patient reports chronic pain in the left hip."]),
    ("LEXICAL", "Patient was prescribed hydroxyzine.",
     "Patient was prescribed hydroxyzine for itching.",
     ["Patient was prescribed hydralazine for hypertension.", "Patient was prescribed hydroxychloroquine for lupus."]),
    ("LEXICAL", "HbA1c was 9.1 percent.",
     "Lab: HbA1c was 9.1 percent.",
     ["Lab: HbA1c was 6.1 percent.", "Lab: HbA1c was 7.9 percent."]),
    ("LEXICAL", "Patient is allergic to penicillin.",
     "Patient is allergic to penicillin causing hives.",
     ["Patient is allergic to pencil shavings dust.", "Patient is allergic to peanuts causing hives."]),
    ("LEXICAL", "Lisinopril 10 mg daily.",
     "Patient's lisinopril dose was set to 10 mg daily.",
     ["Patient's lisinopril dose was set to 40 mg daily.", "Patient's losartan dose was set to 10 mg daily."]),
    ("LEXICAL", "Patient uses 10 units of insulin at bedtime.",
     "Patient injects 10 units of insulin at bedtime.",
     ["Patient injects 100 units of insulin at bedtime.", "Patient injects 20 units of insulin at bedtime."]),
    ("LEXICAL", "Warfarin 5 mg daily.",
     "Patient is anticoagulated with warfarin 5 mg daily.",
     ["Patient is anticoagulated with warfarin 2 mg daily.", "Patient is anticoagulated with warfarin 10 mg daily."]),
    ("LEXICAL", "Patient takes aspirin 25 milligrams every day.",
     "Patient takes aspirin 25mg daily.",
     ["Patient takes aspirin 250mg daily.", "Patient takes aspirin 2.5mg daily."]),
    # ---------------- SEMANTIC: should favour dense ----------------
    ("SEMANTIC", "The patient had a heart attack.",
     "Patient has a history of myocardial infarction.",
     ["Patient has a history of appendectomy.", "Patient has a history of seasonal allergies."]),
    ("SEMANTIC", "The patient has high blood pressure.",
     "Patient has a history of hypertension.",
     ["Patient has a history of hypothyroidism.", "Patient has a history of migraine."]),
    ("SEMANTIC", "The patient takes a water pill.",
     "Patient takes furosemide daily.",
     ["Patient takes sertraline daily.", "Patient takes vitamin D daily."]),
    ("SEMANTIC", "Patient is on Tylenol for pain.",
     "Patient takes acetaminophen as needed for pain.",
     ["Patient takes lisinopril as needed for pain.", "Patient takes omeprazole daily."]),
    ("SEMANTIC", "Patient had a stroke last year.",
     "Patient had a cerebrovascular accident one year ago.",
     ["Patient had a tonsillectomy one year ago.", "Patient had a knee arthroscopy one year ago."]),
    ("SEMANTIC", "Patient has sugar problems.",
     "Patient has type 2 diabetes mellitus.",
     ["Patient has osteoarthritis.", "Patient has chronic sinusitis."]),
    ("SEMANTIC", "Patient used to smoke cigarettes.",
     "Patient is a former tobacco user who quit seven years ago.",
     ["Patient drinks alcohol socially.", "Patient exercises three times a week."]),
    ("SEMANTIC", "Patient had a clot in the leg.",
     "Patient was treated for deep vein thrombosis.",
     ["Patient was treated for cellulitis.", "Patient was treated for strep throat."]),
    # ---------------- NEGATION / TEMPORAL (both likely to be weak) ----------------
    ("NEGATION", "Patient denies any history of diabetes.",
     "Patient has no history of diabetes.",
     ["Patient has a history of diabetes.", "Patient has a family history of diabetes."]),
]


def make_store():
    store = EHRStore()
    for i, (cat, claim, target, distractors) in enumerate(CASES):
        pid = f"P_STRESS{i}"
        for j, desc in enumerate([target] + distractors):
            store.add(EHREvent(doc_id=f"{pid}#{j}", patient_id=pid, encounter_id=f"S{i}", relative_time="T-1 year",
                               event_type="other", description=desc))
    return store


def rank_fns(store):
    bm = BM25Retriever(store)                       # corpus-wide IDF (default)
    bm_local = BM25Retriever(store, global_idf=False)  # IDF from the patient's own few events
    base = DenseRetriever(store, DEFAULT_DENSE_MODEL)
    systems = {
        "BM25(local)": lambda c, p: [e for e, _ in bm_local.search(c, p, 20)],
        "BM25": lambda c, p: [e for e, _ in bm.search(c, p, 20)],
        "Dense(base)": lambda c, p: [e for e, _ in base.search(c, p, 20)],
    }
    dense_for_hybrid = base
    if os.path.isdir(FINETUNED_DIR):
        ft = DenseRetriever(store, FINETUNED_DIR)
        systems["Dense(FT)"] = lambda c, p: [e for e, _ in ft.search(c, p, 20)]
        dense_for_hybrid = ft
    hyb = HybridRetriever(bm, dense_for_hybrid)
    systems["Hybrid"] = lambda c, p: [r[0] for r in hyb.search_events(c, p, 20)]
    return systems


def main():
    store = make_store()
    systems = rank_fns(store)
    names = list(systems)
    results = {n: [] for n in names}
    print(f"{'#':<3}{'cat':<9}" + "".join(f"{n:<13}" for n in names) + "claim")
    for i, (cat, claim, target, _) in enumerate(CASES):
        pid = f"P_STRESS{i}"
        row = f"{i:<3}{cat:<9}"
        for n in names:
            top = systems[n](claim, pid)[0]
            ok = top.doc_id == f"{pid}#0"
            results[n].append((cat, ok))
            row += f"{'PASS' if ok else 'FAIL':<13}"
        print(row + claim)

    print("\nTop-1 accuracy by category")
    cats = ["LEXICAL", "SEMANTIC", "NEGATION"]
    print(f"{'system':<14}" + "".join(f"{c:>10}" for c in cats) + f"{'ALL':>8}")
    for n in names:
        line = f"{n:<14}"
        for c in cats:
            xs = [ok for cc, ok in results[n] if cc == c]
            line += f"{sum(xs)}/{len(xs):>1}".rjust(10)
        allx = [ok for _, ok in results[n]]
        line += f"{sum(allx)}/{len(allx)}".rjust(8)
        print(line)


if __name__ == "__main__":
    main()
