from verify.concepts.attr_type import AttrType
from verify.concepts.facts import TypedFact, extract_facts
from verify.graph.validity import resolve_state, StateStatus
from verify.graph.builder import GraphBuilder, EDGE_SUPERSEDES

def test_acceptance_criteria_1_toy_timeline():
    """
    Acceptance Criteria 1 Toy Test:
    Drug A 10 mg in January, 20 mg in March, discontinued in May.
    Encounter in February: claim '10 mg' is current.
    Encounter in April: claim '10 mg' is STALE.
    Encounter in June: claim 'taking Drug A' is STALE.
    The graph must contain SUPERSEDES edges.
    """
    # 1. Define timeline facts for Drug A (e.g. atorvastatin)
    fact_jan = TypedFact(
        record_id="EHR_JAN",
        source="EHR",
        attr_type=AttrType.MEDICATION,
        key_cui="C0286651", # atorvastatin
        value="10mg",
        unit="mg",
        polarity=True,
        event_type="START",
        timestamp="2024-01-15T09:00:00Z",
    )
    fact_mar = TypedFact(
        record_id="EHR_MAR",
        source="EHR",
        attr_type=AttrType.MEDICATION,
        key_cui="C0286651",
        value="20mg",
        unit="mg",
        polarity=True,
        event_type="CHANGE",
        timestamp="2024-03-15T09:00:00Z",
    )
    fact_may = TypedFact(
        record_id="EHR_MAY",
        source="EHR",
        attr_type=AttrType.MEDICATION,
        key_cui="C0286651",
        value="discontinued",
        unit=None,
        polarity=False,
        event_type="STOP",
        timestamp="2024-05-15T09:00:00Z",
    )

    all_facts = [fact_jan, fact_mar, fact_may]

    # Encounter in February: claim '10 mg' is CURRENT
    state_feb = resolve_state(all_facts, encounter_time="2024-02-15T09:00:00Z", mode="typed")
    entry_feb = state_feb[(fact_jan.key_cui, AttrType.MEDICATION)]
    assert entry_feb.status == StateStatus.CURRENT
    assert entry_feb.effective_value == "10mg"

    # Encounter in April: 20 mg is current, so 10 mg is SUPERSEDED (STALE)
    state_apr = resolve_state(all_facts, encounter_time="2024-04-15T09:00:00Z", mode="typed")
    entry_apr = state_apr[(fact_jan.key_cui, AttrType.MEDICATION)]
    assert entry_apr.status == StateStatus.CURRENT
    assert entry_apr.effective_value == "20mg"
    assert fact_jan in entry_apr.superseded_facts

    # Encounter in June: Discontinued in May, so overall status is SUPERSEDED / STALE
    state_jun = resolve_state(all_facts, encounter_time="2024-06-15T09:00:00Z", mode="typed")
    entry_jun = state_jun[(fact_jan.key_cui, AttrType.MEDICATION)]
    assert entry_jun.status == StateStatus.SUPERSEDED
    assert entry_jun.effective_value == "discontinued"

    # Verify Graph contains SUPERSEDES edges between newer (March) and older (January) records
    builder = GraphBuilder()
    candidates = [
        {
            "record_id": "EHR_JAN",
            "claim_text": "Patient takes Atorvastatin 10 mg daily.",
            "valid_from": "2024-01-15T09:00:00Z",
            "source": "EHR",
            "score": 0.9,
            "dose_agreement": "match",
        },
        {
            "record_id": "EHR_MAR",
            "claim_text": "Atorvastatin increased to 20 mg daily.",
            "valid_from": "2024-03-15T09:00:00Z",
            "source": "EHR",
            "score": 0.85,
            "dose_agreement": "mismatch",
        },
    ]

    claim_10mg = {"claim_id": "CLM-001", "claim_text": "Patient is taking Atorvastatin 10 mg."}
    graph_apr = builder.build_graph(claim_10mg, candidates, encounter_time="2024-04-15T09:00:00Z")

    supersedes_edges = [e for e in graph_apr.edges if e.edge_type == EDGE_SUPERSEDES]
    assert len(supersedes_edges) > 0, "Graph must contain SUPERSEDES edges between newer and older doses!"
