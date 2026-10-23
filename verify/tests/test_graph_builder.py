from verify.graph.builder import (
    GraphBuilder,
    EDGE_RETRIEVED,
    EDGE_RETRIEVED_REV,
    EDGE_MENTIONS,
    EDGE_IS_A,
    EDGE_SUPERSEDES,
    EDGE_SAME_KEY,
)

def test_evidence_graph_structure():
    builder = GraphBuilder()
    claim = {
        "claim_id": "CLM-100",
        "claim_text": "Patient is taking Lisinopril 20 mg oral daily for hypertension.",
    }
    candidates = [
        {
            "record_id": "EHR_01",
            "claim_text": "Lisinopril increased to 20 mg daily.",
            "valid_from": "2024-01-01T09:00:00Z",
            "source": "EHR",
            "score": 0.95,
            "dose_agreement": "match",
        },
        {
            "record_id": "EHR_02",
            "claim_text": "Patient started on Lisinopril 10 mg daily.",
            "valid_from": "2023-06-01T09:00:00Z",
            "source": "EHR",
            "score": 0.88,
            "dose_agreement": "mismatch",
        },
        {
            "record_id": "TRANS_01",
            "claim_text": "[Doctor]: How is your blood pressure on Lisinopril?",
            "timestamp": "2024-01-15T09:00:00Z",
            "source": "TRANSCRIPT",
            "score": 0.70,
            "dose_agreement": "n/a",
        },
    ]

    graph = builder.build_graph(claim, candidates, encounter_time="2024-01-15T09:00:00Z")

    assert len(graph.nodes) >= 4 # 1 claim + 3 evidence + concept nodes
    assert graph.nodes[0].node_type == "CLAIM"
    assert graph.nodes[1].node_type == "EVIDENCE"
    assert graph.nodes[1].record_id == "EHR_01"

    # Check edge types exist
    edge_types = {e.edge_type for e in graph.edges}
    assert EDGE_RETRIEVED in edge_types
    assert EDGE_RETRIEVED_REV in edge_types
    assert EDGE_MENTIONS in edge_types
    assert EDGE_SUPERSEDES in edge_types

    # Convert to tensors
    tensors = graph.to_tensors()
    assert tensors["edge_index"].shape[0] == 2
    assert tensors["edge_index"].shape[1] == len(graph.edges)
    assert tensors["node_types"].shape[0] == len(graph.nodes)
    assert len(tensors["record_ids"]) == len(graph.nodes)
