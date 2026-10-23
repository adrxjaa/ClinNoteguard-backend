from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
import math
import torch

from verify.concepts.attr_type import AttrType, infer_attr_type
from verify.concepts.facts import TypedFact, extract_facts
from verify.concepts.hierarchy import ConceptHierarchy
from verify.concepts.linker import ConceptLinker, CuratedDictionaryLinker
from verify.graph.validity import StateStatus, resolve_state, parse_iso_datetime

# Edge Type Constants
EDGE_RETRIEVED = 0
EDGE_RETRIEVED_REV = 1
EDGE_MENTIONS = 2
EDGE_MENTIONED_BY = 3
EDGE_IS_A = 4
EDGE_SUBCLASS_OF = 5
EDGE_SUPERSEDES = 6
EDGE_SUPERSEDED_BY = 7
EDGE_SAME_KEY = 8
NUM_EDGE_TYPES = 9

DATE_GAP_BUCKETS = ["0", "1-7", "8-30", "31-180", "181-365", ">365", "unknown"]

def get_date_gap_bucket(gap_days: Optional[int]) -> int:
    if gap_days is None:
        return 6 # unknown
    if gap_days == 0:
        return 0
    elif 1 <= gap_days <= 7:
        return 1
    elif 8 <= gap_days <= 30:
        return 2
    elif 31 <= gap_days <= 180:
        return 3
    elif 181 <= gap_days <= 365:
        return 4
    elif gap_days > 365:
        return 5
    return 6

@dataclass
class GraphNode:
    node_id: int
    node_type: str # "CLAIM" | "EVIDENCE" | "CONCEPT"
    label: str # text or CUI
    record_id: Optional[str] = None
    source: str = "OTHER" # "CLAIM" | "EHR" | "TRANSCRIPT" | "CONCEPT"
    attr_type: AttrType = AttrType.OTHER
    status: StateStatus = StateStatus.UNKNOWN
    date_gap_bucket: int = 6
    text: str = ""

@dataclass
class GraphEdge:
    src: int
    dst: int
    edge_type: int
    features: List[float] = field(default_factory=list)

@dataclass
class EvidenceGraph:
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    claim_node_idx: int = 0
    evidence_node_indices: List[int] = field(default_factory=list)
    concept_node_indices: List[int] = field(default_factory=list)

    def to_tensors(self) -> Dict[str, Any]:
        """
        Converts the EvidenceGraph into pure PyTorch tensors for GNN / Transformer models.
        """
        num_nodes = len(self.nodes)
        node_type_map = {"CLAIM": 0, "EVIDENCE": 1, "CONCEPT": 2}
        source_map = {"CLAIM": 0, "EHR": 1, "TRANSCRIPT": 2, "CONCEPT": 3, "OTHER": 4}
        attr_map = {a.value: i for i, a in enumerate(AttrType)}
        status_map = {s.value: i for i, s in enumerate(StateStatus)}

        node_types = torch.tensor([node_type_map.get(n.node_type, 1) for n in self.nodes], dtype=torch.long)
        source_types = torch.tensor([source_map.get(n.source, 4) for n in self.nodes], dtype=torch.long)
        attr_types = torch.tensor([attr_map.get(n.attr_type.value, 6) for n in self.nodes], dtype=torch.long)
        statuses = torch.tensor([status_map.get(n.status.value, 3) for n in self.nodes], dtype=torch.long)
        date_buckets = torch.tensor([n.date_gap_bucket for n in self.nodes], dtype=torch.long)

        edge_index_list = []
        edge_types_list = []
        edge_attr_list = []

        max_feat_len = 10
        for e in self.edges:
            edge_index_list.append([e.src, e.dst])
            edge_types_list.append(e.edge_type)
            feats = e.features + [0.0] * (max_feat_len - len(e.features))
            edge_attr_list.append(feats[:max_feat_len])

        if edge_index_list:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_type = torch.tensor(edge_types_list, dtype=torch.long)
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long)
            edge_attr = torch.empty((0, max_feat_len), dtype=torch.float32)

        record_ids = [n.record_id for n in self.nodes]

        return {
            "num_nodes": num_nodes,
            "node_types": node_types,
            "source_types": source_types,
            "attr_types": attr_types,
            "statuses": statuses,
            "date_buckets": date_buckets,
            "edge_index": edge_index,
            "edge_type": edge_type,
            "edge_attr": edge_attr,
            "record_ids": record_ids,
            "claim_node_idx": self.claim_node_idx,
            "evidence_node_indices": self.evidence_node_indices,
            "concept_node_indices": self.concept_node_indices,
        }

    def to_pyg(self) -> Any:
        """Attempts conversion to torch_geometric.data.Data if installed."""
        tensors = self.to_tensors()
        try:
            from torch_geometric.data import Data
            return Data(
                edge_index=tensors["edge_index"],
                edge_type=tensors["edge_type"],
                edge_attr=tensors["edge_attr"],
                node_types=tensors["node_types"],
                source_types=tensors["source_types"],
                attr_types=tensors["attr_types"],
                statuses=tensors["statuses"],
                date_buckets=tensors["date_buckets"],
                record_ids=tensors["record_ids"],
                claim_node_idx=tensors["claim_node_idx"],
                evidence_node_indices=tensors["evidence_node_indices"],
                concept_node_indices=tensors["concept_node_indices"],
            )
        except ImportError:
            # Fallback returns the pure tensor dictionary
            return tensors

class GraphBuilder:
    """
    Constructs multi-relational EvidenceGraph instances from claims and retrieved candidates.
    """

    def __init__(
        self,
        linker: Optional[ConceptLinker] = None,
        hierarchy: Optional[ConceptHierarchy] = None,
    ):
        self.hierarchy = hierarchy or ConceptHierarchy(backend="curated")
        self.linker = linker or CuratedDictionaryLinker(hierarchy=self.hierarchy)

    def build_graph(
        self,
        claim: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        encounter_time: str,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> EvidenceGraph:
        cfg = cfg or {}
        top_k = cfg.get("top_k", 20)
        validity_mode = cfg.get("validity_mode", "typed")

        enc_dt = parse_iso_datetime(encounter_time) or datetime(2024, 1, 15, 9, 0, 0)
        claim_text = claim.get("claim_text", "")

        # 1. Truncate candidates to top_k
        cand_list = candidates[:top_k]

        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []

        # Node 0: CLAIM
        claim_mentions = self.linker.link(claim_text)
        claim_attr = infer_attr_type(claim_text)
        nodes.append(
            GraphNode(
                node_id=0,
                node_type="CLAIM",
                label="CLAIM",
                record_id=claim.get("claim_id", "CLM-000"),
                source="CLAIM",
                attr_type=claim_attr,
                status=StateStatus.CURRENT,
                date_gap_bucket=0,
                text=claim_text,
            )
        )
        claim_node_idx = 0

        # 2. Extract facts from candidates and resolve temporal validity
        all_candidate_facts: List[TypedFact] = []
        cand_facts_by_cid: Dict[str, List[TypedFact]] = {}

        for c in cand_list:
            c_text = c.get("claim_text", "")
            c_id = c.get("record_id") or c.get("claim_id", "")
            c_source = c.get("source", "EHR")
            c_ts = c.get("valid_from") or c.get("timestamp") or encounter_time
            facts = extract_facts(
                record_id=c_id,
                text=c_text,
                source=c_source,
                timestamp=c_ts,
                linker=self.linker,
            )
            cand_facts_by_cid[c_id] = facts
            all_candidate_facts.extend(facts)

        resolved_states = resolve_state(
            facts=all_candidate_facts,
            encounter_time=encounter_time,
            mode=validity_mode,
        )

        # 3. Create EVIDENCE nodes
        evidence_indices: List[int] = []
        cui_to_node_idx: Dict[str, int] = {}
        concept_indices: List[int] = []

        # Mean and std for z-normalizing cross-encoder scores
        ce_scores = [float(c.get("cross_encoder_score", c.get("score", 0.0))) for c in cand_list]
        mean_ce = sum(ce_scores) / len(ce_scores) if ce_scores else 0.0
        std_ce = math.sqrt(sum((x - mean_ce) ** 2 for x in ce_scores) / len(ce_scores)) if len(ce_scores) > 1 else 1.0
        if std_ce < 1e-6:
            std_ce = 1.0

        for c in cand_list:
            node_idx = len(nodes)
            evidence_indices.append(node_idx)
            c_id = c.get("record_id") or c.get("claim_id", "")
            c_text = c.get("claim_text", "")
            c_source = c.get("source", "EHR")
            c_ts = c.get("valid_from") or c.get("timestamp") or encounter_time

            c_dt = parse_iso_datetime(c_ts)
            gap_days = (enc_dt - c_dt).days if c_dt else None
            bucket = get_date_gap_bucket(gap_days)

            # Determine dominant status & attr_type
            c_facts = cand_facts_by_cid.get(c_id, [])
            dominant_status = StateStatus.UNKNOWN
            dominant_attr = infer_attr_type(c_text)

            if c_facts:
                dominant_attr = c_facts[0].attr_type
                # Look up resolved state for the primary fact
                entry = resolved_states.get(c_facts[0].key)
                if entry:
                    # If this specific record is superseded by a newer fact or discontinued
                    if any(sf.record_id == c_id for sf in entry.superseded_facts):
                        dominant_status = StateStatus.SUPERSEDED
                    elif entry.current_fact and entry.current_fact.record_id == c_id:
                        dominant_status = StateStatus.CURRENT
                    else:
                        dominant_status = entry.status
            elif c_source == "TRANSCRIPT":
                dominant_status = StateStatus.CURRENT

            # Also check explicit valid_to on the record
            if c.get("valid_to"):
                vto_dt = parse_iso_datetime(c["valid_to"])
                if vto_dt and vto_dt <= enc_dt:
                    dominant_status = StateStatus.SUPERSEDED

            nodes.append(
                GraphNode(
                    node_id=node_idx,
                    node_type="EVIDENCE",
                    label=c_id,
                    record_id=c_id,
                    source=c_source,
                    attr_type=dominant_attr,
                    status=dominant_status,
                    date_gap_bucket=bucket,
                    text=c_text,
                )
            )

            # RETRIEVED Edge: claim <-> evidence
            raw_ce = float(c.get("cross_encoder_score", c.get("score", 0.0)))
            z_ce = (raw_ce - mean_ce) / std_ce
            rrf_score = float(c.get("rrf_score", 1.0 / (60 + c.get("rank", 1))))
            bm25_score = float(c.get("bm25_score", 0.0))
            dense_score = float(c.get("dense_score", 0.0))

            dose_flag = c.get("dose_agreement", "n/a").lower()
            dose_one_hot = [
                1.0 if dose_flag == "match" else 0.0,
                1.0 if dose_flag == "mismatch" else 0.0,
                1.0 if dose_flag not in ["match", "mismatch"] else 0.0,
            ]

            # Concept Jaccard
            cand_mentions = self.linker.link(c_text)
            claim_cuis = {m.cui for m in claim_mentions}
            cand_cuis = {m.cui for m in cand_mentions}
            union_cuis = claim_cuis | cand_cuis
            jaccard = (len(claim_cuis & cand_cuis) / len(union_cuis)) if union_cuis else 0.0
            log_gap = math.log1p(max(0, gap_days)) if gap_days is not None else 0.0

            edge_feats = [
                z_ce,
                rrf_score,
                bm25_score,
                dense_score,
                dose_one_hot[0],
                dose_one_hot[1],
                dose_one_hot[2],
                jaccard,
                log_gap,
            ]

            edges.append(GraphEdge(src=claim_node_idx, dst=node_idx, edge_type=EDGE_RETRIEVED, features=edge_feats))
            edges.append(GraphEdge(src=node_idx, dst=claim_node_idx, edge_type=EDGE_RETRIEVED_REV, features=edge_feats))

        # 4. Helper to get or add CONCEPT node
        def get_or_add_concept(cui: str, label: str) -> int:
            if cui in cui_to_node_idx:
                return cui_to_node_idx[cui]
            idx = len(nodes)
            cui_to_node_idx[cui] = idx
            concept_indices.append(idx)
            nodes.append(
                GraphNode(
                    node_id=idx,
                    node_type="CONCEPT",
                    label=label,
                    record_id=cui,
                    source="CONCEPT",
                    attr_type=infer_attr_type(label),
                    status=StateStatus.UNKNOWN,
                    date_gap_bucket=6,
                    text=label,
                )
            )
            return idx

        # Connect CLAIM & EVIDENCE to CONCEPT nodes via MENTIONS edges
        all_text_nodes = [(0, claim_mentions)] + [
            (ev_idx, self.linker.link(cand_list[i].get("claim_text", "")))
            for i, ev_idx in enumerate(evidence_indices)
        ]

        active_cuis: Set[str] = set()
        for t_idx, mentions in all_text_nodes:
            for m in mentions:
                c_idx = get_or_add_concept(m.cui, m.canonical_name or m.text)
                active_cuis.add(m.canonical_name or m.cui)
                edges.append(GraphEdge(src=t_idx, dst=c_idx, edge_type=EDGE_MENTIONS, features=[1.0]))
                edges.append(GraphEdge(src=c_idx, dst=t_idx, edge_type=EDGE_MENTIONED_BY, features=[1.0]))

        # Add IS_A hierarchy edges for concepts
        for concept_str in list(active_cuis):
            ancestors = self.hierarchy.get_ancestors(concept_str, max_hops=2)
            cui_child = self.hierarchy.resolve_cui(concept_str)
            child_idx = get_or_add_concept(cui_child, concept_str)

            for anc in ancestors:
                cui_parent = self.hierarchy.resolve_cui(anc)
                parent_idx = get_or_add_concept(cui_parent, anc)
                edges.append(GraphEdge(src=child_idx, dst=parent_idx, edge_type=EDGE_IS_A, features=[1.0]))
                edges.append(GraphEdge(src=parent_idx, dst=child_idx, edge_type=EDGE_SUBCLASS_OF, features=[1.0]))

        # 5. Evidence-to-Evidence Edges: SUPERSEDES and SAME_KEY
        for i in range(len(evidence_indices)):
            for j in range(i + 1, len(evidence_indices)):
                idx_i = evidence_indices[i]
                idx_j = evidence_indices[j]
                facts_i = cand_facts_by_cid.get(cand_list[i].get("record_id", ""), [])
                facts_j = cand_facts_by_cid.get(cand_list[j].get("record_id", ""), [])

                # Check shared key
                for fi in facts_i:
                    for fj in facts_j:
                        if fi.key == fj.key:
                            dt_i = parse_iso_datetime(fi.timestamp) or datetime.min
                            dt_j = parse_iso_datetime(fj.timestamp) or datetime.min
                            gap = abs((dt_i - dt_j).days)

                            # If different value and one is strictly newer -> SUPERSEDES
                            if fi.value != fj.value and dt_i != dt_j:
                                newer_idx, older_idx = (idx_i, idx_j) if dt_i > dt_j else (idx_j, idx_i)
                                edges.append(
                                    GraphEdge(
                                        src=newer_idx,
                                        dst=older_idx,
                                        edge_type=EDGE_SUPERSEDES,
                                        features=[float(gap)],
                                    )
                                )
                                edges.append(
                                    GraphEdge(
                                        src=older_idx,
                                        dst=newer_idx,
                                        edge_type=EDGE_SUPERSEDED_BY,
                                        features=[float(gap)],
                                    )
                                )
                            elif fi.value == fj.value:
                                # Corroborating assertion
                                edges.append(
                                    GraphEdge(
                                        src=idx_i,
                                        dst=idx_j,
                                        edge_type=EDGE_SAME_KEY,
                                        features=[float(gap)],
                                    )
                                )
                                edges.append(
                                    GraphEdge(
                                        src=idx_j,
                                        dst=idx_i,
                                        edge_type=EDGE_SAME_KEY,
                                        features=[float(gap)],
                                    )
                                )

        return EvidenceGraph(
            nodes=nodes,
            edges=edges,
            claim_node_idx=claim_node_idx,
            evidence_node_indices=evidence_indices,
            concept_node_indices=concept_indices,
        )
