from verify.concepts.hierarchy import ConceptHierarchy
from verify.concepts.linker import CuratedDictionaryLinker
from verify.concepts.attr_type import AttrType, infer_attr_type
from verify.concepts.facts import extract_facts

def test_concept_hierarchy_ancestors():
    hier = ConceptHierarchy(backend="curated")
    ancestors = hier.get_ancestors("amoxicillin", max_hops=2)
    assert "penicillin" in ancestors
    assert "beta_lactam_antibiotic" in ancestors or "antibiotic" in ancestors
    assert hier.is_a("amoxicillin", "penicillin")
    assert hier.is_a("atorvastatin", "statin")

def test_dictionary_linker_layman_and_brands():
    linker = CuratedDictionaryLinker()
    # Test brand to generic
    mentions = linker.link("Patient takes Lipitor 20 mg and Glucophage 500 mg daily.")
    cui_list = [m.cui for m in mentions]
    names = [m.canonical_name for m in mentions]
    assert "atorvastatin" in names
    assert "metformin" in names

    # Test layman terms
    lay_mentions = linker.link("Patient was prescribed a water pill for high blood pressure.")
    lay_names = [m.canonical_name for m in lay_mentions]
    assert "furosemide" in lay_names
    assert "hypertension" in lay_names

def test_fact_extraction_ingredient_normalization():
    # Lipitor 10 mg should normalize key_cui to atorvastatin
    facts = extract_facts(
        record_id="REC_01",
        text="Patient started on Lipitor 10 mg oral tablet daily.",
        source="EHR",
        timestamp="2023-01-15T09:00:00Z",
    )
    assert len(facts) >= 1
    fact = facts[0]
    assert fact.attr_type == AttrType.MEDICATION
    assert "10mg" in fact.value
    assert fact.polarity is True
    assert fact.event_type == "START"
