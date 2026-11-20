"""
Build a Demo SNOMED-CT / RxNorm / LOINC Ontology from Synthea CSVs.

Extracts real SNOMED-CT (conditions, procedures), RxNorm (medications), and
LOINC (observations) concepts from Synthea exports, maps them to hierarchical
clinical parents (IS_A relations), and enriches:
  - verify/concepts/snomed_ct_demo.json (full code-mapped demo ontology)
  - verify/concepts/concept_hierarchy.json (consumed by CuratedDictionaryLinker & GraphBuilder)
"""

import csv
import json
import re
from pathlib import Path
from typing import Dict, List


SYNTHEA_CSV_DIR = Path("/home/statixlo/Develop/Final_year_project/ClinNoteguard-dataset/synthea/output/csv")
CONCEPTS_DIR = Path(__file__).parent.parent / "verify" / "concepts"


def _canon(name: str) -> str:
    # Strip trailing SNOMED semantic tags like "(disorder)", "(finding)", "(situation)"
    s = re.sub(r"\s*\([^)]*\)\s*", " ", name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


# Clinical domain rules mapping canonical keywords -> parent classes
INGREDIENT_PARENTS: Dict[str, List[str]] = {
    # Cardiovascular & Antihypertensives
    "amlodipine": ["calcium_channel_blocker", "antihypertensive"],
    "calcium_channel_blocker": ["antihypertensive", "cardiovascular_agent"],
    "metoprolol": ["beta_blocker", "antihypertensive"],
    "atenolol": ["beta_blocker", "antihypertensive"],
    "carvedilol": ["beta_blocker", "antihypertensive"],
    "beta_blocker": ["antihypertensive", "cardiovascular_agent"],
    "warfarin": ["anticoagulant", "antithrombotic"],
    "apixaban": ["anticoagulant", "antithrombotic"],
    "rivaroxaban": ["anticoagulant", "antithrombotic"],
    "heparin": ["anticoagulant", "antithrombotic"],
    "anticoagulant": ["antithrombotic", "cardiovascular_agent"],
    "digoxin": ["cardiac_glycoside", "cardiovascular_agent"],
    "amiodarone": ["antiarrhythmic", "cardiovascular_agent"],
    # Analgesics & Opioids
    "acetaminophen": ["non_opioid_analgesic", "analgesic"],
    "oxycodone": ["opioid_analgesic", "analgesic"],
    "hydrocodone": ["opioid_analgesic", "analgesic"],
    "tramadol": ["opioid_analgesic", "analgesic"],
    "morphine": ["opioid_analgesic", "analgesic"],
    "opioid_analgesic": ["analgesic"],
    "non_opioid_analgesic": ["analgesic"],
    # Endocrine / Diabetes / Bone / Renal
    "insulin": ["insulin_therapy", "antidiabetic_agent"],
    "insulin_therapy": ["antidiabetic_agent"],
    "alendronic_acid": ["bisphosphonate", "bone_metabolism_agent"],
    "bisphosphonate": ["bone_metabolism_agent"],
    "epoetin_alfa": ["erythropoiesis_stimulating_agent", "hematologic_agent"],
    "sodium_fluoride": ["dental_caries_preventive", "dental_agent"],
    "methotrexate": ["dmard", "immunosuppressive_agent"],
    "lithium_carbonate": ["mood_stabilizer", "psychiatric_agent"],
    # Conditions (SNOMED-CT)
    "viral_sinusitis": ["upper_respiratory_infection", "respiratory_disease"],
    "acute_viral_pharyngitis": ["upper_respiratory_infection", "respiratory_disease"],
    "acute_bronchitis": ["lower_respiratory_infection", "respiratory_disease"],
    "pneumonia": ["lower_respiratory_infection", "respiratory_disease"],
    "upper_respiratory_infection": ["acute_respiratory_disorder", "respiratory_disease"],
    "lower_respiratory_infection": ["acute_respiratory_disorder", "respiratory_disease"],
    "acute_respiratory_disorder": ["respiratory_disease"],
    "gingivitis": ["periodontal_disease", "dental_disorder"],
    "gingival_disease": ["periodontal_disease", "dental_disorder"],
    "primary_dental_caries": ["dental_disorder"],
    "periodontal_disease": ["dental_disorder"],
    "prediabetes": ["glycemic_disorder", "metabolic_disease"],
    "hyperlipidemia": ["lipid_disorder", "metabolic_disease"],
    "obesity": ["metabolic_disease"],
    "anemia": ["hematologic_disorder"],
    "coronary_heart_disease": ["cardiovascular_disease", "heart_disease"],
    "myocardial_infarction": ["cardiovascular_disease", "heart_disease"],
    "stroke": ["cerebrovascular_disease", "cardiovascular_disease"],
    "osteoarthritis": ["musculoskeletal_disorder"],
    "sprain": ["musculoskeletal_injury", "musculoskeletal_disorder"],
    "stress": ["psychosocial_finding"],
    "social_isolation": ["psychosocial_finding"],
    "limited_social_contact": ["psychosocial_finding"],
    "full_time_employment": ["employment_finding"],
    "part_time_employment": ["employment_finding"],
    "medication_review_due": ["clinical_administration"],
    # Distractor conditions
    "systemic_lupus_erythematosus": ["autoimmune_disease"],
    "rheumatoid_arthritis": ["autoimmune_disease", "musculoskeletal_disorder"],
    "pulmonary_embolism": ["vascular_disorder", "respiratory_disease"],
    "acute_pancreatitis": ["gastrointestinal_disorder"],
    "ulcerative_colitis": ["gastrointestinal_disorder"],
    "cirrhosis_of_liver": ["hepatic_disorder", "gastrointestinal_disorder"],
    # Labs & Vitals (LOINC)
    "systolic_blood_pressure": ["blood_pressure", "vital_sign"],
    "diastolic_blood_pressure": ["blood_pressure", "vital_sign"],
    "blood_pressure": ["vital_sign", "clinical_measurement"],
    "heart_rate": ["vital_sign", "clinical_measurement"],
    "respiratory_rate": ["vital_sign", "clinical_measurement"],
    "body_weight": ["anthropometry", "vital_sign"],
    "body_height": ["anthropometry", "vital_sign"],
    "body_mass_index": ["anthropometry", "clinical_measurement"],
    "pain_severity": ["symptom_score", "clinical_measurement"],
    "hemoglobin_a1c": ["glycemic_lab", "lab_test"],
    "glucose": ["glycemic_lab", "lab_test"],
    "creatinine": ["renal_panel", "lab_test"],
    "urea_nitrogen": ["renal_panel", "lab_test"],
    "potassium": ["electrolyte_panel", "lab_test"],
    "sodium": ["electrolyte_panel", "lab_test"],
    "calcium": ["electrolyte_panel", "lab_test"],
    "total_cholesterol": ["lipid_panel", "lab_test"],
    "ldl_cholesterol": ["lipid_panel", "lab_test"],
    "hdl_cholesterol": ["lipid_panel", "lab_test"],
    "triglycerides": ["lipid_panel", "lab_test"],
    "renal_panel": ["lab_test"],
    "electrolyte_panel": ["lab_test"],
    "glycemic_lab": ["lab_test"],
    "lipid_panel": ["lab_test"],
}


def build_ontology():
    hier_path = CONCEPTS_DIR / "concept_hierarchy.json"
    with open(hier_path, "r", encoding="utf-8") as f:
        hier = json.load(f)

    parents: Dict[str, List[str]] = dict(hier.get("parents", {}))
    cui_map: Dict[str, str] = dict(hier.get("cui_map", {}))
    snomed_codes: Dict[str, Dict[str, str]] = {}

    # 1. Merge curated ingredient/condition/lab parents
    for k, p_list in INGREDIENT_PARENTS.items():
        existing = parents.get(k, [])
        merged = list(dict.fromkeys(existing + p_list))
        parents[k] = merged
        if k not in cui_map:
            cui_map[k] = f"SYN_{abs(hash(k)) % 1000000:06d}"
        for p in merged:
            if p not in cui_map:
                cui_map[p] = f"SYN_{abs(hash(p)) % 1000000:06d}"

    # 2. Extract Synthea SNOMED-CT, RxNorm, LOINC codes from CSVs
    if SYNTHEA_CSV_DIR.exists():
        sources = [
            ("conditions.csv", "SNOMED-CT", "condition"),
            ("procedures.csv", "SNOMED-CT", "procedure"),
            ("medications.csv", "RxNorm", "medication"),
            ("observations.csv", "LOINC", "observation"),
        ]
        for fname, system, kind in sources:
            fpath = SYNTHEA_CSV_DIR / fname
            if not fpath.exists():
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    code = (row.get("CODE") or "").strip()
                    desc = (row.get("DESCRIPTION") or "").strip()
                    if not code or not desc:
                        continue
                    canon = _canon(desc)
                    if not canon or len(canon) > 64:
                        continue
                    snomed_codes[canon] = {"code": code, "system": system, "kind": kind, "label": desc}
                    if canon not in cui_map:
                        cui_map[canon] = f"{system}_{code}"
                    # Link multi-word Synthea drug/lab descriptions to ingredient/analyte parents
                    inferred_parents = list(parents.get(canon, []))
                    for ing in INGREDIENT_PARENTS:
                        if ing in canon and ing != canon:
                            inferred_parents.append(ing)
                    if not inferred_parents:
                        inferred_parents.append(f"synthea_{kind}")
                    parents[canon] = list(dict.fromkeys(inferred_parents))
                    if f"synthea_{kind}" not in cui_map:
                        cui_map[f"synthea_{kind}"] = f"SYN_CAT_{kind.upper()}"

    # Write snomed_ct_demo.json
    demo_out = CONCEPTS_DIR / "snomed_ct_demo.json"
    with open(demo_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "description": "Demo SNOMED-CT, RxNorm, and LOINC Ontology derived from Synthea CSVs",
                "total_concepts": len(cui_map),
                "coded_terms": len(snomed_codes),
                "codes": snomed_codes,
            },
            f,
            indent=2,
        )

    # Update concept_hierarchy.json
    with open(hier_path, "w", encoding="utf-8") as f:
        json.dump({"parents": parents, "cui_map": cui_map}, f, indent=2)

    print(f"[+] Built Demo SNOMED-CT/RxNorm/LOINC Ontology: {len(snomed_codes)} coded Synthea terms, "
          f"{len(cui_map)} total concepts, {len(parents)} IS_A parent rules.")


if __name__ == "__main__":
    build_ontology()
