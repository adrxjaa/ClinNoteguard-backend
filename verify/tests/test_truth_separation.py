import ast
from pathlib import Path

def test_truth_separation_enforcement():
    """
    Enforces truth separation:
    Nothing under the verify/ package may import or read truth_timelines.jsonl or truth generator structures.
    The validity engine and graph builder must work only from rendered EHR rows.
    """
    verify_root = Path(__file__).parent.parent
    py_files = list(verify_root.rglob("*.py"))
    assert len(py_files) > 0, "No python files found in verify/"

    forbidden_tokens = ["truth_timelines", "truth_generator", "truth_timelines.jsonl"]

    violations = []
    for p in py_files:
        # Skip this test file itself
        if p.name == "test_truth_separation.py":
            continue

        with open(p, "r", encoding="utf-8") as f:
            code = f.read()

        # 1. Direct string scan
        for token in forbidden_tokens:
            if token in code:
                violations.append(f"{p}: references forbidden token '{token}'")

        # 2. AST parsing to verify imports
        try:
            tree = ast.parse(code, filename=str(p))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if any(tok in alias.name for tok in forbidden_tokens):
                            violations.append(f"{p}: imports '{alias.name}'")
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    if any(tok in mod for tok in forbidden_tokens):
                        violations.append(f"{p}: imports from '{mod}'")
        except Exception as e:
            violations.append(f"{p}: failed to parse AST ({e})")

    assert not violations, "Truth separation violated in verify/:\n" + "\n".join(violations)
