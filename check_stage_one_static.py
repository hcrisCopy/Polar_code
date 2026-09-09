"""AST/config-only checks. Does not import torch or execute any stage/model."""

import argparse
import ast
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from stage_one.storage import atomic_json, clean_stage, output_path, recover_pending, stage_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    folder = stage_dir(args.run_name, "environment")
    if args.clean:
        clean_stage(args.run_name, "environment")
    recover_pending(folder)
    repository = Path("Polar_code")
    added = sorted((repository / "stage_one").glob("*.py")) + [repository / name for name in (
        "run_stage_one.py", "train_stage_one_predictor.py", "check_stage_one_static.py")]
    report = {"check_type": "static AST and configuration only", "model_executed": False,
              "runtime_imports_tested": False, "files": [], "errors": []}
    trees = {}
    for path in added:
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(path))
            compile(tree, str(path), "exec")  # In memory, no .pyc output.
            trees[path] = tree
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level:
                    target = path.parent / ((node.module or "").replace(".", "/") + ".py")
                    if node.level != 1 or not target.is_file():
                        raise ValueError(f"Unresolved relative import at line {node.lineno}")
                    target_tree = ast.parse(target.read_text(encoding="utf-8"))
                    symbols = {n.name for n in target_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
                    for statement in target_tree.body:
                        if isinstance(statement, ast.Assign):
                            symbols.update(n.id for n in statement.targets if isinstance(n, ast.Name))
                    for symbol in node.names:
                        if symbol.name not in symbols:
                            raise ValueError(f"Unresolved imported symbol {symbol.name} at line {node.lineno}")
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    value = node.value
                    if len(value) > 2 and value[1:3] in {":\\", ":/"}:
                        raise ValueError(f"Hard-coded drive path at line {node.lineno}")
            report["files"].append(str(path))
        except Exception as exc:
            report["errors"].append(f"{path}: {exc}")

    official_tree = ast.parse((repository / "run_polar.py").read_text(encoding="utf-8"))
    official_fields = set()
    for node in ast.walk(official_tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            names = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            if names:
                kw = {k.arg: k.value for k in node.keywords}
                field = ast.literal_eval(kw["dest"]) if "dest" in kw else names[0].removeprefix("--").replace("-", "_")
                official_fields.add(field)
    config = json.loads((repository / "stage_one/predictor_config.json").read_text(encoding="utf-8"))
    if set(config) != official_fields:
        report["errors"].append("Predictor configuration does not enumerate all official parser fields")
    for key in ("data_root", "save_dir", "hf_cache_dir"):
        output_path(config[key])
    report["explicit_predictor_fields"] = len(official_fields)
    # Check the exact direct interfaces reused from the original release.
    interfaces = {"polar/eval.py": ["_online_eval_math_single"],
                  "polar/data.py": ["parse_path_to_seg_and_ops", "PolarDataset", "extract_question_and_gt"],
                  "llm_depth_router/model.py": ["get_model", "get_tokenizer", "_supported_model_key"],
                  "polar/train.py": ["train_polar"]}
    for name, symbols in interfaces.items():
        tree = ast.parse((repository / name).read_text(encoding="utf-8"))
        found = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        if not set(symbols) <= found:
            report["errors"].append(f"Official interface missing: {name}: {set(symbols) - found}")
    report["official_interfaces"] = interfaces
    report["passed"] = not report["errors"]
    atomic_json(folder / "static_report.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
