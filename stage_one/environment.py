"""Offline environment checks; never instantiate a model or touch a GPU."""

import importlib
import importlib.metadata
import platform
from pathlib import Path

from .model_runner import model_inventory
from .storage import atomic_json, clean_stage, recover_pending, stage_dir


def check_environment(args):
    folder = stage_dir(args.run_name, "environment")
    if args.clean:
        clean_stage(args.run_name, "environment")
    recover_pending(folder)
    problems, versions = [], {}
    requirements = []
    for requirement_file in (Path("Polar_code/requirements.txt"),
                             Path("Polar_code/stage_one/requirements.txt")):
        requirements.extend(requirement_file.read_text(encoding="utf-8").splitlines())
    for spec in requirements:
        if "==" not in spec:
            continue
        name, expected = spec.split("==", 1)
        try:
            actual = importlib.metadata.version(name)
            versions[name] = actual
            if actual != expected:
                problems.append(f"{name}: expected {expected}, found {actual}")
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"Missing dependency: {spec}")
    if platform.system() != "Linux":
        problems.append("Execution stages require Linux (fcntl locks and torchrun)")
    if not problems:
        # Imports do not call from_pretrained or torch.cuda.*. Loading a predictor
        # embedding model is deliberately not part of this check.
        for module in ("polar.data", "polar.eval", "llm_depth_router.model", "dart_math.eval"):
            try:
                importlib.import_module(module)
            except Exception as exc:
                problems.append(f"Import failed: {module}: {exc}")
    try:
        from .prepare import input_files
        inventory = model_inventory(args.model_path)
        files = [str(path) for path in input_files(args.data_path)]
    except Exception as exc:
        inventory, files = [], []
        problems.append(str(exc))
    report = {"passed": not problems, "versions": versions, "problems": problems,
              "model_path": args.model_path, "model_files": inventory, "data_files": files,
              "model_executed": False, "gpu_checked": False}
    atomic_json(folder / "report.json", report)
    if problems:
        raise RuntimeError("Environment not ready:\n" + "\n".join(problems))
    print(f"Offline environment/import checks passed: {folder / 'report.json'}")


def configure_runtime(run_name):
    """Keep implicit Python/HF/Torch/temp artifacts in Polar_data as well."""
    import os
    import sys
    import tempfile
    from .storage import ROOT, output_path
    stage_dir(run_name, "environment")
    sys.dont_write_bytecode = True
    cache = output_path(ROOT / "cache")
    temporary = output_path(ROOT / "runtime" / run_name / f"rank_{os.environ.get('RANK', '0')}")
    temporary.mkdir(parents=True, exist_ok=True)
    for name, value in {
        "HF_HOME": cache / "huggingface", "HF_HUB_CACHE": cache / "huggingface" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache / "huggingface" / "hub",
        "HF_DATASETS_CACHE": cache / "datasets", "HF_MODULES_CACHE": cache / "modules",
        "TORCH_HOME": cache / "torch", "TRITON_CACHE_DIR": cache / "triton",
        "TORCHINDUCTOR_CACHE_DIR": cache / "torchinductor", "XDG_CACHE_HOME": cache,
        "TMPDIR": temporary, "TMP": temporary, "TEMP": temporary,
    }.items():
        os.environ[name] = str(output_path(value))
    os.environ.pop("TRANSFORMERS_CACHE", None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tempfile.tempdir = str(temporary)
