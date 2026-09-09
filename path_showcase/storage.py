"""Bounded output paths and atomic JSON persistence for path_showcase."""

import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil


OUTPUT_ROOT = Path("Polar_data") / "runs"


def relative_path(value):
    path = Path(value)
    if path.is_absolute() or PureWindowsPath(str(value)).drive or ".." in path.parts:
        raise ValueError(f"Use a relative path without parent traversal: {value}")
    return path


def run_dir(run_name):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", run_name):
        raise ValueError("run-name must contain only letters, digits, _ or -")
    path = OUTPUT_ROOT / run_name / "path_showcase"
    resolved_root = Path("Polar_data").resolve()
    if not path.resolve().is_relative_to(resolved_root):
        raise ValueError(f"Unsafe output path: {path}")
    return path


def clean_run(run_name):
    path = run_dir(run_name)
    if not path.exists():
        return
    resolved_root = Path("Polar_data").resolve()
    if not path.resolve().is_relative_to(resolved_root) or path.resolve() == resolved_root:
        raise ValueError(f"Refusing unsafe cleanup: {path}")
    for child in path.rglob("*"):
        if child.is_symlink() or not child.resolve().is_relative_to(resolved_root):
            raise ValueError(f"Refusing cleanup containing unsafe link: {child}")
    shutil.rmtree(path)


def digest(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def read_json(path):
    with relative_path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def atomic_json(path, value):
    path = relative_path(path)
    if not path.resolve().is_relative_to(Path("Polar_data").resolve()):
        raise ValueError(f"Output must be inside ./Polar_data: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def atomic_text(path, text):
    path = relative_path(path)
    if not path.resolve().is_relative_to(Path("Polar_data").resolve()):
        raise ValueError(f"Output must be inside ./Polar_data: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)

