"""Strict JSON, atomic persistence, and bounded stage directories."""

import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import time
from contextlib import contextmanager

ROOT = Path("Polar_data")


def relative_path(value):
    path = Path(value)
    if path.is_absolute() or PureWindowsPath(str(value)).drive or ".." in path.parts:
        raise ValueError(f"Use a relative path without parent traversal: {value}")
    return path


def output_path(value):
    path = relative_path(value)
    if not path.parts or path.parts[0] != ROOT.name:
        raise ValueError(f"Output must be inside ./Polar_data: {value}")
    # Reject links, even links targeting another stage within Polar_data.
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"Output symlink is forbidden: {part}")
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT.resolve()) or resolved == ROOT.resolve():
        raise ValueError(f"Unsafe output: {value}")
    return path


def stage_dir(run, stage):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", run):
        raise ValueError("run-name must contain only letters, digits, _ or -")
    if stage not in {"environment", "prepared", "search", "merged", "validation", "predictor",
                     "program_mining", "universal_eval", "program_report",
                     "representations", "representation_report"}:
        raise ValueError(f"Unknown stage: {stage}")
    return output_path(ROOT / "runs" / run / stage)


def clean_stage(run, stage):
    path = stage_dir(run, stage)
    if path.exists():
        # Check the entire subtree before recursive removal.
        for child in path.rglob("*"):
            output_path(child)
        shutil.rmtree(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with relative_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    try:
        with relative_path(path).open(encoding="utf-8") as stream:
            return json.load(stream, object_pairs_hook=_pairs, parse_constant=invalid)
    except Exception as exc:
        raise ValueError(f"Cannot read complete JSON {path}: {exc}") from exc


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n")


def atomic_text(path, value):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = output_path(path.with_suffix(path.suffix + ".pending"))
    if pending.exists():
        raise ValueError(f"Unrecovered interrupted write: {pending}")
    with pending.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    if os.name == "posix":
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def recover_pending(directory):
    """Preserve interrupted writes as evidence; never treat them as completed data."""
    directory = output_path(directory)
    recovered = []
    if not directory.exists():
        return recovered
    for path in sorted(directory.rglob("*.pending")):
        if "recovery" in path.relative_to(directory).parts:
            continue
        output_path(path)
        try:
            read_json(path)
            reason = "complete JSON but transaction not committed"
        except ValueError:
            reason = "incomplete or corrupt JSON"
        target = output_path(directory / "recovery" / f"{time.time_ns()}-{path.name}.quarantined")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, target)
        recovered.append({"file": str(path), "preserved_at": str(target), "reason": reason})
        print(f"Interrupted write detected: {path}; {reason}; preserved at {target}")
    return recovered


@contextmanager
def run_lock(run):
    """Linux advisory lock auto-releases after crashes; lock file is never deleted."""
    import fcntl
    path = output_path(ROOT / "locks" / f"{run}.lock")
    stage_dir(run, "search")  # Validate run-name before using it.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another stage is using run {run}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
