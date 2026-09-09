"""One full model per rank, deterministic question shards, crash-safe records."""

from datetime import timedelta
import contextlib
import json
import importlib.metadata
import os
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

from tqdm import tqdm

from .model_runner import ModelRunner, model_inventory
from .prepare import check_manifest
from .search_tree import search
from .storage import (atomic_json, clean_stage, digest, file_digest, output_path,
                      read_json, recover_pending, run_lock, stage_dir)
from .validate import summarize, verify_record


def build_config(args, manifest, world):
    from polar.config import infer_original_depth
    options = {key: getattr(args, key) for key in (
        "model_id", "model_path", "model_revision", "seed", "simulations", "exploration",
        "length_penalty", "max_block", "max_repeats", "max_length_factor",
        "max_new_tokens", "temperature")}
    depth = infer_original_depth(args.model_id)
    local_config = read_json(Path(args.model_path) / "config.json")
    if local_config.get("quantization_config"):
        raise ValueError("Quantized snapshots are outside this stage's full-model protocol")
    from llm_depth_router.model import _supported_model_key
    if _supported_model_key(args.model_id) != _supported_model_key(args.model_path):
        raise ValueError("model-id and local model snapshot name disagree")
    if local_config.get("num_hidden_layers") != depth:
        raise ValueError("Local model config does not match model-id depth")
    # Hash weight/config/tokenizer bytes without deserializing any tensors.
    source_files = sorted(Path("Polar_code").rglob("*.py"))
    config = {"schema_version": 1, "args": options, "world_size": world,
              "depth": depth, "max_length": int(depth * args.max_length_factor),
              "manifest_id": manifest["manifest_id"],
              "model_files": model_inventory(args.model_path),
              "code_sha256": {str(p): file_digest(p) for p in source_files}}
    config["dependency_versions"] = {name: importlib.metadata.version(name) for name in
                                     ("torch", "transformers", "numpy", "sympy", "Pebble", "tqdm")}
    if config["dependency_versions"]["transformers"] != "4.52.4":
        raise ValueError("Execution cache adapter targets the repository's transformers==4.52.4")
    config["config_id"] = digest(config)
    return config


def make_record(row, config):
    return {**row, "config_id": config["config_id"], "status": "failed",
            "initial_score": None, "final_valid_transitions": [], "final_invalid_transitions": [],
            "evaluations": [], "search_statistics": {}, "failure_reason": None}


def finish_record(record):
    record.pop("record_digest", None)
    record["record_digest"] = digest(record)
    return record


def process_question(row, args, config, runner, rank):
    from polar.data import parse_path_to_seg_and_ops
    record = make_record(row, config)
    start = time.monotonic()

    def remember(path, score):
        record["evaluations"].append({"path": list(path), "score": score})
        if path == tuple(range(config["depth"])):
            record["initial_score"] = score
        if parse_path_to_seg_and_ops(list(path), config["depth"], max_pack=4, allow_repeat=True) is not None:
            label = "final_valid_transitions" if score == 1 else "final_invalid_transitions"
            record[label].append(list(path))

    try:
        if row["split"] == "test":
            # Held-out test labels never drive tree expansion. Baseline is recorded
            # only for later reporting, and is outside predictor training indices.
            path = tuple(range(config["depth"]))
            remember(path, runner.score(row, list(path)))
            stats = {"simulations_completed": 0, "unique_evaluations": 1, "cache_hits": 0}
        else:
            sample_seed = int(digest([args.seed, row["sample_id"]])[:16], 16)
            stats = search(lambda p: runner.score(row, p), depth=config["depth"],
                           simulations=args.simulations, exploration=args.exploration,
                           length_penalty=args.length_penalty, max_block=args.max_block,
                           max_repeats=args.max_repeats, max_length=config["max_length"],
                           seed=sample_seed, rank=rank, on_evaluation=remember)
        record["search_statistics"] = stats
        record["status"] = "complete"
        if not record["final_valid_transitions"]:
            record["failure_reason"] = "no_predictor_representable_valid_path" if row["split"] != "test" else "baseline_incorrect"
    except Exception as exc:
        record["failure_reason"] = f"{type(exc).__name__}: {exc}"
        record["search_statistics"] = {"unique_evaluations": len(record["evaluations"]),
                                       "simulations_completed": None}
        # Persist known path scores; an infrastructure error is never a negative label.
        record["search_statistics"]["elapsed_seconds"] = time.monotonic() - start
        return finish_record(record), exc
    record["search_statistics"]["elapsed_seconds"] = time.monotonic() - start
    return finish_record(record), None


def run_rank(args, manifest, config, token, rank, local_rank):
    folder = output_path(stage_dir(args.run_name, "search") / f"rank_{rank:05d}")
    folder.mkdir(parents=True, exist_ok=True)
    recovered = recover_pending(folder)
    rows = [r for r in manifest["samples"] if r["global_index"] % config["world_size"] == rank]
    row_by_id = {r["sample_id"]: r for r in rows}
    existing = {}
    # Validate every persisted file, not just files that happen to be selected today.
    for path in tqdm(sorted((folder / "records").glob("*.json")), desc=f"rank {rank} resume scan",
                     position=2 * rank, leave=False):
        record = read_json(path)
        row = row_by_id.get(path.stem)
        if row is None:
            raise ValueError(f"Unexpected record in shard: {path}")
        verify_record(record, row, config)
        existing[path.stem] = record
    completed = {sid for sid, r in existing.items() if r["status"] == "complete"}
    retrying = set(existing) - completed
    summary = {"rank": rank, "world_size": config["world_size"], "invocation": token,
               "assigned": len(rows), "resumed_complete": len(completed),
               "retrying_failed": len(retrying), "interrupted_writes": recovered,
               "newly_completed": 0, "state": "running"}
    atomic_json(folder / "summary.json", summary)
    pending = [r for r in rows if r["sample_id"] not in completed]
    runner = None
    try:
        if pending:
            log_path = output_path(folder / "official_evaluator.log")
            with log_path.open("a", encoding="utf-8", buffering=1) as stream:
                with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                    runner = ModelRunner(args, local_rank, stream)
                if runner.depth != config["depth"]:
                    raise ValueError("Loaded model depth mismatch")
                atomic_json(folder / "model_runtime.json", runner.metadata())
                progress = tqdm(rows, desc=f"rank {rank} questions", position=2 * rank,
                                mininterval=2, leave=True)
                for row in progress:
                    sid = row["sample_id"]
                    if sid in completed:
                        continue
                    record, error = process_question(row, args, config, runner, rank)
                    verify_record(record, row, config)
                    atomic_json(folder / "records" / f"{sid}.json", record)
                    existing[sid] = record
                    if error is not None:
                        raise error
                    summary["newly_completed"] += 1
                    atomic_json(folder / "summary.json", summary)
        summary["state"] = "complete"
        summary["results"] = summarize(list(existing.values()), len(rows))
        atomic_json(folder / "summary.json", summary)
        atomic_json(folder / "done.json", {"invocation": token, "config_id": config["config_id"],
                                          "rank": rank, "count": len(rows)})
    except Exception as exc:
        summary["state"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["results"] = summarize(list(existing.values()), len(rows))
        atomic_json(folder / "summary.json", summary)
        atomic_json(folder / "error.json", {"invocation": token, "error": summary["error"]})
        raise
    finally:
        if runner is not None:
            runner.hook.remove()
            del runner


def wait_for_ranks(args, config, token):
    """No end barrier: empty/resumed ranks also write an invocation-bound done file.

    Torchrun terminates siblings when a worker fails. A configurable deadline is
    also enforced for missing completion markers. Old markers never count.
    """
    folder = stage_dir(args.run_name, "search")
    deadline = time.monotonic() + args.completion_timeout
    remaining = set(range(config["world_size"]))
    with tqdm(total=len(remaining), desc="rank 0 completed shards", mininterval=2) as progress:
        while remaining:
            for rank in sorted(remaining):
                rank_dir = folder / f"rank_{rank:05d}"
                error_path = rank_dir / "error.json"
                if error_path.exists():
                    error = read_json(error_path)
                    if error.get("invocation") == token:
                        raise RuntimeError(f"rank {rank} failed: {error['error']}")
                done_path = rank_dir / "done.json"
                if done_path.exists():
                    done = read_json(done_path)
                    if done.get("invocation") == token and done.get("config_id") == config["config_id"]:
                        remaining.remove(rank)
                        progress.update(1)
            if remaining:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for ranks {sorted(remaining)}; inspect shard logs")
                time.sleep(1)


def distributed_search(args):
    import torch.distributed as dist
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if "RANK" not in os.environ:
        raise ValueError("Use torchrun --standalone --nproc_per_node=1 (or 8)")
    if int(os.environ.get("LOCAL_WORLD_SIZE", world)) != world:
        raise ValueError("Only single-node, one-process-per-GPU execution is supported")
    # This short-lived CPU group coordinates startup only; never wraps a model.
    dist.init_process_group("gloo", timeout=timedelta(seconds=args.completion_timeout))
    lock = None
    packet = [None]
    try:
        if rank == 0:
            try:
                candidate_lock = run_lock(args.run_name)
                candidate_lock.__enter__()
                lock = candidate_lock
                manifest = check_manifest(read_json(stage_dir(args.run_name, "prepared") / "manifest.json"))
                config = build_config(args, manifest, world)
                folder = stage_dir(args.run_name, "search")
                if args.clean:
                    clean_stage(args.run_name, "search")
                # No workers are writing yet: preserve all interrupted writes
                # before broadcasting permission to resume their disjoint shards.
                recovered = recover_pending(folder)
                path = folder / "config.json"
                if path.exists():
                    if read_json(path) != config:
                        raise ValueError("Search inputs/config/code/world_size changed; use a new run-name or search --clean")
                else:
                    if any(folder.glob("rank_*")):
                        raise ValueError("Orphan shards without a config; restore config or use search --clean")
                    atomic_json(path, config)
                if recovered:
                    atomic_json(folder / "recovery_report.json", recovered)
                    print(f"Detected and preserved {len(recovered)} interrupted writes; incomplete questions will restart")
                packet[0] = {"ok": True, "invocation": uuid.uuid4().hex, "config_id": config["config_id"]}
            except Exception as exc:
                packet[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        dist.broadcast_object_list(packet, src=0)
        dist.destroy_process_group()
        if not packet[0]["ok"]:
            raise RuntimeError(packet[0]["error"])
        manifest = check_manifest(read_json(stage_dir(args.run_name, "prepared") / "manifest.json"))
        config = read_json(stage_dir(args.run_name, "search") / "config.json")
        if config["config_id"] != packet[0]["config_id"]:
            raise ValueError("Configuration changed during distributed startup")
        token = packet[0]["invocation"]
        run_rank(args, manifest, config, token, rank, local_rank)
        if rank == 0:
            wait_for_ranks(args, config, token)
            from .merge import merge
            from .validate import validate
            post = SimpleNamespace(run_name=args.run_name, clean=False)
            summary = merge(post)
            validate(post)
            print(json.dumps(summary, ensure_ascii=True, indent=2))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if lock is not None:
            lock.__exit__(None, None, None)
