"""Distributed held-out evaluation of frozen universal layer programs."""

from datetime import timedelta
import contextlib
import importlib.metadata
import os
import time
import uuid

from tqdm import tqdm

from .model_runner import ModelRunner
from .prepare import check_manifest
from .storage import (atomic_json, clean_stage, digest, output_path,
                      read_json, recover_pending, relative_path, run_lock, stage_dir)
from .validate import load_search


def _validated_candidates(payload):
    # Lazy import avoids importing plotting/report code at module startup.
    from .program_analysis import validate_candidate_payload
    return validate_candidate_payload(payload)


def load_evaluation_inputs(run_name, candidate_run_name=None):
    candidate_run_name = candidate_run_name or run_name
    if candidate_run_name == run_name:
        manifest, search_config, search_records = load_search(run_name)
    else:
        manifest = check_manifest(read_json(
            stage_dir(run_name, "prepared") / "manifest.json"))
        _, search_config, search_records = load_search(candidate_run_name)
    candidates = read_json(
        stage_dir(candidate_run_name, "program_mining") / "candidates.json")
    return manifest, search_config, search_records, _validated_candidates(candidates)


def build_config(args, manifest, search_config, search_records, candidates,
                 candidate_run_name, world_size):
    _validated_candidates(candidates)
    if candidates["search_config_id"] != search_config["config_id"]:
        raise ValueError("Candidate set belongs to another MCTS search")
    if (candidates["depth"] != search_config["depth"] or
            candidates["max_length"] != search_config["max_length"] or
            candidates["run_name"] != candidate_run_name):
        raise ValueError("Candidate set source search, depth, or run does not match")
    if candidate_run_name == args.run_name and candidates["manifest_id"] != manifest["manifest_id"]:
        raise ValueError("Candidate set source manifest does not match")
    evaluated_ids = {row["sample_id"] for row in manifest["samples"]
                     if row["split"] in args.evaluation_splits}
    discovery_ids = {row["sample_id"] for row in search_records if row["split"] == "train"}
    if candidate_run_name != args.run_name and evaluated_ids & discovery_ids:
        raise ValueError("Frozen-program evaluation overlaps candidate-discovery train questions")
    expected = search_config["args"]
    requested_model_path = relative_path(args.model_path)
    search_model_path = relative_path(expected["model_path"])
    if not (requested_model_path / "config.json").is_file():
        if (search_model_path / "config.json").is_file():
            print(f"Model path {requested_model_path} is unavailable; using search path {search_model_path}")
            args.model_path = str(search_model_path)
        else:
            raise ValueError(
                f"No local config.json at {requested_model_path} or recorded search path "
                f"{search_model_path}")
    comparable = ("model_id", "seed", "max_new_tokens", "temperature")
    for key in comparable:
        actual_value = getattr(args, key)
        expected_value = expected[key]
        if actual_value != expected_value:
            raise ValueError(
                f"Universal evaluation {key} must match the MCTS search: "
                f"got {actual_value!r}, expected {expected_value!r}")
    if len(candidates["candidates"]) > args.max_eval_candidates:
        raise ValueError("Candidate count exceeds --max-eval-candidates")
    split_counts = {
        split: sum(row["split"] == split for row in manifest["samples"])
        for split in args.evaluation_splits
    }
    if any(count == 0 for count in split_counts.values()):
        raise ValueError("Universal evaluation requires nonempty validation and test splits")
    split_difficulty_counts = {
        split: {str(difficulty): sum(
            row["split"] == split and row["difficulty"] == difficulty
            for row in manifest["samples"])
            for difficulty in manifest["args"]["difficulties"]}
        for split in args.evaluation_splits
    }
    options = {key: getattr(args, key) for key in
               (*comparable, "evaluation_splits", "max_eval_candidates")}
    options["model_path"] = str(relative_path(args.model_path))
    options["search_model_path"] = str(relative_path(expected["model_path"]))
    options["model_revision"] = args.model_revision
    options["search_model_revision"] = expected["model_revision"]
    config = {
        "schema_version": 1,
        "run_name": args.run_name,
        "candidate_run_name": candidate_run_name,
        "args": options,
        "world_size": world_size,
        "depth": search_config["depth"],
        "difficulties": manifest["args"]["difficulties"],
        "split_counts": split_counts,
        "split_difficulty_counts": split_difficulty_counts,
        "manifest_id": manifest["manifest_id"],
        "candidate_manifest_id": candidates["manifest_id"],
        "search_config_id": search_config["config_id"],
        "candidate_set_id": candidates["candidate_set_id"],
        "search_model_inventory_id": digest(search_config.get("model_files", [])),
        "dependency_versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "numpy", "sympy", "Pebble", "tqdm")
        },
    }
    config["evaluation_config_id"] = digest(config)
    return config


def finish_record(record):
    record.pop("record_digest", None)
    record["record_digest"] = digest(record)
    return record


def verify_record(record, row, config, candidate_ids):
    payload = {key: value for key, value in record.items() if key != "record_digest"}
    if digest(payload) != record.get("record_digest"):
        raise ValueError(f"Universal record checksum mismatch: {row['sample_id']}")
    for key in ("sample_id", "question", "gt_ans", "difficulty", "split", "global_index"):
        if record.get(key) != row[key]:
            raise ValueError(f"Universal record source mismatch: {row['sample_id']}")
    if record.get("evaluation_config_id") != config["evaluation_config_id"]:
        raise ValueError("Mixed universal evaluation configurations")
    if record.get("status") not in {"complete", "failed"}:
        raise ValueError("Unknown universal evaluation status")
    if (record.get("baseline_score") not in (0, 1) and
            not (record.get("status") == "failed" and record.get("baseline_score") is None)):
        raise ValueError("Universal baseline score must be binary")
    scores = record.get("scores")
    if not isinstance(scores, dict) or set(scores) - set(candidate_ids):
        raise ValueError("Unknown candidate score in universal evaluation")
    if any(type(value) not in (int, float) or value not in (0, 1)
           for value in scores.values()):
        raise ValueError("Universal candidate scores must be binary")
    if record["status"] == "complete" and set(scores) != set(candidate_ids):
        raise ValueError("Completed universal record lacks candidate scores")
    if record["status"] == "failed" and not record.get("failure_reason"):
        raise ValueError("Failed universal record requires failure_reason")


def evaluate_question(row, baseline_score, candidates, runner, config):
    keys = ("sample_id", "question", "gt_ans", "difficulty", "split", "global_index")
    record = {key: row[key] for key in keys}
    record.update({
        "evaluation_config_id": config["evaluation_config_id"],
        "baseline_score": baseline_score,
        "scores": {},
        "status": "failed",
        "failure_reason": None,
    })
    started = time.monotonic()
    try:
        if baseline_score is None:
            baseline_score = runner.score(row, list(range(config["depth"])))
            record["baseline_score"] = baseline_score
        for candidate in tqdm(candidates,
                              desc=f"fixed programs {row['sample_id'][:8]}", leave=False):
            score = runner.score(row, candidate["path"])
            record["scores"][candidate["candidate_id"]] = score
        record["status"] = "complete"
    except Exception as exc:
        record["failure_reason"] = f"{type(exc).__name__}: {exc}"
        record["elapsed_seconds"] = time.monotonic() - started
        return finish_record(record), exc
    record["elapsed_seconds"] = time.monotonic() - started
    return finish_record(record), None


def run_rank(args, rows, search_records, candidates, config, token, rank, local_rank):
    folder = output_path(stage_dir(args.run_name, "universal_eval") / f"rank_{rank:05d}")
    folder.mkdir(parents=True, exist_ok=True)
    recovered = recover_pending(folder)
    assigned = [row for row in rows
                if row["global_index"] % config["world_size"] == rank]
    by_id = {row["sample_id"]: row for row in assigned}
    search_by_id = {row["sample_id"]: row for row in search_records}
    candidate_ids = [row["candidate_id"] for row in candidates]
    records_dir = folder / "records"
    existing = {}
    for path in tqdm(sorted(records_dir.glob("*.json")),
                     desc=f"rank {rank} universal resume", position=rank, leave=False):
        record = read_json(path)
        row = by_id.get(path.stem)
        if row is None:
            raise ValueError(f"Unexpected universal record: {path}")
        verify_record(record, row, config, candidate_ids)
        existing[path.stem] = record
    completed = {sid for sid, record in existing.items()
                 if record["status"] == "complete"}
    summary = {
        "rank": rank,
        "world_size": config["world_size"],
        "invocation": token,
        "assigned": len(assigned),
        "resumed_complete": len(completed),
        "retrying_failed": len(set(existing) - completed),
        "interrupted_writes": recovered,
        "newly_completed": 0,
        "state": "running",
    }
    atomic_json(folder / "summary.json", summary)
    pending = [row for row in assigned if row["sample_id"] not in completed]
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
                for row in tqdm(assigned, desc=f"rank {rank} held-out questions",
                                position=rank, leave=True):
                    sid = row["sample_id"]
                    if sid in completed:
                        continue
                    result, error = evaluate_question(
                        row, search_by_id[sid]["initial_score"] if sid in search_by_id else None,
                        candidates, runner, config)
                    verify_record(result, row, config, candidate_ids)
                    atomic_json(records_dir / f"{sid}.json", result)
                    existing[sid] = result
                    if error is not None:
                        raise error
                    summary["newly_completed"] += 1
                    atomic_json(folder / "summary.json", summary)
        summary["state"] = "complete"
        summary["completed"] = len(existing)
        atomic_json(folder / "summary.json", summary)
        atomic_json(folder / "done.json", {
            "invocation": token,
            "evaluation_config_id": config["evaluation_config_id"],
            "rank": rank,
            "count": len(assigned),
        })
    except Exception as exc:
        summary["state"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        atomic_json(folder / "summary.json", summary)
        atomic_json(folder / "error.json", {"invocation": token, "error": summary["error"]})
        raise
    finally:
        if runner is not None:
            runner.hook.remove()
            del runner


def wait_for_ranks(args, config, token):
    folder = stage_dir(args.run_name, "universal_eval")
    remaining = set(range(config["world_size"]))
    deadline = time.monotonic() + args.completion_timeout
    with tqdm(total=len(remaining), desc="rank 0 universal shards") as progress:
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
                    if (done.get("invocation") == token and
                            done.get("evaluation_config_id") == config["evaluation_config_id"]):
                        remaining.remove(rank)
                        progress.update(1)
            if remaining:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for ranks {sorted(remaining)}")
                time.sleep(1)


def load_universal_evaluation(run_name, require_complete=True):
    folder = stage_dir(run_name, "universal_eval")
    config = read_json(folder / "config.json")
    payload = {key: value for key, value in config.items()
               if key != "evaluation_config_id"}
    if digest(payload) != config.get("evaluation_config_id"):
        raise ValueError("Universal evaluation configuration checksum mismatch")
    candidate_run_name = config.get("candidate_run_name", run_name)
    manifest, _, _, candidates = load_evaluation_inputs(run_name, candidate_run_name)
    if (config.get("run_name", run_name) != run_name or
            config.get("manifest_id") != manifest["manifest_id"] or
            config.get("candidate_manifest_id", candidates["manifest_id"]) != candidates["manifest_id"]):
        raise ValueError("Universal evaluation source manifest changed")
    if candidates["candidate_set_id"] != config["candidate_set_id"]:
        raise ValueError("Universal results use another candidate set")
    candidate_ids = [row["candidate_id"] for row in candidates["candidates"]]
    source = {row["sample_id"]: row for row in manifest["samples"]
               if row["split"] in config["args"]["evaluation_splits"]}
    expected_dirs = {f"rank_{rank:05d}" for rank in range(config["world_size"])}
    actual_dirs = {path.name for path in folder.glob("rank_*") if path.is_dir()}
    if actual_dirs - expected_dirs or (require_complete and actual_dirs != expected_dirs):
        raise ValueError("Missing or unexpected universal rank directories")
    pending = [path for path in folder.rglob("*.pending") if "recovery" not in path.parts]
    if pending:
        raise ValueError(f"Interrupted universal write present: {pending[0]}")
    records, seen = [], set()
    for rank in tqdm(range(config["world_size"]), desc="Validate universal shards"):
        rank_dir = folder / f"rank_{rank:05d}"
        for path in sorted((rank_dir / "records").glob("*.json")):
            record = read_json(path)
            sid = record.get("sample_id")
            if (sid not in source or sid in seen or path.stem != sid or
                    source[sid]["global_index"] % config["world_size"] != rank):
                raise ValueError(f"Unknown, duplicate, or misplaced universal record: {path}")
            verify_record(record, source[sid], config, candidate_ids)
            records.append(record)
            seen.add(sid)
    failed = sum(row["status"] == "failed" for row in records)
    missing = set(source) - seen
    if require_complete and (missing or failed):
        raise ValueError(f"Universal evaluation incomplete: {len(missing)} missing, {failed} failed")
    records.sort(key=lambda row: row["global_index"])
    return config, records


def distributed_evaluate_programs(args):
    import multiprocessing as mp
    import torch.distributed as dist

    mp.set_start_method("spawn", force=True)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if "RANK" not in os.environ:
        raise ValueError("Use torchrun --standalone --nproc_per_node=1 (or 8)")
    if int(os.environ.get("LOCAL_WORLD_SIZE", world)) != world:
        raise ValueError("Only single-node, one-process-per-GPU evaluation is supported")
    dist.init_process_group("gloo", timeout=timedelta(seconds=args.completion_timeout))
    lock = None
    packet = [None]
    try:
        if rank == 0:
            try:
                candidate_lock = run_lock(args.run_name)
                candidate_lock.__enter__()
                lock = candidate_lock
                candidate_run_name = args.candidate_run_name or args.run_name
                manifest, search_config, search_records, candidates = load_evaluation_inputs(
                    args.run_name, candidate_run_name)
                config = build_config(args, manifest, search_config, search_records,
                                      candidates, candidate_run_name, world)
                folder = stage_dir(args.run_name, "universal_eval")
                if args.clean:
                    clean_stage(args.run_name, "universal_eval")
                recovered = recover_pending(folder)
                config_path = folder / "config.json"
                if config_path.exists():
                    if read_json(config_path) != config:
                        raise ValueError("Universal inputs changed; use a new run or --clean")
                else:
                    if any(folder.glob("rank_*")):
                        raise ValueError("Universal shards exist without config.json")
                    atomic_json(config_path, config)
                if recovered:
                    atomic_json(folder / "recovery_report.json", recovered)
                packet[0] = {"ok": True, "invocation": uuid.uuid4().hex,
                             "evaluation_config_id": config["evaluation_config_id"]}
            except Exception as exc:
                packet[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        dist.broadcast_object_list(packet, src=0)
        dist.destroy_process_group()
        if not packet[0]["ok"]:
            raise RuntimeError(packet[0]["error"])
        config = read_json(stage_dir(args.run_name, "universal_eval") / "config.json")
        if config["evaluation_config_id"] != packet[0]["evaluation_config_id"]:
            raise ValueError("Universal configuration changed during startup")
        manifest, _, search_records, candidate_data = load_evaluation_inputs(
            args.run_name, config.get("candidate_run_name", args.run_name))
        rows = [row for row in manifest["samples"]
                if row["split"] in args.evaluation_splits]
        run_rank(args, rows, search_records, candidate_data["candidates"], config,
                 packet[0]["invocation"], rank, local_rank)
        if rank == 0:
            wait_for_ranks(args, config, packet[0]["invocation"])
            _, records = load_universal_evaluation(args.run_name)
            print(f"Universal evaluation complete: {len(records)} held-out questions")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if lock is not None:
            lock.__exit__(None, None, None)
