"""Incremental 32-path search with per-generation crash recovery."""

from pathlib import Path
import time

from tqdm import tqdm

from polar.config import infer_original_depth
from stage_one.storage import file_digest

from .candidates import generate_candidate_pool
from .model_runner import ShowcaseModelRunner
from .storage import atomic_json, digest, read_json, run_dir


def _counts(results):
    return (sum(row["correct"] for row in results),
            sum(not row["correct"] for row in results))


def _light_model_inventory(model_path):
    """Validate a local snapshot without hashing every multi-GB weight shard."""
    path = Path(model_path)
    required = [path / "config.json", path / "tokenizer_config.json"]
    if not all(item.is_file() for item in required):
        raise ValueError(f"Incomplete model config/tokenizer at {path}")
    weights = sorted(item for item in path.glob("*")
                     if item.is_file() and item.suffix in {".safetensors", ".bin"})
    if not weights:
        raise ValueError(f"No local model weights at {path}")
    for item in weights:
        with item.open("rb") as stream:
            if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/"):
                raise ValueError(f"Weight is only a Git LFS pointer: {item}")
    return {"verified_small_files": [
                {"file": item.name, "bytes": item.stat().st_size,
                 "sha256": file_digest(item)} for item in required],
            "weight_files": [
                {"file": item.name, "bytes": item.stat().st_size} for item in weights]}


def _config(args, manifest):
    depth = infer_original_depth(args.model_id)
    local_config = read_json(Path(args.model_path) / "config.json")
    if local_config.get("num_hidden_layers") != depth:
        raise ValueError("Local model config does not match Qwen3-8B depth")
    if local_config.get("quantization_config"):
        raise ValueError("This checkpoint requires the full, unquantized model")
    return {"schema_version": 1, "model_id": args.model_id,
            "model_path": args.model_path, "model_revision": args.model_revision,
            "model_files": _light_model_inventory(args.model_path),
            "manifest_id": manifest["manifest_id"],
            "seed": args.seed, "candidate_limit": args.candidate_limit,
            "batch_size": args.batch_size, "target_per_label": args.target_per_label,
            "max_block": args.max_block, "max_length_factor": args.max_length_factor,
            "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
            "depth": depth, "max_length": int(depth * args.max_length_factor)}


def _new_question_state(row):
    return {"sample_id": row["sample_id"], "difficulty": row["difficulty"],
            "results": [], "pending_batch": [], "status": "pending"}


def run_search(args):
    folder = run_dir(args.run_name)
    manifest = read_json(folder / "questions.json")
    config = _config(args, manifest)
    config["config_id"] = digest(config)
    config_path = folder / "search_config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError("Search configuration changed; use a new run-name or --clean")
    if not config_path.exists():
        atomic_json(config_path, config)

    pools = {}
    for row in manifest["questions"]:
        pools[str(row["difficulty"])] = generate_candidate_pool(
            depth=config["depth"], maximum=args.candidate_limit,
            seed=int(digest([args.seed, row["sample_id"]])[:16], 16),
            max_block=args.max_block, max_length=config["max_length"])
    pool_payload = {"config_id": config["config_id"], "by_difficulty": pools}
    pool_path = folder / "candidate_pools.json"
    if pool_path.exists() and read_json(pool_path) != pool_payload:
        raise ValueError("Candidate pool changed; use a new run-name or --clean")
    if not pool_path.exists():
        atomic_json(pool_path, pool_payload)

    state_path = folder / "search_state.json"
    if state_path.exists():
        state = read_json(state_path)
        if state["config_id"] != config["config_id"]:
            raise ValueError("Search state belongs to another configuration")
    else:
        state = {"config_id": config["config_id"], "questions": {
            str(row["difficulty"]): _new_question_state(row)
            for row in manifest["questions"]}}
        atomic_json(state_path, state)

    log_path = folder / "model.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log_stream:
        runner = ShowcaseModelRunner(args, log_stream)
        try:
            if runner.depth != config["depth"]:
                raise ValueError("Loaded model depth differs from configured Qwen3-8B depth")
            atomic_json(folder / "model_runtime.json", runner.metadata())
            for row in tqdm(manifest["questions"], desc="Difficulty questions"):
                key = str(row["difficulty"])
                question_state = state["questions"][key]
                if question_state["status"] in {"quota_reached", "candidate_limit_reached"}:
                    continue
                pool = pools[key]
                while True:
                    correct_count, error_count = _counts(question_state["results"])
                    if min(correct_count, error_count) >= args.target_per_label:
                        question_state["status"] = "quota_reached"
                        break
                    completed_ids = {item["candidate_id"] for item in question_state["results"]}
                    pending_ids = {item["candidate_id"] for item in question_state["pending_batch"]}
                    if not question_state["pending_batch"]:
                        remaining = [item for item in pool if item["candidate_id"] not in completed_ids]
                        question_state["pending_batch"] = [
                            {**item, "generated_text": None, "generation_seconds": None}
                            for item in remaining[:args.batch_size]
                        ]
                        atomic_json(state_path, state)
                    if not question_state["pending_batch"]:
                        question_state["status"] = "candidate_limit_reached"
                        break

                    for item in tqdm(question_state["pending_batch"],
                                     desc=f"DM-{key} generate batch", leave=False):
                        if item["candidate_id"] in pending_ids and item["generated_text"] is not None:
                            continue
                        started = time.monotonic()
                        item["generated_text"] = runner.generate(
                            row["question"], row["sample_id"], item["path"])
                        item["generation_seconds"] = time.monotonic() - started
                        atomic_json(state_path, state)

                    judged = runner.judge_batch(
                        [item["generated_text"] for item in question_state["pending_batch"]],
                        row["gt_ans"])
                    for item, (extracted, correct) in zip(question_state["pending_batch"], judged):
                        question_state["results"].append({
                            **item, "extracted_answer": extracted, "correct": correct,
                            "evaluation_index": len(question_state["results"])
                        })
                    question_state["pending_batch"] = []
                    correct_count, error_count = _counts(question_state["results"])
                    print(f"DM-{key}: evaluated={len(question_state['results'])}, "
                          f"correct={correct_count}, error={error_count}")
                    atomic_json(state_path, state)
                atomic_json(state_path, state)
        finally:
            runner.close()
    return state
