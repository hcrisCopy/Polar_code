"""Stage-one MCTS with quota checks every 32 newly evaluated paths."""

from pathlib import Path
import time

from tqdm import tqdm

from polar.config import infer_original_depth
from stage_one.search_tree import search as mcts_search
from stage_one.storage import file_digest

from .model_runner import ShowcaseModelRunner
from .storage import atomic_json, digest, read_json, run_dir


def _counts(results):
    return (sum(row["correct"] for row in results),
            sum(not row["correct"] for row in results))


def _model_inventory(model_path):
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
    return {
        "verified_small_files": [
            {"file": item.name, "bytes": item.stat().st_size,
             "sha256": file_digest(item)} for item in required
        ],
        "weight_files": [
            {"file": item.name, "bytes": item.stat().st_size} for item in weights
        ],
    }


def _config(args, manifest):
    depth = infer_original_depth(args.model_id)
    local_config = read_json(Path(args.model_path) / "config.json")
    if local_config.get("num_hidden_layers") != depth:
        raise ValueError("Local model config does not match Qwen3-8B depth")
    if local_config.get("quantization_config"):
        raise ValueError("This checkpoint requires the full, unquantized model")
    return {
        "schema_version": 2,
        "search_method": "stage_one MCTS",
        "model_id": args.model_id,
        "model_path": args.model_path,
        "model_revision": args.model_revision,
        "model_files": _model_inventory(args.model_path),
        "manifest_id": manifest["manifest_id"],
        "seed": args.seed,
        "simulations": args.simulations,
        "check_interval": args.check_interval,
        "max_question_seconds": args.max_question_seconds,
        "target_per_label": args.target_per_label,
        "exploration": args.exploration,
        "length_penalty": args.length_penalty,
        "max_block": args.max_block,
        "max_repeats": args.max_repeats,
        "max_length_factor": args.max_length_factor,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "depth": depth,
        "max_length": int(depth * args.max_length_factor),
    }


def _new_question_state(row):
    return {
        "sample_id": row["sample_id"],
        "difficulty": row["difficulty"],
        "results": [],
        "status": "pending",
        "search_statistics": None,
    }


def _search_question(row, question_state, args, config, runner, state, state_path):
    result_by_path = {tuple(item["path"]): item for item in question_state["results"]}
    baseline = tuple(range(config["depth"]))
    stop_reason = {"value": None}

    def evaluate(path):
        path_tuple = tuple(path)
        cached = result_by_path.get(path_tuple)
        if cached is not None:
            return float(cached["correct"])

        started = time.monotonic()
        generated, unexpected_thinking = runner.generate(
            row["question"], row["sample_id"], path
        )
        extracted, correct = runner.judge(generated, row["gt_ans"])
        item = {
            "candidate_id": f"mcts_{len(question_state['results']):04d}",
            "path": path,
            "length": len(path),
            "path_digest": digest(path),
            "generated_text": generated,
            "unexpected_thinking": unexpected_thinking,
            "extracted_answer": extracted,
            "correct": correct,
            "generation_seconds": time.monotonic() - started,
            "evaluation_index": len(question_state["results"]),
        }
        question_state["results"].append(item)
        result_by_path[path_tuple] = item
        atomic_json(state_path, state)
        return float(correct)

    def should_stop(*, unique_evaluations, simulations_completed):
        del simulations_completed
        consumed = sum(
            item.get("generation_seconds", 0.0) for item in question_state["results"]
        )
        if consumed >= args.max_question_seconds:
            stop_reason["value"] = "time_limit_reached"
            print(
                f"DM-{row['difficulty']}: stop after {consumed:.1f}s cumulative "
                "generation/judging time"
            )
            return True
        # The root baseline is evaluated outside the N MCTS simulations.
        new_path_count = unique_evaluations - 1
        if new_path_count <= 0 or new_path_count % args.check_interval:
            return False
        replayed_results = question_state["results"][:unique_evaluations]
        correct_count, error_count = _counts(replayed_results)
        print(
            f"DM-{row['difficulty']}: MCTS new_paths={new_path_count}, "
            f"correct={correct_count}, error={error_count}"
        )
        reached = min(correct_count, error_count) >= args.target_per_label
        if reached:
            stop_reason["value"] = "quota_reached"
        return reached

    sample_seed = int(digest([args.seed, row["sample_id"]])[:16], 16)
    statistics = mcts_search(
        evaluate,
        depth=config["depth"],
        simulations=args.simulations,
        exploration=args.exploration,
        length_penalty=args.length_penalty,
        max_block=args.max_block,
        max_repeats=args.max_repeats,
        max_length=config["max_length"],
        seed=sample_seed,
        rank=0,
        on_evaluation=lambda path, reward: None,
        should_stop=should_stop,
    )
    question_state["search_statistics"] = statistics
    correct_count, error_count = _counts(question_state["results"])
    question_state["status"] = stop_reason["value"] or (
        "quota_reached" if min(correct_count, error_count) >= args.target_per_label
        else "simulation_limit_reached"
    )
    question_state["baseline_correct"] = result_by_path[baseline]["correct"]
    atomic_json(state_path, state)


def run_search(args):
    folder = run_dir(args.run_name)
    manifest = read_json(folder / "questions.json")
    config = _config(args, manifest)
    config["config_id"] = digest(config)
    config_path = folder / "search_config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError(
            "Search configuration changed; use a new run-name or prepare --clean"
        )
    if not config_path.exists():
        atomic_json(config_path, config)

    state_path = folder / "search_state.json"
    if state_path.exists():
        state = read_json(state_path)
        if state["config_id"] != config["config_id"]:
            raise ValueError("Search state belongs to another configuration")
    else:
        state = {
            "config_id": config["config_id"],
            "questions": {
                str(row["difficulty"]): _new_question_state(row)
                for row in manifest["questions"]
            },
        }
        atomic_json(state_path, state)

    log_path = folder / "model.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log_stream:
        runner = ShowcaseModelRunner(args, log_stream)
        try:
            if runner.depth != config["depth"]:
                raise ValueError("Loaded model depth differs from Qwen3-8B depth")
            atomic_json(folder / "model_runtime.json", runner.metadata())
            for row in tqdm(manifest["questions"], desc="Difficulty questions"):
                question_state = state["questions"][str(row["difficulty"])]
                if question_state["status"] in {
                    "quota_reached", "simulation_limit_reached", "time_limit_reached"
                }:
                    continue
                _search_question(
                    row, question_state, args, config, runner, state, state_path
                )
        finally:
            runner.close()
    return state
