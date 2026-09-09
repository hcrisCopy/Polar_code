"""Independent, CPU-only checks of manifests, shards, and official loader format."""

from collections import Counter

from tqdm import tqdm

from .prepare import ROW_FIELDS, check_manifest
from .storage import atomic_json, clean_stage, digest, read_json, recover_pending, stage_dir


def verify_record(record, row, config):
    from polar.data import parse_path_to_seg_and_ops
    if digest({k: v for k, v in record.items() if k != "record_digest"}) != record.get("record_digest"):
        raise ValueError(f"Record checksum mismatch: {row['sample_id']}")
    if any(record.get(key) != row[key] for key in ROW_FIELDS):
        raise ValueError(f"Record source/split mismatch: {row['sample_id']}")
    if record.get("config_id") != config["config_id"]:
        raise ValueError("Mixed search configurations")
    required = {"initial_score", "final_valid_transitions", "final_invalid_transitions",
                "search_statistics", "failure_reason", "status", "evaluations"}
    if not required <= record.keys():
        raise ValueError("Missing result fields")
    depth = config["depth"]
    maximum = config["max_length"]
    evaluated = {}
    for entry in record["evaluations"]:
        path = entry["path"]
        if (not isinstance(path, list) or not path or len(path) > maximum
                or any(type(i) is not int or not 0 <= i < depth for i in path)):
            raise ValueError("Empty, too long, or out-of-range execution path")
        key = tuple(path)
        if key in evaluated or type(entry["score"]) not in (int, float) or entry["score"] not in (0, 1):
            raise ValueError("Duplicate path or invalid reward")
        evaluated[key] = entry["score"]
    original = tuple(range(depth))
    if record["initial_score"] != evaluated.get(original):
        raise ValueError("initial_score does not match the actually evaluated baseline")
    expected = {1: set(), 0: set()}
    for path, score in evaluated.items():
        if parse_path_to_seg_and_ops(list(path), depth, max_pack=4, allow_repeat=True) is not None:
            expected[int(score)].add(path)
    sets = []
    for label, score in (("final_valid_transitions", 1), ("final_invalid_transitions", 0)):
        paths = record[label]
        if (not isinstance(paths, list) or any(not isinstance(p, list) or not p
                or any(type(i) is not int or not 0 <= i < depth for i in p) for p in paths)):
            raise ValueError("Official predictor transitions must be list[list[int]]")
        actual = {tuple(p) for p in paths}
        if len(actual) != len(paths) or actual != expected[score]:
            raise ValueError(f"{label} differs from actual scored, representable paths")
        sets.append(actual)
    if sets[0] & sets[1]:
        raise ValueError("Valid and invalid paths overlap")
    status = record["status"]
    if status not in {"complete", "failed"}:
        raise ValueError("Unknown record completion state")
    stats = record["search_statistics"]
    if stats.get("unique_evaluations") != len(evaluated):
        raise ValueError("Evaluation count differs from recorded real model calls")
    if status == "complete":
        if record["initial_score"] not in (0.0, 1.0):
            raise ValueError("Completed question lacks evaluated baseline")
        expected_sim = 0 if row["split"] == "test" else config["args"]["simulations"]
        if stats.get("simulations_completed") != expected_sim:
            raise ValueError("Completed question has incomplete simulation budget")
    elif not record["failure_reason"]:
        raise ValueError("Failed record requires failure_reason")
    if row["split"] == "test" and set(evaluated) - {original}:
        raise ValueError("Test question was used for MCTS reward/search")


def load_search(run_name, *, require_complete=True):
    manifest = check_manifest(read_json(stage_dir(run_name, "prepared") / "manifest.json"))
    folder = stage_dir(run_name, "search")
    config = read_json(folder / "config.json")
    if digest({k: v for k, v in config.items() if k != "config_id"}) != config.get("config_id"):
        raise ValueError("Search configuration checksum mismatch")
    if config["manifest_id"] != manifest["manifest_id"]:
        raise ValueError("Search results do not belong to this prepared manifest")
    pending = [p for p in folder.rglob("*.pending") if "recovery" not in p.parts]
    if pending:
        raise ValueError(f"Interrupted writes present; resume search first: {pending[0]}")
    by_id = {row["sample_id"]: row for row in manifest["samples"]}
    records, seen = [], set()
    world = config["world_size"]
    rank_dirs = {p.name for p in folder.glob("rank_*") if p.is_dir()}
    expected_dirs = {f"rank_{rank:05d}" for rank in range(world)}
    if rank_dirs - expected_dirs or (require_complete and rank_dirs != expected_dirs):
        raise ValueError("Missing or unexpected rank directories")
    for rank in tqdm(range(world), desc="Validate shards"):
        rank_dir = folder / f"rank_{rank:05d}"
        for path in tqdm(sorted((rank_dir / "records").glob("*.json")), desc=f"Read rank {rank}", leave=False):
            record = read_json(path)
            sid = record.get("sample_id")
            if sid not in by_id or sid in seen or path.stem != sid:
                raise ValueError(f"Unknown, duplicate, or misnamed sample: {path}")
            row = by_id[sid]
            if row["global_index"] % world != rank:
                raise ValueError(f"Sample in incorrect rank shard: {path}")
            verify_record(record, row, config)
            seen.add(sid)
            records.append(record)
    missing = set(by_id) - seen
    failed = sum(r["status"] == "failed" for r in records)
    if require_complete and (missing or failed):
        raise ValueError(f"Incomplete shards: {len(missing)} missing, {failed} failed. Resume search before merging.")
    records.sort(key=lambda row: row["global_index"])
    return manifest, config, records


def summarize(records, total):
    valid = [p for r in records for p in r["final_valid_transitions"]]
    invalid = [p for r in records for p in r["final_invalid_transitions"]]
    return {"total_questions": total, "persisted_questions": len(records),
            "missing_questions": total - len(records),
            "completed_questions": sum(r["status"] == "complete" for r in records),
            "failed_questions": sum(r["status"] == "failed" for r in records),
            "questions_with_valid_paths": sum(bool(r["final_valid_transitions"]) for r in records),
            "complete_without_valid_paths": sum(r["status"] == "complete" and not r["final_valid_transitions"] for r in records),
            "mean_valid_paths": len(valid) / max(1, len(records)),
            "mean_invalid_paths": len(invalid) / max(1, len(records)),
            "mean_valid_path_length": sum(map(len, valid)) / max(1, len(valid)),
            "mean_invalid_path_length": sum(map(len, invalid)) / max(1, len(invalid)),
            "mean_path_length": sum(map(len, valid + invalid)) / max(1, len(valid + invalid)),
            "split_counts": dict(Counter(r["split"] for r in records)),
            "failure_reasons": dict(Counter(r["failure_reason"] for r in records if r["failure_reason"]))}


def merged_file(run, model_id, difficulty):
    from .storage import relative_path, output_path
    return output_path(stage_dir(run, "merged") / relative_path(model_id)
                       / f"dart-math-diff-{difficulty}" / "merged_mcts_samples.json")


def check_loader(path, samples, depth):
    from polar.data import PolarDataset, extract_question_and_gt
    for row in samples:
        if extract_question_and_gt(row) != (row["question"], row["gt_ans"]):
            raise ValueError("Official question/answer extraction differs")
    indices = [i for i, row in enumerate(samples) if row["split"] == "train"]
    dataset = PolarDataset(str(path), 0, len(samples), depth, indices=indices,
                           max_paths_per_sample=10**9, seed=42)
    expected = sum(len(samples[i]["final_valid_transitions"]) for i in indices)
    if len(dataset) != expected:
        raise ValueError("Official PolarDataset silently dropped training paths")
    return expected


def validate(args):
    folder = stage_dir(args.run_name, "validation")
    if args.clean:
        clean_stage(args.run_name, "validation")
    recover_pending(folder)
    try:
        manifest, config, records = load_search(args.run_name)
        expected_examples = 0
        for difficulty in tqdm(manifest["args"]["difficulties"], desc="Validate final JSON/official loader"):
            path = merged_file(args.run_name, config["args"]["model_id"], difficulty)
            data = read_json(path)
            expected = [r for r in records if r["difficulty"] == difficulty]
            if (data.get("samples") != expected or data.get("config_id") != config["config_id"]
                    or data.get("manifest_id") != manifest["manifest_id"]):
                raise ValueError(f"Merged JSON differs from complete source shards: {path}")
            expected_examples += check_loader(path, data["samples"], config["depth"])
        report = {"passed": True, "config_id": config["config_id"],
                  "manifest_id": manifest["manifest_id"], "training_examples": expected_examples,
                  "model_executed": False, "summary": summarize(records, len(manifest["samples"]))}
        atomic_json(folder / "report.json", report)
        print(f"Data validation passed; {expected_examples} official-loader training examples")
        return report
    except Exception as exc:
        atomic_json(folder / "report.json", {"passed": False, "error": str(exc), "model_executed": False})
        raise
