"""Read local DART-Math files, deduplicate questions, and freeze split manifests."""

import json
import re
import unicodedata
from pathlib import Path

from tqdm import tqdm

from .storage import (atomic_json, clean_stage, digest, file_digest, read_json,
                      recover_pending, relative_path, stage_dir)

SOURCE_ID = "hkust-nlp/dart-math-pool-math"
ROW_FIELDS = ("sample_id", "question", "gt_ans", "difficulty", "split", "position",
              "global_index", "source_query_id")


def question_key(question):
    return " ".join(unicodedata.normalize("NFKC", question).split())


def input_files(data_path):
    path = relative_path(data_path)
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.rglob("*.parquet"))
        if not files:
            files = sorted(path.rglob("*.jsonl"))
    if not files:
        raise ValueError(f"No local parquet/JSONL files at {path}")
    numbered = [re.fullmatch(r"train-(\d+)-of-(\d+)\.parquet", p.name) for p in files]
    matches = [m for m in numbered if m is not None]
    if matches:
        totals = {int(m[2]) for m in matches}
        if len(totals) != 1 or len(matches) != len(files):
            raise ValueError("Inconsistent source parquet shard names")
        total = totals.pop()
        if len(matches) != total or {int(m[1]) for m in matches} != set(range(total)):
            raise ValueError(f"Incomplete source parquet snapshot; expected {total} shards")
    return files


def rows_from_file(path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        table = pq.ParquetFile(path)
        wanted = [c for c in ("query", "gt_ans", "query_metadata", "query_id", "query4test")
                  if c in table.schema_arrow.names]
        for batch in table.iter_batches(batch_size=4096, columns=wanted):
            yield from batch.to_pylist()
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    raise ValueError(f"Blank JSONL row at {path}:{line_number}")
                try:
                    from .storage import _pairs
                    def reject_constant(value):
                        raise ValueError(f"Non-finite JSON value {value}")
                    yield json.loads(line, object_pairs_hook=_pairs, parse_constant=reject_constant)
                except ValueError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    elif path.suffix == ".json":
        content = read_json(path)
        if not isinstance(content, list):
            raise ValueError("Local JSON input must be a list of raw DART-Math records")
        yield from content
    else:
        raise ValueError(f"Unsupported input file: {path}")


def check_manifest(manifest):
    payload = {k: v for k, v in manifest.items() if k != "manifest_id"}
    if digest(payload) != manifest.get("manifest_id"):
        raise ValueError("Prepared manifest checksum mismatch")
    ids, questions, source_ids = set(), set(), set()
    positions = {d: 0 for d in manifest["args"]["difficulties"]}
    for index, row in enumerate(manifest["samples"]):
        if any(k not in row for k in ROW_FIELDS):
            raise ValueError("Missing manifest fields")
        qkey = question_key(row["question"])
        if row["sample_id"] in ids or qkey in questions or row["source_query_id"] in source_ids:
            raise ValueError("Duplicate question or source ID across prepared splits")
        ids.add(row["sample_id"])
        questions.add(qkey)
        source_ids.add(row["source_query_id"])
        if row["sample_id"] != digest(qkey):
            raise ValueError("Question ID is inconsistent with question text")
        d = row["difficulty"]
        if d not in positions or row["position"] != positions[d] or row["global_index"] != index:
            raise ValueError("Prepared ordering is not contiguous")
        positions[d] += 1
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        if not isinstance(row["gt_ans"], str) or not row["gt_ans"].strip():
            raise ValueError("Empty ground truth")
    for d in positions:
        rows = [r for r in manifest["samples"] if r["difficulty"] == d]
        counts = manifest["split_counts"][str(d)]
        expected = (["train"] * counts["train"] + ["validation"] * counts["validation"]
                    + ["test"] * counts["test"])
        if [r["split"] for r in rows] != expected:
            raise ValueError("Split order/count mismatch")
    return manifest


def prepare(args):
    folder = stage_dir(args.run_name, "prepared")
    if args.clean:
        clean_stage(args.run_name, "prepared")
    recovered = recover_pending(folder)
    options = {key: getattr(args, key) for key in (
        "data_path", "data_source", "source_revision", "difficulties", "seed",
        "split_policy", "train_fraction", "validation_fraction", "max_questions_per_diff")}
    excluded_ids = set()
    if args.exclude_run_name is not None:
        excluded_manifest = check_manifest(read_json(
            stage_dir(args.exclude_run_name, "prepared") / "manifest.json"))
        excluded_ids = {row["sample_id"] for row in excluded_manifest["samples"]}
        options["exclude_run_name"] = args.exclude_run_name
        options["excluded_manifest_id"] = excluded_manifest["manifest_id"]
        options["excluded_sample_ids_digest"] = digest(sorted(excluded_ids))
    sources = [{"path": str(path), "sha256": file_digest(path), "bytes": path.stat().st_size}
               for path in tqdm(input_files(args.data_path), desc="Fingerprint local data")]
    target = folder / "manifest.json"
    if target.exists():
        manifest = check_manifest(read_json(target))
        if manifest["args"] != options or manifest["sources"] != sources:
            raise ValueError("Prepared inputs/config changed; use a different run-name or explicit prepare --clean")
        if recovered:
            atomic_json(folder / "recovery_report.json", recovered)
        print(f"Prepared manifest verified; resume: {len(manifest['samples'])} questions")
        return manifest

    unique, source_id_to_question = {}, {}
    raw_count = 0
    for source in tqdm(sources, desc="Read files"):
        for raw in tqdm(rows_from_file(Path(source["path"])), desc="Read/deduplicate questions", leave=False):
            raw_count += 1
            if not isinstance(raw, dict):
                raise ValueError(f"Non-object record {source['path']} row {raw_count}")
            # These are exactly the query/gt_ans fields used by dart_math/data.py.
            question, gt = raw.get("query"), raw.get("gt_ans")
            level = raw.get("query_metadata", {}).get("level")
            if not isinstance(question, str) or not question.strip() or not isinstance(gt, str) or not gt.strip():
                raise ValueError(f"Missing query/gt_ans at input row {raw_count}")
            if type(level) is not int or level not in range(1, 6):
                raise ValueError(f"Missing/invalid query_metadata.level at input row {raw_count}; never infer difficulty")
            if raw.get("query4test", False) is not False:
                raise ValueError("query4test is not explicitly false; this pipeline conservatively rejects marked test-source records")
            key = question_key(question)
            source_id = str(raw.get("query_id") or digest(key))
            if source_id in source_id_to_question and source_id_to_question[source_id] != key:
                raise ValueError(f"Conflicting query_id {source_id}")
            source_id_to_question[source_id] = key
            if key in unique:
                if unique[key]["gt_ans"] != gt or unique[key]["difficulty"] != level:
                    raise ValueError(f"Conflicting answer/difficulty for question {source_id}")
                continue
            unique[key] = {"sample_id": digest(key), "question": question, "gt_ans": gt,
                           "difficulty": level, "source_query_id": source_id}

    samples, split_counts = [], {}
    for diff in args.difficulties:
        bucket = [r for r in unique.values()
                  if r["difficulty"] == diff and r["sample_id"] not in excluded_ids]
        # Content-based ordering does not depend on input shard order or Python RNG version.
        bucket.sort(key=lambda r: digest([args.seed, r["sample_id"]]))
        if args.max_questions_per_diff:
            bucket = bucket[:args.max_questions_per_diff]
        if args.split_policy == "official":
            if len(bucket) < 2000:
                raise ValueError(f"diff {diff} has {len(bucket)} unique questions; official fixed split needs 2000. "
                                 "No duplication/padding is allowed; use proportional with the split adapter.")
            bucket = bucket[:2000]
            train_count, val_count = 1250, 250
        else:
            train_count = int(len(bucket) * args.train_fraction)
            val_count = int(len(bucket) * args.validation_fraction)
        test_count = len(bucket) - train_count - val_count
        if min(train_count, val_count, test_count) < 1:
            raise ValueError(f"diff {diff}: at least one unique question per split is required")
        split_counts[str(diff)] = {"train": train_count, "validation": val_count, "test": test_count}
        for position, row in enumerate(bucket):
            split = "train" if position < train_count else "validation" if position < train_count + val_count else "test"
            samples.append({**row, "position": position, "global_index": len(samples), "split": split})
    manifest = {"schema_version": 1, "args": options, "sources": sources,
                "raw_rows": raw_count, "unique_questions_in_source": len(unique),
                "split_counts": split_counts, "samples": samples}
    manifest["manifest_id"] = digest(manifest)
    check_manifest(manifest)
    atomic_json(target, manifest)
    atomic_json(folder / "summary.json", {"total_questions": len(samples), "raw_rows": raw_count,
                                        "split_counts": split_counts,
                                        "excluded_source_run_questions": len(excluded_ids),
                                        "recovery": recovered})
    print(f"Prepared {len(samples)} unique questions at {target}")
    return manifest
