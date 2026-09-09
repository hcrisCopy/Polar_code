"""Select one deterministic, unique DART-Math question per difficulty."""

from pathlib import Path

from tqdm import tqdm

from stage_one.prepare import input_files, question_key, rows_from_file
from stage_one.storage import file_digest

from .storage import atomic_json, digest, read_json, run_dir


def prepare_questions(args):
    folder = run_dir(args.run_name)
    target = folder / "questions.json"
    sources = [{"path": str(path), "bytes": path.stat().st_size,
                "sha256": file_digest(path)}
               for path in tqdm(input_files(args.data_path), desc="Fingerprint DART-Math files")]
    config = {"data_path": args.data_path, "source_revision": args.source_revision,
              "difficulties": args.difficulties, "seed": args.seed,
              "sources": sources}
    if target.exists():
        payload = read_json(target)
        if payload["config"] != config:
            raise ValueError("Prepared inputs changed; use a new run-name or --clean")
        print(f"Prepared questions verified: {target}")
        return payload

    unique = {}
    for source in tqdm(sources, desc="Read local DART-Math"):
        for raw in tqdm(rows_from_file(Path(source["path"])),
                        desc="Deduplicate questions", leave=False):
            if not isinstance(raw, dict):
                raise ValueError("DART-Math row must be an object")
            question, answer = raw.get("query"), raw.get("gt_ans")
            metadata = raw.get("query_metadata") or {}
            difficulty = metadata.get("level")
            if (not isinstance(question, str) or not question.strip()
                    or not isinstance(answer, str) or not answer.strip()
                    or type(difficulty) is not int or difficulty not in range(1, 6)):
                raise ValueError("DART-Math row has invalid query/gt_ans/difficulty")
            if raw.get("query4test", False) is not False:
                raise ValueError("This diagnostic refuses query4test-marked source rows")
            key = question_key(question)
            old = unique.get(key)
            if old and (old["gt_ans"] != answer or old["difficulty"] != difficulty):
                raise ValueError("Duplicate question has conflicting answer or difficulty")
            unique[key] = {"sample_id": digest(key), "question": question,
                           "gt_ans": answer, "difficulty": difficulty,
                           "source_query_id": str(raw.get("query_id") or digest(key))}

    selected = []
    for difficulty in args.difficulties:
        rows = [row for row in unique.values() if row["difficulty"] == difficulty]
        if not rows:
            raise ValueError(f"No unique question for difficulty {difficulty}")
        rows.sort(key=lambda row: digest([args.seed, difficulty, row["sample_id"]]))
        selected.append(rows[0])
    payload = {"schema_version": 1, "config": config, "questions": selected}
    payload["manifest_id"] = digest(payload)
    atomic_json(target, payload)
    print(f"Prepared {len(selected)} questions: {target}")
    return payload

