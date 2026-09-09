"""Re-evaluate already visualized path pairs with Qwen3 thinking enabled."""

import csv
import io
import time

from tqdm import tqdm

from .model_runner import ShowcaseModelRunner
from .report import save_thinking_comparison
from .storage import atomic_json, atomic_text, digest, read_json, run_dir


PAIR_KEYS = (
    ("simplest", "simplest_correct_with_nearest_wrong"),
    ("most_complex", "most_complex_correct_with_nearest_wrong"),
)


def _selected_paths(selection):
    selected = []
    seen = set()
    for figure, key in PAIR_KEYS:
        if key not in selection:
            raise ValueError(
                "Path selections predate paired visualization; regenerate the report first"
            )
        for pair_index, pair in enumerate(selection[key], start=1):
            for role in ("correct", "wrong"):
                row = pair[role]
                candidate_id = row["candidate_id"]
                if candidate_id in seen:
                    raise ValueError(f"Visualized candidate is reused: {candidate_id}")
                seen.add(candidate_id)
                selected.append({
                    "candidate_id": candidate_id,
                    "path": row["path"],
                    "figure": figure,
                    "pair_index": pair_index,
                    "nonthinking_role": "C" if role == "correct" else "W",
                    "nonthinking_correct": bool(row["correct"]),
                })
    return selected


def _write_outputs(folder, difficulty, selected, state, elapsed_seconds):
    by_id = state["results"]
    counts = {"correct": 0, "wrong": 0, "truncated": 0, "pending": 0}
    transitions = {
        "correct_to_correct": 0,
        "correct_to_wrong": 0,
        "wrong_to_correct": 0,
        "wrong_to_wrong": 0,
    }
    table_rows = []
    outcomes = {}
    for item in selected:
        result = by_id.get(item["candidate_id"])
        if result is None:
            outcome = "pending"
            extracted = ""
            token_count = ""
            hit_limit = ""
        else:
            outcome = result["outcome"]
            extracted = result["extracted_answer"]
            token_count = result["generated_token_count"]
            hit_limit = result["hit_token_limit"]
            if outcome in counts:
                counts[outcome] += 1
            if outcome in {"correct", "wrong"}:
                source = "correct" if item["nonthinking_correct"] else "wrong"
                transitions[f"{source}_to_{outcome}"] += 1
        if result is None:
            counts["pending"] += 1
        outcomes[item["candidate_id"]] = {
            "correct": "C", "wrong": "W", "truncated": "T"
        }.get(outcome, "?")
        table_rows.append({
            "figure": item["figure"],
            "pair": item["pair_index"],
            "role_nonthinking": item["nonthinking_role"],
            "candidate_id": item["candidate_id"],
            "nonthinking_correct": item["nonthinking_correct"],
            "thinking_outcome": outcome,
            "thinking_extracted_answer": extracted,
            "thinking_boundary_status": (
                "" if result is None else result["thinking_boundary_status"]
            ),
            "thinking_tokens": (
                "" if result is None else result["thinking_token_count"]
            ),
            "answer_tokens": (
                "" if result is None else result["answer_token_count"]
            ),
            "generated_tokens": token_count,
            "hit_token_limit": hit_limit,
        })

    summary = {
        "schema_version": 1,
        "difficulty": difficulty,
        "thinking_enabled": True,
        "selected_paths": len(selected),
        "evaluated_paths": len(by_id),
        "elapsed_seconds_this_invocation": elapsed_seconds,
        "outcomes": counts,
        "transitions": transitions,
    }
    atomic_json(folder / "summary.json", summary)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=table_rows[0].keys())
    writer.writeheader()
    writer.writerows(table_rows)
    atomic_text(folder / "comparison.csv", stream.getvalue())

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        f"# DM-{difficulty}：相同路径的 Qwen3 Thinking 复评",
        "",
        (f"共选择 {len(selected)} 条可视化路径，已评估 {len(by_id)} 条；"
         f"正确 {counts['correct']}，错误 {counts['wrong']}，"
         f"截断 {counts['truncated']}，待评估 {counts['pending']}。"),
        "",
        "| 图 | Pair | 非思考角色 | 候选 | Thinking 结果 | 抽取答案 | 思考 Tokens | 答案 Tokens | 边界 | 撞上限 |",
        "| --- | ---: | --- | --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in table_rows:
        lines.append(
            f"| {cell(row['figure'])} | {row['pair']} | {row['role_nonthinking']} | "
            f"{cell(row['candidate_id'])} | {row['thinking_outcome']} | "
            f"{cell(row['thinking_extracted_answer'])} | "
            f"{row['thinking_tokens']} | {row['answer_tokens']} | "
            f"{row['thinking_boundary_status']} | "
            f"{row['hit_token_limit']} |"
        )
    lines.extend([
        "",
        "`truncated` 表示生成达到 token 上限且尚未给出 boxed answer；该项不计为错误。",
        "`missing_end_marker` 表示未生成 Qwen3 的 `</think>` 边界；完整原文保存在 state.json。",
    ])
    atomic_text(folder / "summary.md", "\n".join(lines) + "\n")
    return outcomes


def evaluate_selected_paths(args):
    source_folder = run_dir(args.run_name)
    search_config = read_json(source_folder / "search_config.json")
    search_state = read_json(source_folder / "search_state.json")
    if search_state["config_id"] != search_config["config_id"]:
        raise ValueError("Search state/config mismatch")
    selection_path = source_folder / f"path_selections_dm{args.difficulty}.json"
    selections = read_json(selection_path)
    key = str(args.difficulty)
    if key not in selections:
        raise ValueError(f"No paired visualization selection for DM-{args.difficulty}")
    selection = selections[key]
    selected = _selected_paths(selection)
    if not selected:
        raise ValueError(f"DM-{args.difficulty} paired visualization contains no paths")

    question_state = search_state["questions"].get(key)
    if question_state is None:
        raise ValueError(f"DM-{args.difficulty} is absent from search state")
    source_by_id = {
        row["candidate_id"]: row for row in question_state["results"]
    }
    for item in selected:
        source = source_by_id.get(item["candidate_id"])
        if source is None or source["path"] != item["path"]:
            raise ValueError(
                f"Selection/search-state mismatch for {item['candidate_id']}"
            )
        if bool(source["correct"]) != item["nonthinking_correct"]:
            raise ValueError(
                f"Non-thinking label mismatch for {item['candidate_id']}"
            )

    questions = read_json(source_folder / "questions.json")["questions"]
    question = next(
        (row for row in questions if int(row["difficulty"]) == args.difficulty), None
    )
    if question is None or question["sample_id"] != question_state["sample_id"]:
        raise ValueError("Question manifest/search-state mismatch")
    if args.model_id != search_config["model_id"]:
        raise ValueError("Thinking evaluation must use the same model-id as search")
    if args.model_path != search_config["model_path"]:
        raise ValueError("Thinking evaluation must use the same model-path as search")
    if args.model_revision != search_config["model_revision"]:
        raise ValueError("Thinking evaluation must use the same model revision as search")

    folder = source_folder / "thinking_eval" / f"dm{args.difficulty}"
    config = {
        "schema_version": 1,
        "search_config_id": search_config["config_id"],
        "selection_digest": digest(selection),
        "model_id": args.model_id,
        "model_path": args.model_path,
        "model_revision": args.model_revision,
        "device": args.device,
        "seed": args.seed,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "max_total_seconds": args.max_total_seconds,
        "thinking_enabled": True,
        "selected_candidate_ids": [item["candidate_id"] for item in selected],
    }
    config["config_id"] = digest(config)
    config_path = folder / "config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError("Thinking evaluation configuration changed; rerun with --clean")
    if not config_path.exists():
        atomic_json(config_path, config)
    state_path = folder / "state.json"
    if state_path.exists():
        state = read_json(state_path)
        if state["config_id"] != config["config_id"]:
            raise ValueError("Thinking evaluation state/config mismatch")
    else:
        state = {"config_id": config["config_id"], "results": {}}
        atomic_json(state_path, state)

    started = time.monotonic()
    pending = [
        item for item in selected if item["candidate_id"] not in state["results"]
    ]
    runner = None
    try:
        if pending:
            log_path = folder / "model.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8", buffering=1) as log_stream:
                runner = ShowcaseModelRunner(args, log_stream)
                if runner.depth != search_config["depth"]:
                    raise ValueError("Thinking model depth differs from search model depth")
                metadata = runner.metadata()
                metadata.update({
                    "thinking": True,
                    "max_new_tokens": args.max_new_tokens,
                })
                atomic_json(folder / "model_runtime.json", metadata)
                for item in tqdm(pending, desc=f"DM-{args.difficulty} thinking paths"):
                    if time.monotonic() - started >= args.max_total_seconds:
                        print(
                            f"DM-{args.difficulty}: thinking evaluation reached "
                            f"{args.max_total_seconds}s total limit"
                        )
                        break
                    generation_started = time.monotonic()
                    details = runner.generate_detailed(
                        question["question"], question["sample_id"], item["path"],
                        enable_thinking=True,
                        max_new_tokens=args.max_new_tokens,
                    )
                    no_boxed_answer = "oxed{" not in details["answer_text"]
                    truncated = details["hit_token_limit"] and no_boxed_answer
                    if truncated:
                        extracted, correct, outcome = "", None, "truncated"
                    else:
                        extracted, correct = runner.judge(
                            details["answer_text"], question["gt_ans"]
                        )
                        outcome = "correct" if correct else "wrong"
                    state["results"][item["candidate_id"]] = {
                        **item,
                        **details,
                        "nonthinking_generated_text": source_by_id[
                            item["candidate_id"]
                        ]["generated_text"],
                        "nonthinking_extracted_answer": source_by_id[
                            item["candidate_id"]
                        ]["extracted_answer"],
                        "extracted_answer": extracted,
                        "correct": correct,
                        "outcome": outcome,
                        "generation_seconds": time.monotonic() - generation_started,
                    }
                    atomic_json(state_path, state)
    finally:
        if runner is not None:
            runner.close()

    elapsed = time.monotonic() - started
    outcomes = _write_outputs(
        folder, args.difficulty, selected, state, elapsed
    )
    save_thinking_comparison(
        folder, args.difficulty, selection, search_config["depth"], outcomes
    )
    print(f"Thinking comparison written to {folder}")
    return state
