"""Create the two requested path figures for every selected question."""

import csv
import os

from matplotlib.colors import ListedColormap
import numpy as np
from tqdm import tqdm

from stage_one.plot_utils import pyplot

from .storage import atomic_json, atomic_text, read_json, run_dir


def _complexity_key(row):
    return (row["length"], len(set(row["path"])), row["evaluation_index"])


def _select(results, largest, count):
    correct = sorted((row for row in results if row["correct"]),
                     key=_complexity_key, reverse=largest)[:count]
    errors = sorted((row for row in results if not row["correct"]),
                    key=_complexity_key, reverse=largest)[:count]
    return correct + errors


def _execution_counts(path, depth):
    counts = np.zeros(depth, dtype=int)
    for layer in path:
        if layer < 0 or layer >= depth:
            raise ValueError(f"Invalid layer index {layer}")
        counts[layer] += 1
    return counts


def _save_figure(path, rows, depth, title):
    plt = pyplot()
    matrix = np.stack([np.clip(_execution_counts(row["path"], depth), 0, 2)
                       for row in rows])
    fig, axis = plt.subplots(figsize=(11.5, max(4.2, 0.34 * len(rows) + 1.6)),
                             constrained_layout=True)
    cmap = ListedColormap(["#E6E6E6", "#56B4E9", "#D55E00"])
    axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap,
                vmin=-0.5, vmax=2.5)
    axis.set_xticks(range(depth))
    axis.set_xticklabels(range(depth), fontsize=6)
    labels = []
    for row in rows:
        verdict = "C" if row["correct"] else "W"
        labels.append(f"{verdict}  {row['candidate_id']}  len={row['length']}  "
                      f"found={row['evaluation_index'] + 1}")
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels(labels, fontsize=7)
    axis.set_xlabel("Original transformer layer (gray=skip, blue=once, orange=loop)")
    axis.set_title(title)
    split = sum(row["correct"] for row in rows)
    if 0 < split < len(rows):
        axis.axhline(split - 0.5, color="black", linewidth=1.2)
    for index, row in enumerate(rows):
        color = "#009E73" if row["correct"] else "#CC3311"
        axis.get_yticklabels()[index].set_color(color)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    fig.savefig(pending, format="png", dpi=180)
    with pending.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(pending, path)
    plt.close(fig)


def build_report(args):
    folder = run_dir(args.run_name)
    manifest = read_json(folder / "questions.json")
    config = read_json(folder / "search_config.json")
    state = read_json(folder / "search_state.json")
    if state["config_id"] != config["config_id"]:
        raise ValueError("Search state/config mismatch")
    if config["target_per_label"] < 2 * args.paths_per_label:
        raise ValueError(
            "Distinct simplest/most-complex figures require target-per-label to be "
            "at least twice paths-per-label"
        )
    question_by_diff = {str(row["difficulty"]): row for row in manifest["questions"]}
    report_rows = []
    selections = {}
    for key, question_state in tqdm(state["questions"].items(), desc="Render path figures"):
        results = question_state["results"]
        correct_count = sum(row["correct"] for row in results)
        error_count = len(results) - correct_count
        required_count = 2 * args.paths_per_label
        if min(correct_count, error_count) < required_count:
            raise ValueError(
                f"DM-{key} has only {correct_count} correct and {error_count} wrong paths; "
                f"need {required_count} of each to render disjoint extremes"
            )
        simple = _select(results, largest=False, count=args.paths_per_label)
        complex_rows = _select(results, largest=True, count=args.paths_per_label)
        if not simple or not complex_rows:
            raise ValueError(f"DM-{key} has no paths to visualize")
        figure_dir = folder / "figures"
        _save_figure(figure_dir / f"dm{key}_simplest.png", simple, config["depth"],
                     f"DM-{key}: simplest correct and wrong paths")
        _save_figure(figure_dir / f"dm{key}_most_complex.png", complex_rows,
                     config["depth"], f"DM-{key}: most complex correct and wrong paths")
        simple_ids = {row["candidate_id"] for row in simple}
        complex_ids = {row["candidate_id"] for row in complex_rows}
        overlap = len(simple_ids & complex_ids)
        if overlap:
            raise RuntimeError(
                f"DM-{key} simplest and most-complex path sets unexpectedly overlap"
            )
        selections[key] = {
            "sample_id": question_state["sample_id"],
            "first_discovered_correct": [row for row in results if row["correct"]][
                :args.paths_per_label],
            "first_discovered_wrong": [row for row in results if not row["correct"]][
                :args.paths_per_label],
            "simplest_correct_and_wrong": simple,
            "most_complex_correct_and_wrong": complex_rows,
        }
        report_rows.append({"difficulty": int(key), "sample_id": question_state["sample_id"],
                            "status": question_state["status"], "evaluated": len(results),
                            "correct": correct_count, "wrong": error_count,
                            "simple_complex_overlap": overlap,
                            "question": question_by_diff[key]["question"],
                            "ground_truth": question_by_diff[key]["gt_ans"]})

    summary = {"schema_version": 1, "run_name": args.run_name,
               "complexity_metric": "execution path length",
               "paths_per_label_per_figure": args.paths_per_label,
               "simplest_and_most_complex_sets_disjoint": True,
               "questions": report_rows}
    atomic_json(folder / "report.json", summary)
    atomic_json(folder / "path_selections.json", selections)
    csv_path = folder / "report.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pending = csv_path.with_suffix(".csv.pending")
    with pending.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, csv_path)
    lines = ["# 前 10 条正确/错误路径展示", "",
             "复杂度暂按执行路径长度衡量；每个难度一题，每题输出最简单和最复杂两张图。", "",
             "| 难度 | 已评估 | 正确 | 错误 | 状态 | 两图重合路径数 |",
             "| --- | ---: | ---: | ---: | --- | ---: |"]
    for row in report_rows:
        lines.append(f"| DM-{row['difficulty']} | {row['evaluated']} | {row['correct']} | "
                     f"{row['wrong']} | {row['status']} | {row['simple_complex_overlap']} |")
    lines.extend(["", "每类至少收集 20 条；最简单 10 条与最复杂 10 条严格不重合。"])
    atomic_text(folder / "summary.md", "\n".join(lines) + "\n")
    print(f"Report written to {folder / 'summary.md'}")
    return summary
