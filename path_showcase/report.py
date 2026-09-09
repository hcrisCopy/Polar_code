"""Create the two requested path figures for every selected question."""

import csv
import os

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
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch, Patch

    navy = "#062A3A"
    arrow_blue = "#176B8A"
    input_blue = "#58C3E7"
    output_orange = "#F4A77D"
    recurrent_green = "#D8F0CD"
    skipped_gray = "#8A8A8A"
    correct_green = "#00875A"
    wrong_red = "#C43C39"

    fig, axis = plt.subplots(figsize=(14.5, max(4.8, 0.44 * len(rows) + 2.0)),
                             constrained_layout=True)
    row_positions = list(reversed(range(len(rows))))
    cell_width = 0.72
    cell_height = 0.56
    for y, row in zip(row_positions, rows):
        counts = _execution_counts(row["path"], depth)
        axis.annotate(
            "", xy=(depth + 1.35, y), xytext=(-1.35, y),
            arrowprops={"arrowstyle": "-|>", "color": arrow_blue,
                        "linewidth": 0.9, "mutation_scale": 7},
            zorder=0,
        )
        axis.add_patch(FancyBboxPatch(
            (-1.72, y - 0.28), 0.62, 0.56,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            facecolor=input_blue, edgecolor=navy, linewidth=1.1, zorder=2,
        ))
        axis.text(-1.41, y, "x", ha="center", va="center", fontsize=8,
                  color=navy, fontweight="bold", zorder=3)
        axis.add_patch(FancyBboxPatch(
            (depth + 1.08, y - 0.28), 0.72, 0.56,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            facecolor=output_orange, edgecolor=navy, linewidth=1.1, zorder=2,
        ))
        axis.text(depth + 1.44, y, "out", ha="center", va="center",
                  fontsize=8, color=navy, fontweight="bold", zorder=3)

        for layer, count in enumerate(counts):
            skipped = count == 0
            repeated = count > 1
            cell = FancyBboxPatch(
                (layer - cell_width / 2, y - cell_height / 2),
                cell_width, cell_height,
                boxstyle="round,pad=0.01,rounding_size=0.07",
                facecolor=recurrent_green if repeated else "white",
                edgecolor=skipped_gray if skipped else navy,
                linewidth=0.85 if skipped else 1.0,
                linestyle=(0, (3, 2)) if skipped else "solid",
                zorder=2,
            )
            axis.add_patch(cell)
            if repeated:
                axis.text(layer, y, f"×{count}", ha="center", va="center",
                          fontsize=8, color=navy, fontweight="bold", zorder=3)

    axis.set_xticks(range(depth))
    axis.set_xticklabels(range(depth), fontsize=8)
    labels = []
    for row in rows:
        verdict = "C" if row["correct"] else "W"
        labels.append(f"{verdict}  {row['candidate_id']}  len={row['length']}  "
                      f"found={row['evaluation_index'] + 1}")
    axis.set_yticks(row_positions)
    row_labels = axis.set_yticklabels(labels, fontsize=8)
    axis.set_xlabel("Frozen pretrained layer index")
    axis.set_title(title)
    split = sum(row["correct"] for row in rows)
    if 0 < split < len(rows):
        separator = len(rows) - split - 0.5
        axis.axhline(separator, color=navy, linewidth=1.0, linestyle=(0, (5, 3)))
    for row_label, row in zip(row_labels, rows):
        color = correct_green if row["correct"] else wrong_red
        row_label.set_color(color)
    axis.set_xlim(-2.0, depth + 2.05)
    axis.set_ylim(-0.75, len(rows) - 0.25)
    axis.tick_params(axis="both", length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.legend(
        handles=[
            Patch(facecolor="white", edgecolor=navy, label="Keep (execute once)"),
            Patch(facecolor="white", edgecolor=skipped_gray, linestyle=(0, (3, 2)),
                  label="Skip"),
            Patch(facecolor=recurrent_green, edgecolor=navy,
                  label="Recurrent (×n executions)"),
            Line2D([0], [0], color=correct_green, linewidth=2, label="C: correct"),
            Line2D([0], [0], color=wrong_red, linewidth=2, label="W: wrong"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=5,
        frameon=False, fontsize=8,
    )

    stem = path.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, figure_format in ((".png", "png"), (".svg", "svg"), (".pdf", "pdf")):
        target = stem.with_suffix(suffix)
        pending = target.with_suffix(target.suffix + ".pending")
        save_options = {"dpi": 180} if figure_format == "png" else {}
        fig.savefig(pending, format=figure_format, **save_options)
        with pending.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(pending, target)
    plt.close(fig)


def build_report(args):
    folder = run_dir(args.run_name)
    manifest = read_json(folder / "questions.json")
    config = read_json(folder / "search_config.json")
    state = read_json(folder / "search_state.json")
    if state["config_id"] != config["config_id"]:
        raise ValueError("Search state/config mismatch")
    partial = args.difficulties is not None
    if not partial and config["target_per_label"] < 2 * args.paths_per_label:
        raise ValueError(
            "Distinct simplest/most-complex figures require target-per-label to be "
            "at least twice paths-per-label"
        )
    question_by_diff = {str(row["difficulty"]): row for row in manifest["questions"]}
    selected_difficulties = (
        [str(value) for value in args.difficulties]
        if args.difficulties is not None
        else sorted(state["questions"], key=int)
    )
    missing = [key for key in selected_difficulties if key not in state["questions"]]
    if missing:
        raise ValueError(f"Difficulties absent from this run: {missing}")
    report_rows = []
    selections = {}
    for key in tqdm(selected_difficulties, desc="Render path figures"):
        question_state = state["questions"][key]
        if not partial and question_state["status"] != "quota_reached":
            raise ValueError(
                f"DM-{key} is not ready for visualization: "
                f"status={question_state['status']}"
            )
        results = question_state["results"]
        if not results:
            raise ValueError(f"DM-{key} has no evaluated paths to visualize")
        correct_count = sum(row["correct"] for row in results)
        error_count = len(results) - correct_count
        required_count = 2 * args.paths_per_label
        if not partial and min(correct_count, error_count) < required_count:
            raise ValueError(
                f"DM-{key} has only {correct_count} correct and {error_count} wrong paths; "
                f"need {required_count} of each to render disjoint extremes"
            )
        simple = _select(results, largest=False, count=args.paths_per_label)
        complex_rows = _select(results, largest=True, count=args.paths_per_label)
        figure_dir = folder / "figures"
        _save_figure(figure_dir / f"dm{key}_simplest.png", simple, config["depth"],
                     f"DM-{key}: simplest correct and wrong paths")
        _save_figure(figure_dir / f"dm{key}_most_complex.png", complex_rows,
                     config["depth"], f"DM-{key}: most complex correct and wrong paths")
        simple_ids = {row["candidate_id"] for row in simple}
        complex_ids = {row["candidate_id"] for row in complex_rows}
        overlap = len(simple_ids & complex_ids)
        if overlap and not partial:
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

    suffix = "" if not partial else "_dm" + "-".join(selected_difficulties)
    summary = {"schema_version": 1, "run_name": args.run_name,
               "reported_difficulties": [int(key) for key in selected_difficulties],
               "partial_report": partial,
               "complexity_metric": "execution path length",
               "paths_per_label_per_figure": args.paths_per_label,
               "simplest_and_most_complex_sets_disjoint": all(
                   row["simple_complex_overlap"] == 0 for row in report_rows
               ),
               "questions": report_rows}
    atomic_json(folder / f"report{suffix}.json", summary)
    atomic_json(folder / f"path_selections{suffix}.json", selections)
    csv_path = folder / f"report{suffix}.csv"
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
    if partial:
        lines.extend([
            "",
            "这是基于当前已评估路径生成的阶段性快照；每类不足 10 条时有多少画多少，"
            "不足 20 条时最简单与最复杂集合允许重合。",
        ])
    else:
        lines.extend([
            "", "每类至少收集 20 条；最简单 10 条与最复杂 10 条严格不重合。"
        ])
    summary_path = folder / f"summary{suffix}.md"
    atomic_text(summary_path, "\n".join(lines) + "\n")
    print(f"Report written to {summary_path}")
    return summary
