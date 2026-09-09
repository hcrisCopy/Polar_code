"""Pair extreme correct paths with structurally nearest wrong paths and plot them."""

import csv
import os

from tqdm import tqdm

from stage_one.plot_utils import pyplot

from .storage import atomic_json, atomic_text, read_json, run_dir


def _complexity_key(row):
    return (row["length"], len(set(row["path"])), row["evaluation_index"])


def _path_segments(path, depth):
    """Return canonical contiguous (start, end, operation) path segments."""
    from polar.config import OP_EXECUTE, OP_REPEAT, OP_SKIP
    from polar.data import parse_path_to_seg_and_ops

    parsed = parse_path_to_seg_and_ops(
        list(path), depth, max_pack=4, allow_repeat=True
    )
    if parsed is None:
        raise ValueError("Path is not representable by official 2x loop segments")
    boundaries, operations = parsed
    starts = [0] + [index for index in range(1, depth) if boundaries[index] == 1]
    ends = starts[1:] + [depth]
    valid_operations = {OP_SKIP, OP_EXECUTE, OP_REPEAT}
    segments = []
    for start, end in zip(starts, ends):
        operation = operations[start]
        if operation not in valid_operations:
            raise ValueError(f"Missing path operation at layer {start}")
        segments.append((start, end, operation))
    return segments


def _path_operations(path, depth):
    operations = [None] * depth
    for start, end, operation in _path_segments(path, depth):
        operations[start:end] = [operation] * (end - start)
    if any(operation is None for operation in operations):
        raise ValueError("Canonical path segments do not cover every original layer")
    return operations


def _is_official_path(path, depth):
    try:
        _path_segments(path, depth)
        return True
    except ValueError:
        return False


def _levenshtein(left, right):
    """Integer edit distance over the actually executed layer sequence."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_value != right_value),
            ))
        previous = current
    return previous[-1]


def _distance_components(correct, wrong, operation_cache):
    from polar.config import OP_REPEAT, OP_SKIP

    correct_ops = operation_cache[correct["candidate_id"]]
    wrong_ops = operation_cache[wrong["candidate_id"]]
    operation_distance = 0
    for correct_op, wrong_op in zip(correct_ops, wrong_ops):
        if correct_op == wrong_op:
            continue
        operation_distance += (
            2 if {correct_op, wrong_op} == {OP_SKIP, OP_REPEAT} else 1
        )
    return {
        "operation_distance": operation_distance,
        "execution_edit_distance": _levenshtein(correct["path"], wrong["path"]),
        "length_difference": abs(correct["length"] - wrong["length"]),
    }


def _minimum_cost_assignment(costs):
    """Rectangular Hungarian assignment; rows are matched to distinct columns."""
    row_count = len(costs)
    column_count = len(costs[0]) if costs else 0
    if row_count > column_count:
        raise ValueError("One-to-one matching requires at least as many wrong paths")
    if row_count == 0:
        return []

    row_potential = [0] * (row_count + 1)
    column_potential = [0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    predecessor = [0] * (column_count + 1)
    for row in range(1, row_count + 1):
        matched_row[0] = row
        minimum = [float("inf")] * (column_count + 1)
        used = [False] * (column_count + 1)
        column = 0
        while True:
            used[column] = True
            active_row = matched_row[column]
            delta = float("inf")
            next_column = 0
            for candidate in range(1, column_count + 1):
                if used[candidate]:
                    continue
                reduced = (
                    costs[active_row - 1][candidate - 1]
                    - row_potential[active_row]
                    - column_potential[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    predecessor[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(column_count + 1):
                if used[candidate]:
                    row_potential[matched_row[candidate]] += delta
                    column_potential[candidate] -= delta
                elif candidate:
                    minimum[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = predecessor[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break

    assignment = [-1] * row_count
    for column in range(1, column_count + 1):
        if matched_row[column]:
            assignment[matched_row[column] - 1] = column - 1
    if any(column < 0 for column in assignment):
        raise RuntimeError("Hungarian assignment did not match every correct path")
    return assignment


def _pair_paths(correct_rows, wrong_rows, depth):
    """Globally match correct paths using lexicographic structural distance."""
    if not correct_rows:
        return []
    if len(wrong_rows) < len(correct_rows):
        raise ValueError("Not enough distinct wrong paths for one-to-one matching")
    wrong_rows = sorted(wrong_rows, key=lambda row: row["evaluation_index"])
    operation_cache = {
        row["candidate_id"]: _path_operations(row["path"], depth)
        for row in correct_rows + wrong_rows
    }
    components = [
        [_distance_components(correct, wrong, operation_cache) for wrong in wrong_rows]
        for correct in correct_rows
    ]
    pair_count = len(correct_rows)
    maximum_tie = max(row["evaluation_index"] for row in wrong_rows)
    maximum_length = max(
        value["length_difference"] for row in components for value in row
    )
    maximum_edit = max(
        value["execution_edit_distance"] for row in components for value in row
    )
    length_weight = pair_count * maximum_tie + 1
    edit_weight = pair_count * (
        maximum_length * length_weight + maximum_tie
    ) + 1
    operation_weight = pair_count * (
        maximum_edit * edit_weight
        + maximum_length * length_weight
        + maximum_tie
    ) + 1
    costs = []
    for component_row in components:
        costs.append([
            value["operation_distance"] * operation_weight
            + value["execution_edit_distance"] * edit_weight
            + value["length_difference"] * length_weight
            + wrong_rows[column]["evaluation_index"]
            for column, value in enumerate(component_row)
        ])
    assignment = _minimum_cost_assignment(costs)
    return [
        {"correct": correct, "wrong": wrong_rows[column],
         "distance": components[row][column]}
        for row, (correct, column) in enumerate(zip(correct_rows, assignment))
    ]


def _select_correct_extremes(correct_rows, count):
    ordered = sorted(correct_rows, key=_complexity_key)
    simplest = ordered[:count]
    simplest_ids = {row["candidate_id"] for row in simplest}
    most_complex = [
        row for row in reversed(ordered) if row["candidate_id"] not in simplest_ids
    ][:count]
    return simplest, most_complex


def _figure_rows(pairs):
    rows = []
    for pair_index, pair in enumerate(pairs, start=1):
        for role in ("correct", "wrong"):
            rows.append({
                **pair[role],
                "pair_index": pair_index,
                "pair_role": "C" if role == "correct" else "W",
                "pair_distance": pair["distance"],
            })
    return rows


def _save_figure(path, pairs, depth, title, thinking_outcomes=None):
    plt = pyplot()
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
    from polar.config import OP_REPEAT, OP_SKIP

    navy = "#062A3A"
    arrow_blue = "#176B8A"
    input_blue = "#58C3E7"
    output_orange = "#F4A77D"
    recurrent_green = "#D8F0CD"
    skipped_gray = "#8A8A8A"
    correct_green = "#00875A"
    wrong_red = "#C43C39"
    pair_gray = "#F2F4F5"

    rows = _figure_rows(pairs)
    fig, axis = plt.subplots(figsize=(14.5, max(5.2, 0.44 * len(rows) + 2.0)),
                             constrained_layout=True)
    row_positions = list(reversed(range(len(rows))))
    cell_width = 0.72
    cell_height = 0.56
    for pair_offset in range(len(pairs)):
        correct_y = len(rows) - 1 - 2 * pair_offset
        wrong_y = correct_y - 1
        if pair_offset % 2 == 0:
            axis.axhspan(wrong_y - 0.48, correct_y + 0.48,
                         facecolor=pair_gray, edgecolor="none", zorder=-3)
        bracket_x = -2.05
        axis.plot([bracket_x, bracket_x], [wrong_y, correct_y],
                  color=navy, linewidth=1.0, zorder=4)
        axis.plot([bracket_x, bracket_x + 0.16], [wrong_y, wrong_y],
                  color=navy, linewidth=1.0, zorder=4)
        axis.plot([bracket_x, bracket_x + 0.16], [correct_y, correct_y],
                  color=navy, linewidth=1.0, zorder=4)
        if wrong_y > 0:
            axis.axhline(wrong_y - 0.5, color="#C8CDD0", linewidth=0.6, zorder=-2)

    for y, row in zip(row_positions, rows):
        segments = _path_segments(row["path"], depth)
        layer_operations = {}
        for start, end, operation in segments:
            for layer in range(start, end):
                layer_operations[layer] = operation
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

        for start, end, operation in segments:
            if operation != OP_REPEAT:
                continue
            left = start - 0.45
            right = end - 1 + 0.45
            axis.add_patch(FancyBboxPatch(
                (left, y - 0.34), right - left, 0.68,
                boxstyle="round,pad=0.02,rounding_size=0.10",
                facecolor=recurrent_green, edgecolor=navy,
                linewidth=1.35, zorder=1,
            ))
            axis.add_patch(FancyArrowPatch(
                (end - 1 + 0.24, y + 0.16), (start - 0.24, y + 0.16),
                connectionstyle=(
                    "arc3,rad=0.42" if end - start > 1 else "arc3,rad=1.1"
                ),
                arrowstyle="-|>", mutation_scale=6.5,
                color=navy, linewidth=0.8, zorder=4,
            ))
            axis.text(
                (start + end - 1) / 2, y + 0.37, "×2",
                ha="center", va="center", fontsize=8, color=navy,
                fontweight="bold", zorder=5,
                bbox={"boxstyle": "round,pad=0.08", "facecolor": "white",
                      "edgecolor": "none", "alpha": 0.9},
            )

        for layer in range(depth):
            operation = layer_operations[layer]
            skipped = operation == OP_SKIP
            repeated = operation == OP_REPEAT
            axis.add_patch(FancyBboxPatch(
                (layer - cell_width / 2, y - cell_height / 2),
                cell_width, cell_height,
                boxstyle="round,pad=0.01,rounding_size=0.07",
                facecolor="#EDF7E8" if repeated else "white",
                edgecolor=skipped_gray if skipped else navy,
                linewidth=0.85 if skipped else 1.0,
                linestyle=(0, (3, 2)) if skipped else "solid",
                zorder=2,
            ))

    labels = []
    for row in rows:
        label = (
            f"P{row['pair_index']:02d}-{row['pair_role']}  "
            f"{row['candidate_id']}  len={row['length']}"
        )
        if row["pair_role"] == "W":
            distance = row["pair_distance"]
            label += (
                f"  Δop={distance['operation_distance']}"
                f"  edit={distance['execution_edit_distance']}"
                f"  Δlen={distance['length_difference']}"
            )
        if thinking_outcomes is not None:
            label += f"  Think={thinking_outcomes.get(row['candidate_id'], '?')}"
        labels.append(label)
    axis.set_xticks(range(depth))
    axis.set_xticklabels(range(depth), fontsize=8)
    axis.set_yticks(row_positions)
    row_labels = axis.set_yticklabels(labels, fontsize=8)
    axis.set_xlabel(
        "Frozen pretrained layer index"
        + (" (Think: C=correct, W=wrong, T=truncated, ?=pending)"
           if thinking_outcomes is not None else "")
    )
    axis.set_title(title)
    for row_label, row in zip(row_labels, rows):
        row_label.set_color(correct_green if row["correct"] else wrong_red)
        if row["pair_role"] == "C":
            row_label.set_fontweight("bold")
    axis.set_xlim(-2.3, depth + 2.05)
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
                  label="Contiguous loop segment (×2)"),
            Line2D([0], [0], color=correct_green, linewidth=2, label="C: correct"),
            Line2D([0], [0], color=wrong_red, linewidth=2,
                   label="W: nearest wrong counterpart"),
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


def save_thinking_comparison(folder, difficulty, selection, depth, outcomes):
    """Render the original paired paths annotated with thinking-mode outcomes."""
    figure_dir = folder / "figures"
    _save_figure(
        figure_dir / f"dm{difficulty}_simplest_thinking.png",
        selection["simplest_correct_with_nearest_wrong"],
        depth,
        f"DM-{difficulty}: simplest pairs re-evaluated with Qwen3 thinking",
        thinking_outcomes=outcomes,
    )
    _save_figure(
        figure_dir / f"dm{difficulty}_most_complex_thinking.png",
        selection["most_complex_correct_with_nearest_wrong"],
        depth,
        f"DM-{difficulty}: most-complex pairs re-evaluated with Qwen3 thinking",
        thinking_outcomes=outcomes,
    )


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
            "Two disjoint correct-path extremes require target-per-label to be "
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
    for key in tqdm(selected_difficulties, desc="Render paired path figures"):
        question_state = state["questions"][key]
        if not partial and question_state["status"] != "quota_reached":
            raise ValueError(
                f"DM-{key} is not ready for final visualization: "
                f"status={question_state['status']}"
            )
        all_results = question_state["results"]
        results = [
            row for row in all_results
            if _is_official_path(row["path"], config["depth"])
        ]
        official_ids = {row["candidate_id"] for row in results}
        excluded_rows = [
            row for row in all_results if row["candidate_id"] not in official_ids
        ]
        correct_rows = [row for row in results if row["correct"]]
        wrong_rows = [row for row in results if not row["correct"]]
        if partial:
            pair_count = min(
                args.paths_per_label,
                len(correct_rows) // 2,
                len(wrong_rows) // 2,
            )
        else:
            pair_count = args.paths_per_label
            required = 2 * pair_count
            if len(correct_rows) < required or len(wrong_rows) < required:
                raise ValueError(
                    f"DM-{key} has {len(correct_rows)} correct and {len(wrong_rows)} wrong "
                    f"official paths; need {required} of each"
                )

        simplest_correct, complex_correct = _select_correct_extremes(
            correct_rows, pair_count
        )
        selected_correct = simplest_correct + complex_correct
        all_pairs = _pair_paths(selected_correct, wrong_rows, config["depth"])
        pair_by_correct = {
            pair["correct"]["candidate_id"]: pair for pair in all_pairs
        }
        simplest_pairs = [
            pair_by_correct[row["candidate_id"]] for row in simplest_correct
        ]
        complex_pairs = [
            pair_by_correct[row["candidate_id"]] for row in complex_correct
        ]
        if pair_count:
            figure_dir = folder / "figures"
            _save_figure(
                figure_dir / f"dm{key}_simplest.png", simplest_pairs, config["depth"],
                f"DM-{key}: simplest correct paths with nearest wrong counterparts",
            )
            _save_figure(
                figure_dir / f"dm{key}_most_complex.png", complex_pairs,
                config["depth"],
                f"DM-{key}: most complex correct paths with nearest wrong counterparts",
            )

        simple_ids = {
            pair[role]["candidate_id"]
            for pair in simplest_pairs for role in ("correct", "wrong")
        }
        complex_ids = {
            pair[role]["candidate_id"]
            for pair in complex_pairs for role in ("correct", "wrong")
        }
        overlap = len(simple_ids & complex_ids)
        if overlap:
            raise RuntimeError(
                f"DM-{key} paired simplest and most-complex sets unexpectedly overlap"
            )
        selections[key] = {
            "sample_id": question_state["sample_id"],
            "excluded_noncanonical_candidate_ids": [
                row["candidate_id"] for row in excluded_rows
            ],
            "distance_priority": [
                "operation_distance",
                "execution_edit_distance",
                "length_difference",
            ],
            "simplest_correct_with_nearest_wrong": simplest_pairs,
            "most_complex_correct_with_nearest_wrong": complex_pairs,
        }
        report_rows.append({
            "difficulty": int(key),
            "sample_id": question_state["sample_id"],
            "status": question_state["status"],
            "evaluated": len(all_results),
            "official_paths": len(results),
            "excluded_noncanonical": len(excluded_rows),
            "correct": len(correct_rows),
            "wrong": len(wrong_rows),
            "pairs_per_figure": pair_count,
            "cross_figure_overlap": overlap,
            "question": question_by_diff[key]["question"],
            "ground_truth": question_by_diff[key]["gt_ans"],
        })

    suffix = "" if not partial else "_dm" + "-".join(selected_difficulties)
    summary = {
        "schema_version": 2,
        "run_name": args.run_name,
        "reported_difficulties": [int(key) for key in selected_difficulties],
        "partial_report": partial,
        "path_grammar": "contiguous skip/keep/loop segments; loop executes exactly 2x",
        "search_layer_scope": "all original layers",
        "complexity_metric": "correct-path execution length",
        "pairing": "global one-to-one lexicographic minimum-cost assignment",
        "distance_priority": [
            "weighted per-layer operation distance (skip-loop costs 2; others cost 1)",
            "executed-layer-sequence Levenshtein distance",
            "absolute path-length difference",
        ],
        "target_pairs_per_figure": args.paths_per_label,
        "questions": report_rows,
    }
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

    lines = [
        "# 正确路径与最相似错误路径配对展示",
        "",
        "每题分别选择最短和最长的正确路径，并为每条正确路径全局一对一匹配结构最相似的错误路径。",
        "距离依次比较逐层 S/K/L 操作差异、实际执行序列编辑距离和路径长度差。",
        "",
        "| 难度 | 已评估 | 规范路径 | 正确 | 错误 | 每图配对数 | 状态 | 跨图重合 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in report_rows:
        lines.append(
            f"| DM-{row['difficulty']} | {row['evaluated']} | {row['official_paths']} | "
            f"{row['correct']} | {row['wrong']} | {row['pairs_per_figure']} | "
            f"{row['status']} | {row['cross_figure_overlap']} |"
        )
    if partial:
        lines.extend([
            "",
            "这是难度完成后的阶段性快照；不足 20 条正确或错误路径时，两张图按可组成的"
            "不重合配对数等量缩减，不复用错误路径。",
        ])
    else:
        lines.extend([
            "",
            "最终报告每张图包含 10 个正确—错误对；两图中的正确和错误路径均不重复。",
        ])
    summary_path = folder / f"summary{suffix}.md"
    atomic_text(summary_path, "\n".join(lines) + "\n")
    print(f"Report written to {summary_path}")
    return summary
