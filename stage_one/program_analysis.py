"""Mine train-only programs and report held-out universal-program robustness."""

from collections import Counter, defaultdict
import csv
import io
import json
import math
import random

from tqdm import tqdm

from .storage import (atomic_json, atomic_text, clean_stage, digest, output_path,
                      read_json, recover_pending, stage_dir)
from .validate import load_search


ACTIONS = ("skip", "keep", "loop")


def layer_actions(path, depth):
    """Map a predictor-representable path to per-layer execution counts.

    A repeated multi-layer segment keeps its true order in ``path``; this
    summary deliberately records only whether each original layer ran 0/1/2x.
    """
    counts = Counter(path)
    if any(index not in range(depth) or count not in (1, 2)
           for index, count in counts.items()):
        raise ValueError("Program is outside skip/keep/single-loop analysis vocabulary")
    return ["skip" if counts[i] == 0 else "keep" if counts[i] == 1 else "loop"
            for i in range(depth)]


def actions_path(actions):
    path = []
    for index, action in enumerate(actions):
        path.extend([index] * {"skip": 0, "keep": 1, "loop": 2}[action])
    if not path:
        raise ValueError("Consensus program cannot skip every layer")
    return path


def add_candidate(pool, path, source):
    key = tuple(path)
    if key not in pool:
        pool[key] = {"path": list(path), "sources": []}
    if source not in pool[key]["sources"]:
        pool[key]["sources"].append(source)


def validate_candidate_payload(payload):
    """Reject corrupt, mixed, duplicate, or unrepresentable candidate sets."""
    required = {"schema_version", "run_name", "search_config_id", "manifest_id",
                "depth", "max_length", "discovery_split", "train_questions", "args",
                "candidates", "candidate_set_id"}
    if not required.issubset(payload):
        raise ValueError("Candidate set lacks required fields")
    unsigned = {key: value for key, value in payload.items() if key != "candidate_set_id"}
    if digest(unsigned) != payload["candidate_set_id"]:
        raise ValueError("Candidate set checksum mismatch")
    if payload["schema_version"] != 1 or payload["discovery_split"] != "train":
        raise ValueError("Unsupported candidate schema or discovery split")
    depth = payload["depth"]
    if type(depth) is not int or depth <= 0:
        raise ValueError("Candidate depth must be a positive integer")
    if type(payload["max_length"]) is not int or payload["max_length"] < depth:
        raise ValueError("Candidate maximum length is invalid")
    seen_ids, seen_paths = set(), set()
    from polar.data import parse_path_to_seg_and_ops
    for row in payload["candidates"]:
        path = row.get("path")
        if (not isinstance(path, list) or not path or
                any(type(index) is not int for index in path)):
            raise ValueError("Every candidate path must be a nonempty integer list")
        if len(path) > payload["max_length"]:
            raise ValueError("Candidate exceeds the MCTS program-length limit")
        layer_actions(path, depth)
        if parse_path_to_seg_and_ops(path, depth, max_pack=4, allow_repeat=True) is None:
            raise ValueError("Candidate path is not representable by the official predictor")
        candidate_id = row.get("candidate_id")
        if candidate_id != digest(path)[:16]:
            raise ValueError("Candidate ID does not match its path")
        path_key = tuple(path)
        if candidate_id in seen_ids or path_key in seen_paths:
            raise ValueError("Duplicate candidate ID or path")
        seen_ids.add(candidate_id)
        seen_paths.add(path_key)
        if not row.get("sources") or not all(isinstance(value, str) for value in row["sources"]):
            raise ValueError("Every candidate requires a nonempty source list")
        if row.get("length") != len(path):
            raise ValueError("Candidate length metadata does not match its path")
    if not seen_ids:
        raise ValueError("Candidate set is empty")
    maximum = payload["args"].get("max_candidates")
    if type(maximum) is not int or len(seen_ids) > maximum:
        raise ValueError("Candidate count exceeds its recorded limit")
    return payload


def _write_csv(path, rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(output_path(path), stream.getvalue())


def mine_programs(args):
    """Use only train traces to freeze exact programs for held-out evaluation."""
    folder = stage_dir(args.run_name, "program_mining")
    if args.clean:
        clean_stage(args.run_name, "program_mining")
    recovered = recover_pending(folder)
    manifest, config, records = load_search(args.run_name)
    train = [row for row in records if row["split"] == "train"]
    if not train:
        raise ValueError("No train questions are available for candidate mining")
    depth = config["depth"]
    full = tuple(range(depth))

    path_observations = defaultdict(lambda: {"valid": set(), "invalid": set(), "difficulties": Counter()})
    # Each question contributes total mass one per outcome, preventing questions
    # with many paths from dominating the layer propensity visualization.
    weights = defaultdict(float)
    for row in tqdm(train, desc="Mine train-only program evidence"):
        for label, outcome in (("final_valid_transitions", "valid"),
                               ("final_invalid_transitions", "invalid")):
            paths = row[label]
            weight = 1.0 / max(1, len(paths))
            for path in paths:
                key = tuple(path)
                path_observations[key][outcome].add(row["sample_id"])
                if outcome == "valid":
                    path_observations[key]["difficulties"][row["difficulty"]] += 1
                for layer, action in enumerate(layer_actions(path, depth)):
                    weights[(row["difficulty"], outcome, layer, action)] += weight
                    weights[("all", outcome, layer, action)] += weight

    layer_rows = []
    difficulty_groups = sorted({row["difficulty"] for row in train})
    groups = difficulty_groups + ["all"]
    for group in groups:
        for layer in range(depth):
            for action in ACTIONS:
                valid = weights[(group, "valid", layer, action)]
                invalid = weights[(group, "invalid", layer, action)]
                valid_total = sum(weights[(group, "valid", layer, a)] for a in ACTIONS)
                invalid_total = sum(weights[(group, "invalid", layer, a)] for a in ACTIONS)
                valid_fraction = valid / valid_total if valid_total else 0.0
                invalid_fraction = invalid / invalid_total if invalid_total else 0.0
                # Descriptive log propensity, not a causal layer effect.
                lift = math.log((valid_fraction + args.smoothing) /
                                (invalid_fraction + args.smoothing))
                layer_rows.append({"group": group, "layer": layer, "action": action,
                                   "valid_weight": valid, "invalid_weight": invalid,
                                   "valid_fraction": valid_fraction,
                                   "invalid_fraction": invalid_fraction,
                                   "log_propensity_lift": lift})

    row_lookup = {(row["group"], row["layer"], row["action"]): row
                  for row in layer_rows}
    robust_hypotheses = []
    for action in ("skip", "loop"):
        for layer in range(depth):
            group_deltas = {
                str(group): (row_lookup[(group, layer, action)]["log_propensity_lift"] -
                             row_lookup[(group, layer, "keep")]["log_propensity_lift"])
                for group in difficulty_groups
            }
            deltas = list(group_deltas.values())
            robust_hypotheses.append({
                "action": action,
                "layer": layer,
                "worst_difficulty_lift_over_keep": min(deltas),
                "macro_lift_over_keep": sum(deltas) / len(deltas),
                "all_train_lift_over_keep": (
                    row_lookup[("all", layer, action)]["log_propensity_lift"] -
                    row_lookup[("all", layer, "keep")]["log_propensity_lift"]),
                "by_difficulty_lift_over_keep": group_deltas,
            })
    robust_hypotheses.sort(key=lambda row: (
        row["action"], -row["worst_difficulty_lift_over_keep"],
        -row["macro_lift_over_keep"], row["layer"]))
    for action in ("skip", "loop"):
        ranked = [row for row in robust_hypotheses if row["action"] == action]
        for rank, row in enumerate(ranked, start=1):
            row["rank_within_action"] = rank

    if args.max_candidates < 2 * args.top_layers_per_action:
        raise ValueError("--max-candidates must cover both skip/loop single-layer hypotheses")
    if args.top_layers_per_action > depth:
        raise ValueError("--top-layers-per-action cannot exceed model depth")

    candidates = {}
    for row in robust_hypotheses:
        if row["rank_within_action"] > args.top_layers_per_action:
            continue
        actions = ["keep"] * depth
        actions[row["layer"]] = row["action"]
        add_candidate(candidates, actions_path(actions),
                      f"single_{row['action']}_layer_{row['layer']}")

    ranked_exact = []
    for path, observed in path_observations.items():
        if path == full or not observed["valid"]:
            continue
        valid_support = len(observed["valid"])
        observed_support = len(observed["valid"] | observed["invalid"])
        ranked_exact.append((path, valid_support, observed_support,
                             valid_support / observed_support, observed["difficulties"]))
    ranked_exact.sort(key=lambda item: (-item[1], -item[3], len(item[0]), item[0]))
    # Layerwise consensus candidates are hypotheses only. Progressive edit
    # budgets expose whether a small stable set of changes transfers.
    for group in groups:
        row_map = {(r["layer"], r["action"]): r for r in layer_rows if r["group"] == group}
        changes = []
        for layer in range(depth):
            keep_lift = row_map[(layer, "keep")]["log_propensity_lift"]
            options = [(row_map[(layer, action)]["log_propensity_lift"] - keep_lift, action)
                       for action in ("skip", "loop")]
            gain, action = max(options)
            if gain > 0:
                changes.append((gain, layer, action))
        changes.sort(key=lambda item: (-item[0], item[1], item[2]))
        for budget in range(1, min(args.max_consensus_edits, len(changes)) + 1):
            actions = ["keep"] * depth
            for _, layer, action in changes[:budget]:
                actions[layer] = action
            if all(action == "skip" for action in actions):
                continue
            consensus_path = actions_path(actions)
            if len(consensus_path) <= config["max_length"]:
                add_candidate(candidates, consensus_path, f"consensus_{group}_top{budget}")

    for path, support, _, _, _ in ranked_exact:
        if support >= args.min_train_support:
            add_candidate(candidates, path, "exact_train_support")

    # Fill remaining capacity with best-supported exact programs even if the
    # requested support threshold is too strict for exact cross-query matches.
    for path, _, _, _, _ in ranked_exact:
        if len(candidates) >= args.max_candidates:
            break
        add_candidate(candidates, path, "exact_train_fallback")

    candidate_rows = []
    for path, candidate in candidates.items():
        observed = path_observations.get(path, {"valid": set(), "invalid": set(), "difficulties": Counter()})
        valid_support = len(observed["valid"])
        observed_support = len(observed["valid"] | observed["invalid"])
        candidate_rows.append({**candidate, "candidate_id": digest(list(path))[:16],
                               "length": len(path), "depth_ratio": len(path) / depth,
                               "train_valid_support": valid_support,
                               "train_observed_support": observed_support,
                               "train_observed_accuracy": valid_support / observed_support if observed_support else None,
                               "train_valid_support_by_difficulty": dict(observed["difficulties"])})
    ranked_rows = sorted(candidate_rows, key=lambda row: (
        -row["train_valid_support"], -(row["train_observed_accuracy"] or -1),
        row["length"], row["candidate_id"]))
    single_rows = [row for row in ranked_rows
                   if any(source.startswith("single_") for source in row["sources"])]
    consensus_rows = [row for row in ranked_rows
                      if any(source.startswith("consensus_") for source in row["sources"])]
    candidate_rows = single_rows[:args.max_candidates]
    selected_ids = {row["candidate_id"] for row in candidate_rows}
    for row in consensus_rows + ranked_rows:
        if len(candidate_rows) >= args.max_candidates:
            break
        if row["candidate_id"] not in selected_ids:
            candidate_rows.append(row)
            selected_ids.add(row["candidate_id"])
    candidate_rows.sort(key=lambda row: (-row["train_valid_support"],
                                         row["length"], row["candidate_id"]))
    if not candidate_rows:
        raise ValueError("No non-baseline program candidates were mined")

    payload = {"schema_version": 1, "run_name": args.run_name,
               "search_config_id": config["config_id"],
               "manifest_id": manifest["manifest_id"], "depth": depth,
               "max_length": config["max_length"],
               "discovery_split": "train", "train_questions": len(train),
               "args": {key: getattr(args, key) for key in
                        ("max_candidates", "min_train_support", "max_consensus_edits",
                         "top_layers_per_action", "smoothing")},
               "candidates": candidate_rows}
    payload["candidate_set_id"] = digest(payload)
    validate_candidate_payload(payload)
    candidate_path = folder / "candidates.json"
    if candidate_path.exists() and read_json(candidate_path) != payload:
        raise ValueError("Candidate mining inputs changed; use --clean or a new run-name")
    atomic_json(candidate_path, payload)
    atomic_json(folder / "layer_statistics.json", {"candidate_set_id": payload["candidate_set_id"],
                                                    "rows": layer_rows})
    _write_csv(folder / "layer_statistics.csv", layer_rows, list(layer_rows[0]))
    atomic_json(folder / "robust_layer_hypotheses.json", {
        "candidate_set_id": payload["candidate_set_id"],
        "warning": "Train MCTS propensity ranking is descriptive; held-out fixed-path evaluation is required",
        "rows": robust_hypotheses,
    })
    hypothesis_csv = [{**{key: row[key] for key in (
        "action", "layer", "rank_within_action", "worst_difficulty_lift_over_keep",
        "macro_lift_over_keep", "all_train_lift_over_keep")},
                       "by_difficulty_lift_over_keep": json.dumps(
                           row["by_difficulty_lift_over_keep"], sort_keys=True)}
                      for row in robust_hypotheses]
    _write_csv(folder / "robust_layer_hypotheses.csv", hypothesis_csv,
               list(hypothesis_csv[0]))
    _write_csv(folder / "candidates.csv", [
        {**{k: row[k] for k in ("candidate_id", "length", "depth_ratio", "train_valid_support",
                                "train_observed_support", "train_observed_accuracy")},
         "sources": ";".join(row["sources"]), "path": json.dumps(row["path"])}
        for row in candidate_rows],
        ["candidate_id", "length", "depth_ratio", "train_valid_support", "train_observed_support",
         "train_observed_accuracy", "sources", "path"])
    plot_layer_heatmaps(folder, layer_rows, groups, depth)
    atomic_text(folder / "FIGURE_CAPTIONS.md", "\n".join([
        "# Figure captions", "",
        "- `layer_action_heatmaps`: Per-question-normalized fractions of train-valid MCTS paths "
        "that skip or loop each layer, split by DART-Math difficulty. These adaptive-search "
        "frequencies generate hypotheses and are not causal layer effects.",
    ]) + "\n")
    atomic_json(folder / "summary.json", {"candidate_set_id": payload["candidate_set_id"],
                                          "train_questions": len(train),
                                          "candidate_count": len(candidate_rows),
                                          "recovery": recovered,
                                          "warning": "MCTS propensities are adaptive-search descriptions, not causal effects"})
    print(f"Mined {len(candidate_rows)} frozen candidates from {len(train)} train questions")
    return payload


def plot_layer_heatmaps(folder, rows, groups, depth):
    from .plot_utils import pyplot, save_vector
    plt = pyplot()
    display = [f"DM-{g}" if g != "all" else "All train" for g in groups]
    lookup = {(r["group"], r["layer"], r["action"]): r["valid_fraction"] for r in rows}
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 2.8), constrained_layout=True)
    for axis, action, label in zip(axes, ("skip", "loop"), ("Skip propensity", "Loop propensity")):
        matrix = [[lookup[(group, layer, action)] for layer in range(depth)] for group in groups]
        image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                            interpolation="nearest")
        axis.set_ylabel("Difficulty")
        axis.set_yticks(range(len(groups)), display)
        axis.set_xticks(range(0, depth, 2))
        axis.set_xticklabels(range(0, depth, 2))
        axis.text(0.005, 1.03, label, transform=axis.transAxes, fontsize=8)
        fig.colorbar(image, ax=axis, label="Train-valid path fraction", pad=0.01, fraction=0.025)
    axes[-1].set_xlabel("Layer index")
    save_vector(fig, folder / "layer_action_heatmaps")
    plt.close(fig)


def _metrics(records, candidate_id, split, difficulties):
    rows = [row for row in records if row["split"] == split]
    by_group = {}
    for difficulty in difficulties:
        group = [row for row in rows if row["difficulty"] == difficulty]
        if not group:
            continue
        baseline = sum(row["baseline_score"] for row in group) / len(group)
        accuracy = sum(row["scores"][candidate_id] for row in group) / len(group)
        by_group[str(difficulty)] = {"questions": len(group), "baseline_accuracy": baseline,
                                     "program_accuracy": accuracy, "gain": accuracy - baseline}
    baseline = sum(row["baseline_score"] for row in rows) / len(rows)
    accuracy = sum(row["scores"][candidate_id] for row in rows) / len(rows)
    w2c = sum(row["baseline_score"] == 0 and row["scores"][candidate_id] == 1 for row in rows)
    c2w = sum(row["baseline_score"] == 1 and row["scores"][candidate_id] == 0 for row in rows)
    gains = [item["gain"] for item in by_group.values()]
    return {"questions": len(rows), "baseline_accuracy": baseline, "program_accuracy": accuracy,
            "gain": accuracy - baseline, "wrong_to_correct": w2c, "correct_to_wrong": c2w,
            "net_corrections": w2c - c2w, "macro_gain": sum(gains) / len(gains),
            "worst_group_gain": min(gains), "by_difficulty": by_group}


def _bootstrap_gain(records, candidate_id, split, samples, seed):
    rows = [row for row in records if row["split"] == split]
    if not rows or samples <= 0:
        return None
    rng = random.Random(seed)
    deltas = [row["scores"][candidate_id] - row["baseline_score"] for row in rows]
    estimates = []
    for _ in tqdm(range(samples), desc=f"Bootstrap {split} paired gain", leave=False):
        estimates.append(sum(rng.choice(deltas) for _ in deltas) / len(deltas))
    estimates.sort()
    low = estimates[int(0.025 * (samples - 1))]
    high = estimates[int(0.975 * (samples - 1))]
    return [low, high]


def report_programs(args):
    from .universal_eval import load_universal_evaluation
    folder = stage_dir(args.run_name, "program_report")
    if args.clean:
        clean_stage(args.run_name, "program_report")
    recovered = recover_pending(folder)
    eval_config, records = load_universal_evaluation(args.run_name, require_complete=True)
    candidate_run_name = eval_config.get("candidate_run_name", args.run_name)
    candidates = read_json(
        stage_dir(candidate_run_name, "program_mining") / "candidates.json")
    validate_candidate_payload(candidates)
    candidate_map = {row["candidate_id"]: row for row in candidates["candidates"]}
    split_groups = eval_config["split_difficulty_counts"]
    difficulties = [difficulty for difficulty in eval_config["difficulties"]
                    if split_groups["validation"].get(str(difficulty), 0) > 0 and
                    split_groups["test"].get(str(difficulty), 0) > 0]
    if not difficulties:
        raise ValueError("Validation and test have no shared difficulty group")
    metrics = {}
    for candidate_id in tqdm(candidate_map, desc="Aggregate fixed-program metrics"):
        metrics[candidate_id] = {
            split: _metrics(records, candidate_id, split, difficulties)
            for split in ("validation", "test")
        }
    # Test values never participate in this ordering.
    selected_id = max(candidate_map, key=lambda cid: (
        metrics[cid]["validation"]["worst_group_gain"],
        metrics[cid]["validation"]["macro_gain"],
        metrics[cid]["validation"]["program_accuracy"],
        -candidate_map[cid]["length"], cid))
    selected = candidate_map[selected_id]
    per_difficulty_selected = {}
    for difficulty in difficulties:
        key = str(difficulty)
        difficulty_id = max(candidate_map, key=lambda cid: (
            metrics[cid]["validation"]["by_difficulty"][key]["gain"],
            metrics[cid]["validation"]["by_difficulty"][key]["program_accuracy"],
            -candidate_map[cid]["length"], cid))
        per_difficulty_selected[key] = {
            "candidate": candidate_map[difficulty_id],
            "validation": metrics[difficulty_id]["validation"]["by_difficulty"][key],
            "test": metrics[difficulty_id]["test"]["by_difficulty"][key],
        }
    intervals = {split: _bootstrap_gain(records, selected_id, split, args.bootstrap_samples,
                                        args.bootstrap_seed + offset)
                 for offset, split in enumerate(("validation", "test"))}
    validation = metrics[selected_id]["validation"]
    test = metrics[selected_id]["test"]
    verdict = {
        "validation_robust": (validation["gain"] > 0 and
                              validation["worst_group_gain"] >= 0),
        "held_out_robust": (validation["gain"] > 0 and
                            validation["worst_group_gain"] >= 0 and
                            test["gain"] > 0 and test["worst_group_gain"] >= 0),
        "test_paired_gain_ci_excludes_zero": intervals["test"][0] > 0,
    }
    result = {"schema_version": 1, "run_name": args.run_name,
              "candidate_run_name": candidate_run_name,
              "selection_split": "validation", "held_out_report_split": "test",
              "selection_objective": ["validation worst_group_gain", "validation macro_gain",
                                      "validation program_accuracy", "shorter length"],
              "selected_candidate": selected, "selected_metrics": metrics[selected_id],
              "per_difficulty_selected_on_validation": per_difficulty_selected,
              "paired_gain_95ci": intervals, "verdict": verdict,
              "all_candidate_metrics": metrics,
              "candidate_set_id": candidates["candidate_set_id"],
              "evaluation_config_id": eval_config["evaluation_config_id"],
              "recovery": recovered,
              "interpretation_limit": "One run estimates transfer across DART-Math difficulty groups; repeat seeds/models for stronger robustness claims"}
    atomic_json(folder / "report.json", result)
    metric_rows = []
    for cid, split_metrics in metrics.items():
        for split, values in split_metrics.items():
            metric_rows.append({"candidate_id": cid, "split": split,
                                "length": candidate_map[cid]["length"],
                                "sources": ";".join(candidate_map[cid]["sources"]),
                                **{k: values[k] for k in ("questions", "baseline_accuracy", "program_accuracy",
                                                          "gain", "macro_gain", "worst_group_gain",
                                                          "wrong_to_correct", "correct_to_wrong", "net_corrections")}})
    _write_csv(folder / "candidate_metrics.csv", metric_rows, list(metric_rows[0]))
    plot_program_report(folder, candidates, metrics, selected_id, difficulties,
                        per_difficulty_selected)
    actions = layer_actions(selected["path"], candidates["depth"])
    action_summary = {action: [i for i, value in enumerate(actions) if value == action]
                      for action in ACTIONS}
    single_layer_rows = []
    for candidate_id, candidate in candidate_map.items():
        actions = layer_actions(candidate["path"], candidates["depth"])
        edits = [(layer, action) for layer, action in enumerate(actions) if action != "keep"]
        if len(edits) == 1:
            layer, action = edits[0]
            single_layer_rows.append({
                "candidate_id": candidate_id, "layer": layer, "action": action,
                "validation_worst_group_gain": metrics[candidate_id]["validation"]["worst_group_gain"],
                "validation_macro_gain": metrics[candidate_id]["validation"]["macro_gain"],
                "test_gain": metrics[candidate_id]["test"]["gain"],
                "test_worst_group_gain": metrics[candidate_id]["test"]["worst_group_gain"],
            })
    single_layer_rows.sort(key=lambda row: (
        -row["validation_worst_group_gain"], -row["validation_macro_gain"],
        row["action"], row["layer"]))
    atomic_json(folder / "single_layer_transfer.json", {
        "selection_rule": "ranked by validation only; test values are reports",
        "rows": single_layer_rows,
    })
    _write_csv(folder / "single_layer_transfer.csv", single_layer_rows,
               list(single_layer_rows[0]) if single_layer_rows else [
                   "candidate_id", "layer", "action", "validation_worst_group_gain",
                   "validation_macro_gain", "test_gain", "test_worst_group_gain"])
    selected_rows = []
    selected_sets = [("universal", selected, validation, test)] + [
        (f"DM-{difficulty}", values["candidate"], values["validation"], values["test"])
        for difficulty, values in per_difficulty_selected.items()
    ]
    for task, candidate, validation_values, test_values in selected_sets:
        candidate_actions = layer_actions(candidate["path"], candidates["depth"])
        selected_rows.append({
            "task": task, "candidate_id": candidate["candidate_id"],
            "length": candidate["length"],
            "skip_layers": json.dumps([i for i, value in enumerate(candidate_actions)
                                        if value == "skip"]),
            "loop_layers": json.dumps([i for i, value in enumerate(candidate_actions)
                                        if value == "loop"]),
            "validation_gain": validation_values["gain"],
            "test_gain": test_values["gain"],
            "path": json.dumps(candidate["path"]),
        })
    _write_csv(folder / "selected_programs.csv", selected_rows, list(selected_rows[0]))
    lines = ["# 通用 Skip-Loop 结构评估", "",
             f"候选仅由 train MCTS 轨迹生成；最终候选 `{selected_id}` 仅按 validation 指标选择。",
             ("结论：本次划分上找到满足判据的通用候选。" if verdict["held_out_robust"] else
              "结论：本次划分上未找到同时满足 validation 与 test 跨难度判据的通用候选。"),
             ("统计证据：test 配对增益的 95% bootstrap CI 排除 0。" if
              verdict["test_paired_gain_ci_excludes_zero"] else
              "统计证据：test 配对增益的 95% bootstrap CI 未排除 0。"),
             f"固定路径：`{selected['path']}`", f"Skip 层：`{action_summary['skip']}`",
             f"Loop 层：`{action_summary['loop']}`", "",
             f"Validation：{validation['program_accuracy']:.4f}，相对 baseline {validation['gain']:+.4f}，"
             f"最差难度增益 {validation['worst_group_gain']:+.4f}。",
             f"Test：{test['program_accuracy']:.4f}，相对 baseline {test['gain']:+.4f}，"
             f"最差难度增益 {test['worst_group_gain']:+.4f}，95% paired bootstrap CI {intervals['test']}。",
             "", "## 各任务在 validation 上的最优路径", ""]
    for difficulty, values in per_difficulty_selected.items():
        candidate_actions = layer_actions(values["candidate"]["path"], candidates["depth"])
        skip_layers = [i for i, value in enumerate(candidate_actions) if value == "skip"]
        loop_layers = [i for i, value in enumerate(candidate_actions) if value == "loop"]
        lines.append(
            f"- DM-{difficulty}：候选 `{values['candidate']['candidate_id']}`，skip {skip_layers}，"
            f"loop {loop_layers}；validation gain {values['validation']['gain']:+.4f}，"
            f"同难度 test gain {values['test']['gain']:+.4f}。")
    lines.extend(["", "## 单层操作的迁移排序", "",
                  "下面仅按 validation 最差难度增益排序；test 数值只作一次最终报告，不参与选择。"])
    for row in single_layer_rows[:10]:
        lines.append(
            f"- {row['action']} layer {row['layer']}：validation worst "
            f"{row['validation_worst_group_gain']:+.4f}，test worst "
            f"{row['test_worst_group_gain']:+.4f}。")
    lines.extend(["", "注意：层频率来自自适应 MCTS 轨迹，只是描述性证据；"
                  "固定路径的 held-out test 结果才用于判断迁移。单次数据划分只能支持本次划分上的迁移结论，"
                  "需要重复种子和模型才能称为更强的鲁棒规律。"])
    atomic_text(folder / "summary.md", "\n".join(lines) + "\n")
    captions = [
        "# Figure captions", "",
        "- `selected_program_accuracy`: Full-depth baseline and the validation-selected fixed "
        "program on each DART-Math difficulty. Candidate generation uses train only; the test "
        "panel is never used for selection.",
        "- `accuracy_depth_tradeoff`: Validation macro accuracy gain versus executed-depth ratio "
        "for every frozen candidate; the star marks the validation-selected program.",
        "- `all_candidate_program_layers`: All frozen candidate paths. Rows are ordered using "
        "validation metrics only; S/K/L mean skip/keep/loop once. The right panel reports "
        "validation and held-out test accuracy gains, and the star marks the selected program.",
        "- `selected_program_layers`: Exact per-layer action of the selected fixed program. "
        "S/K/L mean skip/keep/loop once.",
        "- `per_difficulty_program_layers`: Per-layer actions for the universal candidate and "
        "each difficulty-specific candidate, all selected using validation only.",
        "- `single_layer_transfer`: For train-ranked single-layer hypotheses, validation and test "
        "worst-difficulty gains relative to the full-depth baseline. Test values "
        "are reports, not selection criteria.",
    ]
    atomic_text(folder / "FIGURE_CAPTIONS.md", "\n".join(captions) + "\n")
    print(f"Selected validation-robust candidate {selected_id}; test gain {test['gain']:+.4f}")
    return result


def plot_program_report(folder, candidates, metrics, selected_id, difficulties,
                        per_difficulty_selected):
    from .plot_utils import (BASELINE_COLOR, KEEP_COLOR, LOOP_COLOR, PROGRAM_COLOR,
                             SKIP_COLOR, pyplot, save_vector)
    plt = pyplot()
    selected = next(row for row in candidates["candidates"] if row["candidate_id"] == selected_id)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5), sharey=True, constrained_layout=True)
    for axis, split in zip(axes, ("validation", "test")):
        values = metrics[selected_id][split]["by_difficulty"]
        groups = [d for d in difficulties if str(d) in values]
        x = list(range(len(groups)))
        baseline = [values[str(d)]["baseline_accuracy"] for d in groups]
        program = [values[str(d)]["program_accuracy"] for d in groups]
        axis.bar([i - 0.19 for i in x], baseline, width=0.38, color=BASELINE_COLOR,
                 edgecolor="black", linewidth=0.5, hatch="//", label="Full-depth baseline")
        axis.bar([i + 0.19 for i in x], program, width=0.38, color=PROGRAM_COLOR,
                 edgecolor="black", linewidth=0.5, label="Selected fixed program")
        axis.axhline(0, color="black", linewidth=0.6)
        axis.set_xticks(x, [f"DM-{d}" for d in groups])
        axis.set_xlabel("DART-Math difficulty")
        axis.set_ylim(0, 1)
        axis.text(0.02, 0.96, split.capitalize(), transform=axis.transAxes, va="top")
    axes[0].set_ylabel("Accuracy")
    axes[1].legend(frameon=False, loc="upper right")
    save_vector(fig, folder / "selected_program_accuracy")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
    for row in candidates["candidates"]:
        cid = row["candidate_id"]
        axis.scatter(row["depth_ratio"], metrics[cid]["validation"]["macro_gain"],
                     color=PROGRAM_COLOR if cid == selected_id else BASELINE_COLOR,
                     marker="*" if cid == selected_id else "o",
                     s=70 if cid == selected_id else 18,
                     edgecolor="black", linewidth=0.4)
    axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
    axis.axvline(1, color="black", linestyle=":", linewidth=0.8)
    axis.set_xlabel("Executed depth / full depth")
    axis.set_ylabel("Validation macro accuracy gain")
    axis.annotate("Selected", (selected["depth_ratio"], metrics[selected_id]["validation"]["macro_gain"]),
                  xytext=(4, 5), textcoords="offset points")
    save_vector(fig, folder / "accuracy_depth_tradeoff")
    plt.close(fig)

    actions = layer_actions(selected["path"], candidates["depth"])
    mapping = {"skip": 0, "keep": 1, "loop": 2}
    from matplotlib.colors import ListedColormap

    # Keep the selected row first, then rank the remaining candidates using validation only.
    # Test gains are displayed for held-out reporting and never affect the row order.
    ranked_candidates = sorted(
        candidates["candidates"],
        key=lambda row: (
            row["candidate_id"] != selected_id,
            -metrics[row["candidate_id"]]["validation"]["worst_group_gain"],
            -metrics[row["candidate_id"]]["validation"]["macro_gain"],
            row["length"],
            row["candidate_id"],
        ),
    )
    all_actions = [layer_actions(row["path"], candidates["depth"])
                   for row in ranked_candidates]
    figure_height = max(3.4, 1.3 + 0.32 * len(ranked_candidates))
    fig, (path_axis, gain_axis) = plt.subplots(
        1, 2, figsize=(9.4, figure_height), sharey=True, constrained_layout=True,
        gridspec_kw={"width_ratios": [4.8, 2.0]},
    )
    path_axis.imshow(
        [[mapping[action] for action in row] for row in all_actions],
        aspect="auto", cmap=ListedColormap([SKIP_COLOR, KEEP_COLOR, LOOP_COLOR]),
        vmin=-0.5, vmax=2.5,
    )
    row_labels = [
        f"{'* ' if row['candidate_id'] == selected_id else '  '}{row['candidate_id']}  L={row['length']}"
        for row in ranked_candidates
    ]
    path_axis.set_yticks(range(len(ranked_candidates)), row_labels)
    path_axis.set_xticks(range(candidates["depth"]), range(candidates["depth"]))
    path_axis.set_xlabel("Original layer index")
    path_axis.set_title("Frozen candidate paths")
    path_axis.tick_params(axis="y", labelsize=7)
    for tick, row in zip(path_axis.get_yticklabels(), ranked_candidates):
        if row["candidate_id"] == selected_id:
            tick.set_fontweight("bold")
    for row_index, actions_row in enumerate(all_actions):
        for layer, action in enumerate(actions_row):
            path_axis.text(
                layer, row_index, {"skip": "S", "keep": "K", "loop": "L"}[action],
                ha="center", va="center", fontsize=5.2,
                color="white" if action != "keep" else "black",
            )

    y_positions = list(range(len(ranked_candidates)))
    validation_gains = [metrics[row["candidate_id"]]["validation"]["gain"]
                        for row in ranked_candidates]
    test_gains = [metrics[row["candidate_id"]]["test"]["gain"]
                  for row in ranked_candidates]
    gain_axis.scatter(validation_gains, y_positions, color=PROGRAM_COLOR, marker="o", s=28,
                      edgecolor="black", linewidth=0.4, label="Validation")
    gain_axis.scatter(test_gains, y_positions, color=SKIP_COLOR, marker="s", s=24,
                      edgecolor="black", linewidth=0.4, label="Test")
    gain_axis.axvline(0, color="black", linestyle=":", linewidth=0.8)
    gain_axis.set_xlabel("Accuracy gain vs full depth")
    gain_axis.set_title("Transfer")
    gain_axis.tick_params(axis="y", left=False, labelleft=False)
    gain_axis.legend(frameon=False, loc="best")
    save_vector(fig, folder / "all_candidate_program_layers")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.1, 1.05), constrained_layout=True)
    axis.imshow([[mapping[action] for action in actions]], aspect="auto",
                cmap=ListedColormap([SKIP_COLOR, KEEP_COLOR, LOOP_COLOR]), vmin=-0.5, vmax=2.5)
    axis.set_yticks([0], ["Operation"])
    axis.set_xticks(range(candidates["depth"]), range(candidates["depth"]))
    axis.set_xlabel("Layer index")
    for layer, action in enumerate(actions):
        axis.text(layer, 0, {"skip": "S", "keep": "K", "loop": "L"}[action],
                  ha="center", va="center", fontsize=6,
                  color="white" if action != "keep" else "black")
    save_vector(fig, folder / "selected_program_layers")
    plt.close(fig)

    comparison = [("Universal", selected)] + [
        (f"DM-{difficulty}", values["candidate"])
        for difficulty, values in per_difficulty_selected.items()
    ]
    action_matrix = [layer_actions(candidate["path"], candidates["depth"])
                     for _, candidate in comparison]
    fig, axis = plt.subplots(figsize=(7.1, 1.0 + 0.28 * len(comparison)),
                             constrained_layout=True)
    axis.imshow([[mapping[action] for action in row] for row in action_matrix],
                aspect="auto", cmap=ListedColormap([SKIP_COLOR, KEEP_COLOR, LOOP_COLOR]),
                vmin=-0.5, vmax=2.5)
    axis.set_yticks(range(len(comparison)), [label for label, _ in comparison])
    axis.set_xticks(range(candidates["depth"]), range(candidates["depth"]))
    axis.set_xlabel("Layer index")
    for row_index, actions_row in enumerate(action_matrix):
        for layer, action in enumerate(actions_row):
            axis.text(layer, row_index, {"skip": "S", "keep": "K", "loop": "L"}[action],
                      ha="center", va="center", fontsize=5.5,
                      color="white" if action != "keep" else "black")
    save_vector(fig, folder / "per_difficulty_program_layers")
    plt.close(fig)

    single_points = {"skip": [], "loop": []}
    for row in candidates["candidates"]:
        actions = layer_actions(row["path"], candidates["depth"])
        edits = [(layer, action) for layer, action in enumerate(actions) if action != "keep"]
        if len(edits) == 1:
            layer, action = edits[0]
            single_points[action].append((
                layer,
                metrics[row["candidate_id"]]["validation"]["worst_group_gain"],
                metrics[row["candidate_id"]]["test"]["worst_group_gain"],
            ))
    if any(single_points.values()):
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), sharey=True,
                                 constrained_layout=True)
        for axis, action, color in zip(axes, ("skip", "loop"),
                                       (SKIP_COLOR, LOOP_COLOR)):
            points = sorted(single_points[action])
            layers = [point[0] for point in points]
            validation_gains = [point[1] for point in points]
            test_gains = [point[2] for point in points]
            axis.scatter(layers, validation_gains, color=color, marker="o", s=28,
                         edgecolor="black", linewidth=0.4,
                         label="Validation worst-difficulty gain")
            axis.scatter(layers, test_gains, color="black", marker="s", s=22,
                         label="Test worst-difficulty gain")
            axis.axhline(0, color=BASELINE_COLOR, linestyle=":", linewidth=0.8)
            axis.set_xticks(layers)
            axis.set_xlabel("Layer index")
            axis.set_title(f"Single-layer {action}")
        axes[0].set_ylabel("Accuracy gain vs full depth")
        axes[1].legend(frameon=False, loc="best")
        save_vector(fig, folder / "single_layer_transfer")
        plt.close(fig)
