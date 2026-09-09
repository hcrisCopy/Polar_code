"""Capture residual streams and compare program geometry with mutual k-NN."""

from datetime import timedelta
import contextlib
import csv
import io
import os
import time
import uuid

import numpy as np
from tqdm import tqdm

from .model_runner import ModelRunner
from .program_analysis import validate_candidate_payload
from .storage import (atomic_json, atomic_text, clean_stage, digest, output_path,
                      read_json, recover_pending, relative_path, run_lock, stage_dir)
from .universal_eval import load_universal_evaluation
from .validate import load_search


def _write_csv(path, rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def _atomic_npz(path, arrays):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = output_path(path.with_suffix(path.suffix + ".pending"))
    if pending.exists():
        raise ValueError(f"Interrupted representation write exists: {pending}")
    with pending.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def _selected_programs(run_name, maximum):
    eval_config = read_json(stage_dir(run_name, "universal_eval") / "config.json")
    candidate_run_name = eval_config.get("candidate_run_name", run_name)
    candidates = read_json(
        stage_dir(candidate_run_name, "program_mining") / "candidates.json")
    validate_candidate_payload(candidates)
    report = read_json(stage_dir(run_name, "program_report") / "report.json")
    candidate_map = {row["candidate_id"]: row for row in candidates["candidates"]}
    ordered = []

    def add(candidate_id, source):
        if candidate_id not in candidate_map or any(row["program_id"] == candidate_id for row in ordered):
            return
        candidate = candidate_map[candidate_id]
        ordered.append({"program_id": candidate_id, "path": candidate["path"],
                        "source": source, "length": candidate["length"]})

    selected = report["selected_candidate"]
    add(selected["candidate_id"], "universal_validation_selected")
    for difficulty, values in report["per_difficulty_selected_on_validation"].items():
        add(values["candidate"]["candidate_id"], f"DM-{difficulty}_validation_selected")
    metrics = report["all_candidate_metrics"]
    ranked = sorted(candidate_map, key=lambda cid: (
        -metrics[cid]["validation"]["worst_group_gain"],
        -metrics[cid]["validation"]["macro_gain"],
        candidate_map[cid]["length"], cid))
    for candidate_id in ranked:
        add(candidate_id, "validation_ranked_fill")
    baseline = {"program_id": "baseline", "path": list(range(candidates["depth"])),
                "source": "full_depth", "length": candidates["depth"]}
    return candidates, [baseline] + ordered[:max(0, maximum - 1)]


def _choose_questions(rows, maximum, seed):
    ranked = sorted(rows, key=lambda row: digest([seed, row["sample_id"]]))
    return ranked if maximum == 0 else ranked[:maximum]


def _build_capture_config(args, manifest, search_config, programs, rows, world_size):
    model_path = relative_path(args.model_path)
    if not (model_path / "config.json").is_file():
        recorded = relative_path(search_config["args"]["model_path"])
        if (recorded / "config.json").is_file():
            args.model_path = str(recorded)
            model_path = recorded
        else:
            raise ValueError(f"No local model config.json at {model_path}")
    if args.model_id != search_config["args"]["model_id"]:
        raise ValueError("Representation model-id differs from the MCTS model")
    config = {
        "schema_version": 2,
        "run_name": args.run_name,
        "candidate_run_name": read_json(
            stage_dir(args.run_name, "universal_eval") / "config.json").get(
                "candidate_run_name", args.run_name),
        "world_size": world_size,
        "manifest_id": manifest["manifest_id"],
        "search_config_id": search_config["config_id"],
        "model_id": args.model_id,
        "model_path": str(model_path),
        "model_revision": args.model_revision,
        "seed": args.seed,
        "pooling": args.pooling,
        "representation_splits": args.representation_splits,
        "max_representation_questions": args.max_representation_questions,
        "max_programs": args.max_programs,
        "programs": programs,
        "sample_ids": [row["sample_id"] for row in rows],
    }
    config["capture_config_id"] = digest(config)
    return config


def _read_npz(path, config, residual_sample_ids=None):
    try:
        with np.load(path, allow_pickle=False) as data:
            sample_id = str(data["sample_id"].item())
            split = str(data["split"].item())
            difficulty = str(data["difficulty"].item())
            program_ids = [str(value) for value in data["program_ids"].tolist()]
            scores = data["scores"].astype(np.int8)
            token_count = int(data["token_count"].item())
            load_residuals = residual_sample_ids is None or sample_id in residual_sample_ids
            residuals = [] if load_residuals else None
            shapes = []
            for index in range(len(program_ids)):
                matrix = data[f"residual_{index:03d}"]
                shapes.append(matrix.shape)
                if load_residuals:
                    residuals.append(matrix.astype(np.float32))
    except Exception as exc:
        raise ValueError(f"Cannot read representation record {path}: {exc}") from exc
    expected_ids = [row["program_id"] for row in config["programs"]]
    if program_ids != expected_ids or len(scores) != len(program_ids):
        raise ValueError(f"Program mismatch in representation record {path}")
    for shape, program in zip(shapes, config["programs"]):
        if len(shape) != 2 or shape[0] != len(program["path"]):
            raise ValueError(f"Residual shape mismatch in {path}")
    return {"sample_id": sample_id, "split": split, "difficulty": difficulty,
            "program_ids": program_ids, "scores": scores,
            "token_count": token_count, "residuals": residuals}


def _capture_rank(args, rows, score_records, config, token, rank, local_rank):
    folder = stage_dir(args.run_name, "representations") / f"rank_{rank:05d}"
    folder = output_path(folder)
    records_dir = output_path(folder / "records")
    records_dir.mkdir(parents=True, exist_ok=True)
    recovered = recover_pending(folder)
    assigned = [row for index, row in enumerate(rows) if index % config["world_size"] == rank]
    source_scores = {row["sample_id"]: row for row in score_records}
    complete = set()
    for path in tqdm(sorted(records_dir.glob("*.npz")), desc=f"rank {rank} residual resume",
                     position=rank, leave=False):
        record = _read_npz(path, config)
        if record["sample_id"] != path.stem or record["sample_id"] not in {
                row["sample_id"] for row in assigned}:
            raise ValueError(f"Unexpected representation record {path}")
        complete.add(record["sample_id"])
    summary = {"rank": rank, "assigned": len(assigned), "resumed_complete": len(complete),
               "newly_completed": 0, "recovery": recovered, "state": "running"}
    atomic_json(folder / "summary.json", summary)
    runner = None
    try:
        pending = [row for row in assigned if row["sample_id"] not in complete]
        if pending:
            with output_path(folder / "capture.log").open("a", encoding="utf-8", buffering=1) as stream:
                with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                    runner = ModelRunner(args, local_rank, stream)
                for row in tqdm(pending, desc=f"rank {rank} residual questions",
                                position=rank, leave=True):
                    scored = source_scores[row["sample_id"]]
                    arrays = {
                        "sample_id": np.asarray(row["sample_id"]),
                        "split": np.asarray(row["split"]),
                        "difficulty": np.asarray(str(row["difficulty"])),
                        "program_ids": np.asarray([p["program_id"] for p in config["programs"]]),
                        "scores": np.asarray([
                            scored["baseline_score"] if p["program_id"] == "baseline"
                            else scored["scores"][p["program_id"]]
                            for p in config["programs"]], dtype=np.int8),
                    }
                    token_count = None
                    for index, program in enumerate(config["programs"]):
                        matrix, current_tokens = runner.residual_stream(
                            row, program["path"], config["pooling"])
                        token_count = current_tokens if token_count is None else token_count
                        arrays[f"residual_{index:03d}"] = matrix
                    arrays["token_count"] = np.asarray(token_count, dtype=np.int32)
                    _atomic_npz(records_dir / f"{row['sample_id']}.npz", arrays)
                    summary["newly_completed"] += 1
                    atomic_json(folder / "summary.json", summary)
        summary["state"] = "complete"
        atomic_json(folder / "summary.json", summary)
        atomic_json(folder / "done.json", {"invocation": token,
                                            "capture_config_id": config["capture_config_id"],
                                            "rank": rank})
    except Exception as exc:
        summary["state"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        atomic_json(folder / "summary.json", summary)
        atomic_json(folder / "error.json", {"invocation": token, "error": summary["error"]})
        raise
    finally:
        if runner is not None:
            runner.hook.remove()
            del runner


def _wait_capture(args, config, token):
    folder = stage_dir(args.run_name, "representations")
    remaining = set(range(config["world_size"]))
    deadline = time.monotonic() + args.completion_timeout
    with tqdm(total=len(remaining), desc="rank 0 residual shards") as progress:
        while remaining:
            for rank in list(remaining):
                rank_dir = folder / f"rank_{rank:05d}"
                error_path = rank_dir / "error.json"
                if error_path.exists() and read_json(error_path).get("invocation") == token:
                    raise RuntimeError(read_json(error_path)["error"])
                done_path = rank_dir / "done.json"
                if done_path.exists():
                    done = read_json(done_path)
                    if (done.get("invocation") == token and
                            done.get("capture_config_id") == config["capture_config_id"]):
                        remaining.remove(rank)
                        progress.update(1)
            if remaining:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for residual ranks {sorted(remaining)}")
                time.sleep(1)


def load_representations(run_name, residual_sample_ids=None):
    folder = stage_dir(run_name, "representations")
    config = read_json(folder / "config.json")
    unsigned = {key: value for key, value in config.items() if key != "capture_config_id"}
    if digest(unsigned) != config["capture_config_id"]:
        raise ValueError("Representation configuration checksum mismatch")
    records, seen = [], set()
    for rank in tqdm(range(config["world_size"]), desc="Load residual shards"):
        for path in sorted((folder / f"rank_{rank:05d}" / "records").glob("*.npz")):
            record = _read_npz(path, config, residual_sample_ids)
            if record["sample_id"] in seen:
                raise ValueError("Duplicate representation sample")
            seen.add(record["sample_id"])
            records.append(record)
    missing = set(config["sample_ids"]) - seen
    if missing:
        raise ValueError(f"Missing {len(missing)} representation samples")
    order = {sample_id: index for index, sample_id in enumerate(config["sample_ids"])}
    records.sort(key=lambda row: order[row["sample_id"]])
    return config, records


def distributed_capture_representations(args):
    import multiprocessing as mp
    import torch.distributed as dist

    mp.set_start_method("spawn", force=True)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if "RANK" not in os.environ:
        raise ValueError("Use torchrun for representation capture")
    if int(os.environ.get("LOCAL_WORLD_SIZE", world)) != world:
        raise ValueError("Representation capture supports one local process per GPU")
    dist.init_process_group("gloo", timeout=timedelta(seconds=args.completion_timeout))
    packet, lock = [None], None
    try:
        if rank == 0:
            try:
                lock = run_lock(args.run_name)
                lock.__enter__()
                from .universal_eval import load_evaluation_inputs
                eval_config = read_json(
                    stage_dir(args.run_name, "universal_eval") / "config.json")
                manifest, search_config, _, _ = load_evaluation_inputs(
                    args.run_name, eval_config.get("candidate_run_name", args.run_name))
                _, score_records = load_universal_evaluation(args.run_name)
                _, programs = _selected_programs(args.run_name, args.max_programs)
                available = [row for row in manifest["samples"]
                             if row["split"] in args.representation_splits]
                rows = _choose_questions(available, args.max_representation_questions, args.seed)
                config = _build_capture_config(args, manifest, search_config, programs, rows, world)
                folder = stage_dir(args.run_name, "representations")
                if args.clean:
                    clean_stage(args.run_name, "representations")
                recover_pending(folder)
                if (folder / "config.json").exists() and read_json(folder / "config.json") != config:
                    raise ValueError("Representation inputs changed; use --clean")
                atomic_json(folder / "config.json", config)
                packet[0] = {"ok": True, "invocation": uuid.uuid4().hex,
                             "capture_config_id": config["capture_config_id"]}
            except Exception as exc:
                packet[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        dist.broadcast_object_list(packet, src=0)
        dist.destroy_process_group()
        if not packet[0]["ok"]:
            raise RuntimeError(packet[0]["error"])
        from .universal_eval import load_evaluation_inputs
        eval_config = read_json(stage_dir(args.run_name, "universal_eval") / "config.json")
        manifest, _, _, _ = load_evaluation_inputs(
            args.run_name, eval_config.get("candidate_run_name", args.run_name))
        _, score_records = load_universal_evaluation(args.run_name)
        config = read_json(stage_dir(args.run_name, "representations") / "config.json")
        args.model_path = config["model_path"]
        row_map = {row["sample_id"]: row for row in manifest["samples"]}
        rows = [row_map[sample_id] for sample_id in config["sample_ids"]]
        _capture_rank(args, rows, score_records, config, packet[0]["invocation"], rank, local_rank)
        if rank == 0:
            _wait_capture(args, config, packet[0]["invocation"])
            _, records = load_representations(args.run_name, residual_sample_ids=set())
            print(f"Residual capture complete: {len(records)} questions")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if lock is not None:
            lock.__exit__(None, None, None)


def _normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def knn_indices(matrix, k):
    similarity = _normalize(matrix) @ _normalize(matrix).T
    np.fill_diagonal(similarity, -np.inf)
    return np.argpartition(similarity, -k, axis=1)[:, -k:]


def neighbor_overlap(first_neighbors, second_neighbors):
    k = first_neighbors.shape[1]
    scores = []
    for left, right in zip(first_neighbors, second_neighbors):
        left = set(int(value) for value in left)
        right = set(int(value) for value in right)
        scores.append(len(left & right) / k)
    return float(np.mean(scores))


def mutual_knn(first, second, k):
    return neighbor_overlap(knn_indices(first, k), knn_indices(second, k))


def linear_cka(first, second):
    first = first - first.mean(axis=0, keepdims=True)
    second = second - second.mean(axis=0, keepdims=True)
    gram_first = first @ first.T
    gram_second = second @ second.T
    numerator = float(np.sum(gram_first * gram_second))
    denominator = float(np.sqrt(np.sum(gram_first ** 2) * np.sum(gram_second ** 2)))
    return numerator / denominator if denominator else 0.0


def _pca(matrix):
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    u, singular, _ = np.linalg.svd(centered, full_matrices=False)
    return u[:, :2] * singular[:2]


def _plot_search_trajectories(folder, run_name, maximum_questions, maximum_paths, seed):
    from .plot_utils import LOOP_COLOR, SKIP_COLOR, pyplot, save_vector
    plt = pyplot()
    _, config, records = load_search(run_name)
    eligible = [row for row in records if len(row.get("evaluations", [])) > 1]
    selected = _choose_questions(eligible, maximum_questions, seed)
    outputs = []
    for row in tqdm(selected, desc="Plot MCTS evaluation trajectories"):
        evaluations = row["evaluations"][:maximum_paths]
        features = np.asarray([[path.count(layer) for layer in range(config["depth"])]
                               for path in (item["path"] for item in evaluations)], dtype=float)
        projection = _pca(features) if len(features) >= 2 else np.zeros((len(features), 2))
        rewards = np.asarray([item["score"] for item in evaluations])
        lengths = np.asarray([len(item["path"]) for item in evaluations])
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), constrained_layout=True)
        order = np.arange(len(lengths))
        axes[0].plot(order, lengths, color="#777777", linewidth=0.7,
                     label="Temporal order")
        axes[1].plot(projection[:, 0], projection[:, 1], color="#BBBBBB", linewidth=0.6)
        for reward, color, marker, label in (
                (1, LOOP_COLOR, "o", "Correct (reward=1)"),
                (0, SKIP_COLOR, "x", "Incorrect (reward=0)")):
            mask = rewards == reward
            if not np.any(mask):
                continue
            axes[0].scatter(order[mask], lengths[mask], color=color, marker=marker,
                            s=14, label=label)
            axes[1].scatter(projection[mask, 0], projection[mask, 1], color=color,
                            marker=marker, s=16)
        axes[0].set_xlabel("Evaluation order")
        axes[0].set_ylabel("Executed layers")
        axes[0].set_title("MCTS evaluated-program sequence")
        axes[0].legend(frameon=False)
        axes[1].set_xlabel("Structural PC 1")
        axes[1].set_ylabel("Structural PC 2")
        axes[1].set_title("Path execution-count geometry")
        stem = folder / f"search_trajectory_{row['sample_id']}"
        save_vector(fig, stem)
        plt.close(fig)

        from matplotlib.colors import ListedColormap
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [20, 1]})
        image = axes[0].imshow(features, aspect="auto", interpolation="nearest",
                               vmin=0, vmax=max(2, float(features.max())), cmap="cividis")
        axes[0].set_xlabel("Original layer index")
        axes[0].set_ylabel("MCTS evaluation order")
        axes[0].set_title("Explored program execution counts (0=skip, 2+=loop)")
        fig.colorbar(image, ax=axes[0], label="Execution count")
        axes[1].imshow(rewards[:, None], aspect="auto", interpolation="nearest",
                       vmin=0, vmax=1, cmap=ListedColormap([SKIP_COLOR, LOOP_COLOR]))
        axes[1].set_xticks([0], ["Reward"], rotation=45, ha="right")
        axes[1].set_yticks([])
        save_vector(fig, folder / f"search_path_matrix_{row['sample_id']}")
        plt.close(fig)
        outputs.append(row["sample_id"])
    return outputs


def _plot_representation_report(folder, config, projection_records, rows, group_rows):
    from .plot_utils import BASELINE_COLOR, PROGRAM_COLOR, pyplot, save_vector
    plt = pyplot()
    labels = [row["program_id"] for row in config["programs"]]
    max_steps = max(len(row["path"]) for row in config["programs"])
    alignment = np.full((len(labels), max_steps), np.nan)
    for row in rows:
        alignment[row["program_index"], :len(row["step_mnn"])] = row["step_mnn"]
    fig, axis = plt.subplots(figsize=(7.1, 1.5 + 0.28 * len(labels)), constrained_layout=True)
    image = axis.imshow(alignment, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Execution slot")
    axis.set_ylabel("Program")
    fig.colorbar(image, ax=axis, label="mNN vs baseline final residual")
    save_vector(fig, folder / "residual_alignment_heatmap")
    plt.close(fig)

    for row in rows:
        layerwise = np.asarray(row["layerwise_mnn"])
        fig, axis = plt.subplots(
            figsize=(5.2, max(2.5, 1.2 + 0.12 * layerwise.shape[0])),
            constrained_layout=True)
        image = axis.imshow(layerwise, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        axis.set_xlabel("Baseline execution slot")
        axis.set_ylabel("Program execution slot")
        axis.set_title(f"Layerwise mNN: {row['program_id']}")
        fig.colorbar(image, ax=axis, label="Mutual k-NN overlap")
        save_vector(fig, folder / f"layer_alignment_{row['program_id']}")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), constrained_layout=True)
    x = np.arange(len(labels))
    axes[0].bar(x, [row["final_mnn"] for row in rows], color=PROGRAM_COLOR,
                edgecolor="black", linewidth=0.4)
    axes[1].bar(x, [row["final_cka"] for row in rows], color=BASELINE_COLOR,
                edgecolor="black", linewidth=0.4, hatch="//")
    for axis, title in zip(axes, ("Mutual k-NN", "Linear CKA")):
        axis.set_xticks(x, labels, rotation=45, ha="right")
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.set_ylabel("Alignment with baseline final residual")
    save_vector(fig, folder / "final_geometry_alignment")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(4.5, 3.0), constrained_layout=True)
    markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "*")
    palette = plt.get_cmap("tab10")
    for program_index, row in enumerate(rows):
        axis.scatter(row["final_mnn"], row["accuracy_gain"], s=28,
                     color=palette(program_index % 10),
                     marker=markers[program_index % len(markers)],
                     label=labels[program_index])
    axis.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    axis.set_xlim(0, 1.02)
    axis.set_xlabel("Final-residual mNN vs baseline")
    axis.set_ylabel("Accuracy gain over baseline")
    axis.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_vector(fig, folder / "alignment_accuracy_tradeoff")
    plt.close(fig)

    group_labels = list(dict.fromkeys(row["group_label"] for row in group_rows))
    if group_labels:
        group_lookup = {(row["program_id"], row["group_label"]): row["final_mnn"]
                        for row in group_rows}
        group_matrix = np.asarray([
            [group_lookup.get((program_id, group_label), np.nan)
             for group_label in group_labels]
            for program_id in labels
        ])
        fig, axis = plt.subplots(
            figsize=(4.2, 1.5 + 0.28 * len(labels)), constrained_layout=True)
        image = axis.imshow(group_matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        axis.set_xticks(range(len(group_labels)), group_labels, rotation=35, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xlabel("Held-out group")
        axis.set_ylabel("Program")
        fig.colorbar(image, ax=axis, label="Final-residual mNN vs baseline")
        save_vector(fig, folder / "final_alignment_by_group")
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.0, 2.8), constrained_layout=True)
    for program_index, row in enumerate(rows):
        axis.plot(range(1, len(row["step_residual_norm"]) + 1),
                  row["step_residual_norm"], marker=markers[program_index % len(markers)],
                  color=palette(program_index % 10), markersize=2.5,
                  label=labels[program_index])
    axis.set_xlabel("Execution slot")
    axis.set_ylabel("Mean residual L2 norm")
    axis.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_vector(fig, folder / "residual_norm_trajectories")
    plt.close(fig)

    per_program = len(projection_records)
    matrices = []
    for program_index, label in enumerate(labels):
        matrices.append(np.stack([record["residuals"][program_index][-1]
                                  for record in projection_records]))
    projection = _pca(np.concatenate(matrices, axis=0))
    fig, axis = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
    offset = 0
    for program_index, label in enumerate(labels):
        points = projection[offset:offset + per_program]
        axis.scatter(points[:, 0], points[:, 1], s=14, color=palette(program_index % 10),
                     marker=markers[program_index % len(markers)], label=label, alpha=0.75)
        offset += per_program
    axis.set_xlabel("Residual PC 1")
    axis.set_ylabel("Residual PC 2")
    axis.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_vector(fig, folder / "final_residual_pca")
    plt.close(fig)


def _heldout_groups(records, alignment_samples, neighbor_k, seed):
    specifications = []
    difficulties = sorted(
        {row["difficulty"] for row in records},
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value))
    for difficulty in difficulties:
        specifications.append(("difficulty", difficulty, f"DM-{difficulty}",
                               [row for row in records if row["difficulty"] == difficulty]))
    for split in ("validation", "test"):
        members = [row for row in records if row["split"] == split]
        if members:
            specifications.append(("split", split, split, members))

    inventory, eligible = [], []
    for kind, value, label, members in specifications:
        used = min(alignment_samples, len(members))
        row = {"group_kind": kind, "group_value": value, "group_label": label,
               "questions_captured": len(members), "questions_used_for_alignment": used,
               "mnn_available": used > neighbor_k}
        inventory.append(row)
        if used > neighbor_k:
            sampled = _choose_questions(members, used, seed)
            row["alignment_sample_ids"] = [item["sample_id"] for item in sampled]
            eligible.append((row, members, sampled))
    return inventory, eligible


def report_representations(args):
    folder = stage_dir(args.run_name, "representation_report")
    if args.clean:
        clean_stage(args.run_name, "representation_report")
    recover_pending(folder)
    config, metadata = load_representations(args.run_name, residual_sample_ids=set())
    sample_count = min(args.alignment_samples, len(metadata))
    if args.neighbor_k >= sample_count:
        raise ValueError(
            f"--neighbor-k must be smaller than the {sample_count} questions used for alignment")
    sampled_metadata = _choose_questions(metadata, sample_count, args.seed)
    baseline_index = next(index for index, row in enumerate(config["programs"])
                          if row["program_id"] == "baseline")
    group_inventory, eligible_groups = _heldout_groups(
        metadata, args.alignment_samples, args.neighbor_k, args.seed)
    projection_metadata = _choose_questions(
        metadata, min(args.projection_samples, len(metadata)), args.seed)
    needed_ids = {row["sample_id"] for row in sampled_metadata + projection_metadata}
    for _, _, group_sampled in eligible_groups:
        needed_ids.update(row["sample_id"] for row in group_sampled)
    _, loaded = load_representations(args.run_name, residual_sample_ids=needed_ids)
    residual_map = {row["sample_id"]: row for row in loaded if row["residuals"] is not None}
    sampled = [residual_map[row["sample_id"]] for row in sampled_metadata]
    projection_records = [residual_map[row["sample_id"]] for row in projection_metadata]
    baseline_layers = [
        np.stack([row["residuals"][baseline_index][step] for row in sampled])
        for step in range(len(config["programs"][baseline_index]["path"]))
    ]
    baseline_neighbors = [knn_indices(matrix, args.neighbor_k) for matrix in baseline_layers]
    baseline_final = baseline_layers[-1]
    baseline_final_neighbors = baseline_neighbors[-1]
    group_runtime = []
    for group, members, group_sampled_metadata in eligible_groups:
        group_sampled = [residual_map[row["sample_id"]] for row in group_sampled_metadata]
        group_baseline = np.stack([
            row["residuals"][baseline_index][-1] for row in group_sampled])
        group_runtime.append((group, members, group_sampled, group_baseline,
                              knn_indices(group_baseline, args.neighbor_k)))
    metric_rows, group_rows = [], []
    for program_index, program in enumerate(tqdm(config["programs"], desc="Compare residual geometry")):
        steps = []
        norms = []
        layerwise = []
        for step in range(len(program["path"])):
            matrix = np.stack([row["residuals"][program_index][step] for row in sampled])
            neighbors = (baseline_neighbors[step] if program_index == baseline_index
                         else knn_indices(matrix, args.neighbor_k))
            steps.append(neighbor_overlap(neighbors, baseline_final_neighbors))
            layerwise.append([neighbor_overlap(neighbors, reference)
                              for reference in baseline_neighbors])
            norms.append(float(np.mean(np.linalg.norm(matrix, axis=1))))
        final = np.stack([row["residuals"][program_index][-1] for row in sampled])
        metric_rows.append({
            "program_index": program_index,
            "program_id": program["program_id"],
            "source": program["source"],
            "length": program["length"],
            "accuracy": float(np.mean([row["scores"][program_index] for row in metadata])),
            "final_mnn": neighbor_overlap(neighbors, baseline_final_neighbors),
            "final_cka": linear_cka(final, baseline_final),
            "step_mnn": steps,
            "layerwise_mnn": layerwise,
            "step_residual_norm": norms,
        })
        for group, members, group_sampled, group_baseline, group_baseline_neighbors in group_runtime:
            group_final = np.stack([
                row["residuals"][program_index][-1] for row in group_sampled])
            group_neighbors = (group_baseline_neighbors if program_index == baseline_index
                               else knn_indices(group_final, args.neighbor_k))
            group_rows.append({
                "program_id": program["program_id"],
                "source": program["source"],
                "length": program["length"],
                "group_kind": group["group_kind"],
                "group_value": group["group_value"],
                "group_label": group["group_label"],
                "questions_captured": len(members),
                "questions_used_for_alignment": len(group_sampled),
                "accuracy": float(np.mean([row["scores"][program_index] for row in members])),
                "final_mnn": neighbor_overlap(group_neighbors, group_baseline_neighbors),
                "final_cka": linear_cka(group_final, group_baseline),
            })
    baseline_accuracy = metric_rows[baseline_index]["accuracy"]
    for row in metric_rows:
        row["accuracy_gain"] = row["accuracy"] - baseline_accuracy
    group_baseline_accuracy = {
        (row["group_kind"], row["group_value"]): row["accuracy"]
        for row in group_rows if row["program_id"] == "baseline"
    }
    for row in group_rows:
        key = (row["group_kind"], row["group_value"])
        row["accuracy_gain"] = row["accuracy"] - group_baseline_accuracy[key]
    result = {
        "schema_version": 1,
        "run_name": args.run_name,
        "capture_config_id": config["capture_config_id"],
        "paper_method": "mutual k-nearest-neighbor overlap on corresponding samples",
        "adaptation": "same base model under different fixed layer-execution programs",
        "questions_captured": len(metadata),
        "questions_used_for_alignment": len(sampled),
        "neighbor_k": args.neighbor_k,
        "pooling": config["pooling"],
        "report_args": {
            "alignment_samples": args.alignment_samples,
            "projection_samples": args.projection_samples,
            "max_search_questions": args.max_search_questions,
            "max_paths_per_question": args.max_paths_per_question,
            "seed": args.seed,
        },
        "alignment_sample_ids": [row["sample_id"] for row in sampled_metadata],
        "projection_sample_ids": [row["sample_id"] for row in projection_metadata],
        "program_metrics": metric_rows,
        "group_inventory": group_inventory,
        "program_group_metrics": group_rows,
        "interpretation_limit": "Program alignment is an adaptation of PRH, not a cross-model Platonic convergence claim",
    }
    atomic_json(folder / "report.json", result)
    flat_rows = [{key: row[key] for key in (
        "program_id", "source", "length", "accuracy", "accuracy_gain",
        "final_mnn", "final_cka")}
        for row in metric_rows]
    _write_csv(folder / "program_geometry.csv", flat_rows, list(flat_rows[0]))
    group_fields = ["program_id", "source", "length", "group_kind", "group_value",
                    "group_label", "questions_captured", "questions_used_for_alignment",
                    "accuracy", "accuracy_gain", "final_mnn", "final_cka"]
    _write_csv(folder / "program_geometry_by_group.csv", group_rows, group_fields)
    _plot_representation_report(folder, config, projection_records, metric_rows, group_rows)
    trajectories = _plot_search_trajectories(
        folder, config.get("candidate_run_name", args.run_name),
        args.max_search_questions, args.max_paths_per_question, args.seed)
    atomic_text(folder / "summary.md", "\n".join([
        "# MCTS 路径与残差流表征", "",
        f"捕获题目数：{len(metadata)}；mNN 实际使用：{len(sampled)}；k={args.neighbor_k}。",
        f"Pooling：{config['pooling']}；固定程序数：{len(config['programs'])}。",
        "", "`residual_alignment_heatmap` 展示每个执行槽位相对 baseline 最终残差几何的 mNN。",
        "`layer_alignment_<program_id>` 展示候选槽位与 baseline 槽位的完整 mNN 矩阵。",
        "`final_geometry_alignment` 对比最终 residual 的 mNN 与 linear CKA。",
        "`alignment_accuracy_tradeoff` 对比表征对齐度与相对 baseline 的正确率变化。",
        "`final_alignment_by_group` 分别展示各 DART-Math 难度及 held-out split 的 mNN。",
        "`final_residual_pca` 仅用于观察聚类，不作为相似性结论。",
        f"生成了 {len(trajectories)} 组 MCTS 评估顺序轨迹图。",
        "", "这里比较的是同一模型的不同 layer programs，属于柏拉图表征方法的改造应用，"
        "不能表述为不同模型或模态已经收敛到同一个柏拉图表征。",
    ]) + "\n")
    atomic_text(folder / "FIGURE_CAPTIONS.md", "\n".join([
        "# Figure captions", "",
        "- `search_trajectory_*`: Actual MCTS evaluation order. Point color is binary reward; "
        "the connecting line is temporal order, not a stored parent-child tree edge.",
        "- `search_path_matrix_*`: Original-layer execution counts for every evaluated path "
        "in temporal order; 0 means skip and values of 2 or more mean loop/re-execution.",
        "- `residual_alignment_heatmap`: Mutual k-NN overlap between each program execution "
        "slot and the full-depth program's final residual geometry over corresponding questions.",
        "- `layer_alignment_<program_id>`: Full mutual-kNN matrix between the program's "
        "execution slots and the full-depth baseline's execution slots.",
        "- `final_geometry_alignment`: Final residual geometry alignment measured by the PRH "
        "paper's mutual k-NN and supplementary linear CKA.",
        "- `alignment_accuracy_tradeoff`: Held-out accuracy gain versus final-residual mNN; "
        "alignment is descriptive and is not treated as a causal explanation of correctness.",
        "- `final_alignment_by_group`: Final-residual mutual k-NN split by DART-Math "
        "difficulty and held-out data split; groups with at most k samples are omitted.",
        "- `residual_norm_trajectories`: Mean L2 norm of pooled post-block residuals over questions.",
        "- `final_residual_pca`: Shared two-dimensional PCA for visual inspection only; mNN is "
        "the primary representation comparison.",
    ]) + "\n")
    print(f"Representation report written to {folder}")
    return result
