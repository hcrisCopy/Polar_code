"""Train the original predictor using explicit split indices; no test evaluation."""

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from stage_one.environment import configure_runtime
from stage_one.storage import (atomic_json, clean_stage, file_digest, output_path, read_json, relative_path,
                               recover_pending, run_lock, stage_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--predictor-config", required=True)
    parser.add_argument("--clean", action="store_true")
    outer = parser.parse_args()
    configure_runtime(outer.run_name)
    from run_polar import build_arg_parser, normalize_args
    from polar.data import PolarDataset
    import polar.train as official_train
    from stage_one.validate import load_search, validate
    from types import SimpleNamespace
    with run_lock(outer.run_name):
        manifest, config, records = load_search(outer.run_name)
        validate(SimpleNamespace(run_name=outer.run_name, clean=False))
        values = read_json(relative_path(outer.predictor_config))
        official_parser = build_arg_parser()
        defaults = vars(official_parser.parse_args([]))
        if set(values) != set(defaults):
            raise ValueError(f"Explicit predictor config must contain every official CLI field. "
                             f"Missing: {set(defaults) - set(values)}; extra: {set(values) - set(defaults)}")
        # Apply the official CLI's own type/choice rules to explicit JSON values.
        # This avoids silently bypassing argparse when training through a config.
        for action in official_parser._actions:
            if action.dest == "help":
                continue
            value = values[action.dest]
            if value is None:
                if defaults[action.dest] is not None:
                    raise ValueError(f"Null is not an official default for {action.dest}")
                continue
            if action.type is not None:
                converted = action.type(value)
                if converted != value:
                    raise ValueError(f"Wrong JSON value type for {action.dest}")
            if action.choices is not None and value not in action.choices:
                raise ValueError(f"Invalid official choice for {action.dest}: {value}")
            if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)) and type(value) is not bool:
                raise ValueError(f"Expected JSON boolean for {action.dest}")
        if values["no_validation"] or values["eval"] or values["use_wandb"]:
            raise ValueError("This training adapter requires validation enabled, eval=false, use_wandb=false")
        folder = stage_dir(outer.run_name, "predictor")
        if Path(values["save_dir"]) != folder:
            raise ValueError(f"predictor save_dir must be {folder}")
        if Path(values["data_root"]) != stage_dir(outer.run_name, "merged"):
            raise ValueError("predictor data_root must be this run's merged directory")
        if values["model_path"] != config["args"]["model_id"]:
            raise ValueError("Predictor model_path must match search model-id")
        for key in ("data_root", "save_dir", "hf_cache_dir"):
            output_path(values[key])
        if values["checkpoint_path"] is not None or values["delete_checkpoint_after_eval"]:
            raise ValueError("Training adapter does not accept checkpoint/evaluation cleanup options")
        diffs = [values["target_diff"]] if values["target_diff"] else list(range(1, 6))
        if not set(diffs) <= set(manifest["args"]["difficulties"]):
            raise ValueError("Predictor requests unprepared difficulties")
        for diff in diffs:
            for split in ("train", "validation"):
                if not any(r["difficulty"] == diff and r["split"] == split
                           and r["final_valid_transitions"] for r in records):
                    raise ValueError(f"diff {diff} {split} has no valid supervision")
        if outer.clean:
            clean_stage(outer.run_name, "predictor")
        recover_pending(folder)
        if folder.exists():
            for child in folder.rglob("*"):
                output_path(child)
        identity = {"config_id": config["config_id"], "predictor_config": values,
                    "split_adapter": "explicit indices; original train_polar unchanged"}
        provenance = folder / "training_config.json"
        if provenance.exists() and read_json(provenance) != identity:
            raise ValueError("Existing predictor output has different provenance; use --clean or another run")
        atomic_json(provenance, identity)

        class SplitDataset(PolarDataset):
            def __init__(self, *positional, **kwargs):
                if positional:
                    raise ValueError("Unexpected change in official train dataset call signature")
                start, end = kwargs["start_idx"], kwargs["end_idx"]
                if (start, end) == (0, 1250):
                    wanted = "train"
                elif (start, end) == (1250, 1500):
                    wanted = "validation"
                else:
                    raise ValueError("Official split logic changed; inspect the adapter before training")
                # Official resolve_dart_base_path normalizes internally; convert
                # back to a root-relative path for our guarded IO interface.
                local_path = Path(kwargs["merged_samples_json"]).relative_to(Path.cwd())
                data = read_json(local_path)["samples"]
                kwargs["indices"] = [i for i, row in enumerate(data) if row["split"] == wanted]
                super().__init__(**kwargs)

        # Scope the adapter to this process and this call. The model, optimizer,
        # train/eval modes, objective, and original parameter values are untouched.
        original_dataset = official_train.PolarDataset
        official_train.PolarDataset = SplitDataset
        try:
            args = normalize_args(argparse.Namespace(**values))
            from polar.config import checkpoint_path_for_args
            target = Path(checkpoint_path_for_args(args))
            relative_target = target.relative_to(Path.cwd())
            if target.exists():
                completion = read_json(folder / "training_complete.json")
                if (completion.get("config_id") != config["config_id"]
                        or completion.get("checkpoint_sha256") != file_digest(relative_target)):
                    raise ValueError("Incomplete or corrupted predictor checkpoint; inspect it or explicitly --clean")
                print(f"Existing final predictor checkpoint; training skipped: {target.name}")
                return
            checkpoint = official_train.train_polar(args)
            atomic_json(folder / "training_complete.json", {"checkpoint": str(Path(checkpoint).relative_to(Path.cwd())),
                                                             "checkpoint_sha256": file_digest(relative_target),
                                                             "config_id": config["config_id"]})
        finally:
            official_train.PolarDataset = original_dataset


if __name__ == "__main__":
    main()
