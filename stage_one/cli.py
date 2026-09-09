"""Explicit stage CLI. Non-model stages import no inference implementation eagerly."""

import argparse
import math
import os
from pathlib import Path

from .storage import relative_path, run_lock, stage_dir


MODELS = ["meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen1.5-MoE-A2.7B-Chat",
          "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen3-8B"]


def parser():
    result = argparse.ArgumentParser(description="Offline PoLar MCTS supervision stages")
    commands = result.add_subparsers(dest="stage", required=True)
    for name in ("environment", "prepare", "search", "merge", "validate",
                 "mine-programs", "evaluate-programs", "report-programs",
                 "capture-representations", "report-representations"):
        sub = commands.add_parser(name)
        sub.add_argument("--run-name", required=True)
        sub.add_argument("--clean", action="store_true", help="Explicitly clear ONLY this stage's run directory")
        if name not in {"search", "evaluate-programs", "capture-representations"}:
            sub.add_argument("--clean-only", action="store_true", help="With --clean, clear this stage then exit")
        if name in {"environment", "prepare"}:
            sub.add_argument("--data-path", required=True)
        if name in {"environment", "search"}:
            sub.add_argument("--model-path", required=True, help="Complete local snapshot, relative to project root")
        if name == "prepare":
            sub.add_argument("--data-source", required=True, choices=["hkust-nlp/dart-math-pool-math"])
            sub.add_argument("--source-revision", required=True, help="Record the downloaded dataset revision/tag")
            sub.add_argument("--difficulties", type=int, nargs="+", required=True, choices=range(1, 6))
            sub.add_argument("--seed", type=int, required=True)
            sub.add_argument("--split-policy", choices=["proportional", "official"], required=True)
            sub.add_argument("--train-fraction", type=float, required=True)
            sub.add_argument("--validation-fraction", type=float, required=True)
            sub.add_argument("--max-questions-per-diff", type=int, required=True, help="0 uses all available unique questions")
            sub.add_argument("--exclude-run-name", default=None,
                             help="Exclude train questions from a program-discovery run")
        if name == "search":
            sub.add_argument("--model-id", required=True, choices=MODELS)
            sub.add_argument("--model-revision", required=True)
            sub.add_argument("--seed", type=int, required=True)
            sub.add_argument("--simulations", type=int, required=True)
            sub.add_argument("--exploration", type=float, required=True)
            sub.add_argument("--length-penalty", type=float, required=True)
            sub.add_argument("--max-block", type=int, required=True, choices=range(1, 5))
            sub.add_argument("--max-repeats", type=int, required=True, choices=range(1, 5))
            sub.add_argument("--max-length-factor", type=float, required=True)
            sub.add_argument("--max-new-tokens", type=int, required=True)
            sub.add_argument("--temperature", type=float, required=True)
            sub.add_argument("--completion-timeout", type=int, required=True)
        if name == "mine-programs":
            sub.add_argument("--max-candidates", type=int, required=True)
            sub.add_argument("--min-train-support", type=int, required=True)
            sub.add_argument("--max-consensus-edits", type=int, required=True)
            sub.add_argument("--top-layers-per-action", type=int, required=True)
            sub.add_argument("--smoothing", type=float, required=True)
        if name == "evaluate-programs":
            sub.add_argument("--candidate-run-name", default=None,
                             help="Read frozen candidates from another completed MCTS run")
            sub.add_argument("--model-id", required=True, choices=MODELS)
            sub.add_argument("--model-path", required=True)
            sub.add_argument("--model-revision", required=True)
            sub.add_argument("--seed", type=int, required=True)
            sub.add_argument("--max-new-tokens", type=int, required=True)
            sub.add_argument("--temperature", type=float, required=True)
            sub.add_argument("--evaluation-splits", nargs="+", required=True,
                             choices=["validation", "test"])
            sub.add_argument("--max-eval-candidates", type=int, required=True)
            sub.add_argument("--completion-timeout", type=int, required=True)
        if name == "report-programs":
            sub.add_argument("--bootstrap-samples", type=int, required=True)
            sub.add_argument("--bootstrap-seed", type=int, required=True)
        if name == "capture-representations":
            sub.add_argument("--model-id", required=True, choices=MODELS)
            sub.add_argument("--model-path", required=True)
            sub.add_argument("--model-revision", required=True)
            sub.add_argument("--seed", type=int, required=True)
            sub.add_argument("--representation-splits", nargs="+", required=True,
                             choices=["validation", "test"])
            sub.add_argument("--max-representation-questions", type=int, required=True,
                             help="0 captures every available held-out question")
            sub.add_argument("--max-programs", type=int, required=True)
            sub.add_argument("--pooling", required=True, choices=["last-token", "mean"])
            sub.add_argument("--completion-timeout", type=int, required=True)
        if name == "report-representations":
            sub.add_argument("--neighbor-k", type=int, required=True)
            sub.add_argument("--alignment-samples", type=int, required=True)
            sub.add_argument("--projection-samples", type=int, required=True)
            sub.add_argument("--max-search-questions", type=int, required=True)
            sub.add_argument("--max-paths-per-question", type=int, required=True)
            sub.add_argument("--seed", type=int, required=True)
    return result


def validate_args(args):
    if not Path("Polar_code/polar/data.py").is_file():
        raise ValueError("Run from the project root containing ./Polar_code and ./Polar_data")
    stage_map = {"prepare": "prepared", "merge": "merged", "validate": "validation",
                 "mine-programs": "program_mining", "evaluate-programs": "universal_eval",
                 "report-programs": "program_report",
                 "capture-representations": "representations",
                 "report-representations": "representation_report"}
    stage_dir(args.run_name, stage_map.get(args.stage, args.stage))
    for key in ("data_path", "model_path"):
        if hasattr(args, key):
            relative_path(getattr(args, key))
    if args.stage == "prepare":
        if len(set(args.difficulties)) != len(args.difficulties):
            raise ValueError("Duplicate difficulty arguments")
        args.difficulties = sorted(args.difficulties)
        if args.exclude_run_name is not None:
            stage_dir(args.exclude_run_name, "prepared")
        if (not 0 < args.train_fraction < 1 or not 0 < args.validation_fraction < 1
                or args.train_fraction + args.validation_fraction >= 1 or args.max_questions_per_diff < 0):
            raise ValueError("Invalid split fractions or question limit")
    if args.stage == "search":
        if min(args.simulations, args.max_new_tokens, args.completion_timeout) <= 0:
            raise ValueError("Simulations, generated tokens, and timeout must be positive")
        if any(not math.isfinite(v) or v < 0 for v in (args.exploration, args.length_penalty, args.temperature)):
            raise ValueError("UCB coefficients and temperature must be finite and nonnegative")
        if not math.isfinite(args.max_length_factor) or args.max_length_factor < 1:
            raise ValueError("Max program length must include the baseline (factor >= 1)")
    if args.stage == "mine-programs":
        if min(args.max_candidates, args.min_train_support, args.max_consensus_edits,
               args.top_layers_per_action) <= 0:
            raise ValueError("Candidate, support, and consensus-edit limits must be positive")
        if not math.isfinite(args.smoothing) or args.smoothing <= 0:
            raise ValueError("Smoothing must be finite and positive")
    if args.stage == "evaluate-programs":
        if args.candidate_run_name is not None:
            stage_dir(args.candidate_run_name, "program_mining")
        if set(args.evaluation_splits) != {"validation", "test"} or len(args.evaluation_splits) != 2:
            raise ValueError("Use validation and test exactly once for universal evaluation")
        args.evaluation_splits = ["validation", "test"]
        if min(args.max_new_tokens, args.max_eval_candidates, args.completion_timeout) <= 0:
            raise ValueError("Universal evaluation limits must be positive")
        if not math.isfinite(args.temperature) or args.temperature < 0:
            raise ValueError("Evaluation temperature must be finite and nonnegative")
    if args.stage == "report-programs" and args.bootstrap_samples <= 0:
        raise ValueError("Bootstrap sample count must be positive")
    if args.stage == "capture-representations":
        if args.max_representation_questions < 0 or min(
                args.max_programs, args.completion_timeout) <= 0:
            raise ValueError("Invalid representation capture limits")
        args.representation_splits = sorted(set(args.representation_splits))
    if args.stage == "report-representations":
        if min(args.neighbor_k, args.alignment_samples, args.projection_samples,
               args.max_search_questions, args.max_paths_per_question) <= 0:
            raise ValueError("Representation report limits must be positive")


def main():
    args = parser().parse_args()
    validate_args(args)
    from .environment import configure_runtime
    configure_runtime(args.run_name)
    if args.stage in {"search", "evaluate-programs", "capture-representations"}:
        if args.stage == "capture-representations":
            from .representation_analysis import distributed_capture_representations
            distributed_capture_representations(args)
            return
        if args.stage == "evaluate-programs":
            from .universal_eval import distributed_evaluate_programs
            distributed_evaluate_programs(args)
            return
        from .distributed_search import distributed_search
        distributed_search(args)
        return
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Only model-execution stages accept multi-rank torchrun")
    with run_lock(args.run_name):
        if args.clean_only:
            if not args.clean:
                raise ValueError("--clean-only requires explicit --clean")
            from .storage import clean_stage
            clean_stage(args.run_name, {"prepare": "prepared", "merge": "merged",
                                       "validate": "validation", "mine-programs": "program_mining",
                                       "report-programs": "program_report",
                                       "report-representations": "representation_report"}.get(
                                           args.stage, args.stage))
            return
        if args.stage == "environment":
            from .environment import check_environment
            check_environment(args)
        elif args.stage == "prepare":
            from .prepare import prepare
            prepare(args)
        elif args.stage == "merge":
            from .merge import merge
            merge(args)
        elif args.stage == "validate":
            from .validate import validate
            validate(args)
        elif args.stage == "mine-programs":
            from .program_analysis import mine_programs
            mine_programs(args)
        elif args.stage == "report-programs":
            from .program_analysis import report_programs
            report_programs(args)
        elif args.stage == "report-representations":
            from .representation_analysis import report_representations
            report_representations(args)
