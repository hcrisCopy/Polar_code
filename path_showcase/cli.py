"""Command-line entry point for the fast path-showcase checkpoint."""

import argparse
import math
from pathlib import Path

from .storage import clean_run, relative_path


def build_parser():
    parser = argparse.ArgumentParser(description="Fast Qwen3 path showcase")
    commands = parser.add_subparsers(dest="stage", required=True)
    for name in ("prepare", "search", "report"):
        command = commands.add_parser(name)
        command.add_argument("--run-name", required=True)
        if name == "prepare":
            command.add_argument("--clean", action="store_true",
                                 help="Clear this run's path_showcase outputs before preparing")
            command.add_argument("--data-path", required=True)
            command.add_argument("--source-revision", required=True)
            command.add_argument("--difficulties", nargs="+", type=int, required=True,
                                 choices=range(1, 6))
            command.add_argument("--seed", type=int, required=True)
        elif name == "search":
            command.add_argument("--model-id", required=True,
                                 choices=["Qwen/Qwen3-8B"])
            command.add_argument("--model-path", required=True)
            command.add_argument("--model-revision", required=True)
            command.add_argument("--device", type=int, required=True)
            command.add_argument("--seed", type=int, required=True)
            command.add_argument("--simulations", type=int, required=True)
            command.add_argument("--check-interval", type=int, required=True)
            command.add_argument("--max-question-seconds", type=int, required=True)
            command.add_argument("--target-per-label", type=int, required=True)
            command.add_argument("--exploration", type=float, required=True)
            command.add_argument("--length-penalty", type=float, required=True)
            command.add_argument("--max-block", type=int, required=True)
            command.add_argument("--max-repeats", type=int, required=True)
            command.add_argument("--max-length-factor", type=float, required=True)
            command.add_argument("--max-new-tokens", type=int, required=True)
            command.add_argument("--temperature", type=float, required=True)
        else:
            command.add_argument("--paths-per-label", type=int, required=True)
    return parser


def validate(args):
    if not Path("Polar_code/polar/eval.py").is_file():
        raise ValueError("Run from the directory containing ./Polar_code and ./Polar_data")
    if args.stage == "prepare":
        relative_path(args.data_path)
        args.difficulties = sorted(set(args.difficulties))
    elif args.stage == "search":
        relative_path(args.model_path)
        if min(args.simulations, args.check_interval, args.max_question_seconds,
               args.target_per_label,
               args.max_block, args.max_repeats, args.max_new_tokens) <= 0:
            raise ValueError("Search limits must be positive")
        if args.simulations < args.check_interval:
            raise ValueError("simulations must be at least check-interval")
        if args.max_block > 4 or args.max_repeats > 4:
            raise ValueError("Paper search constraints require max-block/max-repeats <= 4")
        if any(not math.isfinite(value) or value < 0 for value in
               (args.exploration, args.length_penalty)):
            raise ValueError("MCTS coefficients must be finite and nonnegative")
        if (not math.isfinite(args.max_length_factor)
                or args.max_length_factor < 1.0):
            raise ValueError("max-length-factor must be finite and >= 1")
        if not math.isfinite(args.temperature) or args.temperature != 0:
            raise ValueError("This checkpoint requires Qwen3 non-thinking greedy inference")
    elif args.paths_per_label <= 0:
        raise ValueError("paths-per-label must be positive")


def main():
    args = build_parser().parse_args()
    validate(args)
    if getattr(args, "clean", False):
        clean_run(args.run_name)
    if args.stage == "prepare":
        from .data import prepare_questions
        prepare_questions(args)
    elif args.stage == "search":
        from .search import run_search
        run_search(args)
    else:
        from .report import build_report
        build_report(args)
