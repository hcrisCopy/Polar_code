"""Deterministic complete-only merge; never compact away failed/missing rows."""

from tqdm import tqdm

from .storage import atomic_json, atomic_text, clean_stage, read_json, recover_pending, stage_dir
from .validate import load_search, merged_file, summarize


def merge(args):
    folder = stage_dir(args.run_name, "merged")
    if args.clean:
        clean_stage(args.run_name, "merged")
    recovered = recover_pending(folder)
    manifest, config, records = load_search(args.run_name, require_complete=False)
    summary = summarize(records, len(manifest["samples"]))
    summary["config_id"] = config["config_id"]
    summary["merge_recovery"] = recovered
    search_recovery = stage_dir(args.run_name, "search") / "recovery_report.json"
    summary["search_recovery"] = read_json(search_recovery) if search_recovery.exists() else []
    summary["rank_resume_states"] = [read_json(p) for p in sorted(
        stage_dir(args.run_name, "search").glob("rank_*/summary.json"))]
    write_summary(folder, summary)
    if summary["missing_questions"] or summary["failed_questions"]:
        raise ValueError(f"Refusing incomplete merged_mcts_samples.json: {summary['missing_questions']} missing, "
                         f"{summary['failed_questions']} failed. Summary saved; resume search first.")
    for difficulty in tqdm(manifest["args"]["difficulties"], desc="Merge difficulty files"):
        samples = [row for row in records if row["difficulty"] == difficulty]
        result = {"samples": samples, "schema_version": 1, "config_id": config["config_id"],
                  "manifest_id": manifest["manifest_id"], "split_policy": manifest["args"]["split_policy"]}
        target = merged_file(args.run_name, config["args"]["model_id"], difficulty)
        if target.exists():
            if read_json(target) != result:
                raise ValueError(f"Existing merged file differs; use merge --clean: {target}")
        else:
            atomic_json(target, result)
    print(f"Merged {len(records)} questions into {folder}")
    return summary


def write_summary(folder, summary):
    atomic_json(folder / "summary.json", summary)
    labels = {"total_questions": "总题数", "completed_questions": "完成数",
              "failed_questions": "执行失败数", "missing_questions": "未完成且未持久化的题数",
              "complete_without_valid_paths": "完成但未找到有效路径的题数",
              "questions_with_valid_paths": "找到有效路径的题数", "mean_valid_paths": "平均有效路径数",
              "mean_invalid_paths": "平均无效路径数", "mean_path_length": "平均路径长度"}
    lines = ["# 第一阶段汇总", "", *[f"- {label}：{summary[key]}" for key, label in labels.items()],
             "", "测试题只评估完整路径；以上均值包含测试题。详细分组见逐题 split 字段。", "", "恢复状态："]
    lines.append(f"- 搜索中断写入留证 {len(summary['search_recovery'])} 项，"
                 f"本次合并中断写入留证 {len(summary['merge_recovery'])} 项。")
    for rank in summary["rank_resume_states"]:
        lines.append(f"- rank {rank['rank']}：{rank['state']}，跳过已完成 {rank['resumed_complete']}，"
                     f"重试失败 {rank['retrying_failed']}，本次完成 {rank['newly_completed']}。")
    atomic_text(folder / "summary.md", "\n".join(lines) + "\n")
