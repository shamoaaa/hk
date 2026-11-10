#!/usr/bin/env python3
"""
基线对比报告生成脚本（独立运行）

从已有的评估结果目录中读取 eval_summary.json，与基线结果对比，
生成 baseline_comparison_report.md。

不需要重新跑模型推理，只需要已有的评估结果。

目录结构约定：
    新模型评估结果:  /data1/eval_output/v0.1.0/<ckpt_name>/<label_source>/<dataset>/<split>/eval_summary.json
    基线评估结果:    /data1/eval_output/base/{augment,reflow,speech}_eval_result/{seaco,finetuned,online_model}/<dataset>/<split>/eval_summary.json

用法:
    # 为指定评估结果目录生成基线对比报告
    python gen_baseline_report.py --eval-dir /data1/eval_output/v0.1.0/model.pt.best

    # 为整个版本目录下所有模型生成对比报告
    python gen_baseline_report.py --eval-dir /data1/eval_output/v0.1.0

    # 指定基线目录
    python gen_baseline_report.py --eval-dir /data1/eval_output/v0.1.0 \
        --baseline-dir /data1/eval_output/base

    # 只对比指定基线模式
    python gen_baseline_report.py --eval-dir /data1/eval_output/v0.1.0 \
        --baseline-modes seaco online_model

    # 指定输出文件名
    python gen_baseline_report.py --eval-dir /data1/eval_output/v0.1.0/model.pt.best \
        --output baseline_comparison_report.md
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ============ 默认配置 ============
DEFAULT_BASELINE_DIR = "/data1/eval_output/base"
# ==================================


def load_baseline_results(baseline_dir: Path, baseline_modes: list = None) -> dict:
    """
    从基线评估结果目录加载所有 eval_summary.json。

    返回:
      baselines_by_mode: {mode: {eval_type/dataset/split: summary}}
      其中 eval_type 映射为对应的 label_source:
        augment_eval_result -> augment_label
        reflow_eval_result  -> reflow_label
        speech_eval_result  -> speech_label
    """
    # eval_type 目录名到 label_source 的映射
    EVAL_TYPE_TO_LABEL = {
        "augment_eval_result": "augment_label",
        "reflow_eval_result": "reflow_label",
        "speech_eval_result": "speech_label",
    }

    baselines_by_mode = {}

    if not baseline_dir.exists():
        print(f"[错误] 基线目录不存在: {baseline_dir}")
        return {}

    for eval_type_dir in sorted(baseline_dir.iterdir()):
        if not eval_type_dir.is_dir():
            continue
        eval_type = eval_type_dir.name
        label_source = EVAL_TYPE_TO_LABEL.get(eval_type, eval_type)

        for mode_dir in sorted(eval_type_dir.iterdir()):
            if not mode_dir.is_dir():
                continue
            mode = mode_dir.name

            if baseline_modes and mode not in baseline_modes:
                continue

            if mode not in baselines_by_mode:
                baselines_by_mode[mode] = {}

            for dataset_dir in sorted(mode_dir.iterdir()):
                if not dataset_dir.is_dir():
                    continue
                dataset = dataset_dir.name

                for split_dir in sorted(dataset_dir.iterdir()):
                    if not split_dir.is_dir():
                        continue
                    split = split_dir.name

                    summary_file = split_dir / "eval_summary.json"
                    if not summary_file.exists():
                        continue

                    try:
                        with open(summary_file, "r", encoding="utf-8") as f:
                            summary = json.load(f)
                        # key 带上 label_source 前缀，确保不同来源的同名数据集不冲突
                        task_name = f"{label_source}/{dataset}/{split}"
                        baselines_by_mode[mode][task_name] = summary
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"  [警告] 加载失败: {summary_file}: {e}")

    return baselines_by_mode


def load_model_results(eval_dir: Path) -> dict:
    """
    从模型评估结果目录递归加载所有 eval_summary.json。

    支持两种目录结构:
      1. eval_dir/<label_source>/<dataset>/<split>/eval_summary.json
      2. eval_dir/<dataset>/<split>/eval_summary.json

    返回: {task_name: summary_dict}
    """
    results = {}

    for root, dirs, files in os.walk(eval_dir):
        # 跳过 staging 目录（里面是符号链接的 wav 文件，非常大）
        dirs[:] = [d for d in dirs if d != "staging" and not d.startswith(".")]

        if "eval_summary.json" in files:
            summary_file = Path(root) / "eval_summary.json"
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            # 从路径推断 task_name
            rel_path = summary_file.parent.relative_to(eval_dir)
            parts = rel_path.parts
            task_name = "/".join(parts)
            results[task_name] = summary

    return results


def _has_eval_summary(directory: Path, max_depth: int = 4) -> bool:
    """快速检查目录下是否有 eval_summary.json（限制深度，跳过 staging）"""
    def _search(d, depth):
        if depth > max_depth:
            return False
        try:
            for item in d.iterdir():
                if item.name == "eval_summary.json" and item.is_file():
                    return True
                if item.is_dir() and item.name != "staging" and not item.name.startswith("."):
                    if _search(item, depth + 1):
                        return True
        except (PermissionError, OSError):
            pass
        return False
    return _search(directory, 0)


def discover_model_dirs(eval_dir: Path) -> list:
    """
    判断 eval_dir 是单个模型目录还是包含多个模型的父目录。

    返回: [(model_name, model_dir), ...]
    """
    model_dirs = []
    for d in sorted(eval_dir.iterdir()):
        if not d.is_dir():
            continue
        # 只检查名字像模型权重的目录
        if not (d.name.startswith("model.pt") or d.name in ("baseline", "finetuned", "seaco")):
            continue
        # 快速检查是否有评估结果（跳过 staging 目录）
        if _has_eval_summary(d):
            model_dirs.append((d.name, d))

    if model_dirs:
        return model_dirs

    # eval_dir 本身可能就是一个模型目录
    if _has_eval_summary(eval_dir):
        return [(eval_dir.name, eval_dir)]

    return []


def _match_task_to_baseline(task_name: str, baseline_data: dict):
    """将当前模型的 task_name 匹配到基线的 dataset/split key"""
    parts = task_name.split("/")
    # 策略1: 去掉 label_source 前缀
    if len(parts) >= 3:
        short_key = "/".join(parts[1:])
        if short_key in baseline_data:
            return short_key
    # 策略2: 直接匹配
    if task_name in baseline_data:
        return task_name
    # 策略3: 最后两段
    if len(parts) >= 2:
        short_key = "/".join(parts[-2:])
        if short_key in baseline_data:
            return short_key
    return None


def _format_delta(new_val: float, base_val: float) -> str:
    """格式化 delta 值"""
    delta = new_val - base_val
    if abs(delta) < 0.0001:
        return "="
    sign = "↓" if delta < 0 else "↑"
    return f"{sign}{abs(delta):.2%}"


def _calc_weighted_metrics(summaries: list) -> dict:
    """从 summary 列表计算加权 CER/SER"""
    t_sub = sum(s.get("substitutions", 0) for s in summaries)
    t_ins = sum(s.get("insertions", 0) for s in summaries)
    t_del = sum(s.get("deletions", 0) for s in summaries)
    t_ref = sum(s.get("total_ref_chars", 0) for s in summaries)
    t_correct = sum(s.get("correct_sentences", 0) for s in summaries)
    t_count = sum(s.get("total_count", 0) for s in summaries)
    cer = (t_sub + t_ins + t_del) / t_ref if t_ref > 0 else 0
    ser = 1 - t_correct / t_count if t_count > 0 else 0
    return {"cer": cer, "ser": ser, "count": t_count, "ref": t_ref}


def generate_baseline_comparison(model_name: str, model_results: dict,
                                  baselines_by_mode: dict, output_dir: Path,
                                  output_filename: str = "baseline_comparison_report.md"):
    """
    生成单个模型与所有基线模式的对比报告。
    """
    report_path = output_dir / output_filename
    lines = []
    lines.append(f"# {model_name} vs 基线模型 对比报告\n")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 确定有数据的基线模式
    mode_order = ["seaco", "online_model", "finetuned"]
    active_modes = [m for m in mode_order if m in baselines_by_mode and baselines_by_mode[m]]

    if not active_modes:
        lines.append("> ⚠️ 未找到任何基线数据\n")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  [警告] 未找到任何基线数据")
        return

    mode_labels = {"seaco": "SeACo基准", "online_model": "线上模型", "finetuned": "旧微调"}

    # 建立匹配关系
    task_baselines = {}
    unmatched_current = []

    for task_name in sorted(model_results.keys()):
        matched_any = False
        task_baselines[task_name] = {}
        for mode in active_modes:
            bl_key = _match_task_to_baseline(task_name, baselines_by_mode[mode])
            if bl_key is not None:
                task_baselines[task_name][mode] = baselines_by_mode[mode][bl_key]
                matched_any = True
        if not matched_any:
            unmatched_current.append(task_name)

    matched_tasks = [t for t in sorted(model_results.keys()) if task_baselines.get(t)]

    if not matched_tasks:
        lines.append("> ⚠️ 未找到可匹配的基线数据集\n")
        lines.append(f"当前模型数据集: {sorted(model_results.keys())}\n")
        lines.append(f"基线模式: {active_modes}\n")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  [警告] 未找到可匹配的基线数据集")
        return

    # === 1. 总览 ===
    lines.append("## 总览\n")

    cur_metrics = _calc_weighted_metrics([model_results[t] for t in matched_tasks])

    header_cols = [model_name]
    for mode in active_modes:
        header_cols.append(mode_labels.get(mode, mode))
    header_cols_delta = []
    for mode in active_modes:
        header_cols_delta.append(f"vs {mode_labels.get(mode, mode)}")

    lines.append("| 指标 | " + " | ".join(header_cols) + " | " + " | ".join(header_cols_delta) + " |")
    lines.append("|------|" + "|".join(["------" for _ in header_cols]) + "|" + "|".join(["------" for _ in header_cols_delta]) + "|")

    # CER 行
    row = f"| 加权 CER | {cur_metrics['cer']:.2%}"
    for mode in active_modes:
        mode_summaries = [baselines_by_mode[mode][_match_task_to_baseline(t, baselines_by_mode[mode])]
                          for t in matched_tasks
                          if _match_task_to_baseline(t, baselines_by_mode[mode]) is not None]
        m = _calc_weighted_metrics(mode_summaries)
        row += f" | {m['cer']:.2%}"
    for mode in active_modes:
        mode_summaries = [baselines_by_mode[mode][_match_task_to_baseline(t, baselines_by_mode[mode])]
                          for t in matched_tasks
                          if _match_task_to_baseline(t, baselines_by_mode[mode]) is not None]
        m = _calc_weighted_metrics(mode_summaries)
        row += f" | {_format_delta(cur_metrics['cer'], m['cer'])}"
    lines.append(row + " |")

    # SER 行
    row = f"| 加权 SER | {cur_metrics['ser']:.2%}"
    for mode in active_modes:
        mode_summaries = [baselines_by_mode[mode][_match_task_to_baseline(t, baselines_by_mode[mode])]
                          for t in matched_tasks
                          if _match_task_to_baseline(t, baselines_by_mode[mode]) is not None]
        m = _calc_weighted_metrics(mode_summaries)
        row += f" | {m['ser']:.2%}"
    for mode in active_modes:
        mode_summaries = [baselines_by_mode[mode][_match_task_to_baseline(t, baselines_by_mode[mode])]
                          for t in matched_tasks
                          if _match_task_to_baseline(t, baselines_by_mode[mode]) is not None]
        m = _calc_weighted_metrics(mode_summaries)
        row += f" | {_format_delta(cur_metrics['ser'], m['ser'])}"
    lines.append(row + " |")
    lines.append("")

    # === 2. 逐数据集 CER 对比 ===
    lines.append("## 逐数据集 CER 对比\n")
    header = "| 数据集 | " + model_name + " | " + " | ".join(mode_labels.get(m, m) for m in active_modes) + " | " + " | ".join(f"vs {mode_labels.get(m, m)}" for m in active_modes) + " |"
    sep = "|--------|--------" + "|--------" * len(active_modes) + "|--------" * len(active_modes) + "|"
    lines.append(header)
    lines.append(sep)

    for task in matched_tasks:
        cur_cer = model_results[task].get("overall_cer", 0)
        row = f"| {task} | {cur_cer:.2%}"
        for mode in active_modes:
            bl = task_baselines[task].get(mode)
            if bl:
                row += f" | {bl.get('overall_cer', 0):.2%}"
            else:
                row += " | -"
        for mode in active_modes:
            bl = task_baselines[task].get(mode)
            if bl:
                row += f" | {_format_delta(cur_cer, bl.get('overall_cer', 0))}"
            else:
                row += " | -"
        lines.append(row + " |")
    lines.append("")

    # === 3. 逐数据集 SER 对比 ===
    lines.append("## 逐数据集 SER 对比\n")
    lines.append(header.replace("CER", "SER"))
    lines.append(sep)

    for task in matched_tasks:
        cur_ser = model_results[task].get("ser", 0)
        row = f"| {task} | {cur_ser:.2%}"
        for mode in active_modes:
            bl = task_baselines[task].get(mode)
            if bl:
                row += f" | {bl.get('ser', 0):.2%}"
            else:
                row += " | -"
        for mode in active_modes:
            bl = task_baselines[task].get(mode)
            if bl:
                row += f" | {_format_delta(cur_ser, bl.get('ser', 0))}"
            else:
                row += " | -"
        lines.append(row + " |")
    lines.append("")

    # === 4. 改善/退化汇总 ===
    lines.append("## 改善/退化汇总\n")

    for mode in active_modes:
        label = mode_labels.get(mode, mode)
        improved = []
        degraded = []
        unchanged = []

        for task in matched_tasks:
            bl = task_baselines[task].get(mode)
            if bl is None:
                continue
            cur_cer = model_results[task].get("overall_cer", 0)
            base_cer = bl.get("overall_cer", 0)
            delta = cur_cer - base_cer
            if delta < -0.0001:
                improved.append((task, delta))
            elif delta > 0.0001:
                degraded.append((task, delta))
            else:
                unchanged.append(task)

        lines.append(f"### vs {label}\n")
        if improved:
            improved.sort(key=lambda x: x[1])
            lines.append(f"✅ 改善 ({len(improved)} 个):")
            for task, delta in improved:
                lines.append(f"- {task}: CER ↓{abs(delta):.2%}")
        if degraded:
            degraded.sort(key=lambda x: x[1], reverse=True)
            lines.append(f"\n❌ 退化 ({len(degraded)} 个):")
            for task, delta in degraded:
                lines.append(f"- {task}: CER ↑{abs(delta):.2%}")
        if unchanged:
            lines.append(f"\n➖ 持平 ({len(unchanged)} 个): {', '.join(unchanged)}")
        lines.append("")

    if unmatched_current:
        lines.append(f"### ⚠️ 无基线对比 ({len(unmatched_current)} 个)\n")
        for task in unmatched_current:
            lines.append(f"- {task}")
        lines.append("")

    report_content = "\n".join(lines)
    report_path.write_text(report_content, encoding="utf-8")
    print(f"  报告 -> {report_path}")

    # JSON 格式
    json_path = output_dir / output_filename.replace(".md", "_summary.json")
    json_data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "baseline_modes": active_modes,
        "overall": {mode: {} for mode in active_modes},
        "per_dataset": {},
    }
    # 计算各基线模式的总体指标
    for mode in active_modes:
        mode_summaries = [baselines_by_mode[mode][_match_task_to_baseline(t, baselines_by_mode[mode])]
                          for t in matched_tasks
                          if _match_task_to_baseline(t, baselines_by_mode[mode]) is not None]
        m = _calc_weighted_metrics(mode_summaries)
        json_data["overall"][mode] = {
            "baseline_cer": round(m["cer"], 4),
            "baseline_ser": round(m["ser"], 4),
            "cer_delta": round(cur_metrics["cer"] - m["cer"], 4),
            "ser_delta": round(cur_metrics["ser"] - m["ser"], 4),
        }
    json_data["overall"]["model"] = {
        "cer": round(cur_metrics["cer"], 4),
        "ser": round(cur_metrics["ser"], 4),
    }

    for task in matched_tasks:
        task_data = {"model": model_results[task], "baselines": {}}
        for mode in active_modes:
            bl = task_baselines[task].get(mode)
            if bl:
                task_data["baselines"][mode] = {
                    "summary": bl,
                    "cer_delta": round(model_results[task].get("overall_cer", 0) - bl.get("overall_cer", 0), 4),
                    "ser_delta": round(model_results[task].get("ser", 0) - bl.get("ser", 0), 4),
                }
        json_data["per_dataset"][task] = task_data

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"  JSON  -> {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="基线对比报告生成（从已有评估结果）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 为单个模型评估结果生成对比报告
  python gen_baseline_report.py --eval-dir /data1/eval_output/v0.1.0/model.pt.best

  # 为整个版本目录下所有模型生成对比报告
  python gen_baseline_report.py --eval-dir /data1/eval_output/v0.1.0

  # 指定基线目录
  python gen_baseline_report.py --eval-dir /data1/eval_output/v0.1.0 \\
      --baseline-dir /data1/eval_output/base

  # 只对比 seaco 和 online_model
  python gen_baseline_report.py --eval-dir /data1/eval_output/v0.1.0 \\
      --baseline-modes seaco online_model
        """,
    )

    parser.add_argument("--eval-dir", required=True,
                        help="模型评估结果目录（单个模型或包含多个模型的父目录）")
    parser.add_argument("--baseline-dir", default=DEFAULT_BASELINE_DIR,
                        help=f"基线评估结果目录 (default: {DEFAULT_BASELINE_DIR})")
    parser.add_argument("--baseline-modes", nargs="+", default=None,
                        help="只对比指定基线模式 (如: seaco online_model finetuned)")
    parser.add_argument("--output", default="baseline_comparison_report.md",
                        help="输出报告文件名 (default: baseline_comparison_report.md)")

    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    baseline_dir = Path(args.baseline_dir)

    if not eval_dir.exists():
        print(f"[错误] 评估结果目录不存在: {eval_dir}")
        return

    # 1. 加载基线
    print(f"加载基线: {baseline_dir}")
    baselines_by_mode = load_baseline_results(baseline_dir, args.baseline_modes)
    if not baselines_by_mode:
        print("[错误] 未找到基线评估结果")
        return

    for mode, results in baselines_by_mode.items():
        print(f"  [{mode}]: {len(results)} 个数据集/集合")

    # 2. 发现模型目录
    model_dirs = discover_model_dirs(eval_dir)
    print(f"\n发现 {len(model_dirs)} 个模型:")
    for name, path in model_dirs:
        print(f"  - {name}: {path}")

    # 3. 逐模型生成报告
    print()
    for model_name, model_dir in model_dirs:
        print(f"处理: {model_name}")
        model_results = load_model_results(model_dir)
        if not model_results:
            print(f"  [跳过] 未找到 eval_summary.json")
            continue

        print(f"  加载到 {len(model_results)} 个评估结果")
        generate_baseline_comparison(
            model_name, model_results,
            baselines_by_mode, model_dir,
            output_filename=args.output,
        )

    print("\n完成!")


if __name__ == "__main__":
    main()
