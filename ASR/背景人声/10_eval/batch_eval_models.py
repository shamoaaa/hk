#!/usr/bin/env python3
"""
多模型权重批量评估脚本

扫描指定目录下的所有模型权重文件（model.pt.*），对 /data1/label 下的数据集
进行批量评估，并自动与基线模型结果对比，生成对比报告。

目录结构约定：
    模型权重:  /data1/eval_output/v0.1.0/model.pt.best, model.pt.ep16.4000, ...
    标签目录:  /data1/label/{augment_label,reflow_label,speech_label}/{train,dev,test}/
    基线结果:  /data1/eval_output/base/{augment,reflow,speech}_eval_result/{seaco,finetuned,online_model}/...

用法:
    # 评估模型目录下所有权重（自动与基线对比）
    python batch_eval_models.py --model-dir /data1/eval_output/v0.1.0

    # 指定标签源和集合
    python batch_eval_models.py --model-dir /data1/eval_output/v0.1.0 \\
        --label-source all_label --splits dev test

    # 只评估指定权重文件
    python batch_eval_models.py --model-dir /data1/eval_output/v0.1.0 \\
        --ckpt-pattern "model.pt.best"

    # 指定数据集
    python batch_eval_models.py --model-dir /data1/eval_output/v0.1.0 \\
        --datasets aidatatang_200zh data_aishell

    # 不与基线对比
    python batch_eval_models.py --model-dir /data1/eval_output/v0.1.0 \\
        --baseline-dir ""

    # 指定其他基线目录
    python batch_eval_models.py --model-dir /data1/eval_output/v0.1.0 \\
        --baseline-dir /data1/eval_output/v0.0.9

    # 采样评估（快速验证）
    python batch_eval_models.py --model-dir /data1/eval_output/v0.1.0 --sample 500
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from eval_utils import (
    select_gpus, setup_logging, format_duration,
    load_labels, load_hotwords,
    run_eval_items, build_eval_items_from_scp, generate_combined_report,
)


# ============ 默认配置 ============
DEFAULT_BASE_MODEL = "/data1/pretrained_models/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
DEFAULT_LABEL_ROOT = "/data1/label"
DEFAULT_OUTPUT_ROOT = "/data1/eval_output/v0.2.0"
DEFAULT_BASELINE_DIR = "/data1/eval_output/base"
# ==================================


def discover_checkpoints(model_dir: Path, pattern: str = None) -> list:
    """
    扫描模型目录下的所有权重文件。

    支持的文件名模式:
      - model.pt.best
      - model.pt.ep<N>.<step>
      - model.pt.avg<N>
      - model.pt

    返回: [(ckpt_path, ckpt_name), ...] 按名称排序
    """
    ckpt_files = []
    for f in sorted(model_dir.iterdir()):
        if not f.is_file():
            continue
        name = f.name
        if not name.startswith("model.pt"):
            continue
        # 排除非权重文件
        if name.endswith((".log", ".txt", ".json", ".yaml", ".yml")):
            continue

        if pattern:
            patterns = [p.strip() for p in pattern.split(",")]
            if not any(p in name for p in patterns):
                continue

        ckpt_files.append((f, name))

    return ckpt_files


def discover_datasets_from_label(label_root: Path, label_sources: list,
                                  splits: list, datasets_filter: list = None) -> dict:
    """
    从标签目录发现可用数据集。

    返回: {dataset_key: {split: (scp_path, text_path), ...}, ...}
    """
    all_datasets = {}

    for source in label_sources:
        source_dir = label_root / source
        if not source_dir.exists():
            print(f"[警告] 标签源目录不存在: {source_dir}")
            continue

        for split in splits:
            split_dir = source_dir / split
            if not split_dir.exists():
                continue

            # 合并后的 all_label 格式（wav.scp + text）
            merged_scp = split_dir / "wav.scp"
            merged_text = split_dir / "text"
            if merged_scp.exists() and merged_text.exists():
                ds_key = f"{source}"
                if ds_key not in all_datasets:
                    all_datasets[ds_key] = {}
                all_datasets[ds_key][split] = (merged_scp, merged_text)
                continue

            # 标准格式：<dataset>_wav.scp + <dataset>_text
            for scp_file in sorted(split_dir.glob("*_wav.scp")):
                dataset_name = scp_file.name.replace("_wav.scp", "")
                text_file = split_dir / f"{dataset_name}_text"
                if not text_file.exists():
                    continue

                if datasets_filter and dataset_name not in datasets_filter:
                    continue

                ds_key = f"{source}/{dataset_name}"
                if ds_key not in all_datasets:
                    all_datasets[ds_key] = {}
                all_datasets[ds_key][split] = (scp_file, text_file)

    return all_datasets


def load_finetuned_model(base_model_dir: str, finetune_ckpt: str, device: str, logger):
    """加载微调模型: 基准模型结构 + 微调权重"""
    import torch
    from funasr import AutoModel

    logger.info(f"基准模型: {base_model_dir}")
    logger.info(f"微调权重: {finetune_ckpt}")

    load_start = time.time()
    auto_model = AutoModel(model=base_model_dir, device=device)

    ckpt_path = Path(finetune_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"微调权重文件不存在: {finetune_ckpt}")

    state_dict = torch.load(str(ckpt_path), map_location=device)

    # 兼容不同的 checkpoint 格式
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]

    model_inner = auto_model.model
    missing, unexpected = model_inner.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"缺少的权重 ({len(missing)} 个): {missing[:5]}...")
    if unexpected:
        logger.warning(f"多余的权重 ({len(unexpected)} 个): {unexpected[:5]}...")
    model_inner.eval()

    logger.info(f"微调模型加载完成，耗时: {format_duration(time.time() - load_start)}")
    return auto_model


def load_baseline_model(base_model_dir: str, device: str, logger):
    """加载基准模型（不加载微调权重）"""
    from funasr import AutoModel

    logger.info(f"加载基准模型: {base_model_dir}")
    load_start = time.time()
    model = AutoModel(model=base_model_dir, device=device)
    logger.info(f"基准模型加载完成，耗时: {format_duration(time.time() - load_start)}")
    return model


def load_baseline_results(baseline_dir: Path, logger) -> dict:
    """
    从基线评估结果目录加载所有 eval_summary.json。

    基线目录结构:
      baseline_dir/
        {eval_type}/          # augment_eval_result, reflow_eval_result, speech_eval_result
          {mode}/             # seaco, finetuned, online_model
            {dataset}/        # data_aishell, aidatatang_200zh, ...
              {split}/        # train, dev, test
                eval_summary.json

    返回:
      baselines_by_mode: {mode: {label_source/dataset/split: summary}} — 按基线模式分组
          key 带 label_source 前缀，确保不同来源的同名数据集不冲突
    """
    # eval_type 目录名到 label_source 的映射
    EVAL_TYPE_TO_LABEL = {
        "augment_eval_result": "augment_label",
        "reflow_eval_result": "reflow_label",
        "speech_eval_result": "speech_label",
    }

    baselines_by_mode = {}  # {mode: {label_source/dataset/split: summary}}

    if not baseline_dir.exists():
        logger.warning(f"基线目录不存在: {baseline_dir}")
        return {}

    logger.info(f"加载基线评估结果: {baseline_dir}")

    for eval_type_dir in sorted(baseline_dir.iterdir()):
        if not eval_type_dir.is_dir():
            continue
        eval_type = eval_type_dir.name
        label_source = EVAL_TYPE_TO_LABEL.get(eval_type, eval_type)

        for mode_dir in sorted(eval_type_dir.iterdir()):
            if not mode_dir.is_dir():
                continue
            mode = mode_dir.name  # seaco, finetuned, online_model

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
                        task_name = f"{label_source}/{dataset}/{split}"
                        baselines_by_mode[mode][task_name] = summary
                    except (json.JSONDecodeError, IOError) as e:
                        logger.warning(f"加载基线结果失败: {summary_file}: {e}")

    for mode, results in baselines_by_mode.items():
        logger.info(f"  基线 [{mode}]: {len(results)} 个数据集/集合")

    return baselines_by_mode


def _format_delta(new_val: float, base_val: float) -> str:
    """格式化 delta 值，负值（变好）显示绿色箭头，正值（变差）显示红色箭头"""
    delta = new_val - base_val
    if abs(delta) < 0.0001:
        return "="
    sign = "↓" if delta < 0 else "↑"
    return f"{sign}{abs(delta):.2%}"


def _generate_model_comparison_report(all_model_results: dict, output_dir: Path, logger):
    """
    生成跨模型对比报告。
    all_model_results: {ckpt_name: {task_name: summary_dict, ...}, ...}
    """
    report_path = output_dir / "model_comparison_report.md"
    lines = []
    lines.append("# 多模型权重对比评估报告\n")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 收集所有任务名和模型名
    all_tasks = set()
    for ckpt_results in all_model_results.values():
        all_tasks.update(ckpt_results.keys())
    all_tasks = sorted(all_tasks)
    ckpt_names = list(all_model_results.keys())

    # === 1. 总览表格 ===
    lines.append("## 模型总览\n")
    lines.append("| 模型权重 | 加权CER | 加权SER | 总条数 | 总字符数 |")
    lines.append("|----------|---------|---------|--------|----------|")

    model_overall = {}
    for ckpt_name in ckpt_names:
        results = all_model_results[ckpt_name]
        t_sub, t_ins, t_del, t_ref = 0, 0, 0, 0
        t_correct, t_count = 0, 0
        for task, s in results.items():
            if s is None:
                continue
            t_sub += s.get("substitutions", 0)
            t_ins += s.get("insertions", 0)
            t_del += s.get("deletions", 0)
            t_ref += s.get("total_ref_chars", 0)
            t_correct += s.get("correct_sentences", 0)
            t_count += s.get("total_count", 0)

        cer = (t_sub + t_ins + t_del) / t_ref if t_ref > 0 else 0
        ser = 1 - t_correct / t_count if t_count > 0 else 0
        model_overall[ckpt_name] = {
            "cer": cer, "ser": ser, "count": t_count, "ref": t_ref
        }
        lines.append(f"| {ckpt_name} | {cer:.2%} | {ser:.2%} | {t_count} | {t_ref} |")

    lines.append("")

    # === 2. 按数据集对比 CER ===
    lines.append("## 按数据集 CER 对比\n")
    header = "| 数据集 | " + " | ".join(ckpt_names) + " |"
    sep = "|--------|" + "|".join(["--------" for _ in ckpt_names]) + "|"
    lines.append(header)
    lines.append(sep)

    for task in all_tasks:
        row = f"| {task} |"
        for ckpt_name in ckpt_names:
            s = all_model_results[ckpt_name].get(task)
            if s and "overall_cer" in s:
                row += f" {s['overall_cer']:.2%} |"
            else:
                row += " - |"
        lines.append(row)
    lines.append("")

    # === 3. 按数据集对比 SER ===
    lines.append("## 按数据集 SER 对比\n")
    lines.append(header)
    lines.append(sep)

    for task in all_tasks:
        row = f"| {task} |"
        for ckpt_name in ckpt_names:
            s = all_model_results[ckpt_name].get(task)
            if s and "ser" in s:
                row += f" {s['ser']:.2%} |"
            else:
                row += " - |"
        lines.append(row)
    lines.append("")

    # === 4. 最佳模型推荐 ===
    if model_overall:
        best_cer = min(model_overall, key=lambda k: model_overall[k]["cer"])
        best_ser = min(model_overall, key=lambda k: model_overall[k]["ser"])
        lines.append("## 最佳模型\n")
        lines.append(f"- 最低加权 CER: **{best_cer}** ({model_overall[best_cer]['cer']:.2%})")
        lines.append(f"- 最低加权 SER: **{best_ser}** ({model_overall[best_ser]['ser']:.2%})")
        lines.append("")

    report_content = "\n".join(lines)
    report_path.write_text(report_content, encoding="utf-8")
    logger.info(f"跨模型对比报告 -> {report_path}")

    # JSON 格式
    json_path = output_dir / "model_comparison_summary.json"
    json_data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": {},
    }
    for ckpt_name in ckpt_names:
        json_data["models"][ckpt_name] = {
            "overall": model_overall.get(ckpt_name, {}),
            "per_dataset": {
                task: all_model_results[ckpt_name].get(task)
                for task in all_tasks
            },
        }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    logger.info(f"跨模型对比 JSON -> {json_path}")


def _match_task_to_baseline(task_name: str, baseline_data: dict):
    """
    将当前模型的 task_name 匹配到基线的 dataset/split key。
    task_name 格式: "augment_label/data_aishell/test" 或 "reflow_label/s10_reflow/test"
    baseline key 格式: "data_aishell/test" 或 "s10_reflow/test"
    """
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


def _generate_baseline_comparison_report(ckpt_name: str, ckpt_results: dict,
                                          baselines_by_mode: dict,
                                          output_dir: Path, logger):
    """
    生成单个模型权重与所有基线模式（seaco / online_model / finetuned）的对比报告。

    ckpt_results: {task_name: summary_dict}  当前模型的评估结果
    baselines_by_mode: {mode: {dataset/split: summary}}  各基线模式的结果
    """
    report_path = output_dir / "baseline_comparison_report.md"
    lines = []
    lines.append(f"# {ckpt_name} vs 基线模型 对比报告\n")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 确定有数据的基线模式，固定显示顺序
    mode_order = ["seaco", "online_model", "finetuned"]
    active_modes = [m for m in mode_order if m in baselines_by_mode and baselines_by_mode[m]]

    if not active_modes:
        lines.append("> ⚠️ 未找到任何基线数据\n")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.warning(f"[{ckpt_name}] 未找到任何基线数据")
        return

    # 建立匹配关系: {cur_task: {mode: (baseline_key, baseline_summary)}}
    task_baselines = {}
    unmatched_current = []

    for task_name, summary in sorted(ckpt_results.items()):
        matched_any = False
        task_baselines[task_name] = {}
        for mode in active_modes:
            bl_key = _match_task_to_baseline(task_name, baselines_by_mode[mode])
            if bl_key is not None:
                task_baselines[task_name][mode] = baselines_by_mode[mode][bl_key]
                matched_any = True
        if not matched_any:
            unmatched_current.append(task_name)

    matched_tasks = [t for t in sorted(ckpt_results.keys()) if task_baselines.get(t)]

    if not matched_tasks:
        lines.append("> ⚠️ 未找到可匹配的基线数据集\n")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.warning(f"[{ckpt_name}] 未找到可匹配的基线数据集")
        return

    mode_labels = {"seaco": "SeACo基准", "online_model": "线上模型", "finetuned": "旧微调"}

    # === 1. 总览 ===
    lines.append("## 总览\n")

    cur_metrics = _calc_weighted_metrics(
        [ckpt_results[t] for t in matched_tasks]
    )

    header_cols = [f"{ckpt_name}"]
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
    header = "| 数据集 | " + ckpt_name + " | " + " | ".join(mode_labels.get(m, m) for m in active_modes) + " | " + " | ".join(f"vs {mode_labels.get(m, m)}" for m in active_modes) + " |"
    sep = "|--------|--------" + "|--------" * len(active_modes) + "|--------" * len(active_modes) + "|"
    lines.append(header)
    lines.append(sep)

    for task in matched_tasks:
        cur_cer = ckpt_results[task].get("overall_cer", 0)
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
        cur_ser = ckpt_results[task].get("ser", 0)
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

    # === 4. 改善/退化汇总（按每个基线模式分别统计） ===
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
            cur_cer = ckpt_results[task].get("overall_cer", 0)
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
    logger.info(f"基线对比报告 -> {report_path}")

    # JSON 格式
    json_path = output_dir / "baseline_comparison_summary.json"
    json_data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": ckpt_name,
        "baseline_modes": active_modes,
        "per_dataset": {},
    }
    for task in matched_tasks:
        task_data = {"model": ckpt_results[task], "baselines": {}}
        for mode in active_modes:
            bl = task_baselines[task].get(mode)
            if bl:
                task_data["baselines"][mode] = {
                    "summary": bl,
                    "cer_delta": round(ckpt_results[task].get("overall_cer", 0) - bl.get("overall_cer", 0), 4),
                    "ser_delta": round(ckpt_results[task].get("ser", 0) - bl.get("ser", 0), 4),
                }
        json_data["per_dataset"][task] = task_data

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    logger.info(f"基线对比 JSON -> {json_path}")


def run_batch_eval_models(args):
    model_dir = Path(args.model_dir)
    label_root = Path(args.label_root)
    output_root = Path(args.output_root)
    splits = args.splits
    label_sources = args.label_source

    # 1. 发现模型权重
    checkpoints = discover_checkpoints(model_dir, args.ckpt_pattern)
    if not checkpoints:
        print(f"[错误] 在 {model_dir} 下未找到模型权重文件")
        print("  支持的文件名: model.pt.best, model.pt.ep*.*, model.pt.avg*, model.pt")
        if args.ckpt_pattern:
            print(f"  当前过滤: {args.ckpt_pattern}")
        return

    # 2. 发现数据集
    all_datasets = discover_datasets_from_label(
        label_root, label_sources, splits, args.datasets
    )
    if not all_datasets:
        print(f"[错误] 在 {label_root} 下未找到可评估的数据集")
        print(f"  标签源: {label_sources}")
        print(f"  集合: {splits}")
        return

    # 3. 构建评估模型列表
    eval_models = []
    if args.with_baseline:
        eval_models.append(("baseline", None))
    for ckpt_path, ckpt_name in checkpoints:
        eval_models.append((ckpt_name, ckpt_path))

    # 3.5 加载基线评估结果（用于对比）
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else None
    baselines_by_mode = {}
    if baseline_dir and baseline_dir.exists():
        _tmp_logger, _ = setup_logging(output_root, "baseline_load")
        baselines_by_mode = load_baseline_results(baseline_dir, _tmp_logger)
        if baselines_by_mode:
            total_bl = sum(len(v) for v in baselines_by_mode.values())
            modes_str = ", ".join(f"{m}({len(v)})" for m, v in baselines_by_mode.items())
            print(f"基线结果:     已加载 [{modes_str}] (来自 {baseline_dir})")
        else:
            print(f"基线结果:     未找到有效结果 ({baseline_dir})")
    elif baseline_dir:
        print(f"基线结果:     目录不存在 ({baseline_dir})")
    else:
        print(f"基线结果:     未指定 (使用 --baseline-dir 指定)")

    # 构建评估任务
    eval_tasks = []
    for ds_key, split_info in sorted(all_datasets.items()):
        for split, (scp_path, text_path) in sorted(split_info.items()):
            task_name = f"{ds_key}/{split}"
            eval_tasks.append((task_name, ds_key, split, scp_path, text_path))

    total_task_count = len(eval_models) * len(eval_tasks)

    # 4. 打印评估计划
    print("\n" + "=" * 70)
    print("多模型权重批量评估计划")
    print("=" * 70)
    print(f"模型目录:     {model_dir}")
    print(f"基准模型:     {args.base_model}")
    print(f"标签根目录:   {label_root}")
    print(f"标签源:       {label_sources}")
    print(f"评估集:       {splits}")
    print(f"输出根目录:   {output_root}")
    print(f"评估基准模型: {'是' if args.with_baseline else '否'}")
    print()
    print(f"模型权重 ({len(eval_models)} 个):")
    for ckpt_name, ckpt_path in eval_models:
        if ckpt_path:
            size_mb = ckpt_path.stat().st_size / (1024 * 1024)
            print(f"  - {ckpt_name} ({size_mb:.0f} MB)")
        else:
            print(f"  - {ckpt_name} (基准模型)")
    print()
    print(f"评估任务 ({len(eval_tasks)} 个数据集/集合):")
    for task_name, _, _, _, _ in eval_tasks:
        print(f"  - {task_name}")
    print()
    print(f"总评估任务: {total_task_count} 个 ({len(eval_models)} 模型 x {len(eval_tasks)} 数据集)")
    print("=" * 70)

    # 加载热词
    hotword_str = ""
    if args.hotword:
        hotword_str = Path(args.hotword).read_text(encoding="utf-8").strip().replace("\n", " ")
        print(f"热词: {args.hotword}")

    # 5. 逐模型评估
    all_model_results = {}  # {ckpt_name: {task_name: summary}}
    total_start = time.time()

    for model_idx, (ckpt_name, ckpt_path) in enumerate(eval_models):
        model_output = output_root / ckpt_name
        logger, log_file = setup_logging(model_output, f"eval_{ckpt_name}")

        logger.info("=" * 60)
        logger.info(f"模型 [{model_idx + 1}/{len(eval_models)}]: {ckpt_name}")
        logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        # 加载模型
        device = select_gpus(args.device, logger)
        try:
            if ckpt_path is None:
                model = load_baseline_model(args.base_model, device, logger)
            else:
                model = load_finetuned_model(
                    args.base_model, str(ckpt_path), device, logger
                )
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            continue

        # 逐数据集评估
        dir_summaries = {}
        for task_idx, (task_name, ds_key, split, scp_path, text_path) in enumerate(eval_tasks):
            logger.info(f"\n{'─' * 50}")
            logger.info(f"任务 [{task_idx + 1}/{len(eval_tasks)}]: {task_name}")
            logger.info(f"SCP:  {scp_path}")
            logger.info(f"TEXT: {text_path}")

            # 加载标签
            labels = load_labels(text_path, logger)
            if not labels:
                logger.warning(f"[{task_name}] 标签为空，跳过")
                continue

            # 直接从 scp + labels 构建 eval_items（无需 staging 目录）
            eval_items = build_eval_items_from_scp(scp_path, labels, logger)
            if not eval_items:
                logger.warning(f"[{task_name}] 无有效评估条目，跳过")
                continue

            # 评估输出目录
            eval_output = model_output / ds_key / split

            summary = run_eval_items(
                eval_items=eval_items,
                output_dir=eval_output,
                model=model,
                hotword_str=hotword_str,
                args=args,
                logger=logger,
                dir_name=task_name,
            )
            if summary:
                dir_summaries[task_name] = summary
                logger.info(f"[{task_name}] CER: {summary.get('overall_cer', 'N/A')}")

        all_model_results[ckpt_name] = dir_summaries

        # 生成单模型汇总报告
        if len(dir_summaries) > 1:
            report_title = f"{ckpt_name} 批量评估汇总"
            generate_combined_report(dir_summaries, model_output, logger,
                                     report_title=report_title)

        # 生成与基线的对比报告
        if baselines_by_mode and dir_summaries:
            _generate_baseline_comparison_report(
                ckpt_name, dir_summaries,
                baselines_by_mode,
                model_output, logger,
            )

        # 打印单模型结果摘要
        logger.info("\n" + "=" * 60)
        logger.info(f"{ckpt_name} 评估结果摘要")
        logger.info("=" * 60)
        logger.info(f"{'数据集/集合':<40} {'CER':>8} {'SER':>8} {'条数':>8}")
        logger.info("-" * 66)
        for task_name, summary in sorted(dir_summaries.items()):
            cer = summary.get("overall_cer", "N/A")
            ser = summary.get("ser", "N/A")
            total = summary.get("total_count", "N/A")
            cer_str = f"{cer:.2%}" if isinstance(cer, (int, float)) else str(cer)
            ser_str = f"{ser:.2%}" if isinstance(ser, (int, float)) else str(ser)
            logger.info(f"{task_name:<40} {cer_str:>8} {ser_str:>8} {total:>8}")
        logger.info("=" * 60)

        # 释放显存
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # 6. 生成跨模型对比报告
    if len(all_model_results) > 1:
        logger, _ = setup_logging(output_root, "model_comparison")
        _generate_model_comparison_report(all_model_results, output_root, logger)

    # 7. 打印最终摘要
    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print("全部评估完成!")
    print("=" * 70)
    print(f"总耗时: {format_duration(total_time)}")
    print(f"输出目录: {output_root}")
    print()
    for ckpt_name, results in all_model_results.items():
        if not results:
            print(f"  {ckpt_name}: 无结果")
            continue
        t_sub = sum(s.get("substitutions", 0) for s in results.values() if s)
        t_ins = sum(s.get("insertions", 0) for s in results.values() if s)
        t_del = sum(s.get("deletions", 0) for s in results.values() if s)
        t_ref = sum(s.get("total_ref_chars", 0) for s in results.values() if s)
        cer = (t_sub + t_ins + t_del) / t_ref if t_ref > 0 else 0
        print(f"  {ckpt_name}: 加权CER = {cer:.2%}")
    if len(all_model_results) > 1:
        print(f"\n对比报告: {output_root / 'model_comparison_report.md'}")
    if baselines_by_mode:
        for ckpt_name in all_model_results:
            bl_report = output_root / ckpt_name / "baseline_comparison_report.md"
            if bl_report.exists():
                print(f"基线对比: {bl_report}")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(
        description="多模型权重批量评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 评估模型目录下所有权重（自动与基线对比）
  python batch_eval_models.py --model-dir /data1/eval_output/v0.1.0

  # 指定标签源和集合
  python batch_eval_models.py --model-dir /data1/eval_output/v0.1.0 \\
      --label-source all_label --splits dev test

  # 只评估指定权重
  python batch_eval_models.py --model-dir /data1/eval_output/v0.1.0 \\
      --ckpt-pattern "model.pt.best"

  # 只评估指定数据集
  python batch_eval_models.py --model-dir /data1/eval_output/v0.1.0 \\
      --datasets aidatatang_200zh data_aishell

  # 不与基线对比
  python batch_eval_models.py --model-dir /data1/eval_output/v0.1.0 \\
      --baseline-dir ""

  # 指定其他基线目录
  python batch_eval_models.py --model-dir /data1/eval_output/v0.1.0 \\
      --baseline-dir /data1/eval_output/v0.0.9

  # 采样快速评估
  python batch_eval_models.py --model-dir /data1/eval_output/v0.1.0 --sample 500

  # 使用多个标签源
  python batch_eval_models.py --model-dir /data1/eval_output/v0.1.0 \\
      --label-source augment_label reflow_label speech_label
        """,
    )

    # 模型
    parser.add_argument("--model-dir", required=True,
                        help="模型权重目录，扫描其中所有 model.pt.* 文件")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                        help=f"基准模型目录 (default: {DEFAULT_BASE_MODEL})")
    parser.add_argument("--ckpt-pattern", default=None,
                        help="权重文件名过滤，逗号分隔多个模式 (如 'best,avg')")
    parser.add_argument("--with-baseline", action="store_true",
                        help="同时评估基准模型作为 baseline 对比")
    parser.add_argument("--baseline-dir", default=DEFAULT_BASELINE_DIR,
                        help=f"基线评估结果目录，用于生成对比报告 (default: {DEFAULT_BASELINE_DIR})")

    # 数据
    parser.add_argument("--label-root", default=DEFAULT_LABEL_ROOT,
                        help=f"标签根目录 (default: {DEFAULT_LABEL_ROOT})")
    parser.add_argument("--label-source", nargs="+",
                        default=["augment_label", "reflow_label", "speech_label"],
                        help="标签源目录名 (default: augment_label reflow_label speech_label)")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="指定数据集名称，不指定则评估全部")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"],
                        help="评估哪些集合 (default: test)")

    # 输出
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                        help=f"输出根目录 (default: {DEFAULT_OUTPUT_ROOT})")

    # 评估参数
    parser.add_argument("--hotword", default=None, help="热词文件路径")
    parser.add_argument("--device", default="cuda:auto", help="设备 (default: cuda:auto)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="推理 batch 大小 (default: 64)")
    parser.add_argument("--file-batch", type=int, default=5000,
                        help="文件批大小 (default: 5000)")
    parser.add_argument("--sample", type=int, default=None,
                        help="随机采样 N 条（快速验证用）")
    parser.add_argument("--resume", action="store_true", help="断点续跑")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_batch_eval_models(args)
