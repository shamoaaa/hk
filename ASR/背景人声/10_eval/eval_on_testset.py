#!/usr/bin/env python3
"""
在 12_test 测试集上批量评估多个模型权重

支持两种目录结构:
  1. 单目录模式: --model-dir 指向包含 model.pt.* 的目录
  2. 多版本模式: --model-dir 指向包含多个版本子目录的父目录
     例如:
       /data1/eval_output/
       ├── v0.1.0/model/model.pt.best, model.pt.ep16.4000
       ├── v0.2.0/model/model.pt.best, model.pt.ep14.4000
       └── v0.3.0/model/model.pt.best, model.pt.ep27.4000

用法:
    # 多版本模式（自动扫描子目录）
    python eval_on_testset.py --model-dir /data1/eval_output

    # 单目录模式
    python eval_on_testset.py --model-dir /data1/eval_output/v0.1.0/model

    # 只评估 best 权重
    python eval_on_testset.py --model-dir /data1/eval_output --ckpt-pattern "best"

    # 只评估指定版本
    python eval_on_testset.py --model-dir /data1/eval_output --versions v0.2.0 v0.3.0

    # 采样快速验证
    python eval_on_testset.py --model-dir /data1/eval_output --sample 100

    # 同时评估基准模型（不加载微调权重）
    python eval_on_testset.py --model-dir /data1/eval_output --with-baseline
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
    load_labels, run_eval_items, build_eval_items_from_scp, generate_combined_report,
)


# ============ 默认配置（根据你的项目结构） ============
DEFAULT_BASE_MODEL = r"/data1/pretrained_models/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
DEFAULT_MODEL_DIR = r"/data1/eval_output"
DEFAULT_TEST_SCP = None
DEFAULT_TEST_TEXT = None
DEFAULT_OUTPUT_DIR = r"/data1/eval_output/testset_results"
# ======================================================


def discover_checkpoints(model_dir: Path, pattern: str = None) -> list:
    """
    扫描模型目录下的所有权重文件（单目录模式）。

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


def discover_checkpoints_multi_version(root_dir: Path, pattern: str = None,
                                        versions_filter: list = None) -> list:
    """
    扫描多版本目录结构下的所有权重文件。

    目录结构:
      root_dir/
      ├── v0.1.0/model/model.pt.best, model.pt.ep16.4000
      ├── v0.2.0/model/model.pt.best, model.pt.ep14.4000
      └── v0.3.0/model/model.pt.best, model.pt.ep27.4000

    返回: [(ckpt_path, display_name), ...] 按版本+名称排序
           display_name 格式: "v0.1.0/model.pt.best"
    """
    ckpt_files = []

    for version_dir in sorted(root_dir.iterdir()):
        if not version_dir.is_dir():
            continue
        version_name = version_dir.name

        # 版本过滤
        if versions_filter and version_name not in versions_filter:
            continue

        # 查找 model 子目录
        model_dir = version_dir / "model"
        if not model_dir.exists():
            # 也支持直接在版本目录下放权重
            model_dir = version_dir

        for f in sorted(model_dir.iterdir()):
            if not f.is_file():
                continue
            name = f.name
            if not name.startswith("model.pt"):
                continue
            if name.endswith((".log", ".txt", ".json", ".yaml", ".yml")):
                continue

            if pattern:
                patterns = [p.strip() for p in pattern.split(",")]
                if not any(p in name for p in patterns):
                    continue

            display_name = f"{version_name}/{name}"
            ckpt_files.append((f, display_name))

    return ckpt_files


def auto_discover_checkpoints(model_dir: Path, pattern: str = None,
                               versions_filter: list = None) -> list:
    """
    自动判断目录结构并发现权重文件。

    - 如果 model_dir 下直接有 model.pt.* 文件 → 单目录模式
    - 如果 model_dir 下有子目录且子目录中有 model/ 或 model.pt.* → 多版本模式
    """
    # 先检查是否是单目录模式（直接包含 model.pt.* 文件）
    direct_ckpts = discover_checkpoints(model_dir, pattern)
    if direct_ckpts:
        return direct_ckpts

    # 多版本模式
    multi_ckpts = discover_checkpoints_multi_version(model_dir, pattern, versions_filter)
    if multi_ckpts:
        return multi_ckpts

    return []


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


def generate_comparison_report(all_results: dict, output_dir: Path, logger):
    """
    生成跨模型对比报告。
    all_results: {ckpt_name: summary_dict}
    """
    report_path = output_dir / "comparison_report.md"
    lines = []
    lines.append("# 12_test 测试集 - 多模型权重对比报告\n")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    ckpt_names = list(all_results.keys())

    # === 总览表格 ===
    lines.append("## 模型性能对比\n")
    lines.append("| 模型权重 | CER | SER | 句子正确率 | 总条数 | 总字符数 | 替换 | 插入 | 删除 |")
    lines.append("|----------|-----|-----|-----------|--------|----------|------|------|------|")

    model_metrics = {}
    for ckpt_name in ckpt_names:
        s = all_results[ckpt_name]
        if s is None:
            lines.append(f"| {ckpt_name} | - | - | - | - | - | - | - | - |")
            continue

        cer = s.get("overall_cer", 0)
        ser = s.get("ser", 0)
        acc = s.get("sentence_accuracy", 0)
        total = s.get("total_count", 0)
        ref_chars = s.get("total_ref_chars", 0)
        sub = s.get("substitutions", 0)
        ins = s.get("insertions", 0)
        dele = s.get("deletions", 0)

        model_metrics[ckpt_name] = {"cer": cer, "ser": ser, "acc": acc}
        lines.append(
            f"| {ckpt_name} | {cer:.2%} | {ser:.2%} | {acc:.2%} | "
            f"{total} | {ref_chars} | {sub} | {ins} | {dele} |"
        )

    lines.append("")

    # === CER 分布对比 ===
    lines.append("## CER 分布对比\n")
    bucket_order = ["0% (完全正确)", "0-5%", "5-10%", "10-20%", "20-50%", ">50%"]
    header = "| CER 区间 | " + " | ".join(ckpt_names) + " |"
    sep = "|----------|" + "|".join(["--------" for _ in ckpt_names]) + "|"
    lines.append(header)
    lines.append(sep)

    for bucket in bucket_order:
        row = f"| {bucket} |"
        for ckpt_name in ckpt_names:
            s = all_results[ckpt_name]
            if s and "cer_distribution" in s:
                cnt = s["cer_distribution"].get(bucket, 0)
                total = s.get("total_count", 1)
                row += f" {cnt} ({cnt/total*100:.1f}%) |"
            else:
                row += " - |"
        lines.append(row)
    lines.append("")

    # === 最佳模型推荐 ===
    if model_metrics:
        best_cer = min(model_metrics, key=lambda k: model_metrics[k]["cer"])
        best_ser = min(model_metrics, key=lambda k: model_metrics[k]["ser"])
        lines.append("## 最佳模型\n")
        lines.append(f"- **最低 CER**: {best_cer} ({model_metrics[best_cer]['cer']:.2%})")
        lines.append(f"- **最低 SER**: {best_ser} ({model_metrics[best_ser]['ser']:.2%})")

        # 与最差模型的差距
        worst_cer = max(model_metrics, key=lambda k: model_metrics[k]["cer"])
        if best_cer != worst_cer:
            delta = model_metrics[worst_cer]["cer"] - model_metrics[best_cer]["cer"]
            lines.append(f"- CER 最大差距: {delta:.2%} ({best_cer} vs {worst_cer})")
        lines.append("")

    report_content = "\n".join(lines)
    report_path.write_text(report_content, encoding="utf-8")
    logger.info(f"对比报告 -> {report_path}")
    print(f"\n📊 对比报告: {report_path}")

    # JSON 格式
    json_path = output_dir / "comparison_summary.json"
    json_data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_dataset": "12_test",
        "models": {},
    }
    for ckpt_name in ckpt_names:
        json_data["models"][ckpt_name] = all_results[ckpt_name]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    logger.info(f"对比 JSON -> {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="在 12_test 测试集上批量评估多个模型权重",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 多版本模式（自动扫描 v0.1.0/model/, v0.2.0/model/, ...）
  python eval_on_testset.py --model-dir /data1/eval_output

  # 单目录模式
  python eval_on_testset.py --model-dir /data1/eval_output/v0.1.0/model

  # 只评估 best 权重
  python eval_on_testset.py --ckpt-pattern "best"

  # 只评估指定版本
  python eval_on_testset.py --versions v0.2.0 v0.3.0

  # 采样快速验证
  python eval_on_testset.py --sample 100

  # 同时评估基准模型
  python eval_on_testset.py --with-baseline
        """,
    )

    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help=f"模型权重目录，支持单目录或多版本父目录 (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                        help="基准模型目录（SeACo-Paraformer）")
    parser.add_argument("--test-scp", default=DEFAULT_TEST_SCP,
                        required=(DEFAULT_TEST_SCP is None),
                        help="测试集 wav.scp 路径")
    parser.add_argument("--test-text", default=DEFAULT_TEST_TEXT,
                        required=(DEFAULT_TEST_TEXT is None),
                        help="测试集 text 路径")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"输出目录 (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--ckpt-pattern", default=None,
                        help="权重文件名过滤，逗号分隔 (如 'best,avg5')")
    parser.add_argument("--versions", nargs="*", default=None,
                        help="指定版本目录名，不指定则评估全部 (如 v0.2.0 v0.3.0)")
    parser.add_argument("--with-baseline", action="store_true",
                        help="同时评估基准模型（不加载微调权重）")
    parser.add_argument("--hotword", default=None, help="热词文件路径")
    parser.add_argument("--device", default="cuda:auto", help="设备 (default: cuda:auto)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="推理 batch 大小 (default: 8)")
    parser.add_argument("--file-batch", type=int, default=5000,
                        help="文件批大小 (default: 5000)")
    parser.add_argument("--sample", type=int, default=None,
                        help="随机采样 N 条（快速验证用）")
    parser.add_argument("--resume", action="store_true", help="断点续跑")

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    test_scp = Path(args.test_scp)
    test_text = Path(args.test_text)

    # 1. 检查测试集
    if not test_scp.exists():
        print(f"[错误] 测试集 wav.scp 不存在: {test_scp}")
        return
    if not test_text.exists():
        print(f"[错误] 测试集 text 不存在: {test_text}")
        return

    # 2. 发现模型权重
    checkpoints = auto_discover_checkpoints(model_dir, args.ckpt_pattern, args.versions)
    if not checkpoints:
        print(f"[错误] 在 {model_dir} 下未找到模型权重文件")
        print("  支持的目录结构:")
        print("    单目录: model_dir/model.pt.best, model.pt.ep13, ...")
        print("    多版本: model_dir/v0.1.0/model/model.pt.best, ...")
        if args.ckpt_pattern:
            print(f"  当前过滤: {args.ckpt_pattern}")
        if args.versions:
            print(f"  版本过滤: {args.versions}")
        return

    # 构建评估列表
    eval_models = []
    if args.with_baseline:
        eval_models.append(("baseline", None))
    for ckpt_path, ckpt_name in checkpoints:
        eval_models.append((ckpt_name, ckpt_path))

    # 3. 打印评估计划
    print("\n" + "=" * 70)
    print("12_test 测试集 - 多模型权重批量评估")
    print("=" * 70)
    print(f"模型目录:   {model_dir}")
    print(f"基准模型:   {args.base_model}")
    print(f"测试集 SCP: {test_scp}")
    print(f"测试集 TXT: {test_text}")
    print(f"输出目录:   {output_dir}")
    print()
    print(f"待评估模型 ({len(eval_models)} 个):")
    for ckpt_name, ckpt_path in eval_models:
        if ckpt_path:
            size_mb = ckpt_path.stat().st_size / (1024 * 1024)
            print(f"  - {ckpt_name} ({size_mb:.0f} MB)")
        else:
            print(f"  - {ckpt_name} (基准模型)")
    if args.sample:
        print(f"\n⚡ 采样模式: 随机抽取 {args.sample} 条")
    print("=" * 70)

    # 4. 加载测试集标签
    logger, _ = setup_logging(output_dir, "testset_eval")
    labels = load_labels(test_text, logger)
    if not labels:
        print("[错误] 测试集标签为空")
        return
    print(f"\n测试集标签: {len(labels)} 条")

    # 构建 eval_items（只需构建一次）
    eval_items_all = build_eval_items_from_scp(test_scp, labels, logger)
    if not eval_items_all:
        print("[错误] 无有效评估条目")
        return
    print(f"有效评估条目: {len(eval_items_all)} 条")

    # 加载热词
    hotword_str = ""
    if args.hotword:
        hotword_str = Path(args.hotword).read_text(encoding="utf-8").strip().replace("\n", " ")
        print(f"热词: {args.hotword}")

    # 5. 逐模型评估
    all_results = {}
    total_start = time.time()

    for model_idx, (ckpt_name, ckpt_path) in enumerate(eval_models):
        print(f"\n{'━' * 70}")
        print(f"[{model_idx + 1}/{len(eval_models)}] 评估: {ckpt_name}")
        print(f"{'━' * 70}")

        model_output = output_dir / ckpt_name.replace("/", "_")
        model_logger, _ = setup_logging(model_output, f"eval_{ckpt_name.replace('/', '_')}")

        # 加载模型
        device = select_gpus(args.device, model_logger)
        try:
            if ckpt_path is None:
                model = load_baseline_model(args.base_model, device, model_logger)
            else:
                model = load_finetuned_model(
                    args.base_model, str(ckpt_path), device, model_logger
                )
        except Exception as e:
            model_logger.error(f"模型加载失败: {e}")
            print(f"  ❌ 模型加载失败: {e}")
            all_results[ckpt_name] = None
            continue

        # 评估
        summary = run_eval_items(
            eval_items=eval_items_all,
            output_dir=model_output,
            model=model,
            hotword_str=hotword_str,
            args=args,
            logger=model_logger,
            dir_name=f"12_test/{ckpt_name}",
        )

        all_results[ckpt_name] = summary

        if summary:
            cer = summary.get("overall_cer", 0)
            ser = summary.get("ser", 0)
            acc = summary.get("sentence_accuracy", 0)
            print(f"  ✅ CER: {cer:.2%}  SER: {ser:.2%}  正确率: {acc:.2%}")
        else:
            print(f"  ⚠️ 评估未产生结果")

        # 释放显存
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # 6. 生成对比报告
    valid_results = {k: v for k, v in all_results.items() if v is not None}
    if valid_results:
        generate_comparison_report(all_results, output_dir, logger)

    # 7. 打印最终摘要
    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print("评估完成!")
    print("=" * 70)
    print(f"总耗时: {format_duration(total_time)}")
    print(f"输出目录: {output_dir}")
    print()
    print(f"{'模型权重':<20s} {'CER':>8s} {'SER':>8s} {'正确率':>8s}")
    print("-" * 50)
    for ckpt_name, summary in all_results.items():
        if summary:
            cer = summary.get("overall_cer", 0)
            ser = summary.get("ser", 0)
            acc = summary.get("sentence_accuracy", 0)
            print(f"{ckpt_name:<20s} {cer:>7.2%} {ser:>7.2%} {acc:>7.2%}")
        else:
            print(f"{ckpt_name:<20s} {'失败':>8s}")

    if valid_results:
        best = min(valid_results, key=lambda k: valid_results[k].get("overall_cer", 1))
        print(f"\n🏆 最佳模型: {best} (CER: {valid_results[best]['overall_cer']:.2%})")
    print("=" * 70)


if __name__ == "__main__":
    main()
