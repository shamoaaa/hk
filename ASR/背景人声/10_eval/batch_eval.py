#!/usr/bin/env python3
"""
公开数据集批量评估脚本

基于 public_dataset_gen_scp_txt.py 生成的标准目录结构，批量运行 SeACo-Paraformer
基准模型 和/或 微调模型 评估。

目录结构约定：
    /data1/code/label/{train,dev,test}/<dataset>_wav.scp
    /data1/code/label/{train,dev,test}/<dataset>_text

用法:
    # 评估所有数据集的 test 集（基准模型）
    python batch_eval.py --mode seaco

    # 评估所有数据集的 test 集（微调模型）
    python batch_eval.py --mode finetuned

    # 同时评估基准和微调模型
    python batch_eval.py --mode both

    # 只评估指定数据集
    python batch_eval.py --mode seaco --datasets aidatatang_200zh data_aishell

    # 评估 dev 集
    python batch_eval.py --mode seaco --splits dev

    # 评估 dev + test
    python batch_eval.py --mode seaco --splits dev test

    # 采样评估
    python batch_eval.py --mode seaco --sample 500

    # 使用热词
    python batch_eval.py --mode seaco --hotword /path/to/hotwords.txt
"""

import time
import argparse
from pathlib import Path
from datetime import datetime

from eval_utils import (
    select_gpus, setup_logging, format_duration,
    load_labels, load_hotwords,
    run_eval_items, build_eval_items_from_scp, generate_combined_report,
)


# ============ 默认配置 ============
DEFAULT_LABEL_ROOT = "/data1/label/augment_label/"
DEFAULT_SEACO_MODEL = r"/data1/pretrained_models/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
DEFAULT_FINETUNE_BASE = r"/data1/pretrained_models/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
DEFAULT_FINETUNE_CKPT = r"/data1/code/09_train/outputs/online_model.pt"
DEFAULT_OUTPUT_ROOT = r"/data1/eval_output/base/augment_eval_result"
# ==================================


def discover_datasets(label_root: Path, splits: list) -> dict:
    """
    自动发现 label_root 下可用的数据集。

    返回: {dataset_name: {split: (scp_path, text_path), ...}, ...}
    """
    datasets = {}
    for split in splits:
        split_dir = label_root / split
        if not split_dir.exists():
            continue
        for scp_file in sorted(split_dir.glob("*_wav.scp")):
            dataset_name = scp_file.name.replace("_wav.scp", "")
            text_file = split_dir / f"{dataset_name}_text"
            if text_file.exists():
                if dataset_name not in datasets:
                    datasets[dataset_name] = {}
                datasets[dataset_name][split] = (scp_file, text_file)
    return datasets


def load_model_seaco(model_dir: str, device: str, logger):
    """加载 SeACo 基准模型"""
    from funasr import AutoModel
    logger.info("正在加载 SeACo-Paraformer 基准模型...")
    load_start = time.time()
    model = AutoModel(model=model_dir, device=device)
    logger.info(f"基准模型加载完成，耗时: {format_duration(time.time() - load_start)}")
    return model


def load_model_finetuned(base_model_dir: str, finetune_ckpt: str, device: str, logger):
    """加载微调模型"""
    import torch
    from funasr import AutoModel

    logger.info("正在加载微调模型...")
    load_start = time.time()

    auto_model = AutoModel(model=base_model_dir, device=device)

    ckpt_path = Path(finetune_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"微调权重文件不存在: {finetune_ckpt}")

    state_dict = torch.load(str(ckpt_path), map_location=device)
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


def _generate_dataset_report(dataset_name: str, split_summaries: dict,
                             output_dir: Path, logger):
    """
    为单个数据集生成汇总报告，对比各 split 的 CER/SER。
    输出到 output_dir/dataset_report.md
    """
    report_path = output_dir / "dataset_report.md"
    lines = []
    lines.append(f"# {dataset_name} 评估报告\n")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 汇总表格
    lines.append("## 各集合对比\n")
    lines.append("| 集合 | CER | SER | 总条数 | 正确数 | 替换 | 插入 | 删除 | 参考字数 |")
    lines.append("|------|-----|-----|--------|--------|------|------|------|----------|")

    for split in sorted(split_summaries.keys()):
        s = split_summaries[split]
        cer = s.get("cer", 0)
        ser = s.get("ser", 0)
        total = s.get("total", 0)
        correct = s.get("correct", 0)
        sub = s.get("total_sub", s.get("sub", 0))
        ins = s.get("total_ins", s.get("ins", 0))
        dele = s.get("total_del", s.get("del", 0))
        ref_len = s.get("total_ref_len", s.get("ref_len", 0))

        cer_str = f"{cer:.2%}" if isinstance(cer, (int, float)) else str(cer)
        ser_str = f"{ser:.2%}" if isinstance(ser, (int, float)) else str(ser)
        lines.append(f"| {split} | {cer_str} | {ser_str} | {total} | {correct} | {sub} | {ins} | {dele} | {ref_len} |")

    # 加权平均
    if len(split_summaries) > 1:
        total_ref = sum(s.get("total_ref_len", s.get("ref_len", 0)) for s in split_summaries.values())
        total_errors = sum(
            s.get("total_sub", s.get("sub", 0)) +
            s.get("total_ins", s.get("ins", 0)) +
            s.get("total_del", s.get("del", 0))
            for s in split_summaries.values()
        )
        total_utts = sum(s.get("total", 0) for s in split_summaries.values())
        total_correct = sum(s.get("correct", 0) for s in split_summaries.values())
        avg_cer = total_errors / total_ref if total_ref > 0 else 0
        avg_ser = 1 - total_correct / total_utts if total_utts > 0 else 0

        lines.append(f"| **加权平均** | **{avg_cer:.2%}** | **{avg_ser:.2%}** | {total_utts} | {total_correct} | - | - | - | {total_ref} |")

    lines.append("")

    report_content = "\n".join(lines)
    report_path.write_text(report_content, encoding="utf-8")
    logger.info(f"[{dataset_name}] 数据集报告 → {report_path}")


def run_batch_eval(args):
    label_root = Path(args.label_root)
    output_root = Path(args.output_root)
    splits = args.splits

    # 发现数据集
    all_datasets = discover_datasets(label_root, splits)

    # 过滤指定数据集
    if args.datasets:
        filtered = {}
        for name in args.datasets:
            if name in all_datasets:
                filtered[name] = all_datasets[name]
            else:
                print(f"[警告] 数据集 '{name}' 未找到，可用: {list(all_datasets.keys())}")
        all_datasets = filtered

    if not all_datasets:
        print("没有找到可评估的数据集。请先运行 public_dataset_gen_scp_txt.py 生成数据。")
        return

    # 确定要运行的模型
    modes = []
    if args.mode in ("seaco", "both"):
        modes.append("seaco")
    if args.mode in ("finetuned", "both"):
        modes.append("finetuned")

    # 打印评估计划
    print("\n" + "=" * 70)
    print("批量评估计划")
    print("=" * 70)
    print(f"数据根目录: {label_root}")
    print(f"评估模式:   {args.mode}")
    print(f"评估集:     {splits}")
    print(f"数据集:     {list(all_datasets.keys())}")
    print(f"输出根目录: {output_root}")

    eval_tasks = []
    for dataset_name, split_info in sorted(all_datasets.items()):
        for split, (scp_path, text_path) in sorted(split_info.items()):
            for mode in modes:
                eval_tasks.append((dataset_name, split, scp_path, text_path, mode))
    print(f"总评估任务: {len(eval_tasks)} 个")
    for ds, sp, _, _, md in eval_tasks:
        print(f"  - [{md}] {ds}/{sp}")
    print("=" * 70)

    # 加载热词
    hotword_str = ""
    if args.hotword:
        hotword_str = Path(args.hotword).read_text(encoding="utf-8").strip().replace("\n", " ")
        print(f"热词: {args.hotword}")

    # 按模型分组执行（避免重复加载模型）
    for mode in modes:
        mode_output = output_root / mode
        logger, log_file = setup_logging(mode_output, f"batch_{mode}")

        logger.info("=" * 60)
        logger.info(f"批量评估 - {mode} 模型")
        logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        # 加载模型
        device = select_gpus(args.device, logger)
        if mode == "seaco":
            model = load_model_seaco(args.seaco_model, device, logger)
        else:
            model = load_model_finetuned(args.finetune_base, args.finetune_ckpt, device, logger)

        # 逐数据集逐 split 评估
        dir_summaries = {}
        mode_tasks = [(ds, sp, scp, txt) for ds, sp, scp, txt, md in eval_tasks if md == mode]

        for dataset_name, split, scp_path, text_path in mode_tasks:
            task_name = f"{dataset_name}/{split}"
            logger.info(f"\n{'─' * 50}")
            logger.info(f"任务: {task_name}")
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
            eval_output = mode_output / dataset_name / split

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
                logger.info(f"[{task_name}] CER: {summary.get('cer', 'N/A')}")

        # 生成数据集级汇总报告（当一个数据集有多个 split 时）
        dataset_splits = {}  # {dataset_name: {split: summary}}
        for task_name, summary in dir_summaries.items():
            ds, sp = task_name.split("/", 1)
            if ds not in dataset_splits:
                dataset_splits[ds] = {}
            dataset_splits[ds][sp] = summary

        for ds_name, split_summaries in dataset_splits.items():
            ds_output = mode_output / ds_name
            ds_output.mkdir(parents=True, exist_ok=True)
            _generate_dataset_report(ds_name, split_summaries, ds_output, logger)

        # 生成全局汇总报告
        if len(dir_summaries) > 1:
            report_title = f"{'SeACo 基准' if mode == 'seaco' else '微调'}模型 批量评估汇总"
            generate_combined_report(dir_summaries, mode_output, logger, report_title=report_title)

        # 打印结果摘要
        logger.info("\n" + "=" * 60)
        logger.info(f"{mode} 模型评估结果摘要")
        logger.info("=" * 60)
        logger.info(f"{'数据集/集合':<35} {'CER':>8} {'SER':>8} {'条数':>8}")
        logger.info("-" * 60)
        for task_name, summary in sorted(dir_summaries.items()):
            cer = summary.get("cer", "N/A")
            ser = summary.get("ser", "N/A")
            total = summary.get("total", "N/A")
            cer_str = f"{cer:.2%}" if isinstance(cer, (int, float)) else str(cer)
            ser_str = f"{ser:.2%}" if isinstance(ser, (int, float)) else str(ser)
            logger.info(f"{task_name:<35} {cer_str:>8} {ser_str:>8} {total:>8}")
        logger.info("=" * 60)

        del model  # 释放显存


def parse_args():
    parser = argparse.ArgumentParser(
        description="公开数据集批量评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 评估所有数据集的 test 集（基准模型）
  python batch_eval.py --mode seaco

  # 只评估指定数据集
  python batch_eval.py --mode seaco --datasets aidatatang_200zh data_aishell

  # 同时评估基准和微调模型的 dev+test
  python batch_eval.py --mode both --splits dev test

  # 采样评估
  python batch_eval.py --mode finetuned --sample 500
        """,
    )

    # 模式
    parser.add_argument("--mode", choices=["seaco", "finetuned", "both"], default="both",
                        help="评估模式: seaco=基准模型, finetuned=微调模型, both=两者 (default: seaco)")

    # 数据
    parser.add_argument("--label-root", default=DEFAULT_LABEL_ROOT,
                        help=f"标签根目录 (default: {DEFAULT_LABEL_ROOT})")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="指定数据集名称，不指定则评估全部")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"],
                        help="评估哪些集合 (default: test)")

    # 模型
    parser.add_argument("--seaco-model", default=DEFAULT_SEACO_MODEL,
                        help="SeACo 基准模型目录")
    parser.add_argument("--finetune-base", default=DEFAULT_FINETUNE_BASE,
                        help="微调模型的基准模型目录")
    parser.add_argument("--finetune-ckpt", default=DEFAULT_FINETUNE_CKPT,
                        help="微调权重路径")

    # 输出
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                        help=f"输出根目录 (default: {DEFAULT_OUTPUT_ROOT})")

    # 评估参数
    parser.add_argument("--hotword", default=None, help="热词文件路径")
    parser.add_argument("--device", default="cuda:auto", help="设备 (default: cuda:auto)")
    parser.add_argument("--batch-size", type=int, default=16, help="推理 batch 大小 (default: 64)")
    parser.add_argument("--file-batch", type=int, default=5000, help="文件批大小 (default: 5000)")
    parser.add_argument("--sample", type=int, default=None, help="随机采样 N 条")
    parser.add_argument("--resume", action="store_true", help="断点续跑")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_batch_eval(args)
