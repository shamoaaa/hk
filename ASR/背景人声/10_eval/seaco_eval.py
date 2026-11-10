#!/usr/bin/env python3
"""
SeACo-Paraformer 基准模型评估脚本

使用本地 speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch 模型，
对指定数据集进行 ASR 识别并计算 CER/SER 等评估指标。

用法:
    python seaco_eval.py --wav-dir /path/to/wav --label /path/to/text.txt
    python seaco_eval.py --wav-dir /path/to/data/wav /path/to/test/wav
    python seaco_eval.py --wav-dir /path/to/wav --hotword /path/to/hotwords.txt
    python seaco_eval.py --wav-dir /path/to/wav --sample 1000 --resume
"""

import time
import argparse
from pathlib import Path

from eval_utils import (
    select_gpus, setup_logging, format_duration,
    load_labels, load_hotwords, auto_discover_labels, get_dir_display_name,
    run_eval_single_dir, generate_combined_report,
)

# ============ 默认配置 ============
DEFAULT_MODEL_DIR = r'E:\8.音频\02 ASR入门\models\speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
DEFAULT_OUTPUT_DIR = r'E:\8.音频\03 s10\10_eval\eval_output'
# ==================================


def run_eval(args):
    output_dir = Path(args.output_dir)
    logger, log_file = setup_logging(output_dir, 'seaco_eval')

    wav_dirs = [Path(d) for d in args.wav_dir]
    multi_mode = len(wav_dirs) > 1

    logger.info("=" * 60)
    logger.info("SeACo-Paraformer 基准模型评估")
    logger.info("=" * 60)
    logger.info(f"模型目录:   {args.model_dir}")
    logger.info(f"音频目录:   {[str(d) for d in wav_dirs]}")
    logger.info(f"标签文件:   {args.label or '自动发现 (trans.txt)'}")
    logger.info(f"输出目录:   {output_dir}")
    logger.info(f"设备:       {args.device}")
    logger.info(f"Batch大小:  {args.batch_size} 条")
    logger.info(f"文件批大小: {args.file_batch}")
    logger.info(f"热词文件:   {args.hotword or '无'}")
    logger.info(f"采样数:     {args.sample or '全部'}")
    logger.info(f"断点续跑:   {args.resume}")
    logger.info(f"多目录模式: {multi_mode}")
    logger.info(f"日志文件:   {log_file}")

    # 1. 加载标签
    global_labels = {}
    per_dir_labels = {}

    if args.label:
        for lbl in args.label:
            label_path = Path(lbl)
            if not label_path.exists():
                logger.error(f"标签文件不存在: {label_path}")
                return
            loaded = load_labels(label_path, logger)
            global_labels.update(loaded)
        logger.info(f"合并标签总数: {len(global_labels)} 条")
    else:
        per_dir_labels = auto_discover_labels(wav_dirs, logger)

    # 2. 加载热词
    hotword_str = ""
    if args.hotword:
        hotword_str = load_hotwords(Path(args.hotword), logger)

    # 3. 加载模型
    device = select_gpus(args.device, logger)
    logger.info("正在加载 SeACo-Paraformer 模型...")
    load_start = time.time()

    from funasr import AutoModel
    model = AutoModel(model=args.model_dir, device=device)
    logger.info(f"模型加载完成，耗时: {format_duration(time.time() - load_start)}")

    # 4. 逐目录评估
    dir_summaries = {}
    for wav_dir in wav_dirs:
        if not wav_dir.exists():
            logger.warning(f"目录不存在，跳过: {wav_dir}")
            continue

        labels = global_labels if global_labels else per_dir_labels.get(str(wav_dir), {})
        dir_name = get_dir_display_name(wav_dir)

        sub_output_dir = output_dir / dir_name if multi_mode else output_dir

        summary = run_eval_single_dir(
            wav_dir=wav_dir, labels=labels, model=model,
            hotword_str=hotword_str, output_dir=sub_output_dir,
            args=args, logger=logger,
        )
        if summary:
            dir_summaries[dir_name] = summary

    # 5. 多目录汇总
    if multi_mode and dir_summaries:
        generate_combined_report(dir_summaries, output_dir, logger)
    elif not dir_summaries and (global_labels or per_dir_labels):
        logger.warning("所有目录均无可评估数据")

    logger.info("评估完成!")


def parse_args():
    parser = argparse.ArgumentParser(
        description="SeACo-Paraformer 基准模型评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR, help='本地模型目录')
    parser.add_argument('--wav-dir', nargs='+', required=True, help='wav 文件目录，可指定多个')
    parser.add_argument('--label', nargs='*', default=None, help='标签文件，可指定多个。不指定则自动发现')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='输出目录')
    parser.add_argument('--hotword', default=None, help='热词文件路径')
    parser.add_argument('--device', default='cuda:auto', help='设备 (default: cuda:auto)')
    parser.add_argument('--batch-size', type=int, default=64, help='每次推理样本数 (default: 64)')
    parser.add_argument('--file-batch', type=int, default=5000, help='每批文件数 (default: 5000)')
    parser.add_argument('--sample', type=int, default=None, help='随机采样 N 条')
    parser.add_argument('--resume', action='store_true', help='断点续跑')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
