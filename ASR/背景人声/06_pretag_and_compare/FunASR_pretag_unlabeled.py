#!/usr/bin/env python3
"""
S10 ASR 未标注数据预标注脚本
使用 FunASR (paraformer-zh) 对 unlabeled/ 目录中的音频进行批量识别，
生成 FunASR 训练格式的 jsonl、wav.scp、text.txt。

依赖安装:
    pip install funasr torch torchaudio

用法:
    python pretag_unlabeled.py
    python pretag_unlabeled.py --model sensevoice  # 使用 SenseVoice 模型
    python pretag_unlabeled.py --device cpu         # 使用 CPU
    python pretag_unlabeled.py --batch-size 128     # 调整batch大小(秒)
    python pretag_unlabeled.py --resume             # 断点续跑
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# ============ 配置 ============
BASE_DIR = Path(r'E:\8.音频\03 s10\s10_data_reflow')
UNLABELED_DIR = BASE_DIR / "unlabeled"
OUTPUT_DIR = BASE_DIR / "pretag_output"

MODEL_CHOICES = {
    "paraformer": "paraformer-zh",
    "sensevoice": "iic/SenseVoiceSmall",
}
# ==============================


def setup_logging(output_dir: Path):
    """配置日志，同时输出到控制台和文件"""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"pretag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # 文件
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    logger = logging.getLogger('pretag')
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger, log_file


def scan_wav_files(unlabeled_dir: Path, logger) -> list:
    """扫描未标注目录中的所有wav文件"""
    logger.info(f"扫描音频目录: {unlabeled_dir}")
    wav_files = sorted(unlabeled_dir.glob("*.wav"))
    logger.info(f"找到 {len(wav_files)} 个wav文件")
    return wav_files


def load_progress(progress_file: Path) -> set:
    """加载已处理的文件ID（用于断点续跑）"""
    done = set()
    if progress_file.exists():
        with open(progress_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(line)
    return done


def format_duration(seconds: float) -> str:
    """格式化时间"""
    return str(timedelta(seconds=int(seconds)))


def run_pretag(args):
    """主流程"""
    output_dir = OUTPUT_DIR
    logger, log_file = setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("S10 ASR 未标注数据预标注")
    logger.info("=" * 60)
    logger.info(f"音频目录: {UNLABELED_DIR}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"模型: {args.model} -> {MODEL_CHOICES[args.model]}")
    logger.info(f"设备: {args.device}")
    logger.info(f"Batch大小: {args.batch_size}s")
    logger.info(f"断点续跑: {args.resume}")
    logger.info(f"日志文件: {log_file}")

    # 1. 扫描wav文件
    wav_files = scan_wav_files(UNLABELED_DIR, logger)
    if not wav_files:
        logger.warning("未找到任何wav文件，退出")
        return

    # 2. 断点续跑：加载已完成的ID
    progress_file = output_dir / "progress.txt"
    done_ids = set()
    if args.resume:
        done_ids = load_progress(progress_file)
        logger.info(f"已完成: {len(done_ids)} 条，将跳过这些文件")

    # 过滤掉已处理的
    pending_files = [f for f in wav_files if f.stem not in done_ids]
    logger.info(f"待处理: {len(pending_files)} 个文件")

    if not pending_files:
        logger.info("所有文件已处理完毕，无需继续")
        return

    # 3. 加载模型
    logger.info("正在加载 FunASR 模型...")
    load_start = time.time()

    from funasr import AutoModel

    model_name = MODEL_CHOICES[args.model]
    model_kwargs = dict(
        model=model_name,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=args.device,
    )
    if args.model == "paraformer":
        model_kwargs["punc_model"] = "ct-punc"

    model = AutoModel(**model_kwargs)
    load_time = time.time() - load_start
    logger.info(f"模型加载完成，耗时: {format_duration(load_time)}")

    # 4. 批量推理
    jsonl_path = output_dir / "pretag.jsonl"
    scp_path = output_dir / "wav.scp"
    text_path = output_dir / "text.txt"

    # 追加模式（支持断点续跑）
    open_mode = 'a' if args.resume and done_ids else 'w'

    total = len(pending_files)
    processed = 0
    failed = 0
    infer_start = time.time()

    # 分批处理
    batch_size = args.file_batch
    logger.info(f"开始推理，每批 {batch_size} 个文件...")

    with open(jsonl_path, open_mode, encoding='utf-8') as f_jsonl, \
         open(scp_path, open_mode, encoding='utf-8') as f_scp, \
         open(text_path, open_mode, encoding='utf-8') as f_text, \
         open(progress_file, open_mode, encoding='utf-8') as f_prog:

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_files = pending_files[batch_start:batch_end]
            batch_paths = [str(f) for f in batch_files]

            batch_t0 = time.time()
            try:
                if args.model == "sensevoice":
                    from funasr.utils.postprocess_utils import rich_transcription_postprocess
                    results = model.generate(
                        input=batch_paths,
                        cache={},
                        language="auto",
                        use_itn=True,
                        batch_size_s=args.batch_size,
                        merge_vad=True,
                        merge_length_s=15,
                    )
                else:
                    results = model.generate(
                        input=batch_paths,
                        cache={},
                        batch_size_s=args.batch_size,
                    )
            except Exception as e:
                logger.error(f"批次推理失败 [{batch_start}-{batch_end}]: {e}")
                failed += len(batch_files)
                continue

            batch_time = time.time() - batch_t0

            # 写入结果
            for i, wav_file in enumerate(batch_files):
                uid = wav_file.stem
                try:
                    text = results[i]["text"]
                    if args.model == "sensevoice":
                        text = rich_transcription_postprocess(text)
                except (IndexError, KeyError, TypeError) as e:
                    logger.warning(f"  结果解析失败 {uid}: {e}")
                    failed += 1
                    continue

                wav_abs = wav_file.resolve()

                # wav.scp
                f_scp.write(f"{uid} {wav_abs}\n")
                # text.txt
                f_text.write(f"{uid} {text}\n")
                # jsonl
                record = {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"语音转写：<|startofspeech|>!{wav_abs}<|endofspeech|>"},
                        {"role": "assistant", "content": text}
                    ],
                    "text_length": len(text)
                }
                f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                # 进度记录
                f_prog.write(f"{uid}\n")
                processed += 1

            # 刷新文件缓冲
            f_jsonl.flush()
            f_scp.flush()
            f_text.flush()
            f_prog.flush()

            # 进度日志
            elapsed = time.time() - infer_start
            speed = processed / elapsed if elapsed > 0 else 0
            eta = (total - batch_end) / speed if speed > 0 else 0
            pct = (batch_end / total) * 100

            logger.info(
                f"进度: {batch_end}/{total} ({pct:.1f}%) | "
                f"本批耗时: {batch_time:.1f}s ({len(batch_files)}条) | "
                f"已处理: {processed} | 失败: {failed} | "
                f"速度: {speed:.1f}条/s | "
                f"已用: {format_duration(elapsed)} | "
                f"预计剩余: {format_duration(eta)}"
            )

    # 5. 汇总
    total_time = time.time() - infer_start
    logger.info("=" * 60)
    logger.info("预标注完成")
    logger.info(f"  总文件数: {total}")
    logger.info(f"  成功: {processed}")
    logger.info(f"  失败: {failed}")
    logger.info(f"  总耗时: {format_duration(total_time)}")
    logger.info(f"  平均速度: {processed / total_time:.1f} 条/s" if total_time > 0 else "")
    logger.info(f"输出文件:")
    logger.info(f"  {jsonl_path}")
    logger.info(f"  {scp_path}")
    logger.info(f"  {text_path}")
    logger.info(f"  {progress_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="S10 ASR 未标注数据预标注")
    parser.add_argument(
        '--model', choices=list(MODEL_CHOICES.keys()),
        default='paraformer',
        help='ASR模型选择 (default: paraformer)'
    )
    parser.add_argument(
        '--device', default='cuda:0',
        help='推理设备 (default: cuda:0)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=300,
        help='动态batch总时长，单位秒 (default: 300)'
    )
    parser.add_argument(
        '--file-batch', type=int, default=500,
        help='每批送入的文件数 (default: 500)'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='断点续跑，跳过已处理的文件'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pretag(args)
