#!/usr/bin/env python3
"""
S10 ASR 音频数据质量扫描脚本
对所有wav音频进行质量检查，输出逐条分析结果 (quality_detail.jsonl)。

采集项（仅采集原始数值，不做阈值判断）：
  1. 文件完整性（能否正常打开）
  2. 时长、采样率、声道数、位深
  3. RMS 能量、峰值
  4. 削波比例
  5. 直流偏移
  6. 信噪比(SNR)估算

依赖: 仅 Python 标准库 (wave, struct, math)

用法:
    python 03_audio_quality_check.py
    python 03_audio_quality_check.py --workers 8
    python 03_audio_quality_check.py --source-only

输出:
    quality_check/quality_detail.jsonl  — 逐条分析结果

后续:
    python 03b_quality_report.py  — 基于 jsonl 生成报告和文件清单
"""

import os
import sys
import wave
import struct
import math
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============ 配置 ============
BASE_DIR = Path(r'E:\8.音频\03 s10\s10_data_reflow')
OUTPUT_DIR = BASE_DIR / "quality_check"

# 扫描目录列表
SCAN_DIRS = [
    BASE_DIR / "labeled" / "wav",
    BASE_DIR / "unlabeled",
]

# SNR 估算参数（采集用，非阈值）
SNR_FRAME_MS = 25
SNR_HOP_MS = 10
# ==============================


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    logger = logging.getLogger('quality')
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger, log_file


def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def analyze_wav(wav_path: str) -> dict:
    """
    分析单个wav文件，只采集原始数值，不做阈值判断。
    在子进程中运行，不能使用 logger。
    """
    result = {
        "path": wav_path,
        "uid": Path(wav_path).stem,
        "error": None,
        "duration": 0.0,
        "sample_rate": 0,
        "channels": 0,
        "sample_width": 0,
        "rms": 0.0,
        "peak": 0,
        "clipping_ratio": 0.0,
        "dc_offset": 0.0,
        "snr_db": 0.0,
        "file_size": 0,
    }

    try:
        result["file_size"] = os.path.getsize(wav_path)
    except OSError:
        result["error"] = "file_not_found"
        return result

    if result["file_size"] == 0:
        result["error"] = "empty_file"
        return result

    try:
        wf = wave.open(wav_path, 'rb')
    except Exception as e:
        result["error"] = f"corrupt_file:{str(e)[:80]}"
        return result

    try:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        duration = n_frames / sample_rate if sample_rate > 0 else 0

        result["channels"] = channels
        result["sample_width"] = sample_width
        result["sample_rate"] = sample_rate
        result["duration"] = round(duration, 3)

        # 读取音频数据做能量分析（只处理16bit PCM）
        if sample_width == 2 and n_frames > 0:
            max_read_frames = min(n_frames, sample_rate * 60)
            raw_data = wf.readframes(max_read_frames)
            n_samples = len(raw_data) // (sample_width * channels)

            if n_samples > 0:
                if channels == 1:
                    samples = struct.unpack(f'<{n_samples}h', raw_data[:n_samples * 2])
                else:
                    all_samples = struct.unpack(f'<{n_samples * channels}h', raw_data[:n_samples * channels * 2])
                    samples = all_samples[::channels]

                # RMS
                sum_sq = sum(s * s for s in samples)
                rms = math.sqrt(sum_sq / len(samples))
                result["rms"] = round(rms, 1)

                # Peak
                result["peak"] = max(abs(s) for s in samples)

                # 直流偏移
                result["dc_offset"] = round(sum(samples) / len(samples), 1)

                # 削波比例
                clip_count = sum(1 for s in samples if abs(s) >= 32000)
                result["clipping_ratio"] = round(clip_count / len(samples), 6)

                # SNR 估算
                frame_len = int(sample_rate * SNR_FRAME_MS / 1000)
                hop_len = int(sample_rate * SNR_HOP_MS / 1000)
                if frame_len > 0 and hop_len > 0 and len(samples) >= frame_len:
                    energies = []
                    for fi in range(0, len(samples) - frame_len, hop_len):
                        frame = samples[fi:fi + frame_len]
                        e = sum(s * s for s in frame) / frame_len
                        energies.append(e)
                    if energies:
                        energies_sorted = sorted(energies)
                        n_e = len(energies_sorted)
                        n_low = max(n_e // 5, 1)
                        n_high = max(n_e // 5, 1)
                        noise_e = sum(energies_sorted[:n_low]) / n_low
                        signal_e = sum(energies_sorted[-n_high:]) / n_high
                        if noise_e > 0:
                            snr_db = 10 * math.log10(signal_e / noise_e)
                        else:
                            snr_db = 99.0
                        result["snr_db"] = round(snr_db, 1)

    except Exception as e:
        result["error"] = f"read_error:{str(e)[:80]}"
    finally:
        wf.close()

    return result


def scan_all_wavs(scan_dirs: list, logger) -> list:
    """递归扫描所有目录中的wav文件"""
    all_wavs = []
    for d in scan_dirs:
        if not d.exists():
            logger.warning(f"目录不存在，跳过: {d}")
            continue
        count = 0
        for root, dirs, files in os.walk(d):
            for f in files:
                if f.lower().endswith('.wav'):
                    all_wavs.append(str(Path(root) / f))
                    count += 1
        logger.info(f"  {d}: {count} 个wav")
    logger.info(f"总计扫描到: {len(all_wavs)} 个wav文件")
    return all_wavs


def run_quality_check(args):
    output_dir = OUTPUT_DIR
    logger, log_file = setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("S10 ASR 音频数据质量扫描（仅采集数值）")
    logger.info("=" * 60)
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"并行进程: {args.workers}")
    logger.info(f"日志文件: {log_file}")

    # 1. 扫描文件
    if args.source_only:
        scan_dirs = [BASE_DIR / "wav" / "source"]
    else:
        scan_dirs = [d for d in SCAN_DIRS]
    logger.info("扫描目录:")
    wav_files = scan_all_wavs(scan_dirs, logger)

    if not wav_files:
        logger.warning("未找到任何wav文件")
        return

    # 2. 并行分析
    total = len(wav_files)
    results = []
    processed = 0
    t_start = time.time()

    logger.info(f"开始质量分析 ({args.workers} 进程)...")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(analyze_wav, p): p
            for p in wav_files
        }

        for future in as_completed(futures):
            try:
                r = future.result()
                results.append(r)
            except Exception as e:
                path = futures[future]
                results.append({
                    "path": path, "uid": Path(path).stem,
                    "error": f"process_error:{e}",
                    "duration": 0, "sample_rate": 0, "channels": 0,
                    "sample_width": 0, "rms": 0, "peak": 0,
                    "clipping_ratio": 0, "dc_offset": 0, "snr_db": 0,
                    "file_size": 0,
                })

            processed += 1
            if processed % 10000 == 0:
                elapsed = time.time() - t_start
                speed = processed / elapsed
                eta = (total - processed) / speed if speed > 0 else 0
                pct = processed / total * 100
                logger.info(
                    f"分析进度: {processed}/{total} ({pct:.1f}%) | "
                    f"速度: {speed:.0f}条/s | "
                    f"已用: {format_duration(elapsed)} | "
                    f"剩余: {format_duration(eta)}"
                )

    elapsed = time.time() - t_start
    logger.info(f"分析完成，耗时: {format_duration(elapsed)}，速度: {total / elapsed:.0f}条/s")

    # 3. 输出 jsonl
    detail_path = output_dir / "quality_detail.jsonl"
    with open(detail_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"逐条详情已保存: {detail_path}")

    valid_count = sum(1 for r in results if not r.get("error"))
    error_count = total - valid_count
    logger.info(f"总计: {total} | 可读: {valid_count} | 读取失败: {error_count}")
    logger.info("运行 python 03b_quality_report.py 生成完整报告")


def parse_args():
    parser = argparse.ArgumentParser(description="S10 ASR 音频数据质量扫描（仅采集数值）")
    parser.add_argument(
        '--workers', type=int, default=4,
        help='并行进程数 (default: 4)'
    )
    parser.add_argument(
        '--source-only', action='store_true',
        help='只扫描 wav/source 目录'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_quality_check(args)
