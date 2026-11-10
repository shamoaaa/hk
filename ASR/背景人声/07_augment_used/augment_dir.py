#!/usr/bin/env python3
"""
基于目录的数据增强（无标签）。

扫描输入目录中的 wav 文件，对每条音频执行增强链，输出到指定目录。
不需要 manifest 和标签文件。

用法：
    python augment_dir.py --config augment_config.yaml \
        --input_dir /path/to/clean_wavs \
        --output_dir /path/to/augmented

    # 指定增强倍数（覆盖 yaml 中的 augment_factor）
    python augment_dir.py --config augment_config.yaml \
        --input_dir /path/to/clean_wavs \
        --output_dir /path/to/augmented \
        --factor 3
"""

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from augment_engine import AudioAugmentor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def scan_wav_files(input_dir: str) -> list:
    """递归扫描目录下所有 wav 文件。"""
    wav_files = sorted(str(p) for p in Path(input_dir).rglob("*.wav"))
    return wav_files


def process_directory(augmentor: AudioAugmentor, input_dir: str,
                      output_dir: str, factor: int = None):
    """扫描目录并增强所有 wav 文件。"""
    if factor is not None:
        augmentor.augment_factor = factor

    os.makedirs(output_dir, exist_ok=True)

    wav_files = scan_wav_files(input_dir)
    if not wav_files:
        logger.error("未找到 wav 文件: %s", input_dir)
        return

    logger.info("扫描到 %d 个 wav 文件", len(wav_files))

    # 预加载所有音频（用于 speaker_mix / concatenate）
    logger.info("预加载音频到内存缓存...")
    t0 = time.time()
    audio_cache = {}
    for wp in wav_files:
        try:
            audio_cache[wp] = sf.read(wp, dtype="float64")
        except Exception as ex:
            logger.warning("读取失败，跳过: %s (%s)", wp, ex)
    logger.info("预加载完成: %d 个文件，耗时 %.1f 秒", len(audio_cache), time.time() - t0)

    candidate_paths = list(audio_cache.keys())
    total = len(candidate_paths) * augmentor.augment_factor

    # 断点重跑：根据配置决定是否跳过已有文件
    existing_files = set()
    if augmentor.resume and os.path.isdir(output_dir):
        existing_files = {f for f in os.listdir(output_dir) if f.endswith(".wav")}
    if existing_files:
        logger.info("检测到输出目录已有 %d 个文件，将跳过已完成的（断点续跑）", len(existing_files))

    processed = 0
    skipped_invalid = 0
    skipped_exist = 0
    log_interval = max(1, total // 20)
    t_start = time.time()

    for idx, wav_path in enumerate(candidate_paths):
        audio, sr = audio_cache[wav_path]
        stem = Path(wav_path).stem

        for aug_idx in range(augmentor.augment_factor):
            out_name = f"{stem}_aug{aug_idx}.wav"

            # 跳过已存在的文件
            if out_name in existing_files:
                skipped_exist += 1
                continue

            aug_audio = augmentor.augment_single(
                audio, sr, wav_path, aug_idx, idx,
                audio_cache=audio_cache,
                candidate_paths=candidate_paths,
            )
            if aug_audio is None:
                logger.warning("增强后音频无效，丢弃: %s (aug_%d)", stem, aug_idx)
                skipped_invalid += 1
                continue

            out_path = os.path.join(output_dir, out_name)
            sf.write(out_path, aug_audio, sr)

            processed += 1
            done = processed + skipped_exist
            if processed % log_interval == 0:
                elapsed = time.time() - t_start
                speed = processed / elapsed if elapsed > 0 else 0
                remaining = total - done
                eta = remaining / speed if speed > 0 else 0
                logger.info(
                    "  %d/%d (%.1f%%) | 跳过 %d 已有 | %.1f 条/秒 | 预计剩余 %.0f 秒",
                    done, total, 100 * done / total, skipped_exist, speed, eta,
                )

    del audio_cache
    elapsed = time.time() - t_start
    logger.info("增强完成: %d 个原始文件 × %d 倍 → 新增 %d 条 + 跳过 %d 条已有（丢弃 %d 条无效），耗时 %.1f 秒",
                len(candidate_paths), augmentor.augment_factor, processed, skipped_exist, skipped_invalid, elapsed)


def main():
    parser = argparse.ArgumentParser(description="基于目录的数据增强（无标签）")

    parser.add_argument("--config", type=str, help="增强配置 YAML 路径", 
    default=os.path.join(os.path.dirname(__file__), "公开数据集增强", "augment_config_speaker_multi.yaml"))

    # parser.add_argument("--input_dir", type=str, help="输入 wav 目录", 
    # default=r'E:\8.音频\02 ASR入门\demo\dataset\test\wav\AT0011')
    parser.add_argument("--input_dir", type=str, help="输入 wav 目录", 
    default=r'E:\8.音频\02 ASR入门\demo\dataset\origin\demo')

    parser.add_argument("--output_dir", type=str, help="输出增强音频目录", 
    default=r'E:\8.音频\02 ASR入门\demo\dataset\augment_multi')

    parser.add_argument("--factor", type=int, default=None,
                        help="增强倍数（覆盖 yaml 中的 augment_factor）")
    args = parser.parse_args()

    augmentor = AudioAugmentor(args.config)
    process_directory(augmentor, args.input_dir, args.output_dir, factor=args.factor)


if __name__ == "__main__":
    main()
