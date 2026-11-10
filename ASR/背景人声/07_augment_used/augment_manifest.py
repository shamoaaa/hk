#!/usr/bin/env python3
"""
基于 manifest 的数据增强（带标签）。

读取 JSONL manifest，按 split 隔离增强，输出增强音频和新 manifest。
与原 augment_data.py 行为一致。

用法：
    python augment_manifest.py --config augment_config.yaml \
        --input_manifest data/manifest.jsonl \
        --output_dir data/augmented \
        --output_manifest data/augmented_manifest.jsonl \
        --splits train
"""

import argparse
import collections
import json
import logging
import os
import time

import numpy as np
import soundfile as sf

from augment_engine import AudioAugmentor


class LRUAudioCache:
    """有界 LRU 音频缓存，避免一次性加载全部音频导致 OOM。"""

    def __init__(self, maxsize: int = 2000):
        self._cache = collections.OrderedDict()
        self._maxsize = maxsize
        self._keys = set()          # 所有已注册的 key（不一定在内存中）

    def register(self, path: str):
        """注册一个路径为合法 key（不加载）。"""
        self._keys.add(path)

    def __contains__(self, path: str):
        return path in self._keys

    def __getitem__(self, path: str):
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        # 按需加载
        data = sf.read(path, dtype="float64")
        self._put(path, data)
        return data

    def _put(self, path, data):
        if path in self._cache:
            self._cache.move_to_end(path)
            return
        while len(self._cache) >= self._maxsize:
            self._cache.popitem(last=False)
        self._cache[path] = data

    def get(self, path, default=None):
        try:
            return self[path]
        except Exception:
            return default

    def __len__(self):
        return len(self._keys)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def process_manifest(augmentor: AudioAugmentor, input_manifest: str,
                     output_dir: str,
                     augment_splits: list = None):
    """处理 manifest，按 split 隔离增强。"""
    if augment_splits is None:
        augment_splits = ["train"]
    active_splits = set(augment_splits) | {"unknown"}

    os.makedirs(output_dir, exist_ok=True)

    entries = []
    with open(input_manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    logger.info("读取 %d 条 manifest 记录", len(entries))

    # 按 split 分组
    split_entries = {}
    for e in entries:
        split = augmentor._detect_split(e["wav"])
        split_entries.setdefault(split, []).append(e)

    for split, items in split_entries.items():
        logger.info("  %s: %d 条", split, len(items))

    augmented_by_split = {}  # {split: [entries]}
    skipped = 0

    for split, items in split_entries.items():
        if split not in active_splits:
            logger.info("跳过 %s 集增强（仅保留原始数据）", split)
            augmented_by_split.setdefault(split, []).extend(items)
            continue

        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)

        # 懒加载 LRU 音频缓存（避免全量预加载导致 OOM）
        logger.info("注册 %s 集音频路径（懒加载模式，LRU 上限 2000）...", split)
        t0 = time.time()
        audio_cache = LRUAudioCache(maxsize=2000)
        failed_paths = set()
        for e in items:
            wp = e["wav"]
            if wp not in audio_cache and wp not in failed_paths:
                try:
                    # 仅验证文件可读，不保留在内存
                    with open(wp, "rb") as _f:
                        _f.read(4)
                    audio_cache.register(wp)
                except Exception as ex:
                    logger.warning("文件不可读，跳过: %s (%s)", wp, ex)
                    failed_paths.add(wp)
        logger.info("注册完成: %d 个文件（跳过 %d 个不可读），耗时 %.1f 秒",
                     len(audio_cache), len(failed_paths), time.time() - t0)

        split_wav_paths = [e["wav"] for e in items if e["wav"] in audio_cache]

        tasks = []
        for idx, entry in enumerate(items):
            if entry["wav"] not in audio_cache:
                skipped += 1
                continue
            tasks.append((idx, entry))

        total = len(tasks) * augmentor.augment_factor

        # 断点重跑：根据配置决定是否跳过已有文件
        existing_files = set()
        if augmentor.resume and os.path.isdir(split_output_dir):
            existing_files = {f for f in os.listdir(split_output_dir) if f.endswith(".wav")}
        if existing_files:
            logger.info("[%s] 检测到已有 %d 个文件，将跳过（断点续跑）", split, len(existing_files))

        processed = 0
        skipped_exist = 0
        log_interval = max(1, total // 20)
        t_start = time.time()

        for idx, entry in tasks:
            wav_path = entry["wav"]
            key = entry["key"]
            txt = entry["txt"]
            audio, sr = audio_cache[wav_path]

            for aug_idx in range(augmentor.augment_factor):
                aug_key = f"{key}_aug{aug_idx}"
                aug_filename = f"{aug_key}.wav"

                # 跳过已存在的文件
                if aug_filename in existing_files:
                    skipped_exist += 1
                    augmented_by_split.setdefault(split, []).append({
                        "key": aug_key,
                        "wav": os.path.abspath(os.path.join(split_output_dir, aug_filename)),
                        "txt": txt,
                    })
                    continue

                aug_audio = augmentor.augment_single(
                    audio, sr, wav_path, aug_idx, idx,
                    audio_cache=audio_cache,
                    candidate_paths=split_wav_paths,
                )
                if aug_audio is None:
                    logger.warning("增强后音频包含 NaN/Inf，丢弃: %s (aug_%d)", key, aug_idx)
                    continue

                aug_path = os.path.join(split_output_dir, aug_filename)
                sf.write(aug_path, aug_audio, sr)

                augmented_by_split.setdefault(split, []).append({
                    "key": aug_key,
                    "wav": os.path.abspath(aug_path),
                    "txt": txt,
                })

                processed += 1
                done = processed + skipped_exist
                if processed % log_interval == 0:
                    elapsed = time.time() - t_start
                    speed = processed / elapsed if elapsed > 0 else 0
                    remaining = total - done
                    eta = remaining / speed if speed > 0 else 0
                    logger.info(
                        "  [%s] %d/%d (%.1f%%) | 跳过 %d 已有 | %.1f 条/秒 | 预计剩余 %.0f 秒",
                        split, done, total, 100 * done / total, skipped_exist, speed, eta,
                    )

        del audio_cache
        elapsed = time.time() - t_start
        logger.info("%s 集增强完成: %d 条，耗时 %.1f 秒", split, processed, elapsed)

    # 按 split 分别写入 wav.scp + text.txt
    total_entries = 0
    for split, split_entries_out in augmented_by_split.items():
        scp_path = os.path.join(output_dir, f"wav_{split}.scp")
        text_path = os.path.join(output_dir, f"text_{split}.txt")

        with open(scp_path, "w", encoding="utf-8") as f_scp, \
             open(text_path, "w", encoding="utf-8") as f_text:
            for entry in split_entries_out:
                key = entry["key"]
                wav = entry["wav"]
                txt = entry.get("txt", "")

                f_scp.write(f"{key} {wav}\n")
                f_text.write(f"{key} {txt}\n")

        total_entries += len(split_entries_out)
        logger.info("[%s] 输出 %d 条: wav_%s.scp  |  text_%s.txt", split, len(split_entries_out), split, split)

    logger.info("增强完成: %d 条原始 → %d 条增强（跳过 %d 条损坏）",
                len(entries), total_entries, skipped)


def main():
    parser = argparse.ArgumentParser(description="基于 manifest 的数据增强（带标签）")
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "公开数据集增强", "augment_config_speaker_multi.yaml"))
    parser.add_argument("--input_manifest", type=str, default=r"E:\8.音频\02 ASR入门\demo\dataset\manifest.json")
    parser.add_argument("--output_dir", type=str, default=r"E:\8.音频\02 ASR入门\demo\dataset\augment")
    parser.add_argument("--splits", type=str, default="dev,test")
    args = parser.parse_args()

    augment_splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    augmentor = AudioAugmentor(args.config)
    process_manifest(augmentor, args.input_manifest, args.output_dir,
                     augment_splits)


if __name__ == "__main__":
    main()
