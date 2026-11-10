#!/usr/bin/env python3
"""
公开数据集批量增强脚本（多进程版）

基于 public_dataset_gen_scp_txt.py 生成的标签目录结构，从每个数据集中随机采样 10%
的数据进行多说话人数据增强。

核心约束：
  1. 每个数据集采样 10% 作为主说话人
  2. 背景说话人从同一 split 的所有数据集中采样 10%
  3. 各 split 完全隔离：train 的主说话人和背景说话人只来自 train，
     dev 只来自 dev，test 只来自 test

多进程策略：
  - 条目级并行：同一数据集内的条目分配到多个 worker 并行增强
  - 每个 worker 独立创建 AudioAugmentor 和 LRUAudioCache 实例
  - 背景说话人池（只读列表）通过 initializer 共享给所有 worker

用法:
    # 使用默认配置增强所有数据集（自动检测 CPU 核数）
    python batch_augment_public.py

    # 指定 worker 数
    python batch_augment_public.py --workers 8

    # 只增强指定数据集
    python batch_augment_public.py --datasets aidatatang_200zh data_aishell

    # 自定义采样比例和增强倍数
    python batch_augment_public.py --sample-ratio 0.2 --factor 3
"""

import argparse
import collections
import json
import logging
import os
import random
import re
import time
import multiprocessing as mp
from functools import partial
from pathlib import Path

import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============ 默认配置 ============
DEFAULT_LABEL_ROOT = "/data1/code/label"
DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "augment_config_speaker_multi.yaml")
DEFAULT_OUTPUT_ROOT = "/data1/code/label/augmented"
DEFAULT_SAMPLE_RATIO = 0.1
DEFAULT_SEED = 42
# ==================================


# ============ Worker 全局变量（每个进程独立初始化） ============
_worker_augmentor = None
_worker_bg_pool = None


def _worker_init(config_path: str, factor: int, bg_pool: list):
    """每个 worker 进程的初始化函数，创建独立的 augmentor 实例。"""
    global _worker_augmentor, _worker_bg_pool
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from augment_engine import AudioAugmentor
    _worker_augmentor = AudioAugmentor(config_path)
    if factor is not None:
        _worker_augmentor.augment_factor = factor
    _worker_bg_pool = bg_pool


def _worker_augment_entry(task: dict) -> list:
    """
    Worker 函数：增强单条音频的所有 augment_factor 个副本。

    task: {key, wav, txt, output_dir, augment_factor, done_keys, entry_idx}
    返回: [(aug_key, aug_abs_path, txt, status), ...]
        status: "done" | "skipped" | "failed"
    """
    global _worker_augmentor, _worker_bg_pool

    key = task["key"]
    wav_path = task["wav"]
    txt = task["txt"]
    output_dir = task["output_dir"]
    augment_factor = task["augment_factor"]
    done_keys = task.get("done_keys", set())
    entry_idx = task["entry_idx"]

    results = []

    # 读取主音频
    try:
        audio, sr = sf.read(wav_path, dtype="float64")
    except Exception as ex:
        for aug_idx in range(augment_factor):
            aug_key = f"{key}_aug{aug_idx}"
            if aug_key not in done_keys:
                results.append((aug_key, "", txt, "failed"))
        return results

    # 构建简易缓存（当前条目 + 按需加载背景）
    audio_cache = _SimpleAudioCache()
    audio_cache.put(wav_path, (audio, sr))
    for p in _worker_bg_pool:
        audio_cache.register(p)

    for aug_idx in range(augment_factor):
        aug_key = f"{key}_aug{aug_idx}"
        aug_filename = f"{aug_key}.wav"

        # 跳过已在 progress 中记录的
        if aug_key in done_keys:
            continue

        try:
            aug_audio = _worker_augmentor.augment_single(
                audio, sr, wav_path, aug_idx, entry_idx,
                audio_cache=audio_cache,
                candidate_paths=_worker_bg_pool,
            )
        except Exception:
            aug_audio = None

        if aug_audio is None:
            results.append((aug_key, "", txt, "failed"))
            continue

        aug_path = os.path.join(output_dir, aug_filename)
        sf.write(aug_path, aug_audio, sr)
        results.append((aug_key, os.path.abspath(aug_path), txt, "done"))

    return results


class _SimpleAudioCache:
    """Worker 内的简易音频缓存，按需加载，无大小限制（单条任务内使用）。"""

    def __init__(self):
        self._cache = {}
        self._keys = set()

    def register(self, path: str):
        self._keys.add(path)

    def put(self, path: str, data):
        self._cache[path] = data
        self._keys.add(path)

    def __contains__(self, path: str):
        return path in self._keys

    def __getitem__(self, path: str):
        if path in self._cache:
            return self._cache[path]
        data = sf.read(path, dtype="float64")
        self._cache[path] = data
        return data

    def get(self, path, default=None):
        try:
            return self[path]
        except Exception:
            return default

    def __len__(self):
        return len(self._keys)


# ============ 主流程函数 ============

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


def load_scp_and_text(scp_path: Path, text_path: Path) -> list:
    """读取 scp + text 文件，返回 [{key, wav, txt}, ...]"""
    wav_map = {}
    with open(scp_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                wav_map[parts[0]] = parts[1]

    text_map = {}
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                text_map[parts[0]] = parts[1]

    entries = []
    for key in wav_map:
        if key in text_map:
            entries.append({"key": key, "wav": wav_map[key], "txt": text_map[key]})
    return entries


def sample_entries(entries: list, ratio: float, rng: random.Random) -> list:
    """从 entries 中随机采样 ratio 比例的数据。"""
    n = max(1, int(len(entries) * ratio))
    if n >= len(entries):
        return list(entries)
    return rng.sample(entries, n)


def _load_progress(progress_path: Path) -> dict:
    """
    从 progress.jsonl 加载已完成的增强记录。
    返回 {aug_key: {"wav": abs_path, "txt": text}, ...}
    """
    done_map = {}
    if not progress_path.exists():
        return done_map
    with open(progress_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("status") == "done":
                    done_map[rec["key"]] = {"wav": rec["wav"], "txt": rec["txt"]}
            except (json.JSONDecodeError, KeyError):
                continue
    return done_map


def _append_progress(progress_path: Path, records: list):
    """
    原子追加增强结果到 progress.jsonl。
    records: [(aug_key, aug_abs_path, txt, status), ...]
    """
    with open(progress_path, "a", encoding="utf-8") as f:
        for aug_key, aug_path, txt, status in records:
            rec = {"key": aug_key, "wav": aug_path, "txt": txt, "status": status}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def augment_dataset_mp(split: str, dataset_name: str, sampled: list,
                       bg_pool: list, output_root: Path, config_path: str,
                       factor: int, resume: bool, num_workers: int):
    """
    使用多进程增强单个数据集，支持 progress.jsonl 断点续跑。

    断点机制：
      - progress.jsonl 记录每条增强结果（原子追加，进程安全）
      - 重跑时从 progress 恢复已完成的条目，跳过不再重复增强
      - scp/text 最终从 progress + 新增结果重建

    返回 (scp_lines, txt_lines, stats_dict)
    """
    ds_output = output_root / split / dataset_name
    ds_output.mkdir(parents=True, exist_ok=True)
    output_dir_str = str(ds_output)
    progress_path = ds_output / "progress.jsonl"

    # 确定增强倍数
    if factor is None:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        aug_factor = cfg.get("augment_factor", 2)
    else:
        aug_factor = factor

    # 断点续跑：加载已完成的记录
    done_map = {}
    if resume:
        done_map = _load_progress(progress_path)
        if done_map:
            logger.info(f"  [断点续跑] 从 progress.jsonl 恢复 {len(done_map)} 条已完成记录")

    # 构建任务列表，过滤已完成的条目
    tasks = []
    already_done_keys = set()
    for idx, entry in enumerate(sampled):
        key = entry["key"]
        all_done = True
        for aug_idx in range(aug_factor):
            aug_key = f"{key}_aug{aug_idx}"
            if aug_key not in done_map:
                all_done = False
                break
        if all_done:
            # 该条目的所有增强副本都已完成
            for aug_idx in range(aug_factor):
                already_done_keys.add(f"{key}_aug{aug_idx}")
        else:
            tasks.append({
                "key": key,
                "wav": entry["wav"],
                "txt": entry["txt"],
                "output_dir": output_dir_str,
                "augment_factor": aug_factor,
                "done_keys": set(k for k in done_map if k.startswith(f"{key}_aug")),
                "entry_idx": idx,
            })

    total_all = len(sampled) * aug_factor
    total_pending = len(tasks) * aug_factor - sum(
        len(t["done_keys"]) for t in tasks
    )
    logger.info(
        f"  [{split}/{dataset_name}] 总计 {total_all} 条: "
        f"已完成 {len(already_done_keys)} + 待处理 {total_pending} 条"
    )

    if not tasks:
        logger.info(f"  [{split}/{dataset_name}] 所有条目已完成，跳过")
        # 从 done_map 重建 scp/text
        scp_lines = [f"{k} {v['wav']}" for k, v in sorted(done_map.items())]
        txt_lines = [f"{k} {v['txt']}" for k, v in sorted(done_map.items())]
        return scp_lines, txt_lines, {"done": 0, "skipped": len(done_map), "failed": 0}

    t_start = time.time()
    stats = {"done": 0, "skipped": len(already_done_keys), "failed": 0}

    # 收集新增结果（用于追加 progress）
    new_results_all = []

    effective_workers = min(num_workers, len(tasks))
    if effective_workers <= 1:
        _worker_init(config_path, factor, bg_pool)
        for task in tasks:
            results = _worker_augment_entry(task)
            new_results_all.extend(results)
            for aug_key, aug_path, txt, status in results:
                stats[status] += 1
    else:
        logger.info(f"  [{split}/{dataset_name}] 启动 {effective_workers} 个 worker")
        with mp.Pool(
            processes=effective_workers,
            initializer=_worker_init,
            initargs=(config_path, factor, bg_pool),
        ) as pool:
            log_interval = max(1, len(tasks) // 20)
            for i, results in enumerate(pool.imap_unordered(_worker_augment_entry, tasks)):
                new_results_all.extend(results)
                for aug_key, aug_path, txt, status in results:
                    stats[status] += 1

                if (i + 1) % log_interval == 0:
                    elapsed = time.time() - t_start
                    done_now = stats["done"] + stats["skipped"]
                    speed = stats["done"] / elapsed if elapsed > 0 else 0
                    remaining = total_all - done_now - stats["failed"]
                    eta = remaining / speed if speed > 0 else 0
                    logger.info(
                        f"  [{split}/{dataset_name}] {done_now}/{total_all} ({100*done_now/total_all:.1f}%) | "
                        f"新增 {stats['done']} 跳过 {stats['skipped']} 失败 {stats['failed']} | "
                        f"{speed:.1f} 条/秒 | 剩余 {eta:.0f}s"
                    )

    # 追加新结果到 progress.jsonl
    if new_results_all:
        _append_progress(progress_path, new_results_all)
        logger.info(f"  进度已保存: {progress_path}")

    elapsed = time.time() - t_start
    logger.info(
        f"  [{split}/{dataset_name}] 完成: 新增 {stats['done']} + 跳过 {stats['skipped']} "
        f"+ 失败 {stats['failed']}, 耗时 {elapsed:.1f}s"
    )

    # 从 done_map + 新增结果 重建 scp/text
    scp_lines = []
    txt_lines = []

    # 已有的
    for k, v in sorted(done_map.items()):
        if v["wav"]:
            scp_lines.append(f"{k} {v['wav']}")
            txt_lines.append(f"{k} {v['txt']}")

    # 新增的
    for aug_key, aug_path, txt, status in new_results_all:
        if status in ("done", "skipped") and aug_path:
            if aug_key not in done_map:  # 避免重复
                scp_lines.append(f"{aug_key} {aug_path}")
                txt_lines.append(f"{aug_key} {txt}")

    return scp_lines, txt_lines, stats


def run_augmentation(args):
    label_root = Path(args.label_root)
    output_root = Path(args.output_root)
    splits = args.splits
    sample_ratio = args.sample_ratio
    num_workers = args.workers
    rng = random.Random(args.seed)

    if num_workers <= 0:
        num_workers = max(1, mp.cpu_count() - 1)
    logger.info(f"Worker 数: {num_workers}")

    # 1. 发现数据集
    all_datasets = discover_datasets(label_root, splits)

    if args.datasets:
        filtered = {}
        for name in args.datasets:
            if name in all_datasets:
                filtered[name] = all_datasets[name]
            else:
                logger.warning(f"数据集 '{name}' 未找到，可用: {list(all_datasets.keys())}")
        all_datasets = filtered

    if not all_datasets:
        logger.error("没有找到可用的数据集。请先运行 public_dataset_gen_scp_txt.py")
        return

    # 2. 加载所有数据集的 entries，按 split 分组
    split_dataset_entries = {}
    for dataset_name, split_info in sorted(all_datasets.items()):
        for split, (scp_path, text_path) in split_info.items():
            entries = load_scp_and_text(scp_path, text_path)
            if entries:
                if split not in split_dataset_entries:
                    split_dataset_entries[split] = {}
                split_dataset_entries[split][dataset_name] = entries
                logger.info(f"  [{split}/{dataset_name}] {len(entries)} 条")

    # 3. 按 split 构建背景说话人池（各 split 完全隔离）
    split_bg_pools = {}
    for split, ds_entries in split_dataset_entries.items():
        all_wavs = []
        for dataset_name, entries in ds_entries.items():
            for e in entries:
                all_wavs.append(e["wav"])
        split_bg_pools[split] = sample_entries(all_wavs, sample_ratio, rng)
        logger.info(f"[{split}] 背景说话人池: {len(split_bg_pools[split])}/{len(all_wavs)} 条 ({sample_ratio:.0%})")

    # 4. 读取 resume 配置（CLI 优先，否则从 yaml 读取）
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    resume = args.resume or cfg.get("resume", False)
    if resume:
        logger.info("断点续跑模式: 已启用")

    # 5. 逐 split 逐数据集增强（多进程）
    for split in splits:
        if split not in split_dataset_entries:
            continue

        bg_pool = split_bg_pools.get(split, [])

        logger.info(f"\n{'=' * 60}")
        logger.info(f"增强 {split} 集 (背景池: {len(bg_pool)} 条，仅来自 {split} 集)")
        logger.info(f"{'=' * 60}")

        for dataset_name, entries in sorted(split_dataset_entries[split].items()):
            sampled = sample_entries(entries, sample_ratio, rng)
            logger.info(f"\n[{split}/{dataset_name}] 采样 {len(sampled)}/{len(entries)} 条 ({sample_ratio:.0%})")

            scp_lines, txt_lines, stats = augment_dataset_mp(
                split=split,
                dataset_name=dataset_name,
                sampled=sampled,
                bg_pool=bg_pool,
                output_root=output_root,
                config_path=args.config,
                factor=args.factor,
                resume=resume,
                num_workers=num_workers,
            )

            # 写出 scp 和 text
            scp_out = output_root / split / f"{dataset_name}_wav.scp"
            txt_out = output_root / split / f"{dataset_name}_text"

            with open(scp_out, "w", encoding="utf-8") as f:
                f.write("\n".join(scp_lines) + "\n")
            with open(txt_out, "w", encoding="utf-8") as f:
                f.write("\n".join(txt_lines) + "\n")

            logger.info(f"  → {scp_out}")
            logger.info(f"  → {txt_out}")

    logger.info("\n===== 全部增强完成 =====")


def parse_args():
    parser = argparse.ArgumentParser(
        description="公开数据集批量增强（多进程版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="增强配置 YAML 路径")
    parser.add_argument("--label-root", default=DEFAULT_LABEL_ROOT,
                        help="标签根目录")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                        help="增强输出根目录")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="指定数据集名称，不指定则增强全部")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"],
                        help="增强哪些集合 (default: train dev test)")
    parser.add_argument("--sample-ratio", type=float, default=DEFAULT_SAMPLE_RATIO,
                        help=f"采样比例 (default: {DEFAULT_SAMPLE_RATIO})")
    parser.add_argument("--factor", type=int, default=None,
                        help="增强倍数（覆盖 yaml 中的 augment_factor）")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"随机种子 (default: {DEFAULT_SEED})")
    parser.add_argument("--workers", type=int, default=0,
                        help="Worker 进程数 (default: 0=自动检测 CPU 核数-1)")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑（从 progress.jsonl 恢复，跳过已完成条目）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_augmentation(args)
