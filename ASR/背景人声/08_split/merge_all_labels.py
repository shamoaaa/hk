"""
将 augment_label / reflow_label / speech_label 中的数据按 train/dev/test 拼接汇总到 all_label/。

输入目录结构：
  /data1/label/augment_label/{train,dev,test}/*_wav.scp, *_text
  /data1/label/reflow_label/{train,dev,test}/*_wav.scp, *_text
  /data1/label/speech_label/{train,dev,test}/*_wav.scp, *_text

输出：
  /data1/label/all_label/{train,dev,test}/wav.scp
  /data1/label/all_label/{train,dev,test}/text

优化：
  - 先从所有 wav.scp 中提取音频目录，一次性扫描建立文件集合（set）
  - 用 set 查找替代逐个 Path.exists()，220w 文件秒级完成
"""

import os
import sys
import time
from pathlib import Path
from collections import defaultdict

LABEL_ROOT = Path("/data1/label")
SOURCE_DIRS = ["augment_label", "reflow_label", "speech_label"]
SPLITS = ["train", "dev", "test"]
OUTPUT_DIR = LABEL_ROOT / "all_label"


def collect_wav_dirs(source_dirs, splits):
    """从所有 wav.scp 中提取音频文件所在的目录集合"""
    wav_dirs = set()
    for src_dir in source_dirs:
        for split in splits:
            split_dir = LABEL_ROOT / src_dir / split
            if not split_dir.exists():
                continue
            for f in split_dir.glob("*_wav.scp"):
                with open(f, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(None, 1)
                        if len(parts) == 2:
                            wav_dirs.add(os.path.dirname(parts[1]))
    return wav_dirs


def scan_existing_files(wav_dirs):
    """一次性扫描所有音频目录，建立文件路径集合"""
    existing = set()
    dirs = sorted(wav_dirs)
    print(f"扫描 {len(dirs)} 个音频目录...")
    for i, d in enumerate(dirs):
        if not os.path.isdir(d):
            print(f"  [跳过] 目录不存在: {d}")
            continue
        count_before = len(existing)
        for fname in os.listdir(d):
            existing.add(os.path.join(d, fname))
        count_new = len(existing) - count_before
        print(f"  [{i+1}/{len(dirs)}] {d} → {count_new} 个文件", flush=True)
    print(f"扫描完成，共 {len(existing)} 个文件\n")
    return existing


def count_total_lines(source_dirs, splits):
    """统计所有 wav.scp 的总行数，用于进度计算"""
    total = 0
    for src_dir in source_dirs:
        for split in splits:
            split_dir = LABEL_ROOT / src_dir / split
            if not split_dir.exists():
                continue
            for f in split_dir.glob("*_wav.scp"):
                with open(f, "r", encoding="utf-8") as fh:
                    total += sum(1 for _ in fh)
    return total


def merge_scp(source_dirs, split, output_path, existing_files, progress):
    """
    合并指定 split 下所有 *_wav.scp 文件。
    用预扫描的 existing_files 集合检查文件存在性。
    progress: [当前已处理行数, 总行数, 起始时间]
    """
    all_lines = []
    seen_keys = set()
    dup_count = 0
    missing_count = 0
    missing_paths = []
    file_count = 0

    for src_dir in source_dirs:
        split_dir = LABEL_ROOT / src_dir / split
        if not split_dir.exists():
            continue

        for f in sorted(split_dir.glob("*_wav.scp")):
            file_count += 1
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(None, 1)
                    if len(parts) < 2:
                        continue
                    key, wav_path = parts

                    progress[0] += 1
                    if progress[0] % 10000 == 0:
                        elapsed = time.time() - progress[2]
                        pct = progress[0] / progress[1] * 100
                        speed = progress[0] / elapsed if elapsed > 0 else 0
                        print(f"    进度: {progress[0]:,}/{progress[1]:,} ({pct:.1f}%) "
                              f"速度: {speed:,.0f} 条/秒 "
                              f"缺失: {missing_count}", flush=True)

                    if key in seen_keys:
                        dup_count += 1
                        continue
                    seen_keys.add(key)
                    if wav_path not in existing_files:
                        missing_count += 1
                        missing_paths.append(wav_path)
                        continue
                    all_lines.append(line)

    valid_keys = {line.split(None, 1)[0] for line in all_lines}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in all_lines:
            f.write(line + "\n")

    return len(all_lines), file_count, dup_count, missing_count, missing_paths, valid_keys


def merge_text(source_dirs, split, output_path, valid_keys):
    """合并指定 split 下所有 *_text 文件，只保留 valid_keys 中的条目。"""
    all_lines = []
    seen_keys = set()
    dup_count = 0
    skip_count = 0
    file_count = 0

    for src_dir in source_dirs:
        split_dir = LABEL_ROOT / src_dir / split
        if not split_dir.exists():
            continue

        for f in sorted(split_dir.glob("*_text")):
            file_count += 1
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    key = line.split(None, 1)[0]
                    if key in seen_keys:
                        dup_count += 1
                        continue
                    seen_keys.add(key)
                    if key not in valid_keys:
                        skip_count += 1
                        continue
                    all_lines.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in all_lines:
            f.write(line + "\n")

    return len(all_lines), file_count, dup_count, skip_count


def main():
    t0 = time.time()
    print(f"合并 {SOURCE_DIRS} → {OUTPUT_DIR}\n")

    # 第一步：收集所有音频目录并扫描文件
    wav_dirs = collect_wav_dirs(SOURCE_DIRS, SPLITS)
    existing_files = scan_existing_files(wav_dirs)

    # 统计总行数用于进度
    total_lines = count_total_lines(SOURCE_DIRS, SPLITS)
    print(f"wav.scp 总行数: {total_lines:,}\n")
    progress = [0, total_lines, time.time()]

    all_missing = []

    for split in SPLITS:
        print(f"=== {split} ===")

        # 合并 wav.scp
        scp_out = OUTPUT_DIR / split / "wav.scp"
        scp_lines, scp_files, scp_dups, scp_missing, missing_paths, valid_keys = merge_scp(
            SOURCE_DIRS, split, scp_out, existing_files, progress
        )
        print(f"  wav.scp: {scp_lines:,} 条 (来自 {scp_files} 个文件, "
              f"去重 {scp_dups:,}, 缺失剔除 {scp_missing:,}) → {scp_out}")
        all_missing.extend(missing_paths)

        # 合并 text
        txt_out = OUTPUT_DIR / split / "text"
        txt_lines, txt_files, txt_dups, txt_skipped = merge_text(
            SOURCE_DIRS, split, txt_out, valid_keys
        )
        print(f"  text:    {txt_lines:,} 条 (来自 {txt_files} 个文件, "
              f"去重 {txt_dups:,}, 跳过 {txt_skipped:,}) → {txt_out}")

        if scp_lines != txt_lines:
            print(f"  ⚠️ wav.scp ({scp_lines:,}) 与 text ({txt_lines:,}) 行数不一致!")

    # 打印缺失文件汇总
    if all_missing:
        print(f"\n=== 缺失文件 ({len(all_missing)} 个) ===")
        for p in all_missing[:50]:
            print(f"  {p}")
        if len(all_missing) > 50:
            print(f"  ... 还有 {len(all_missing) - 50} 个")

    # 总计
    print(f"\n=== 总计 ===")
    total_scp, total_txt = 0, 0
    for split in SPLITS:
        scp_path = OUTPUT_DIR / split / "wav.scp"
        txt_path = OUTPUT_DIR / split / "text"
        if scp_path.exists():
            with open(scp_path) as f:
                n = sum(1 for _ in f)
                total_scp += n
                print(f"  {split}/wav.scp: {n:,}")
        if txt_path.exists():
            with open(txt_path) as f:
                n = sum(1 for _ in f)
                total_txt += n
                print(f"  {split}/text:    {n:,}")
    print(f"  总 wav.scp: {total_scp:,}")
    print(f"  总 text:    {total_txt:,}")
    print(f"\n耗时: {time.time() - t0:.1f} 秒")


if __name__ == "__main__":
    main()
