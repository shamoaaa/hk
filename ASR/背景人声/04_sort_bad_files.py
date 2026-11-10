#!/usr/bin/env python3
"""
将 quality_check/bad_files.txt 中的问题音频按原因分类拷贝到子目录。

输出结构:
  quality_check/bad_sorted/
    too_short/
    silence_or_low_energy/
    clipping/
    ...
  一个文件可能同时存在于多个子目录（有多个问题时）。

用法:
    python sort_bad_files.py
"""

import shutil
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(r'E:\8.音频\03 s10\s10_data_reflow')
BAD_FILE = BASE_DIR / "quality_check" / "bad_files.txt"
OUTPUT_DIR = BASE_DIR / "quality_check" / "bad_sorted"


def main():
    if not BAD_FILE.exists():
        print(f"未找到 {BAD_FILE}，请先运行 audio_quality_check.py")
        return

    # 解析
    entries = []
    with open(BAD_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) < 2:
                continue
            wav_path = parts[0]
            issues = [t.strip() for t in parts[1].split(',')]
            entries.append((wav_path, issues))

    print(f"读取到 {len(entries)} 个问题文件")

    # 按原因分类拷贝
    stats = defaultdict(int)
    copied = 0
    failed = 0

    for wav_path, issues in entries:
        src = Path(wav_path)
        if not src.exists():
            print(f"  跳过(文件不存在): {src}")
            failed += 1
            continue

        for issue in issues:
            # 去掉冒号后的详情作为目录名，如 unexpected_sr:8000 -> unexpected_sr
            folder_name = issue.split(':')[0]
            dst_dir = OUTPUT_DIR / folder_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name
            if not dst.exists():
                shutil.copy2(str(src), str(dst))
            stats[folder_name] += 1

        copied += 1

    # 汇总
    print(f"\n拷贝完成: {copied} 个文件, 跳过: {failed} 个")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"\n分类统计:")
    for folder, cnt in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {folder}/  -> {cnt} 个文件")


if __name__ == "__main__":
    main()
