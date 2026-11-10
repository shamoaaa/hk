#!/usr/bin/env python3
"""
修复结果文件中的 unlabeled 音频路径。
将 .../unlabeled/xxx.wav 替换为 .../unlabeled/wav/xxx.wav

影响的文件类型：
  - wav.scp:     uid /path/to/unlabeled/xxx.wav
  - pretag.jsonl: JSON 中嵌入的绝对路径

用法:
    python fix_unlabeled_paths.py
    python fix_unlabeled_paths.py --base-dir /data/share/dataset/s10_data_reflow
    python fix_unlabeled_paths.py --dry-run   # 只预览不修改
"""

import argparse
import re
from pathlib import Path

BASE_DIR = Path(r'/data/share/dataset/s10_data_reflow')

# 需要修复的目录
TARGET_DIRS = [
    "firered_pretag_output",
    "qwen_pretag_output",
    "compare_output",
]

# 需要修复的文件名
TARGET_FILES = ["wav.scp", "pretag.jsonl"]

# 匹配 unlabeled/xxx.wav（但不匹配已经是 unlabeled/wav/xxx.wav 的）
PATTERN = re.compile(r'(unlabeled)/(?!wav/)([^/\s"]+\.wav)')
REPLACEMENT = r'\1/wav/\2'


def fix_file(filepath: Path, dry_run: bool) -> int:
    """修复单个文件，返回替换行数"""
    if not filepath.exists():
        return 0

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    changed = 0
    new_lines = []
    for line in lines:
        new_line = PATTERN.sub(REPLACEMENT, line)
        if new_line != line:
            changed += 1
        new_lines.append(new_line)

    if changed > 0 and not dry_run:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

    return changed


def main():
    parser = argparse.ArgumentParser(description="修复 unlabeled 路径")
    parser.add_argument('--base-dir', default=str(BASE_DIR), help='数据根目录')
    parser.add_argument('--dry-run', action='store_true', help='只预览不修改')
    args = parser.parse_args()

    base = Path(args.base_dir)
    total_changed = 0

    for dir_name in TARGET_DIRS:
        d = base / dir_name
        if not d.exists():
            continue
        # 递归查找目标文件
        for fname in TARGET_FILES:
            for filepath in d.rglob(fname):
                changed = fix_file(filepath, args.dry_run)
                if changed > 0:
                    action = "[预览]" if args.dry_run else "[已修复]"
                    print(f"  {action} {filepath} — {changed} 行")
                    total_changed += changed

    # 也检查 unlabeled 目录下的 wav.scp
    unlabeled_scp = base / "unlabeled" / "wav.scp"
    if unlabeled_scp.exists():
        changed = fix_file(unlabeled_scp, args.dry_run)
        if changed > 0:
            action = "[预览]" if args.dry_run else "[已修复]"
            print(f"  {action} {unlabeled_scp} — {changed} 行")
            total_changed += changed

    if total_changed == 0:
        print("没有需要修复的路径")
    else:
        action = "预览" if args.dry_run else "修复"
        print(f"\n共{action} {total_changed} 行")


if __name__ == "__main__":
    main()
