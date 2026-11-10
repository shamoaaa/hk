#!/usr/bin/env python3
"""
批量替换 wav.scp 中的音频路径前缀。

用法：
    # 替换路径前缀
    python fix_scp_paths.py wav.scp --old "E:\\old\\path" --new "E:\\new\\path"

    # 预览模式（不写入，只显示变更）
    python fix_scp_paths.py wav.scp --old "/root/data" --new "/data/asr" --dry-run

    # 输出到新文件（不覆盖原文件）
    python fix_scp_paths.py wav.scp --old "/old" --new "/new" --output wav_fixed.scp
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="替换 wav.scp 中的音频路径前缀")
    parser.add_argument("scp", type=str, help="wav.scp 文件路径")
    parser.add_argument("--old", type=str, required=True, help="要替换的旧路径子串")
    parser.add_argument("--new", type=str, required=True, help="替换为的新路径子串")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径（默认覆盖原文件）")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不写入文件")
    args = parser.parse_args()

    with open(args.scp, "r", encoding="utf-8") as f:
        lines = f.readlines()

    changed = 0
    new_lines = []
    for line in lines:
        new_line = line.replace(args.old, args.new)
        if new_line != line:
            changed += 1
            if args.dry_run and changed <= 5:
                print(f"  - {line.strip()}")
                print(f"  + {new_line.strip()}")
                print()
        new_lines.append(new_line)

    print(f"共 {len(lines)} 行，{changed} 行被替换")

    if args.dry_run:
        if changed > 5:
            print(f"  ...（省略 {changed - 5} 条）")
        print("预览模式，未写入文件")
        return

    out_path = args.output or args.scp
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"已写入: {out_path}")


if __name__ == "__main__":
    main()
