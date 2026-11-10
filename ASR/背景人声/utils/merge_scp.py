#!/usr/bin/env python3
"""
合并多个 wav.scp 和 text.txt 文件。

用法：
    # 合并原始 + 增强的 wav.scp 和 text.txt
    python merge_scp.py --scp original/wav.scp augment/wav_train.scp --text original/text.txt augment/text_train.txt --output_dir combined/

    # 只合并 wav.scp
    python merge_scp.py --scp a.scp b.scp c.scp --output_dir combined/
"""

import argparse
import os


def merge_files(input_files: list, output_path: str):
    """合并多个文本文件，去重（按第一列 key 去重，保留首次出现的）。"""
    seen_keys = set()
    lines = []
    for f in input_files:
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                key = line.split(None, 1)[0]
                if key not in seen_keys:
                    seen_keys.add(key)
                    lines.append(line)

    with open(output_path, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + "\n")

    return len(lines), len(seen_keys)


def main():
    parser = argparse.ArgumentParser(description="合并多个 wav.scp / text.txt 文件")
    parser.add_argument("--scp", nargs="+", default=[], help="要合并的 wav.scp 文件列表")
    parser.add_argument("--text", nargs="+", default=[], help="要合并的 text.txt 文件列表")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.scp:
        out = os.path.join(args.output_dir, "wav.scp")
        total, unique = merge_files(args.scp, out)
        print(f"wav.scp: 合并 {len(args.scp)} 个文件 → {total} 条（去重后 {unique} 条）→ {out}")

    if args.text:
        out = os.path.join(args.output_dir, "text.txt")
        total, unique = merge_files(args.text, out)
        print(f"text.txt: 合并 {len(args.text)} 个文件 → {total} 条（去重后 {unique} 条）→ {out}")

    if not args.scp and not args.text:
        print("请指定 --scp 或 --text 参数")


if __name__ == "__main__":
    main()
