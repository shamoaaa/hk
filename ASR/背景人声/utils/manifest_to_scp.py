#!/usr/bin/env python3
"""
将三字段 manifest.jsonl 转换为 wav.scp + text，按 split 切分。

根据 wav 路径中出现的 /train/、/dev/、/test/ 自动判断 split，
输出 wav_{split}.scp 和 text_{split}.txt。

用法：
    python manifest_to_scp.py manifest.jsonl --output_dir ./output

    # 指定输出目录（默认与 manifest 同目录）
    python manifest_to_scp.py manifest.jsonl --output_dir /path/to/output
"""

import argparse
import json
import os
import sys
from collections import defaultdict


def detect_split(wav_path: str) -> str:
    """从路径中检测 split（train/dev/test）。"""
    path = wav_path.replace("\\", "/")
    for split in ("train", "dev", "test"):
        if f"/{split}/" in path:
            return split
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="manifest.jsonl → wav.scp + text（按 split 切分）")
    parser.add_argument("manifest", type=str, help="输入的三字段 manifest.jsonl 路径")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录（默认与 manifest 同目录）")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(args.manifest) or "."
    os.makedirs(output_dir, exist_ok=True)

    # 按 split 分组
    split_entries = defaultdict(list)
    total = 0
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            split = detect_split(entry["wav"])
            split_entries[split].append(entry)
            total += 1

    print(f"读取 {total} 条记录，分布:")
    for split in sorted(split_entries.keys()):
        print(f"  {split}: {len(split_entries[split])} 条")

    # 按 split 写入
    for split, entries in split_entries.items():
        scp_path = os.path.join(output_dir, f"wav_{split}.scp")
        text_path = os.path.join(output_dir, f"text_{split}.txt")

        with open(scp_path, "w", encoding="utf-8") as f_scp, \
             open(text_path, "w", encoding="utf-8") as f_text:
            for entry in entries:
                key = entry["key"]
                wav = entry["wav"]
                txt = entry.get("txt", "")
                f_scp.write(f"{key} {wav}\n")
                f_text.write(f"{key} {txt}\n")

        print(f"  [{split}] {scp_path}  |  {text_path}")

    print("完成！")


if __name__ == "__main__":
    main()
