#!/usr/bin/env python3
"""JSON 与 JSONL 互转工具。

用法：
    # JSON → JSONL
    python json_jsonl_convert.py input.json output.jsonl

    # JSONL → JSON
    python json_jsonl_convert.py input.jsonl output.json

    # 自动推断输出路径（换后缀）
    python json_jsonl_convert.py input.json
    python json_jsonl_convert.py input.jsonl
"""

import json
import sys
from pathlib import Path


def json_to_jsonl(src: str, dst: str):
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    with open(dst, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"JSON → JSONL: {len(data)} 条  {dst}")


def jsonl_to_json(src: str, dst: str):
    data = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"JSONL → JSON: {len(data)} 条  {dst}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    src = sys.argv[1]
    suffix = Path(src).suffix.lower()

    if len(sys.argv) >= 3:
        dst = sys.argv[2]
    else:
        dst = str(Path(src).with_suffix(".jsonl" if suffix == ".json" else ".json"))

    if suffix == ".json":
        json_to_jsonl(src, dst)
    elif suffix == ".jsonl":
        jsonl_to_json(src, dst)
    else:
        print(f"无法识别后缀 '{suffix}'，请使用 .json 或 .jsonl 文件")
        sys.exit(1)


if __name__ == "__main__":
    main()
