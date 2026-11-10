#!/usr/bin/env python3
"""
从 cross_validate_output 中导出不一致样本到 labeled_check 目录。

功能：
1. 读取 validate_detail.jsonl，筛选不一致条目（非 all_agree）
2. 将对应 wav 文件拷贝到 labeled_check/wav/
3. 生成人工标签、FireRedASR、Qwen3-ASR 三方的 text.txt / train.jsonl / wav.scp
4. 生成合并的 validate_detail.json（所有不一致条目的列表）

输出结构:
  labeled_check/
    wav/                    — 不一致样本的 wav 文件
    wav.scp                 — Kaldi 格式音频路径
    human_text.txt          — 人工标注 (uid text)
    firered_text.txt        — FireRedASR 结果
    qwen_text.txt           — Qwen3-ASR 结果
    human_train.jsonl       — 人工标注 ChatML 格式
    firered_train.jsonl     — FireRedASR ChatML 格式
    qwen_train.jsonl        — Qwen3-ASR ChatML 格式
    validate_detail.json    — 所有不一致条目详情

用法:
    python export_labeled_check.py
    python export_labeled_check.py --category models_agree
    python export_labeled_check.py --detail validate_detail.jsonl --wav-dir labeled/wav --output labeled_check
"""

import json
import shutil
import argparse
from pathlib import Path

BASE_DIR = Path(r'E:\8.音频\03 s10\sp4_data')
# Linux 服务器上的 wav 路径前缀（用于 scp/jsonl 中的路径）
LINUX_WAV_PREFIX = "/data1/sp4_data/labeled_check/wav"


def load_detail(detail_path: Path, category: str = None) -> list:
    items = []
    with open(detail_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cat = obj.get("category", "")
            if cat == "all_agree":
                continue
            if category and cat != category:
                continue
            items.append(obj)
    return items


def strip_trailing_punct(text: str) -> str:
    """去除末尾中英文标点"""
    return text.rstrip('，。！？、；：…·,.!?;:')


def write_text(path: Path, items: list, field: str, strip_punct: bool = False):
    """写 text.txt 格式: uid text"""
    with open(path, 'w', encoding='utf-8') as f:
        for item in items:
            uid = item["uid"]
            text = item.get(field, "")
            if strip_punct:
                text = strip_trailing_punct(text)
            f.write(f"{uid} {text}\n")


def write_jsonl(path: Path, items: list, field: str, wav_prefix: str,
                strip_punct: bool = False):
    """写 ChatML 格式 jsonl"""
    with open(path, 'w', encoding='utf-8') as f:
        for item in items:
            uid = item["uid"]
            text = item.get(field, "")
            if strip_punct:
                text = strip_trailing_punct(text)
            wav_path = f"{wav_prefix}/{uid}.wav"
            record = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user",
                     "content": f"语音转写：<|startofspeech|>!{wav_path}<|endofspeech|>"},
                    {"role": "assistant", "content": text}
                ],
                "text_length": len(text)
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_wav_scp(path: Path, items: list, wav_prefix: str):
    """写 wav.scp 格式: uid wav_path"""
    with open(path, 'w', encoding='utf-8') as f:
        for item in items:
            uid = item["uid"]
            f.write(f"{uid} {wav_prefix}/{uid}.wav\n")


def main():
    parser = argparse.ArgumentParser(description="导出不一致样本到 labeled_check")
    parser.add_argument('--detail',
                        default=str(BASE_DIR / "cross_validate_output" / "validate_detail.jsonl"),
                        help='validate_detail.jsonl 路径')
    parser.add_argument('--wav-dir', default=str(BASE_DIR / "labeled" / "wav"),
                        help='源 wav 目录')
    parser.add_argument('--output', default=str(BASE_DIR / "labeled_check"),
                        help='输出目录')
    parser.add_argument('--category', default=None,
                        choices=['models_agree', 'all_differ', 'human_firered', 'human_qwen'],
                        help='只导出指定类别')
    parser.add_argument('--wav-prefix', default=LINUX_WAV_PREFIX,
                        help='scp/jsonl 中使用的 wav 路径前缀')
    parser.add_argument('--no-copy-wav', action='store_true',
                        help='不拷贝 wav 文件（仅生成标签文件）')
    args = parser.parse_args()

    detail_path = Path(args.detail)
    wav_dir = Path(args.wav_dir)
    output_dir = Path(args.output)
    out_wav_dir = output_dir / "wav"

    print(f"加载详情: {detail_path}")
    items = load_detail(detail_path, args.category)
    print(f"  不一致条目: {len(items)}")

    if not items:
        print("没有符合条件的条目")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 拷贝 wav
    if not args.no_copy_wav:
        out_wav_dir.mkdir(parents=True, exist_ok=True)
        copied, missing = 0, 0
        for item in items:
            uid = item["uid"]
            src = wav_dir / f"{uid}.wav"
            dst = out_wav_dir / f"{uid}.wav"
            if dst.exists():
                copied += 1
                continue
            if src.exists():
                shutil.copy2(str(src), str(dst))
                copied += 1
            else:
                missing += 1
        print(f"  wav 拷贝: {copied}, 缺失: {missing}")

    # 2. 生成三方 text.txt
    sources = [
        ("human", "human_text.txt", "human_train.jsonl", False),
        ("firered", "firered_text.txt", "firered_train.jsonl", False),
        ("qwen", "qwen_text.txt", "qwen_train.jsonl", True),
    ]
    for field, txt_name, jsonl_name, strip in sources:
        write_text(output_dir / txt_name, items, field, strip_punct=strip)
        write_jsonl(output_dir / jsonl_name, items, field, args.wav_prefix,
                    strip_punct=strip)
        print(f"  {txt_name} / {jsonl_name}")

    # 3. wav.scp
    write_wav_scp(output_dir / "wav.scp", items, args.wav_prefix)
    print(f"  wav.scp")

    # 4. validate_detail.json（完整列表）
    detail_json_path = output_dir / "validate_detail.json"
    with open(detail_json_path, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"  validate_detail.json")

    # 统计
    cats = {}
    for item in items:
        cat = item.get("category", "")
        cats[cat] = cats.get(cat, 0) + 1

    print(f"\n完成: {len(items)} 条不一致样本")
    print(f"输出目录: {output_dir}")
    for cat, cnt in sorted(cats.items()):
        print(f"  {cat}: {cnt}")


if __name__ == "__main__":
    main()
