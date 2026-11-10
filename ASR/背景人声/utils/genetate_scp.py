#!/usr/bin/env python3
"""
为指定目录生成 wav.scp（Kaldi 格式: <uid> <绝对路径>）。

用法:
    python gen_wav_scp.py
    python gen_wav_scp.py --wav-dir /path/to/wav
    python gen_wav_scp.py --wav-dir /path/to/wav --output /path/to/wav.scp
"""

import argparse
import os
from pathlib import Path

BASE_DIR = Path(r'/data/share/dataset/s10_data_reflow/')
DEFAULT_OUTPUT_DIR = BASE_DIR / "unlabeled" / "wav.scp"
DEFAULT_WAV_DIR = BASE_DIR / "unlabeled" / "wav"


def gen_wav_scp(wav_dir: Path, output: Path):
    wav_dir = wav_dir.resolve()
    print(f"递归扫描目录: {wav_dir}")

    entries = []
    for root, _, files in os.walk(wav_dir):
        for fname in files:
            if fname.lower().endswith('.wav'):
                fpath = Path(root) / fname
                uid = fpath.stem
                entries.append((uid, str(fpath.resolve())))

    entries.sort(key=lambda x: x[0])
    print(f"找到 {len(entries)} 个 wav 文件")

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        for uid, path in entries:
            f.write(f"{uid} {path}\n")

    print(f"已写入: {output} ({len(entries)} 条)")


def main():
    parser = argparse.ArgumentParser(description="生成 wav.scp")
    parser.add_argument('--wav-dir', default=str(DEFAULT_WAV_DIR), help='wav 文件目录')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT_DIR), help='输出路径，默认为 wav-dir/wav.scp')
    args = parser.parse_args()

    wav_dir = Path(args.wav_dir)
    output = Path(args.output) if args.output else wav_dir / "wav.scp"
    gen_wav_scp(wav_dir, output)


if __name__ == "__main__":
    main()
