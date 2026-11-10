#!/usr/bin/env python3
"""从 AISHELL transcript 生成 manifest.jsonl"""

import json
import os
import re
import sys


def find_wav(wav_root, utt_id):
    """根据 utt_id 在 wav_root/train|dev|test/Sxxxx/ 下查找对应 wav 文件"""
    # 从 utt_id 提取说话人 ID，如 BAC009S0002W0122 -> S0002
    m = re.search(r'(S\d+)', utt_id)
    if not m:
        return None
    spk = m.group(1)
    for split in ("train", "dev", "test"):
        path = os.path.join(wav_root, split, spk, f"{utt_id}.wav")
        if os.path.isfile(path):
            return os.path.abspath(path)
    return None


def main():
    transcript = sys.argv[1] if len(sys.argv) > 1 else \
        "/root/demo/dataset/aishell/extracted/data_aishell/transcript/aishell_transcript_v0.8.txt"
    output = sys.argv[2] if len(sys.argv) > 2 else \
        "/root/demo/dataset/aishell/extracted/data_aishell/manifest.jsonl"

    wav_root = os.path.join(os.path.dirname(os.path.dirname(transcript)), "wav")

    total, found, skipped = 0, 0, 0
    with open(transcript, "r", encoding="utf-8") as fin, \
         open(output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            parts = line.split()
            utt_id = parts[0]
            txt = "".join(parts[1:])  # 去掉空格拼成连续文本

            wav_path = find_wav(wav_root, utt_id)
            if wav_path is None:
                skipped += 1
                continue
            found += 1
            entry = {"key": utt_id, "wav": wav_path, "txt": txt}
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"完成: 共 {total} 条, 写入 {found} 条, 跳过 {skipped} 条 (wav 不存在)")
    print(f"输出: {output}")


if __name__ == "__main__":
    main()
