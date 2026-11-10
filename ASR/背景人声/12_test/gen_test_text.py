"""
从 E:\8.音频\03 s10\wavs 目录下各日期文件夹中的 label/label.txt 整合生成测试集 text 文件。

目录结构:
  wavs/<日期范围>/labelled/source/*.wav   -- 音频文件
  wavs/<日期范围>/labelled/label/label.txt -- 标注文件

label.txt 格式:
  <utterance_id> <text>

输出:
  12_test/text  -- 合并后的标注文件（同 label.txt 格式）
  12_test/wav.scp -- 音频路径映射文件（utterance_id -> wav_path）
"""

import os
from pathlib import Path

# 配置路径
WAVS_ROOT = Path(r"E:\8.音频\03 s10\wavs")
OUTPUT_DIR = Path(r"E:\8.音频\03 s10\12_test")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

text_lines = []
wav_scp_lines = []
total_labels = 0
total_wavs = 0

# 遍历所有日期文件夹（按名称排序）
date_dirs = sorted([d for d in WAVS_ROOT.iterdir() if d.is_dir()])

print(f"找到 {len(date_dirs)} 个日期文件夹")

for date_dir in date_dirs:
    label_file = date_dir / "labelled" / "label" / "label.txt"
    source_dir = date_dir / "labelled" / "source"

    if not label_file.exists():
        print(f"  [跳过] {date_dir.name}: 无 label.txt")
        continue

    # 读取 label.txt
    with open(label_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # 收集该文件夹的标注
    folder_count = 0
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            continue
        utt_id, text = parts
        text_lines.append(f"{utt_id} {text}")

        # 生成 wav.scp 条目
        wav_path = source_dir / f"{utt_id}.wav"
        if wav_path.exists():
            wav_scp_lines.append(f"{utt_id} {wav_path}")
            total_wavs += 1

        folder_count += 1

    total_labels += folder_count
    print(f"  [OK] {date_dir.name}: {folder_count} 条标注")

# 写入 text 文件
text_output = OUTPUT_DIR / "text"
with open(text_output, "w", encoding="utf-8") as f:
    f.write("\n".join(text_lines) + "\n")

# 写入 wav.scp 文件
wav_scp_output = OUTPUT_DIR / "wav.scp"
with open(wav_scp_output, "w", encoding="utf-8") as f:
    f.write("\n".join(wav_scp_lines) + "\n")

print(f"\n===== 汇总 =====")
print(f"总标注数: {total_labels}")
print(f"匹配音频数: {total_wavs}")
print(f"缺失音频数: {total_labels - total_wavs}")
print(f"输出 text: {text_output}")
print(f"输出 wav.scp: {wav_scp_output}")
