#!/usr/bin/env python3
"""
S10 ASR 数据整理脚本
功能：
1. 读取所有标签文件，解析 ID -> 文本 映射
2. 在所有音频目录中查找对应的 wav 文件
3. 将匹配到的 wav 复制/移动到 labeled/ 目录
4. 生成 FunASR 训练所需的 jsonl、wav.scp、text.txt
"""

import os
import json
import shutil
import re
from pathlib import Path
from collections import defaultdict

# ============ 配置 ============
# BASE_DIR = Path(__file__).parent.resolve()
BASE_DIR = Path(r'E:\8.音频\03 s10\s10_data_reflow')
LABEL_DIR = BASE_DIR / "label" / "ready"
WAV_SOURCE_DIR = BASE_DIR / "wav" / "source"
OUTPUT_DIR = BASE_DIR / "labeled"
UNLABELED_DIR = BASE_DIR / "unlabeled"
# 是否复制文件（True=复制, False=移动）
COPY_MODE = False
# ==============================


def parse_label_line(line: str):
    """解析标签行，返回 (utterance_id, text)"""
    line = line.strip()
    if not line:
        return None, None
    # 格式: <id> <text>  (第一个空格分隔)
    parts = line.split(None, 1)
    if len(parts) < 2:
        return None, None
    return parts[0], parts[1]


def normalize_label_id(label_id: str) -> str:
    """
    将标签ID规范化，用于匹配wav文件名（不含扩展名）。
    处理特殊后缀如 _agi_stream_data_sequence_X_end_False/True
    """
    # 去掉 _agi_stream_data_sequence_*_end_* 后缀
    label_id = re.sub(r'_agi_stream_data_sequence_\d+_end_(?:True|False)$', '', label_id)
    return label_id


def build_wav_index(wav_source_dir: Path) -> dict:
    """
    递归扫描所有wav文件，建立 文件名(不含.wav) -> 完整路径 的索引。
    """
    print("正在扫描音频文件...")
    wav_index = {}
    count = 0
    for root, dirs, files in os.walk(wav_source_dir):
        for f in files:
            if f.lower().endswith('.wav'):
                stem = f[:-4]  # 去掉 .wav
                full_path = Path(root) / f
                wav_index[stem] = full_path
                count += 1
                if count % 50000 == 0:
                    print(f"  已扫描 {count} 个wav文件...")
    print(f"  扫描完成，共 {count} 个wav文件")
    return wav_index


def load_all_labels(label_dir: Path) -> dict:
    """
    读取所有标签文件，返回 {normalized_id: (original_id, text, label_file)} 字典。
    """
    print("正在读取标签文件...")
    labels = {}
    duplicates = 0
    for label_file in sorted(label_dir.glob("*.txt")):
        count = 0
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                uid, text = parse_label_line(line)
                if uid is None:
                    continue
                norm_id = normalize_label_id(uid)
                if norm_id in labels:
                    duplicates += 1
                    continue
                labels[norm_id] = (uid, text, label_file.name)
                count += 1
        print(f"  {label_file.name}: {count} 条")
    print(f"  标签总计: {len(labels)} 条 (去重后), 重复: {duplicates} 条")
    return labels


def match_labels_to_wav(labels: dict, wav_index: dict) -> list:
    """
    将标签与wav文件匹配。
    返回 [(norm_id, original_id, text, wav_path, label_file)] 列表。
    """
    print("正在匹配标签与音频...")
    matched = []
    unmatched_labels = []

    for norm_id, (original_id, text, label_file) in labels.items():
        if norm_id in wav_index:
            matched.append((norm_id, original_id, text, wav_index[norm_id], label_file))
        else:
            unmatched_labels.append((norm_id, original_id, label_file))

    print(f"  匹配成功: {len(matched)} 条")
    print(f"  未找到音频: {len(unmatched_labels)} 条")

    # 输出未匹配的标签样本
    if unmatched_labels:
        print(f"  未匹配标签样本 (前10条):")
        for norm_id, original_id, lf in unmatched_labels[:10]:
            print(f"    [{lf}] {original_id}")

    return matched


def organize_data(matched: list, output_dir: Path, copy_mode: bool = True):
    """
    将匹配到的数据整理到 labeled/ 目录，生成 FunASR 所需文件：
    - labeled/wav/  存放所有wav文件
    - labeled/train.jsonl  FunASR ChatML格式训练数据
    - labeled/wav.scp      Kaldi格式音频路径列表
    - labeled/text.txt     Kaldi格式文本标注
    """
    wav_dir = output_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "train.jsonl"
    scp_path = output_dir / "wav.scp"
    text_path = output_dir / "text.txt"

    print(f"正在整理数据到 {output_dir} ...")
    action = "复制" if copy_mode else "移动"

    processed = 0
    skipped = 0

    with open(jsonl_path, 'w', encoding='utf-8') as f_jsonl, \
         open(scp_path, 'w', encoding='utf-8') as f_scp, \
         open(text_path, 'w', encoding='utf-8') as f_text:

        for norm_id, original_id, text, src_wav_path, label_file in matched:
            dst_wav = wav_dir / f"{norm_id}.wav"

            # 复制或移动wav文件
            if not dst_wav.exists():
                try:
                    if copy_mode:
                        shutil.copy2(src_wav_path, dst_wav)
                    else:
                        shutil.move(str(src_wav_path), str(dst_wav))
                except Exception as e:
                    print(f"  警告: {action}失败 {src_wav_path}: {e}")
                    skipped += 1
                    continue

            # wav.scp: <utterance_id> <wav_path>
            wav_abs_path = dst_wav.resolve()
            f_scp.write(f"{norm_id} {wav_abs_path}\n")

            # text.txt: <utterance_id> <text>
            f_text.write(f"{norm_id} {text}\n")

            # FunASR ChatML jsonl
            record = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"语音转写：<|startofspeech|>!{wav_abs_path}<|endofspeech|>"},
                    {"role": "assistant", "content": text}
                ],
                "text_length": len(text)
            }
            f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

            processed += 1
            if processed % 10000 == 0:
                print(f"  已处理 {processed} 条...")

    print(f"  处理完成: {processed} 条, 跳过: {skipped} 条")
    print(f"  输出文件:")
    print(f"    {jsonl_path}")
    print(f"    {scp_path}")
    print(f"    {text_path}")
    print(f"    {wav_dir}/ ({processed} 个wav文件)")


def move_unlabeled(wav_index: dict, matched_ids: set, unlabeled_dir: Path, copy_mode: bool = False):
    """
    将没有标签的wav文件复制或移动到 unlabeled/ 目录。
    """
    unlabeled_dir.mkdir(parents=True, exist_ok=True)
    action = "复制" if copy_mode else "移动"
    print(f"\n正在{action}未标注音频到 {unlabeled_dir} ...")

    done = 0
    skipped = 0
    for stem, src_path in wav_index.items():
        if stem in matched_ids:
            continue
        dst_path = unlabeled_dir / f"{stem}.wav"
        if dst_path.exists():
            done += 1
            continue
        try:
            if copy_mode:
                shutil.copy2(src_path, dst_path)
            else:
                shutil.move(str(src_path), str(dst_path))
            done += 1
            if done % 10000 == 0:
                print(f"  已{action} {done} 个...")
        except Exception as e:
            print(f"  警告: {action}失败 {src_path}: {e}")
            skipped += 1

    print(f"  {action}完成: {done} 个未标注音频, 跳过: {skipped} 个")
    print(f"  输出目录: {unlabeled_dir}/")


def gen_wav_scp(wav_dir: Path):
    """为指定目录生成 wav.scp（Kaldi 格式: <uid> <绝对路径>）"""
    wav_dir = wav_dir.resolve()
    scp_path = wav_dir / "wav.scp"
    print(f"\n正在为 {wav_dir} 生成 wav.scp ...")

    entries = []
    for entry in os.scandir(wav_dir):
        if entry.is_file() and entry.name.lower().endswith('.wav'):
            uid = Path(entry.name).stem
            entries.append((uid, str(Path(entry.path).resolve())))

    entries.sort(key=lambda x: x[0])
    with open(scp_path, 'w', encoding='utf-8') as f:
        for uid, path in entries:
            f.write(f"{uid} {path}\n")

    print(f"  已写入: {scp_path} ({len(entries)} 条)")


def print_summary(matched: list):
    """按标签文件统计匹配情况"""
    stats = defaultdict(int)
    for _, _, _, _, label_file in matched:
        stats[label_file] += 1
    print("\n按标签文件统计匹配数:")
    for lf in sorted(stats.keys()):
        print(f"  {lf}: {stats[lf]} 条")
    print(f"  总计: {sum(stats.values())} 条")


def main():
    print("=" * 60)
    print("S10 ASR 数据整理 -> FunASR 训练格式")
    print("=" * 60)
    print(f"基础目录: {BASE_DIR}")
    print(f"标签目录: {LABEL_DIR}")
    print(f"音频目录: {WAV_SOURCE_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"未标注目录: {UNLABELED_DIR}")
    print(f"模式: {'复制' if COPY_MODE else '移动'}")
    print()

    # 1. 建立wav文件索引
    wav_index = build_wav_index(WAV_SOURCE_DIR)

    # 2. 读取所有标签
    labels = load_all_labels(LABEL_DIR)

    # 3. 匹配
    matched = match_labels_to_wav(labels, wav_index)

    if not matched:
        print("没有匹配到任何数据，请检查文件路径和命名格式。")
        return

    # 4. 统计
    print_summary(matched)

    # 5. 整理输出
    organize_data(matched, OUTPUT_DIR, COPY_MODE)

    # 6. 移动未标注音频到 unlabeled/
    matched_ids = {norm_id for norm_id, _, _, _, _ in matched}
    move_unlabeled(wav_index, matched_ids, UNLABELED_DIR, COPY_MODE)

    # 7. 为 unlabeled 目录生成 wav.scp
    gen_wav_scp(UNLABELED_DIR)

    print("\n完成！")


if __name__ == "__main__":
    main()
