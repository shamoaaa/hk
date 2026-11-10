#!/usr/bin/env python3
"""
SP4 ASR 数据整理脚本
功能：
1. 读取已有的 text / wav.scp 文件（含增量 add 文件），合并为完整标签集
2. 将 Linux 服务器路径重映射为本地 Windows 路径，校验文件存在性
3. 将匹配到的 wav 复制/移动到 labeled/wav/ 目录
4. 生成 FunASR 训练所需的 jsonl、wav.scp、text.txt
5. 将未标注音频复制/移动到 unlabeled/ 目录，并生成 wav.scp

与 01_prepare_funasr_data.py 的区别：
- sp4 数据集已有 text / wav.scp，无需从 label/*.txt 扫描
- wav.scp 中路径为 Linux 服务器路径，需重映射到本地 trimed_wav/ 下
- 增量数据 UID 使用 _ 分隔符（主数据集用 -），无需特殊处理
"""

import os
import json
import shutil
from pathlib import Path
from collections import defaultdict, Counter

# ============ 配置 ============
BASE_DIR = Path(r'E:\8.音频\03 s10\sp4_data')
OUTPUT_DIR = BASE_DIR / "labeled"
UNLABELED_DIR = BASE_DIR / "unlabeled"

# 主数据文件
TEXT_FILES = [
    BASE_DIR / "text",
    BASE_DIR / "text_0924_add",
    BASE_DIR / "text_20241204_add",
]
WAV_SCP_FILES = [
    BASE_DIR / "wav.scp",
    BASE_DIR / "wav_0924_add.scp",
    BASE_DIR / "wav_20241204_add.scp",
]
# 全量音频列表（含未标注），用于识别 unlabeled
WAV_ALL_SCP = BASE_DIR / "wav_all.scp"

# Linux -> 本地路径映射
# wav.scp 中路径形如: /home/wangsheng30/train/sp4_data/trimed_wav/...
LINUX_PREFIX = "sp4_data/"

# 是否复制文件（True=复制, False=移动）
COPY_MODE = False
# ==============================


def remap_wav_path(linux_path: str, base_dir: Path) -> Path:
    """将 Linux 服务器路径重映射为本地路径"""
    idx = linux_path.find(LINUX_PREFIX)
    if idx >= 0:
        rel = linux_path[idx + len(LINUX_PREFIX):]
    else:
        rel = linux_path.split("/")[-1]
    return base_dir / rel


def load_text_files(text_files: list) -> dict:
    """
    读取所有 text 文件，返回 {uid: text} 字典。
    格式: <uid> <text>（第一个空格分隔）
    """
    print("正在读取标签文件...")
    labels = {}
    duplicates = 0

    for text_file in text_files:
        if not text_file.exists():
            print(f"  跳过（不存在）: {text_file}")
            continue
        count = 0
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                uid, text = parts[0], parts[1]
                if uid in labels:
                    duplicates += 1
                    continue
                labels[uid] = text
                count += 1
        print(f"  {text_file.name}: {count} 条")

    print(f"  标签总计: {len(labels)} 条 (去重后), 重复: {duplicates} 条")
    return labels


def load_wav_scp_files(wav_scp_files: list, base_dir: Path) -> dict:
    """
    读取所有 wav.scp 文件，重映射路径，返回 {uid: local_path} 字典。
    仅保留本地文件存在的条目。
    """
    print("正在读取 wav.scp 文件...")
    wav_map = {}
    missing = 0
    duplicates = 0

    for scp_file in wav_scp_files:
        if not scp_file.exists():
            print(f"  跳过（不存在）: {scp_file}")
            continue
        count = 0
        file_missing = 0
        with open(scp_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                uid, linux_path = parts[0], parts[1]
                if uid in wav_map:
                    duplicates += 1
                    continue
                local_path = remap_wav_path(linux_path, base_dir)
                if not local_path.exists():
                    file_missing += 1
                    missing += 1
                    continue
                wav_map[uid] = local_path
                count += 1
        status = f", 缺失: {file_missing}" if file_missing else ""
        print(f"  {scp_file.name}: {count} 条{status}")

    print(f"  wav 总计: {len(wav_map)} 条, 重复: {duplicates}, 缺失: {missing}")
    return wav_map


def match_labels_and_wav(labels: dict, wav_map: dict) -> list:
    """
    匹配标签与音频。
    返回 [(uid, text, src_wav_path)] 列表。
    """
    print("正在匹配标签与音频...")
    matched = []
    label_no_wav = 0
    wav_no_label = 0

    for uid, text in labels.items():
        if uid in wav_map:
            matched.append((uid, text, wav_map[uid]))
        else:
            label_no_wav += 1

    for uid in wav_map:
        if uid not in labels:
            wav_no_label += 1

    print(f"  匹配成功: {len(matched)} 条")
    print(f"  有标签无音频: {label_no_wav} 条")
    print(f"  有音频无标签: {wav_no_label} 条")

    if label_no_wav > 0:
        print(f"  无音频标签样本 (前5条):")
        cnt = 0
        for uid in labels:
            if uid not in wav_map:
                print(f"    {uid} -> {labels[uid]}")
                cnt += 1
                if cnt >= 5:
                    break

    return matched


def organize_data(matched: list, output_dir: Path, copy_mode: bool = True):
    """
    将匹配到的数据整理到 labeled/ 目录，生成 FunASR 所需文件：
    - labeled/wav/         存放所有wav文件（从 trimed_wav 复制/移动过来）
    - labeled/train.jsonl  FunASR ChatML格式训练数据
    - labeled/wav.scp      Kaldi格式音频路径列表
    - labeled/text.txt     Kaldi格式文本标注
    """
    wav_dir = output_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "train.jsonl"
    scp_path = output_dir / "wav.scp"
    text_path = output_dir / "text.txt"

    action = "复制" if copy_mode else "移动"
    print(f"正在{action}音频并整理数据到 {output_dir} ...")

    processed = 0
    skipped = 0

    with open(jsonl_path, 'w', encoding='utf-8') as f_jsonl, \
         open(scp_path, 'w', encoding='utf-8') as f_scp, \
         open(text_path, 'w', encoding='utf-8') as f_text:

        for uid, text, src_wav_path in matched:
            dst_wav = wav_dir / f"{uid}.wav"

            # 复制或移动 wav 文件到 labeled/wav/
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
            f_scp.write(f"{uid} {wav_abs_path}\n")

            # text.txt: <utterance_id> <text>
            f_text.write(f"{uid} {text}\n")

            # FunASR ChatML jsonl
            record = {
                "messages": [
                    {"role": "system",
                     "content": "You are a helpful assistant."},
                    {"role": "user",
                     "content": f"语音转写：<|startofspeech|>!{wav_abs_path}"
                                f"<|endofspeech|>"},
                    {"role": "assistant",
                     "content": text}
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


def move_unlabeled(wav_map: dict, labeled_uids: set, wav_all_scp: Path,
                   base_dir: Path, unlabeled_dir: Path,
                   copy_mode: bool = True):
    """
    将没有标签的wav文件复制或移动到 unlabeled/ 目录，并生成 wav.scp。
    未标注来源：
    1. wav_map 中有音频但无标签的
    2. wav_all.scp 中不在 labeled 也不在 wav_map 中的
    """
    unlabeled_dir.mkdir(parents=True, exist_ok=True)
    action = "复制" if copy_mode else "移动"
    print(f"\n正在{action}未标注音频到 {unlabeled_dir} ...")

    # 收集所有未标注的 (uid, src_path)
    unlabeled = {}

    # 来源1: wav_map 中有音频无标签
    for uid, src_path in wav_map.items():
        if uid not in labeled_uids:
            unlabeled[uid] = src_path

    # 来源2: wav_all.scp 中额外的
    if wav_all_scp.exists():
        with open(wav_all_scp, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                uid = parts[0]
                if uid not in labeled_uids and uid not in unlabeled:
                    local_path = remap_wav_path(parts[1], base_dir)
                    if local_path.exists():
                        unlabeled[uid] = local_path

    print(f"  未标注音频总数: {len(unlabeled)}")

    # 复制/移动文件并生成 wav.scp
    scp_path = unlabeled_dir / "wav.scp"
    done = 0
    skipped = 0

    with open(scp_path, 'w', encoding='utf-8') as f_scp:
        for uid in sorted(unlabeled.keys()):
            src_path = unlabeled[uid]
            dst_path = unlabeled_dir / f"{uid}.wav"

            if not dst_path.exists():
                try:
                    if copy_mode:
                        shutil.copy2(src_path, dst_path)
                    else:
                        shutil.move(str(src_path), str(dst_path))
                except Exception as e:
                    print(f"  警告: {action}失败 {src_path}: {e}")
                    skipped += 1
                    continue

            f_scp.write(f"{uid} {dst_path.resolve()}\n")
            done += 1
            if done % 10000 == 0:
                print(f"  已{action} {done} 个...")

    print(f"  {action}完成: {done} 个未标注音频, 跳过: {skipped} 个")
    print(f"  {scp_path}")


def print_summary(matched: list):
    """标签统计摘要"""
    texts = [text for _, text, _ in matched]
    lengths = [len(t) for t in texts]
    tc = Counter(texts)

    print(f"\n=== 标签统计 ===")
    print(f"  总条数: {len(texts)}")
    print(f"  唯一标签: {len(tc)}")
    print(f"  平均长度: {sum(lengths)/len(lengths):.1f} 字")
    print(f"  长度范围: {min(lengths)} ~ {max(lengths)} 字")
    print(f"  Top 10 高频标签:")
    for txt, cnt in tc.most_common(10):
        print(f"    [{cnt:>6}] {txt}")


def main():
    print("=" * 60)
    print("SP4 ASR 数据整理 -> FunASR 训练格式")
    print("=" * 60)
    print(f"基础目录: {BASE_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"未标注目录: {UNLABELED_DIR}")
    print(f"模式: {'复制' if COPY_MODE else '移动'}")
    print()

    # 1. 读取标签
    labels = load_text_files(TEXT_FILES)

    # 2. 读取 wav.scp 并重映射路径
    wav_map = load_wav_scp_files(WAV_SCP_FILES, BASE_DIR)

    # 3. 匹配
    matched = match_labels_and_wav(labels, wav_map)

    if not matched:
        print("没有匹配到任何数据，请检查文件路径。")
        return

    # 4. 统计
    print_summary(matched)

    # 5. 整理输出：复制/移动 wav + 生成 jsonl/wav.scp/text.txt
    organize_data(matched, OUTPUT_DIR, COPY_MODE)

    # 6. 处理未标注音频
    labeled_uids = {uid for uid, _, _ in matched}
    move_unlabeled(wav_map, labeled_uids, WAV_ALL_SCP, BASE_DIR,
                   UNLABELED_DIR, COPY_MODE)

    print(f"\n{'=' * 60}")
    print(f"完成！")
    print(f"  标注数据: {OUTPUT_DIR}/ ({len(matched)} 条)")
    print(f"  未标注数据: {UNLABELED_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
