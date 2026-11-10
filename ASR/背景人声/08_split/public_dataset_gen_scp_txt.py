#!/usr/bin/env python3
"""
公开 ASR 数据集 → wav.scp + text 转换工具

目标格式：
- wav.scp: <utt_id> <wav_absolute_path>
- text:    <utt_id> <transcript>

输出目录：
/data1/code/label/{train,dev,test}/<dataset_name>_wav.scp
/data1/code/label/{train,dev,test}/<dataset_name>_text
"""

import os
import re
import json
import random
from pathlib import Path
from collections import defaultdict


# ===== 全局输出目录 =====
LABEL_ROOT = Path("/data1/code/label")


def write_scp_text(label_root: Path, dataset_name: str, split: str,
                   scp_lines: list, txt_lines: list):
    """写出 wav.scp 和 text 到 label_root/<split>/<dataset_name>_wav.scp / _text"""
    out_dir = label_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    scp_path = out_dir / f"{dataset_name}_wav.scp"
    txt_path = out_dir / f"{dataset_name}_text"

    with open(scp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(scp_lines) + "\n")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines) + "\n")

    print(f"  [{split}] scp: {len(scp_lines)} 条 → {scp_path}")
    print(f"  [{split}] text: {len(txt_lines)} 条 → {txt_path}")


# =============================================================================
# 1. aidatatang_200zh
# =============================================================================
def gen_aidatatang_200zh():
    """
    aidatatang_200zh/
    ├── corpus/{train,dev,test}/<speaker_id>/<utt_id>.wav
    └── transcript/aidatatang_200_zh_transcript.txt   # <utt_id> <transcript>
    """
    print("\n===== aidatatang_200zh =====")
    root = Path("/path/to/aidatatang_200zh")  # ← 修改为实际路径
    dataset_name = "aidatatang_200zh"
    splits = ["train", "dev", "test"]

    # 读取全部转写
    transcript_file = root / "transcript" / "aidatatang_200_zh_transcript.txt"
    utt2text = {}
    with open(transcript_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                utt2text[parts[0]] = parts[1]

    print(f"共读取 {len(utt2text)} 条转写")

    for split in splits:
        corpus_dir = root / "corpus" / split
        if not corpus_dir.exists():
            print(f"  [跳过] {corpus_dir} 不存在")
            continue

        scp_lines, txt_lines, missing = [], [], 0
        for wav_path in sorted(corpus_dir.rglob("*.wav")):
            utt_id = wav_path.stem
            if utt_id in utt2text:
                scp_lines.append(f"{utt_id} {wav_path.resolve()}")
                txt_lines.append(f"{utt_id} {utt2text[utt_id]}")
            else:
                missing += 1

        write_scp_text(LABEL_ROOT, dataset_name, split, scp_lines, txt_lines)
        if missing:
            print(f"  [警告] {split} 有 {missing} 条 wav 缺少转写")


# =============================================================================
# 2. data_aishell (AISHELL-1)
# =============================================================================
def gen_data_aishell():
    """
    data_aishell/
    ├── wav/{train,dev,test}/<speaker_id>/<utt_id>.wav
    └── transcript/aishell_transcript_v0.8.txt
    注意：transcript 是分词格式（空格隔开），需要去掉空格拼成连续文本。
    """
    print("\n===== data_aishell (AISHELL-1) =====")
    root = Path("/path/to/data_aishell")  # ← 修改为实际路径
    dataset_name = "data_aishell"
    splits = ["train", "dev", "test"]

    # 读取转写（分词格式 → 去空格拼接）
    transcript_file = root / "transcript" / "aishell_transcript_v0.8.txt"
    utt2text = {}
    with open(transcript_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                utt_id = parts[0]
                text = "".join(parts[1:])
                utt2text[utt_id] = text

    print(f"共读取 {len(utt2text)} 条转写")

    for split in splits:
        wav_dir = root / "wav" / split
        if not wav_dir.exists():
            print(f"  [跳过] {wav_dir} 不存在")
            continue

        scp_lines, txt_lines, missing = [], [], 0
        for wav_path in sorted(wav_dir.rglob("*.wav")):
            utt_id = wav_path.stem
            if utt_id in utt2text:
                scp_lines.append(f"{utt_id} {wav_path.resolve()}")
                txt_lines.append(f"{utt_id} {utt2text[utt_id]}")
            else:
                missing += 1

        write_scp_text(LABEL_ROOT, dataset_name, split, scp_lines, txt_lines)
        if missing:
            print(f"  [警告] {split} 有 {missing} 条 wav 缺少转写")


# =============================================================================
# 3. data_aishell_1 iOS
# =============================================================================
def gen_data_aishell_1_ios():
    """
    iOS/
    ├── data/  → train    (wav.scp 相对路径 + trans.txt)
    ├── dev/
    └── test/
    """
    print("\n===== data_aishell_1 iOS =====")
    root = Path("/dataset/qiannengfeng/wangsheng30/speech_data/data_aishell_1/iOS")
    dataset_name = "data_aishell_1_ios"

    split_map = {
        "data": "train",
        "dev": "dev",
        "test": "test",
    }

    for src_dir, target_split in split_map.items():
        split_dir = root / src_dir
        scp_file = split_dir / "wav.scp"
        trans_file = split_dir / "trans.txt"

        if not scp_file.exists():
            print(f"  [跳过] {scp_file} 不存在")
            continue

        # 读取 trans.txt
        utt2text = {}
        with open(trans_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    utt2text[parts[0]] = parts[1]

        # 读取 wav.scp，将相对路径转为绝对路径
        scp_lines, txt_lines, missing = [], [], 0
        with open(scp_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    utt_id, rel_path = parts
                    abs_path = (split_dir / rel_path).resolve()
                    scp_lines.append(f"{utt_id} {abs_path}")
                    if utt_id in utt2text:
                        txt_lines.append(f"{utt_id} {utt2text[utt_id]}")
                    else:
                        missing += 1

        write_scp_text(LABEL_ROOT, dataset_name, target_split, scp_lines, txt_lines)
        if missing:
            print(f"  [警告] {target_split} 有 {missing} 条 wav 缺少转写")


# =============================================================================
# 4. MagicData
# =============================================================================
def gen_magicdata():
    """
    magicdata/
    ├── {train,dev,test}/
    │   ├── <speaker_id>/<utt_id>.wav
    │   └── TRANS.txt   # utt_id\tspeaker_id\ttranscript (tab分隔三列)
    """
    print("\n===== MagicData =====")
    root = Path("/dataset/qiannengfeng/wangsheng30/speech_data/magic")
    dataset_name = "magicdata"
    splits = ["train", "dev", "test"]

    for split in splits:
        split_dir = root / split
        trans_file = split_dir / "TRANS.txt"

        if not trans_file.exists():
            print(f"  [跳过] {trans_file} 不存在")
            continue

        scp_lines, txt_lines, missing = [], [], 0
        with open(trans_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    utt_id_raw = parts[0].strip()
                    speaker_id = parts[1].strip()
                    text = parts[2].strip()
                    # 跳过表头行
                    if utt_id_raw.lower() in ("utteranceid", "utterance_id", "id"):
                        continue
                    # utt_id 可能带 .wav 后缀，去掉
                    utt_id = utt_id_raw.replace(".wav", "")
                    # wav 路径: split_dir/<speaker_id>/<utt_id_raw>
                    # 如果原始 utt_id 已带 .wav，直接用；否则加 .wav
                    wav_filename = utt_id_raw if utt_id_raw.endswith(".wav") else f"{utt_id_raw}.wav"
                    wav_path = split_dir / speaker_id / wav_filename
                    if wav_path.exists():
                        scp_lines.append(f"{utt_id} {wav_path.resolve()}")
                        txt_lines.append(f"{utt_id} {text}")
                    else:
                        missing += 1

        write_scp_text(LABEL_ROOT, dataset_name, split, scp_lines, txt_lines)
        if missing:
            print(f"  [警告] {split} 有 {missing} 条 wav 文件不存在")


# =============================================================================
# 5. PrimeWords (primewords_md_2018_set1)
# =============================================================================
def gen_primewords():
    """
    primewords_md_2018_set1/
    ├── audio_files/<hex1>/<hex2>/<uuid>.wav
    └── set1_transcript.json

    无预设划分，按说话人 (user_id) 划分 train 90% / dev 5% / test 5%
    text 是分词格式，需去掉空格
    """
    print("\n===== PrimeWords =====")
    root = Path("/dataset/qiannengfeng/wangsheng30/speech_data/primewords/primewords_md_2018_set1")
    dataset_name = "primewords"
    seed = 42
    dev_ratio = 0.05
    test_ratio = 0.05

    # 读取 JSON
    with open(root / "set1_transcript.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"共 {len(data)} 条, {len(set(d['user_id'] for d in data))} 位说话人")

    # 按说话人分组
    spk2items = defaultdict(list)
    for item in data:
        spk2items[item["user_id"]].append(item)

    # 按说话人随机划分
    speakers = sorted(spk2items.keys())
    random.seed(seed)
    random.shuffle(speakers)

    n_spk = len(speakers)
    n_dev = max(1, int(n_spk * dev_ratio))
    n_test = max(1, int(n_spk * test_ratio))

    test_spks = set(speakers[:n_test])
    dev_spks = set(speakers[n_test:n_test + n_dev])
    train_spks = set(speakers[n_test + n_dev:])

    print(f"说话人划分: train={len(train_spks)}, dev={len(dev_spks)}, test={len(test_spks)}")

    # 生成 scp 和 text
    split_map = {"train": train_spks, "dev": dev_spks, "test": test_spks}
    audio_dir = root / "audio_files"

    for split, spk_set in split_map.items():
        scp_lines, txt_lines, missing = [], [], 0

        for spk in sorted(spk_set):
            for item in spk2items[spk]:
                filename = item["file"]
                utt_id = filename.replace(".wav", "")
                wav_path = audio_dir / filename[0] / filename[0:2] / filename
                text = item["text"].replace(" ", "")

                if wav_path.exists():
                    scp_lines.append(f"{utt_id} {wav_path.resolve()}")
                    txt_lines.append(f"{utt_id} {text}")
                else:
                    missing += 1

        write_scp_text(LABEL_ROOT, dataset_name, split, scp_lines, txt_lines)
        if missing:
            print(f"  [警告] {split} 有 {missing} 条 wav 文件不存在")

    # 统计各 split 时长
    for split, spk_set in split_map.items():
        total_len = sum(float(item["length"]) for spk in spk_set for item in spk2items[spk])
        n_utts = sum(len(spk2items[spk]) for spk in spk_set)
        print(f"  [{split}] {n_utts} 条, {total_len/3600:.1f} 小时")


# =============================================================================
# 6. ST-CMDS
# =============================================================================
def gen_stcmds():
    """
    ST-CMDS-20170001_1-OS/
    ├── 20170001P00001A0001.wav
    ├── 20170001P00001A0001.txt
    └── (102,600 对 .wav + .txt，平铺在同一目录)

    文件名格式：20170001P<speaker_id>A<utt_seq>.wav
    无预设划分，按说话人划分 train 90% / dev 5% / test 5%
    """
    print("\n===== ST-CMDS =====")
    root = Path("/dataset/qiannengfeng/wangsheng30/speech_data/stcmds/ST-CMDS-20170001_1-OS")
    dataset_name = "stcmds"
    seed = 42
    dev_ratio = 0.05
    test_ratio = 0.05

    # 扫描所有 wav，提取说话人 ID
    spk_pattern = re.compile(r"(P\d+)")
    spk2files = defaultdict(list)

    for wav_path in sorted(root.glob("*.wav")):
        m = spk_pattern.search(wav_path.stem)
        if m:
            spk_id = m.group(1)
            spk2files[spk_id].append(wav_path)

    print(f"共 {sum(len(v) for v in spk2files.values())} 条, {len(spk2files)} 位说话人")

    # 按说话人随机划分
    speakers = sorted(spk2files.keys())
    random.seed(seed)
    random.shuffle(speakers)

    n_spk = len(speakers)
    n_dev = max(1, int(n_spk * dev_ratio))
    n_test = max(1, int(n_spk * test_ratio))

    test_spks = set(speakers[:n_test])
    dev_spks = set(speakers[n_test:n_test + n_dev])
    train_spks = set(speakers[n_test + n_dev:])

    print(f"说话人划分: train={len(train_spks)}, dev={len(dev_spks)}, test={len(test_spks)}")

    # 生成 scp 和 text
    split_map = {"train": train_spks, "dev": dev_spks, "test": test_spks}

    for split, spk_set in split_map.items():
        scp_lines, txt_lines, missing = [], [], 0

        for spk in sorted(spk_set):
            for wav_path in spk2files[spk]:
                utt_id = wav_path.stem
                txt_file = wav_path.with_suffix(".txt")
                if txt_file.exists():
                    text = txt_file.read_text(encoding="utf-8").strip()
                    scp_lines.append(f"{utt_id} {wav_path.resolve()}")
                    txt_lines.append(f"{utt_id} {text}")
                else:
                    missing += 1

        write_scp_text(LABEL_ROOT, dataset_name, split, scp_lines, txt_lines)
        if missing:
            print(f"  [警告] {split} 有 {missing} 条缺少 txt 文件")


# =============================================================================
# 7. TAL_ASR (aisolution_data)
# =============================================================================
def gen_tal_asr():
    """
    aisolution_data/
    ├── transcript/transcript.txt       # <utt_id> <transcript>  (utt_id 为纯数字)
    ├── wav/{train,dev,test}/<speaker_id>/<utt_id>.wav
    └── meta/
    """
    print("\n===== TAL_ASR =====")
    root = Path("/dataset/qiannengfeng/wangsheng30/speech_data/TAL_ASR/aisolution_data")
    dataset_name = "tal_asr"
    splits = ["train", "dev", "test"]

    # 读取全部转写
    transcript_file = root / "transcript" / "transcript.txt"
    utt2text = {}
    with open(transcript_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                utt2text[parts[0]] = parts[1]

    print(f"共读取 {len(utt2text)} 条转写")

    for split in splits:
        wav_dir = root / "wav" / split
        if not wav_dir.exists():
            print(f"  [跳过] {wav_dir} 不存在")
            continue

        scp_lines, txt_lines, missing = [], [], 0
        for wav_path in sorted(wav_dir.rglob("*.wav")):
            utt_id = wav_path.stem
            if utt_id in utt2text:
                scp_lines.append(f"{utt_id} {wav_path.resolve()}")
                txt_lines.append(f"{utt_id} {utt2text[utt_id]}")
            else:
                missing += 1

        write_scp_text(LABEL_ROOT, dataset_name, split, scp_lines, txt_lines)
        if missing:
            print(f"  [警告] {split} 有 {missing} 条 wav 缺少转写")


# =============================================================================
# 8. THCHS-30
# =============================================================================
def gen_thchs30():
    """
    data_thchs30/
    ├── train/         # 训练集 wav + .wav.trn
    ├── dev/           # 开发集
    ├── test/          # 测试集
    └── data/          # 全部数据

    .wav.trn 文件三行：第1行汉字（分词），第2行拼音，第3行音素
    只取第1行，去掉分词空格
    """
    print("\n===== THCHS-30 =====")
    root = Path("/dataset/qiannengfeng/wangsheng30/speech_data/thchs/data_thchs30")
    dataset_name = "thchs30"
    splits = ["train", "dev", "test"]

    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            print(f"  [跳过] {split_dir} 不存在")
            continue

        scp_lines, txt_lines, missing = [], [], 0

        for wav_path in sorted(split_dir.glob("*.wav")):
            utt_id = wav_path.stem
            trn_file = split_dir / f"{wav_path.name}.trn"

            if trn_file.exists():
                trn_content = trn_file.read_text(encoding="utf-8").strip()
                if not trn_content:
                    missing += 1
                    continue

                first_line = trn_content.splitlines()[0].strip()

                # trn 文件可能是路径引用（如 ../data/A13_230.wav.trn），需要跟踪到实际文件
                if first_line.endswith(".trn") or first_line.startswith("../"):
                    actual_trn = (split_dir / first_line).resolve()
                    if actual_trn.exists():
                        lines = actual_trn.read_text(encoding="utf-8").strip().splitlines()
                    else:
                        missing += 1
                        continue
                else:
                    lines = trn_content.splitlines()

                if lines:
                    # 第1行是汉字（分词格式），去掉空格
                    text = lines[0].strip().replace(" ", "")
                    scp_lines.append(f"{utt_id} {wav_path.resolve()}")
                    txt_lines.append(f"{utt_id} {text}")
                else:
                    missing += 1
            else:
                missing += 1

        write_scp_text(LABEL_ROOT, dataset_name, split, scp_lines, txt_lines)
        if missing:
            print(f"  [警告] {split} 有 {missing} 条缺少 trn 文件")


# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    gen_aidatatang_200zh()
    gen_data_aishell()
    gen_data_aishell_1_ios()
    gen_magicdata()
    gen_primewords()
    gen_stcmds()
    gen_tal_asr()
    gen_thchs30()

    print("\n===== 全部完成 =====")
