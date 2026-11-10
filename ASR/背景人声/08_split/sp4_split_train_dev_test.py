"""
将 sp4_data/labeled 的 text.txt / wav.scp 随机切分为 train / dev / test。

切分策略：
  - 按条目完全随机划分：train 90% / dev 5% / test 5%
  - seed=42 保证可复现

输出：
  /data1/label/reflow_label/{train,dev,test}/sp4_reflow_wav.scp
  /data1/label/reflow_label/{train,dev,test}/sp4_reflow_text
"""

import os
import random
from pathlib import Path

# ── 配置 ──
BASE_DIR = Path(r"E:\8.音频\03 s10\sp4_data\labeled")
TEXT_FILE = BASE_DIR / "text.txt"
WAV_SCP = BASE_DIR / "wav.scp"
LABEL_ROOT = Path("/data1/label/reflow_label")
DATASET_NAME = "sp4_reflow"

SEED = 42
DEV_RATIO = 0.05
TEST_RATIO = 0.05


def load_data(text_file, wav_scp):
    """加载 text.txt 和 wav.scp"""
    key2text = {}
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                key2text[parts[0]] = parts[1]

    key2wav = {}
    with open(wav_scp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                key2wav[parts[0]] = parts[1]

    # 取交集
    common_keys = sorted(set(key2text) & set(key2wav))
    return key2text, key2wav, common_keys


def split_random(keys, seed, dev_ratio, test_ratio):
    """完全随机划分 train/dev/test"""
    keys = list(keys)
    random.seed(seed)
    random.shuffle(keys)

    n = len(keys)
    n_test = int(n * test_ratio)
    n_dev = int(n * dev_ratio)

    test_keys = keys[:n_test]
    dev_keys = keys[n_test:n_test + n_dev]
    train_keys = keys[n_test + n_dev:]

    return {"train": train_keys, "dev": dev_keys, "test": test_keys}


def write_split(label_root, dataset_name, split, keys, key2text, key2wav):
    """写出一个 split 的 wav.scp 和 text"""
    out_dir = label_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    scp_path = out_dir / f"{dataset_name}_wav.scp"
    txt_path = out_dir / f"{dataset_name}_text"

    keys_sorted = sorted(keys)

    with open(scp_path, "w", encoding="utf-8") as f:
        for key in keys_sorted:
            f.write(f"{key} {key2wav[key]}\n")

    with open(txt_path, "w", encoding="utf-8") as f:
        for key in keys_sorted:
            f.write(f"{key} {key2text[key]}\n")

    print(f"  [{split}] {len(keys_sorted)} 条 → {scp_path}")
    print(f"  [{split}] {len(keys_sorted)} 条 → {txt_path}")

    return len(keys_sorted)


def main():
    print("加载数据...")
    key2text, key2wav, common_keys = load_data(TEXT_FILE, WAV_SCP)
    print(f"共 {len(key2text)} 条文本, {len(key2wav)} 条音频, 交集 {len(common_keys)} 条")

    print(f"\n随机划分 (seed={SEED}, dev={DEV_RATIO}, test={TEST_RATIO})...")
    split_map = split_random(common_keys, SEED, DEV_RATIO, TEST_RATIO)

    for split, keys in split_map.items():
        print(f"  {split}: {len(keys)} 条")

    print(f"\n写出到 {LABEL_ROOT}...")
    total = 0
    for split, keys in split_map.items():
        total += write_split(LABEL_ROOT, DATASET_NAME, split, keys, key2text, key2wav)

    print(f"\n完成! 总计 {total} 条")


if __name__ == "__main__":
    main()
