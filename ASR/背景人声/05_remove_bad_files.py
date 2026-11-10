#!/usr/bin/env python3
"""
从 labeled/ 和 unlabeled/ 中删除 bad_files.txt 里的问题音频。
同时清理 labeled/ 下对应的 text.txt、wav.scp、train.jsonl 中的条目。

用法:
    python remove_bad_files.py --dry-run   # 预览，不实际删除
    python remove_bad_files.py             # 执行删除
"""

import os
import json
import argparse
from pathlib import Path

BASE_DIR = Path(r'E:\8.音频\03 s10\s10_data_reflow')
BAD_FILE = BASE_DIR / "quality_check" / "report" / "bad_files.txt"
LABELED_DIR = BASE_DIR / "labeled"
UNLABELED_DIR = BASE_DIR / "unlabeled"


def load_bad_uids(bad_file: Path) -> set:
    uids = set()
    with open(bad_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            wav_path = line.split('\t')[0]
            uid = Path(wav_path).stem
            uids.add(uid)
    return uids


def remove_wav_files(bad_uids: set, dry_run: bool) -> dict:
    """删除 labeled/wav/ 和 unlabeled/ 中的wav文件，返回统计"""
    stats = {"labeled_deleted": 0, "unlabeled_deleted": 0, "not_found": 0}

    for uid in bad_uids:
        found = False

        labeled_wav = LABELED_DIR / "wav" / f"{uid}.wav"
        if labeled_wav.exists():
            found = True
            if dry_run:
                print(f"  [预览] 删除 {labeled_wav}")
            else:
                labeled_wav.unlink()
            stats["labeled_deleted"] += 1

        unlabeled_wav = UNLABELED_DIR / f"{uid}.wav"
        if unlabeled_wav.exists():
            found = True
            if dry_run:
                print(f"  [预览] 删除 {unlabeled_wav}")
            else:
                unlabeled_wav.unlink()
            stats["unlabeled_deleted"] += 1

        if not found:
            stats["not_found"] += 1

    return stats


def clean_text_file(filepath: Path, bad_uids: set, dry_run: bool) -> int:
    """清理 text.txt 或 wav.scp 中的条目，返回删除行数"""
    if not filepath.exists():
        return 0
    kept = []
    removed = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            uid = line.split(None, 1)[0] if line.strip() else ""
            if uid in bad_uids:
                removed += 1
            else:
                kept.append(line)
    if not dry_run and removed > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(kept)
    return removed


def clean_jsonl(filepath: Path, bad_uids: set, dry_run: bool) -> int:
    """清理 train.jsonl 中的条目，返回删除行数"""
    if not filepath.exists():
        return 0
    kept = []
    removed = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line_s = line.strip()
            if not line_s:
                continue
            try:
                record = json.loads(line_s)
                # 从 user content 中提取文件名作为uid
                user_content = record["messages"][1]["content"]
                # 格式: ...!<path>.wav<|endofspeech|>
                wav_part = user_content.split("!")[-1].split("<|endofspeech|>")[0]
                uid = Path(wav_part).stem
                if uid in bad_uids:
                    removed += 1
                    continue
            except (json.JSONDecodeError, KeyError, IndexError):
                pass
            kept.append(line)
    if not dry_run and removed > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(kept)
    return removed


def main():
    parser = argparse.ArgumentParser(description="删除 bad_files 中的问题音频")
    parser.add_argument('--dry-run', action='store_true', help='预览模式，不实际删除')
    args = parser.parse_args()

    if not BAD_FILE.exists():
        print(f"未找到 {BAD_FILE}，请先运行 audio_quality_check.py")
        return

    bad_uids = load_bad_uids(BAD_FILE)
    print(f"读取到 {len(bad_uids)} 个问题文件ID")
    if args.dry_run:
        print("== 预览模式，不会实际删除 ==\n")

    # 1. 删除wav文件
    print("删除wav文件:")
    stats = remove_wav_files(bad_uids, args.dry_run)
    print(f"  labeled:   {stats['labeled_deleted']} 个")
    print(f"  unlabeled: {stats['unlabeled_deleted']} 个")
    print(f"  未找到:    {stats['not_found']} 个")

    # 2. 清理labeled下的索引文件
    print("\n清理索引文件:")
    for name in ["text.txt", "wav.scp"]:
        n = clean_text_file(LABELED_DIR / name, bad_uids, args.dry_run)
        print(f"  {name}: 移除 {n} 行")

    n = clean_jsonl(LABELED_DIR / "train.jsonl", bad_uids, args.dry_run)
    print(f"  train.jsonl: 移除 {n} 行")

    action = "将删除" if args.dry_run else "已删除"
    print(f"\n完成！{action}以上文件。")
    if args.dry_run:
        print("确认无误后去掉 --dry-run 执行实际删除。")


if __name__ == "__main__":
    main()
