#!/usr/bin/env python3
"""
快速版 scp2jsonl —— 替代 funasr 自带的 scp2jsonl

原版用 librosa.load() 完整加载音频来计算时长，在 NFS 上极慢（14万条需要 15~25 小时）。
本脚本用 soundfile.info() 只读文件头，不加载音频数据，速度提升 100~500 倍。

用法:
    python tools/fast_scp2jsonl.py \
        --wav_scp data/list/wav.scp \
        --text data/list/text \
        --output data/list/train.jsonl \
        --workers 8
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    import soundfile as sf
    USE_SOUNDFILE = True
except ImportError:
    USE_SOUNDFILE = False

import struct
import wave


def get_source_len_soundfile(wav_path: str, frame_shift_ms: int = 10) -> int:
    """用 soundfile.info() 只读文件头，毫秒级完成"""
    info = sf.info(wav_path)
    duration_ms = info.duration * 1000  # 秒 → 毫秒
    return int(duration_ms / frame_shift_ms)


def get_source_len_wavheader(wav_path: str, frame_shift_ms: int = 10) -> int:
    """纯标准库方案：用 wave 模块读 WAV 头"""
    with wave.open(wav_path, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
    duration_ms = frames / rate * 1000
    return int(duration_ms / frame_shift_ms)


def get_source_len(wav_path: str, frame_shift_ms: int = 10) -> int:
    """自动选择最快的可用方案"""
    if USE_SOUNDFILE:
        return get_source_len_soundfile(wav_path, frame_shift_ms)
    else:
        return get_source_len_wavheader(wav_path, frame_shift_ms)


def process_one(key: str, wav_path: str, text: str, frame_shift_ms: int = 10) -> dict:
    """处理单条样本"""
    if os.path.exists(wav_path):
        source_len = get_source_len(wav_path, frame_shift_ms)
    else:
        # 文件不存在时用占位值（与原版行为一致）
        source_len = 1

    target_len = len(text.split()) if " " in text else len(text)

    return {
        "key": key,
        "source": wav_path,
        "source_len": source_len,
        "target": text,
        "target_len": target_len,
    }


def main():
    parser = argparse.ArgumentParser(description="快速版 scp2jsonl（不加载音频数据）")
    parser.add_argument("--wav_scp", required=True, help="wav.scp 文件路径")
    parser.add_argument("--text", required=True, help="text 文件路径")
    parser.add_argument("--output", required=True, help="输出 JSONL 文件路径")
    parser.add_argument("--workers", type=int, default=8, help="并行线程数（默认 8）")
    parser.add_argument("--frame_shift", type=int, default=10, help="帧移毫秒数（默认 10）")
    parser.add_argument("--resume", action="store_true", help="断点续跑：跳过已处理的 key")
    args = parser.parse_args()

    # -------------------- 断点续跑：加载已完成的 key --------------------
    done_keys = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    done_keys.add(data["key"])
                except:
                    pass
        print(f"断点续跑: 已有 {len(done_keys)} 条，将跳过")

    # 读取 wav.scp
    wav_dict = {}
    with open(args.wav_scp, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                wav_dict[parts[0]] = parts[1]

    # 读取 text
    text_dict = {}
    with open(args.text, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                text_dict[parts[0]] = parts[1]
            elif len(parts) == 1:
                text_dict[parts[0]] = ""

    # 取交集
    common_keys = sorted(set(wav_dict.keys()) & set(text_dict.keys()))
    wav_only = set(wav_dict.keys()) - set(text_dict.keys())
    text_only = set(text_dict.keys()) - set(wav_dict.keys())

    if wav_only:
        print(f"警告: {len(wav_only)} 条在 wav.scp 中但不在 text 中，已跳过")
    if text_only:
        print(f"警告: {len(text_only)} 条在 text 中但不在 wav.scp 中，已跳过")

    print(f"共 {len(common_keys)} 条待处理，使用 {args.workers} 线程")
    backend = "soundfile" if USE_SOUNDFILE else "wave(标准库)"
    print(f"音频时长读取方式: {backend}")

    # -------------------- 断点续跑：过滤已完成的 key --------------------
    if done_keys:
        pending_keys = [k for k in common_keys if k not in done_keys]
        print(f"跳过已完成 {len(common_keys) - len(pending_keys)} 条，剩余 {len(pending_keys)} 条")
    else:
        pending_keys = common_keys

    if not pending_keys:
        print("所有条目已处理完毕，无需重新运行")
        return

    # -------------------- 边处理边写入 --------------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_mode = "a" if done_keys else "w"
    success_count = 0
    error_count = 0
    errors = []

    with open(args.output, write_mode, encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_one, key, wav_dict[key], text_dict[key], args.frame_shift
                ): key
                for key in pending_keys
            }

            with tqdm(total=len(pending_keys), desc="处理中", dynamic_ncols=True) as pbar:
                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        result = future.result()
                        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out_f.flush()
                        success_count += 1
                    except Exception as e:
                        errors.append((key, str(e)))
                        error_count += 1
                    pbar.update(1)

    total_done = len(done_keys) + success_count
    print(f"\n完成! 输出: {args.output}")
    if done_keys:
        print(f"本次新增: {success_count} 条, 累计: {total_done} 条, 本次失败: {error_count} 条")
    else:
        print(f"成功: {success_count} 条, 失败: {error_count} 条")

    if errors:
        print(f"\n失败详情（前 10 条）:")
        for key, err in errors[:10]:
            print(f"  {key}: {err}")


if __name__ == "__main__":
    main()
