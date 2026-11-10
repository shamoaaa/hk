#!/usr/bin/env python3
"""
【本地端】根据 trim_log.jsonl 裁剪音频中的唤醒词

读取服务器生成的 trim_log.jsonl（包含唤醒词时间戳边界），
在本地对音频进行精确裁剪，输出裁剪后的音频和新的 wav.scp。

工作流:
  1. 服务器运行 trim_wakeword.py → 生成 trim_log.jsonl
  2. 将 trim_log.jsonl 拷贝到本地 12_test/ 目录
  3. 本地运行本脚本 → 裁剪音频 + 生成新 wav.scp + text

用法:
    # 默认配置（读取 12_test/trim_log.jsonl，裁剪到 12_test/trimmed/）
    python trim_wakeword_local.py

    # 指定 trim_log 路径
    python trim_wakeword_local.py --trim-log path/to/trim_log.jsonl

    # 只裁剪开头（不处理结尾）
    python trim_wakeword_local.py --trim-tail false

    # 调整安全边距
    python trim_wakeword_local.py --margin 50

    # 采样验证
    python trim_wakeword_local.py --sample 20
"""

import os
import sys
import json
import shutil
import argparse
import soundfile as sf
from pathlib import Path


# ============ 默认配置 ============
DEFAULT_TRIM_LOG = r"E:\8.音频\03 s10\12_test\trim_log.jsonl"
DEFAULT_INPUT_TEXT = r"E:\8.音频\03 s10\12_test\text"
DEFAULT_OUTPUT_DIR = r"E:\8.音频\03 s10\12_test\trimmed"
DEFAULT_MARGIN_MS = 100  # 裁剪边距（毫秒），避免切到正文
# ==================================


def trim_audio(wav_path: str, output_path: str,
               head_boundary=None, tail_boundary=None,
               trim_head: bool = True, trim_tail: bool = True,
               margin_ms: int = 100):
    """
    根据唤醒词边界裁剪音频。

    Args:
        wav_path: 输入音频路径
        output_path: 输出音频路径
        head_boundary: [start_ms, end_ms] 开头唤醒词时间范围
        tail_boundary: [start_ms, end_ms] 结尾唤醒词时间范围
        trim_head: 是否裁剪开头
        trim_tail: 是否裁剪结尾
        margin_ms: 安全边距(ms)

    Returns:
        (original_duration, trimmed_duration, action)
    """
    data, sr = sf.read(wav_path)
    total_samples = len(data)

    start_sample = 0
    end_sample = total_samples
    action = "copy"

    # 裁剪开头：从唤醒词结束位置 + margin 开始保留
    if head_boundary and trim_head:
        end_ms = head_boundary[1]
        start_sample = int((end_ms + margin_ms) * sr / 1000)
        start_sample = min(start_sample, total_samples - 1)
        action = "trim_head"

    # 裁剪结尾：保留到唤醒词开始位置 - margin
    if tail_boundary and trim_tail:
        start_ms = tail_boundary[0]
        end_sample = int((start_ms - margin_ms) * sr / 1000)
        end_sample = max(end_sample, start_sample + 1)
        action = "trim_tail" if action == "copy" else "trim_both"

    if action == "copy":
        # 无需裁剪
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wav_path, output_path)
        dur = total_samples / sr
        return dur, dur, "copy"

    trimmed = data[start_sample:end_sample]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, trimmed, sr)

    original_duration = total_samples / sr
    trimmed_duration = len(trimmed) / sr
    return original_duration, trimmed_duration, action


def main():
    parser = argparse.ArgumentParser(
        description="【本地端】根据 trim_log.jsonl 裁剪唤醒词",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--trim-log", default=DEFAULT_TRIM_LOG,
                        help=f"trim_log.jsonl 路径 (default: {DEFAULT_TRIM_LOG})")
    parser.add_argument("--input-text", default=DEFAULT_INPUT_TEXT,
                        help=f"原始 text 文件 (default: {DEFAULT_INPUT_TEXT})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"输出目录 (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--trim-head", default="true", choices=["true", "false"],
                        help="是否裁剪开头唤醒词 (default: true)")
    parser.add_argument("--trim-tail", default="true", choices=["true", "false"],
                        help="是否裁剪结尾唤醒词 (default: true)")
    parser.add_argument("--margin", type=int, default=DEFAULT_MARGIN_MS,
                        help=f"裁剪安全边距(ms) (default: {DEFAULT_MARGIN_MS})")
    parser.add_argument("--sample", type=int, default=None, help="只处理前 N 条")

    args = parser.parse_args()

    trim_log_path = Path(args.trim_log)
    output_dir = Path(args.output_dir)
    do_trim_head = args.trim_head == "true"
    do_trim_tail = args.trim_tail == "true"

    if not trim_log_path.exists():
        print(f"[错误] trim_log.jsonl 不存在: {trim_log_path}")
        print("  请先在服务器运行 trim_wakeword.py 生成检测结果")
        return

    # 1. 读取 trim_log
    entries = []
    with open(trim_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    print(f"读取 trim_log: {len(entries)} 条")

    if args.sample and args.sample < len(entries):
        entries = entries[:args.sample]
        print(f"只处理前 {args.sample} 条")

    # 2. 裁剪音频
    output_wav_dir = output_dir / "wavs"
    output_wav_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total": len(entries),
        "trim_head": 0,
        "trim_tail": 0,
        "trim_both": 0,
        "copy": 0,
        "errors": 0,
        "total_saved_ms": 0,
    }

    new_wav_scp_lines = []
    result_log = []

    print(f"\n开始裁剪...")
    print(f"  裁剪开头: {do_trim_head}")
    print(f"  裁剪结尾: {do_trim_tail}")
    print(f"  安全边距: {args.margin}ms")
    print()

    for idx, entry in enumerate(entries):
        uid = entry["uid"]
        wav_path = entry["wav_path"]
        head_boundary = entry.get("head_boundary")
        tail_boundary = entry.get("tail_boundary")
        output_wav_path = str(output_wav_dir / f"{uid}.wav")

        # 检查音频文件是否存在
        if not Path(wav_path).exists():
            print(f"  [跳过] {uid}: 音频文件不存在 {wav_path}")
            stats["errors"] += 1
            new_wav_scp_lines.append(f"{uid} {wav_path}")
            continue

        try:
            orig_dur, trim_dur, action = trim_audio(
                wav_path, output_wav_path,
                head_boundary=head_boundary,
                tail_boundary=tail_boundary,
                trim_head=do_trim_head,
                trim_tail=do_trim_tail,
                margin_ms=args.margin,
            )

            stats[action] = stats.get(action, 0) + 1
            if action != "copy":
                stats["total_saved_ms"] += int((orig_dur - trim_dur) * 1000)

            new_wav_scp_lines.append(f"{uid} {output_wav_path}")

            result_log.append({
                "uid": uid,
                "action": action,
                "original_duration": round(orig_dur, 3),
                "trimmed_duration": round(trim_dur, 3),
                "head_boundary": head_boundary,
                "tail_boundary": tail_boundary,
            })

        except Exception as e:
            print(f"  [错误] {uid}: {e}")
            stats["errors"] += 1
            new_wav_scp_lines.append(f"{uid} {wav_path}")

        if (idx + 1) % 100 == 0 or idx + 1 == len(entries):
            print(f"  进度: {idx + 1}/{len(entries)}")

    # 3. 写入新的 wav.scp
    new_scp_path = output_dir / "wav.scp"
    with open(new_scp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_wav_scp_lines) + "\n")

    # 4. 复制 text 文件
    text_src = Path(args.input_text)
    if text_src.exists():
        shutil.copy2(str(text_src), str(output_dir / "text"))
        print(f"  text 文件已复制到: {output_dir / 'text'}")

    # 5. 保存裁剪结果日志
    result_log_path = output_dir / "trim_result.jsonl"
    with open(result_log_path, "w", encoding="utf-8") as f:
        for entry in result_log:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 6. 打印统计
    trimmed_count = stats.get("trim_head", 0) + stats.get("trim_tail", 0) + stats.get("trim_both", 0)
    print("\n" + "=" * 60)
    print("裁剪完成")
    print("=" * 60)
    print(f"总条数:         {stats['total']}")
    print(f"裁剪开头:       {stats.get('trim_head', 0)}")
    print(f"裁剪结尾:       {stats.get('trim_tail', 0)}")
    print(f"裁剪两端:       {stats.get('trim_both', 0)}")
    print(f"未裁剪(复制):   {stats.get('copy', 0)}")
    print(f"处理失败:       {stats['errors']}")
    print(f"总节省时长:     {stats['total_saved_ms'] / 1000:.1f}s")
    print()
    print(f"输出目录:       {output_dir}")
    print(f"新 wav.scp:     {new_scp_path}")
    print(f"裁剪日志:       {result_log_path}")
    print()
    print("后续评估时使用裁剪后的数据:")
    print(f"  python eval_on_testset.py --test-scp {new_scp_path} --test-text {output_dir / 'text'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
