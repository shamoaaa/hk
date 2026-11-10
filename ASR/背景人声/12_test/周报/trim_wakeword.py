#!/usr/bin/env python3
"""
【服务器端】使用 FunASR 时间戳对齐检测音频中的唤醒词"你好萤石"位置

只做识别和检测，不做裁剪。输出 trim_log.jsonl 供本地裁剪脚本使用。

原理:
  1. 用带时间戳的 ASR 模型识别每条音频，获取逐字时间戳
  2. 检测开头/结尾是否包含唤醒词"你好萤石"
  3. 输出检测结果到 trim_log.jsonl

用法:
    # 默认配置运行
    python trim_wakeword.py

    # 指定输入
    python trim_wakeword.py --input-scp /data1/12_test/wav.scp

    # 自定义唤醒词
    python trim_wakeword.py --wakeword "你好小萤"

    # 采样快速验证
    python trim_wakeword.py --sample 20

输出:
    trim_log.jsonl - 每行一条检测结果，包含 uid、识别文本、唤醒词边界时间戳
    后续在本地运行 trim_wakeword_local.py 进行裁剪
"""

import os
import sys
import json
import time
import argparse
import soundfile as sf
from pathlib import Path
from datetime import datetime


# ============ 默认配置 ============
DEFAULT_INPUT_SCP = "/data1/12_test/wav.scp"
DEFAULT_OUTPUT_DIR = "/data1/12_test"
DEFAULT_WAKEWORD = "你好萤石"
DEFAULT_ASR_MODEL = "/data1/pretrained_models/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
# ==================================


def load_wav_scp(scp_path: str) -> dict:
    """加载 wav.scp，返回 {uid: wav_path}"""
    wav_map = {}
    with open(scp_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                wav_map[parts[0]] = parts[1]
    return wav_map


def load_model_with_timestamp(model_dir: str, device: str, wakeword: str = ""):
    """加载带时间戳输出的 ASR 模型，支持热词增强"""
    from funasr import AutoModel

    print(f"加载 ASR 模型（带时间戳）: {model_dir}")
    load_start = time.time()
    model = AutoModel(model=model_dir, device=device)
    print(f"模型加载完成，耗时: {time.time() - load_start:.1f}s")
    if wakeword:
        print(f"热词增强: {wakeword}")
    return model


def detect_wakeword_boundaries(result: dict, wakeword: str = "你好萤石"):
    """
    从 FunASR 带时间戳的识别结果中检测唤醒词边界。

    支持模糊匹配：即使模型把"你好萤石"识别成"你好岩石"等变体，
    只要开头符合"你好XX"模式（4个字）且前两字为"你好"，就认为是唤醒词。

    Args:
        result: FunASR generate 返回的单条结果，包含 text 和 timestamp
        wakeword: 唤醒词文本

    Returns:
        {
            "head": (start_ms, end_ms) or None,
            "tail": (start_ms, end_ms) or None,
            "text": str,
            "timestamp": list,
        }
    """
    text = result.get("text", "")
    timestamp = result.get("timestamp", [])

    chars = list(text.replace(" ", "").replace("，", "").replace(",", ""))
    wakeword_chars = list(wakeword)
    wk_len = len(wakeword_chars)

    head_boundary = None
    tail_boundary = None

    if len(chars) < wk_len or len(timestamp) < wk_len:
        return {"head": None, "tail": None, "text": text, "timestamp": timestamp}

    # --- 开头检测（模糊匹配） ---
    # 策略1: 精确匹配
    if chars[:wk_len] == wakeword_chars:
        head_boundary = (timestamp[0][0], timestamp[wk_len - 1][1])
    # 策略2: 前两字是"你好" + 后面跟2个字（变体如：你好岩石、你好英石、你好零食...）
    elif len(chars) >= wk_len and chars[0] == "你" and chars[1] == "好":
        head_boundary = (timestamp[0][0], timestamp[wk_len - 1][1])
    # 策略3: 开头是"好" + 后面的模式（有时"你"被吞掉，识别为"好萤石"等）
    elif len(chars) >= (wk_len - 1) and chars[0] == "好":
        head_boundary = (timestamp[0][0], timestamp[wk_len - 2][1])

    # --- 结尾检测（模糊匹配） ---
    if chars[-wk_len:] == wakeword_chars:
        tail_start_idx = len(chars) - wk_len
        if tail_start_idx < len(timestamp):
            tail_boundary = (timestamp[tail_start_idx][0], timestamp[-1][1])
    elif len(chars) >= wk_len and chars[-wk_len] == "你" and chars[-wk_len + 1] == "好":
        tail_start_idx = len(chars) - wk_len
        if tail_start_idx < len(timestamp):
            tail_boundary = (timestamp[tail_start_idx][0], timestamp[-1][1])

    return {
        "head": head_boundary,
        "tail": tail_boundary,
        "text": text,
        "timestamp": timestamp,
    }


def main():
    parser = argparse.ArgumentParser(
        description="【服务器端】FunASR 检测唤醒词位置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input-scp", default=DEFAULT_INPUT_SCP,
                        help=f"输入 wav.scp (default: {DEFAULT_INPUT_SCP})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"输出目录 (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--wakeword", default=DEFAULT_WAKEWORD,
                        help=f"唤醒词 (default: {DEFAULT_WAKEWORD})")
    parser.add_argument("--model", default=DEFAULT_ASR_MODEL,
                        help="ASR 模型路径（需支持时间戳）")
    parser.add_argument("--device", default="cuda:0", help="设备 (default: cuda:0)")
    parser.add_argument("--batch-size", type=int, default=8, help="推理 batch 大小")
    parser.add_argument("--sample", type=int, default=None, help="采样 N 条快速验证")

    args = parser.parse_args()

    input_scp = Path(args.input_scp)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载 wav.scp
    wav_map = load_wav_scp(str(input_scp))
    print(f"加载 wav.scp: {len(wav_map)} 条")

    # 过滤损坏文件（0KB 或无法读取）
    valid_wav_map = {}
    skipped_bad = 0
    for uid, wav_path in wav_map.items():
        p = Path(wav_path)
        if not p.exists():
            skipped_bad += 1
            continue
        if p.stat().st_size == 0:
            skipped_bad += 1
            continue
        # 尝试读取 wav 头部验证
        try:
            info = sf.info(wav_path)
            if info.duration < 0.1:
                skipped_bad += 1
                continue
        except Exception:
            skipped_bad += 1
            continue
        valid_wav_map[uid] = wav_path

    if skipped_bad > 0:
        print(f"跳过损坏/空文件: {skipped_bad} 个")
    wav_map = valid_wav_map
    print(f"有效音频: {len(wav_map)} 条")

    if args.sample and args.sample < len(wav_map):
        import random
        random.seed(42)
        keys = random.sample(list(wav_map.keys()), args.sample)
        wav_map = {k: wav_map[k] for k in keys}
        print(f"采样模式: {args.sample} 条")

    # 2. 加载模型
    model = load_model_with_timestamp(args.model, args.device, args.wakeword)

    # 3. 批量识别获取时间戳
    uids = list(wav_map.keys())
    wav_paths = [wav_map[uid] for uid in uids]

    # 热词字符串（SeACo-Paraformer 热词增强，提高唤醒词识别准确率）
    hotword_str = args.wakeword

    print(f"\n开始识别（获取时间戳）...")
    print(f"  总条数: {len(wav_paths)}")
    print(f"  唤醒词: {args.wakeword}")
    print(f"  热词增强: {hotword_str}")

    all_results = []
    batch_size = args.batch_size
    infer_start = time.time()

    for i in range(0, len(wav_paths), batch_size):
        batch = wav_paths[i:i + batch_size]
        try:
            results = model.generate(
                input=batch,
                cache={},
                batch_size=batch_size,
                hotword=hotword_str,
                disable_pbar=True,
            )
            all_results.extend(results)
        except Exception as e:
            print(f"  [错误] 批次 {i}-{i+len(batch)} 推理失败: {e}")
            all_results.extend([{"text": "", "timestamp": []}] * len(batch))

        if (i + batch_size) % (batch_size * 10) == 0 or i + batch_size >= len(wav_paths):
            elapsed = time.time() - infer_start
            done = min(i + batch_size, len(wav_paths))
            speed = done / elapsed if elapsed > 0 else 0
            print(f"  进度: {done}/{len(wav_paths)} ({speed:.1f} 条/s)")

    print(f"识别完成，耗时: {time.time() - infer_start:.1f}s")

    # 4. 检测唤醒词并输出日志
    stats = {
        "total": len(uids),
        "head_detected": 0,
        "tail_detected": 0,
        "no_wakeword": 0,
        "errors": 0,
    }

    log_path = output_dir / "trim_log.jsonl"
    print(f"\n检测唤醒词边界...")

    with open(log_path, "w", encoding="utf-8") as f:
        for idx, (uid, result) in enumerate(zip(uids, all_results)):
            wav_path = wav_map[uid]

            try:
                boundaries = detect_wakeword_boundaries(result, args.wakeword)

                has_head = boundaries["head"] is not None
                has_tail = boundaries["tail"] is not None

                if has_head:
                    stats["head_detected"] += 1
                if has_tail:
                    stats["tail_detected"] += 1
                if not has_head and not has_tail:
                    stats["no_wakeword"] += 1

                entry = {
                    "uid": uid,
                    "wav_path": wav_path,
                    "recognized_text": boundaries["text"],
                    "head_boundary": list(boundaries["head"]) if boundaries["head"] else None,
                    "tail_boundary": list(boundaries["tail"]) if boundaries["tail"] else None,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"  [错误] {uid}: {e}")
                stats["errors"] += 1
                entry = {
                    "uid": uid,
                    "wav_path": wav_path,
                    "recognized_text": "",
                    "head_boundary": None,
                    "tail_boundary": None,
                    "error": str(e),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 5. 打印统计
    print("\n" + "=" * 60)
    print("唤醒词检测完成")
    print("=" * 60)
    print(f"总条数:         {stats['total']}")
    print(f"开头检测到:     {stats['head_detected']} ({stats['head_detected']/stats['total']*100:.1f}%)")
    print(f"结尾检测到:     {stats['tail_detected']} ({stats['tail_detected']/stats['total']*100:.1f}%)")
    print(f"无唤醒词:       {stats['no_wakeword']} ({stats['no_wakeword']/stats['total']*100:.1f}%)")
    print(f"处理失败:       {stats['errors']}")
    print()
    print(f"检测结果:       {log_path}")
    print()
    print("下一步: 将 trim_log.jsonl 拷贝到本地，运行 trim_wakeword_local.py 进行裁剪")
    print("=" * 60)


if __name__ == "__main__":
    main()
