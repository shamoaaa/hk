#!/usr/bin/env python3
"""
音频音量 / SNR / 推测距离 分布分析脚本

扫描指定目录下的 wav 文件，统计：
  1. RMS 音量 (dBFS)
  2. SNR (dB) — 基于帧能量排序估算
  3. 推测录音距离 — 基于 RMS + SNR 的简单模型

输出：
  - 控制台统计摘要
  - 分布直方图 (PNG)
  - 详细数据 (CSV)

用法:
    python analyze_audio_distribution.py
    python analyze_audio_distribution.py --wav-dir "E:\8.音频\03 s10\sp4_data\labeled_check\wav"
    python analyze_audio_distribution.py --workers 8
"""

import os
import sys
import wave
import struct
import math
import csv
import time
import argparse
from pathlib import Path
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============ 默认配置 ============
DEFAULT_WAV_DIR = r"E:\8.音频\03 s10\sp4_data\labeled_check\wav"
# ==================================

# SNR 估算参数
SNR_FRAME_MS = 25
SNR_HOP_MS = 10


def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def analyze_wav(wav_path: str) -> dict:
    """分析单个 wav 文件的音量和 SNR"""
    result = {
        "path": wav_path,
        "uid": Path(wav_path).stem,
        "error": None,
        "duration": 0.0,
        "sample_rate": 0,
        "rms": 0.0,
        "rms_dbfs": -999.0,
        "peak": 0,
        "peak_dbfs": -999.0,
        "snr_db": 0.0,
    }

    try:
        file_size = os.path.getsize(wav_path)
        if file_size == 0:
            result["error"] = "empty_file"
            return result
    except OSError:
        result["error"] = "file_not_found"
        return result

    try:
        wf = wave.open(wav_path, 'rb')
    except Exception as e:
        result["error"] = f"corrupt:{str(e)[:80]}"
        return result

    try:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        duration = n_frames / sample_rate if sample_rate > 0 else 0

        result["sample_rate"] = sample_rate
        result["duration"] = round(duration, 3)

        if sample_width != 2 or n_frames == 0:
            result["error"] = f"unsupported:sw={sample_width},frames={n_frames}"
            return result

        # 读取音频（最多 60 秒）
        max_read = min(n_frames, sample_rate * 60)
        raw_data = wf.readframes(max_read)
        n_samples = len(raw_data) // (sample_width * channels)

        if n_samples == 0:
            result["error"] = "no_samples"
            return result

        # 取单声道
        if channels == 1:
            samples = struct.unpack(f'<{n_samples}h', raw_data[:n_samples * 2])
        else:
            all_s = struct.unpack(f'<{n_samples * channels}h',
                                  raw_data[:n_samples * channels * 2])
            samples = all_s[::channels]

        # RMS
        sum_sq = sum(s * s for s in samples)
        rms = math.sqrt(sum_sq / len(samples))
        result["rms"] = round(rms, 1)

        # dBFS (相对 16-bit 满幅 32767)
        if rms > 0:
            result["rms_dbfs"] = round(20 * math.log10(rms / 32767), 1)
        else:
            result["rms_dbfs"] = -999.0

        # Peak
        peak = max(abs(s) for s in samples)
        result["peak"] = peak
        if peak > 0:
            result["peak_dbfs"] = round(20 * math.log10(peak / 32767), 1)

        # SNR 估算（帧能量排序法）
        frame_len = int(sample_rate * SNR_FRAME_MS / 1000)
        hop_len = int(sample_rate * SNR_HOP_MS / 1000)
        if frame_len > 0 and hop_len > 0 and len(samples) >= frame_len:
            energies = []
            for fi in range(0, len(samples) - frame_len, hop_len):
                frame = samples[fi:fi + frame_len]
                e = sum(s * s for s in frame) / frame_len
                energies.append(e)
            if energies:
                energies_sorted = sorted(energies)
                n_e = len(energies_sorted)
                n_low = max(n_e // 5, 1)
                n_high = max(n_e // 5, 1)
                noise_e = sum(energies_sorted[:n_low]) / n_low
                signal_e = sum(energies_sorted[-n_high:]) / n_high
                if noise_e > 0:
                    snr_db = 10 * math.log10(signal_e / noise_e)
                else:
                    snr_db = 99.0
                result["snr_db"] = round(snr_db, 1)

    except Exception as e:
        result["error"] = f"read_error:{str(e)[:80]}"
    finally:
        wf.close()

    return result


def estimate_distance(rms_dbfs: float, snr_db: float) -> str:
    """
    基于 RMS 音量和 SNR 粗略推测录音距离范围。

    模型假设：
    - 正常说话声压约 60 dB SPL @ 1m
    - 声压随距离按 ~6 dB/倍距 衰减（自由场）
    - 高 RMS + 高 SNR → 近距离
    - 低 RMS + 低 SNR → 远距离或嘈杂环境

    距离分档（粗略）：
      < 0.3m  : 贴近麦克风
      0.3-1m  : 近场
      1-3m    : 中距离
      > 3m    : 远场
    """
    # 综合评分：RMS 和 SNR 各占一半权重
    # RMS dBFS 典型范围: -40 ~ -10
    # SNR 典型范围: 5 ~ 40
    rms_score = min(max((rms_dbfs + 40) / 30, 0), 1)   # -40→0, -10→1
    snr_score = min(max((snr_db - 5) / 35, 0), 1)      # 5→0, 40→1
    combined = 0.6 * rms_score + 0.4 * snr_score

    if combined > 0.75:
        return "<0.3m"
    elif combined > 0.5:
        return "0.3-1m"
    elif combined > 0.25:
        return "1-3m"
    else:
        return ">3m"


def estimate_distance_meters(rms_dbfs: float, snr_db: float) -> float:
    """返回推测距离的数值（米），用于统计"""
    rms_score = min(max((rms_dbfs + 40) / 30, 0), 1)
    snr_score = min(max((snr_db - 5) / 35, 0), 1)
    combined = 0.6 * rms_score + 0.4 * snr_score

    if combined > 0.75:
        return 0.15
    elif combined > 0.5:
        return 0.6
    elif combined > 0.25:
        return 2.0
    else:
        return 5.0


def percentile(sorted_list, p):
    """计算百分位数"""
    if not sorted_list:
        return 0
    k = (len(sorted_list) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_list):
        return sorted_list[-1]
    return sorted_list[f] + (k - f) * (sorted_list[c] - sorted_list[f])


def print_stats(values: list, name: str, unit: str):
    """打印统计摘要"""
    if not values:
        print(f"  {name}: 无数据")
        return

    s = sorted(values)
    n = len(s)
    mean = sum(s) / n
    p5 = percentile(s, 5)
    p25 = percentile(s, 25)
    p50 = percentile(s, 50)
    p75 = percentile(s, 75)
    p95 = percentile(s, 95)

    print(f"  {name}:")
    print(f"    数量:  {n}")
    print(f"    均值:  {mean:.1f} {unit}")
    print(f"    范围:  [{s[0]:.1f}, {s[-1]:.1f}] {unit}")
    print(f"    P5:    {p5:.1f} {unit}")
    print(f"    P25:   {p25:.1f} {unit}")
    print(f"    P50:   {p50:.1f} {unit} (中位数)")
    print(f"    P75:   {p75:.1f} {unit}")
    print(f"    P95:   {p95:.1f} {unit}")


def print_histogram(values: list, name: str, unit: str, bins: list):
    """打印文本直方图"""
    if not values:
        return

    counts = [0] * (len(bins) + 1)
    for v in values:
        placed = False
        for i, edge in enumerate(bins):
            if v < edge:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1

    total = len(values)
    max_count = max(counts) if counts else 1
    bar_width = 40

    print(f"\n  {name} 分布直方图:")
    labels = []
    for i in range(len(bins) + 1):
        if i == 0:
            labels.append(f"    <{bins[0]}{unit}")
        elif i == len(bins):
            labels.append(f"    >={bins[-1]}{unit}")
        else:
            labels.append(f"    [{bins[i-1]},{bins[i]}){unit}")

    max_label_len = max(len(l) for l in labels)
    for i, label in enumerate(labels):
        bar_len = int(counts[i] / max_count * bar_width) if max_count > 0 else 0
        pct = counts[i] / total * 100 if total > 0 else 0
        bar = "█" * bar_len
        print(f"{label:<{max_label_len}} | {bar:<{bar_width}} {counts[i]:>6} ({pct:5.1f}%)")


def plot_distributions(rms_dbfs_list, snr_list, dist_labels, output_dir: Path):
    """生成分布图（如果 matplotlib 可用）"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except ImportError:
        print("\n  [提示] matplotlib 未安装，跳过图表生成。安装: pip install matplotlib")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("音频音量 / SNR / 距离分布分析", fontsize=14, fontweight='bold')

    # 1. RMS dBFS 分布
    ax = axes[0, 0]
    ax.hist(rms_dbfs_list, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_xlabel('RMS (dBFS)')
    ax.set_ylabel('文件数')
    ax.set_title('音量分布 (RMS dBFS)')
    ax.axvline(x=sum(rms_dbfs_list)/len(rms_dbfs_list), color='red',
               linestyle='--', label=f'均值: {sum(rms_dbfs_list)/len(rms_dbfs_list):.1f}')
    ax.legend()

    # 2. SNR 分布
    ax = axes[0, 1]
    ax.hist(snr_list, bins=50, color='forestgreen', edgecolor='white', alpha=0.8)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('文件数')
    ax.set_title('信噪比分布 (SNR)')
    ax.axvline(x=sum(snr_list)/len(snr_list), color='red',
               linestyle='--', label=f'均值: {sum(snr_list)/len(snr_list):.1f}')
    ax.legend()

    # 3. RMS vs SNR 散点图
    ax = axes[1, 0]
    scatter = ax.scatter(rms_dbfs_list, snr_list, alpha=0.3, s=5, c='darkorange')
    ax.set_xlabel('RMS (dBFS)')
    ax.set_ylabel('SNR (dB)')
    ax.set_title('音量 vs 信噪比')

    # 4. 距离分布饼图
    ax = axes[1, 1]
    from collections import Counter
    dist_counts = Counter(dist_labels)
    order = ["<0.3m", "0.3-1m", "1-3m", ">3m"]
    labels_ordered = [l for l in order if l in dist_counts]
    sizes = [dist_counts[l] for l in labels_ordered]
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c'][:len(labels_ordered)]
    ax.pie(sizes, labels=labels_ordered, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('推测距离分布')

    plt.tight_layout()
    plot_path = output_dir / "audio_distribution.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  分布图已保存: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="音频音量/SNR/距离分布分析")
    parser.add_argument("--wav-dir", default=DEFAULT_WAV_DIR, help="wav 文件目录")
    parser.add_argument("--workers", type=int, default=4, help="并行进程数 (default: 4)")
    parser.add_argument("--limit", type=int, default=None, help="只分析前 N 个文件（调试用）")
    args = parser.parse_args()

    wav_dir = Path(args.wav_dir)
    output_dir = wav_dir.parent / "audio_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  音频音量 / SNR / 推测距离 分布分析")
    print("=" * 65)
    print(f"  目录: {wav_dir}")

    # 扫描 wav 文件
    wav_files = []
    for root, _, files in os.walk(wav_dir):
        for f in files:
            if f.lower().endswith('.wav'):
                wav_files.append(str(Path(root) / f))
    wav_files.sort()

    if not wav_files:
        print("  未找到 wav 文件！")
        return

    if args.limit:
        wav_files = wav_files[:args.limit]

    print(f"  文件数: {len(wav_files)}")
    print(f"  并行数: {args.workers}")
    print(f"  输出到: {output_dir}")
    print("-" * 65)

    # 并行分析
    results = []
    t_start = time.time()
    total = len(wav_files)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(analyze_wav, p): p for p in wav_files}
        done = 0
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            if done % 2000 == 0 or done == total:
                elapsed = time.time() - t_start
                speed = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / speed if speed > 0 else 0
                print(f"  进度: {done}/{total} ({done/total*100:.1f}%) "
                      f"| {speed:.0f} 条/s | 剩余 {format_duration(eta)}")

    elapsed = time.time() - t_start
    print(f"\n  分析完成，耗时 {format_duration(elapsed)}，"
          f"速度 {total / elapsed:.0f} 条/s")

    # 过滤有效结果
    valid = [r for r in results if not r.get("error")]
    errors = [r for r in results if r.get("error")]
    print(f"  有效: {len(valid)} | 错误: {len(errors)}")

    if not valid:
        print("  没有有效数据可分析！")
        return

    # 提取数值
    rms_dbfs_list = [r["rms_dbfs"] for r in valid if r["rms_dbfs"] > -999]
    snr_list = [r["snr_db"] for r in valid if r["snr_db"] > 0]

    # 推测距离
    dist_labels = []
    dist_meters = []
    for r in valid:
        if r["rms_dbfs"] > -999 and r["snr_db"] > 0:
            dl = estimate_distance(r["rms_dbfs"], r["snr_db"])
            dm = estimate_distance_meters(r["rms_dbfs"], r["snr_db"])
            dist_labels.append(dl)
            dist_meters.append(dm)

    # ===== 统计摘要 =====
    print("\n" + "=" * 65)
    print("  统计摘要")
    print("=" * 65)

    print_stats(rms_dbfs_list, "RMS 音量", "dBFS")
    print_stats(snr_list, "信噪比 (SNR)", "dB")

    # 距离分布统计
    if dist_labels:
        from collections import Counter
        dist_counts = Counter(dist_labels)
        total_d = len(dist_labels)
        print(f"\n  推测距离分布:")
        for label in ["<0.3m", "0.3-1m", "1-3m", ">3m"]:
            c = dist_counts.get(label, 0)
            pct = c / total_d * 100
            print(f"    {label:<10} {c:>6} ({pct:5.1f}%)")

    # ===== 文本直方图 =====
    print_histogram(rms_dbfs_list, "RMS 音量",
                    "dBFS", [-50, -40, -35, -30, -25, -20, -15, -10, -5])
    print_histogram(snr_list, "信噪比",
                    "dB", [5, 10, 15, 20, 25, 30, 35, 40, 50])

    # ===== 保存 CSV =====
    csv_path = output_dir / "audio_analysis.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["uid", "path", "duration", "sample_rate",
                         "rms", "rms_dbfs", "peak", "peak_dbfs",
                         "snr_db", "est_distance", "error"])
        for r in results:
            dist = ""
            if not r.get("error") and r["rms_dbfs"] > -999 and r["snr_db"] > 0:
                dist = estimate_distance(r["rms_dbfs"], r["snr_db"])
            writer.writerow([
                r["uid"], r["path"], r["duration"], r["sample_rate"],
                r["rms"], r["rms_dbfs"], r["peak"], r.get("peak_dbfs", ""),
                r["snr_db"], dist, r.get("error", ""),
            ])
    print(f"\n  详细数据已保存: {csv_path}")

    # ===== 生成图表 =====
    if rms_dbfs_list and snr_list:
        plot_distributions(rms_dbfs_list, snr_list, dist_labels, output_dir)

    print("\n" + "=" * 65)
    print("  分析完成！")
    print(f"  输出目录: {output_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
