#!/usr/bin/env python3
"""
S10 ASR 音频数据质量报告生成脚本
读取 quality_detail.jsonl，按阈值判断问题，生成统计报告和文件清单。

所有过滤阈值在此脚本中配置，调整后只需重跑本脚本，无需重新扫描。

前置:
    先运行 python 03_audio_quality_check.py 生成 quality_detail.jsonl

用法:
    python 03b_quality_report.py
    python 03b_quality_report.py --input quality_check/quality_detail.jsonl
    python 03b_quality_report.py --min-dur 0.5 --max-dur 20

输出:
    quality_check/quality_report.txt   — 文本统计报告
    quality_check/quality_summary.json — JSON 汇总
    quality_check/bad_files.txt        — 问题文件清单
    quality_check/good_files.txt       — 合格文件清单
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ============ 配置 ============
BASE_DIR = Path(r'E:\8.音频\03 s10\s10_data_reflow')
OUTPUT_DIR = BASE_DIR / "quality_check"
DEFAULT_INPUT = OUTPUT_DIR / "quality_detail.jsonl"
EXPECTED_SAMPLE_RATE = 16000

# 过滤阈值（可通过命令行覆盖）
DEFAULT_MIN_DURATION = 0.3    # 秒
DEFAULT_MAX_DURATION = 30.0   # 秒
DEFAULT_RMS_THRESHOLD = 50    # 16bit PCM 下的 RMS 阈值，低于此视为静音
DEFAULT_CLIPPING_RATIO = 0.01 # 削波样本占比阈值
DEFAULT_DC_OFFSET = 500       # 直流偏移绝对值阈值
DEFAULT_SNR_THRESHOLD = 5.0   # dB，低于此视为低信噪比
# ==============================


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger('quality_report')
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    return logger


def load_results(input_path: Path, logger) -> list:
    """从 jsonl 加载分析结果"""
    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        logger.error("请先运行 python 03_audio_quality_check.py")
        sys.exit(1)

    logger.info(f"加载分析结果: {input_path}")
    results = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    logger.info(f"加载 {len(results)} 条记录")
    return results


def apply_thresholds(results: list, args, logger) -> list:
    """根据阈值对每条记录标记 valid 和 issues"""
    logger.info(f"应用过滤阈值: 时长 {args.min_dur}s~{args.max_dur}s, "
                f"RMS<{args.rms_threshold}, 削波>{args.clipping_ratio}, "
                f"DC偏移>{args.dc_offset}, SNR<{args.snr_threshold}dB")

    for r in results:
        issues = []

        # 文件级错误（扫描时已标记）
        if r.get("error"):
            issues.append(r["error"])
        else:
            # 时长
            if r["duration"] < args.min_dur:
                issues.append("too_short")
            if r["duration"] > args.max_dur:
                issues.append("too_long")
            # 采样率
            if r["sample_rate"] != EXPECTED_SAMPLE_RATE and r["sample_rate"] > 0:
                issues.append(f"unexpected_sr:{r['sample_rate']}")
            # 声道
            if r["channels"] != 1 and r["channels"] > 0:
                issues.append(f"not_mono:{r['channels']}ch")
            # 位深
            if r["sample_width"] not in (0, 2) and r["sample_width"] > 0:
                issues.append(f"not_16bit:{r['sample_width'] * 8}bit")
            # 静音/低能量
            if r["rms"] > 0 and r["rms"] < args.rms_threshold:
                issues.append("silence_or_low_energy")
            # 直流偏移
            if abs(r["dc_offset"]) > args.dc_offset:
                issues.append(f"dc_offset:{r['dc_offset']:.0f}")
            # 削波
            if r["clipping_ratio"] > args.clipping_ratio:
                issues.append("clipping")
            # SNR
            if r["snr_db"] > 0 and r["snr_db"] < args.snr_threshold:
                issues.append(f"low_snr:{r['snr_db']:.1f}dB")

        r["issues"] = issues
        r["valid"] = len(issues) == 0

    valid_count = sum(1 for r in results if r["valid"])
    logger.info(f"阈值判断完成: 合格 {valid_count}, 有问题 {len(results) - valid_count}")
    return results


def write_file_lists(results: list, output_dir: Path, logger):
    """输出问题文件和合格文件清单"""
    bad_path = output_dir / "bad_files.txt"
    bad_count = 0
    with open(bad_path, 'w', encoding='utf-8') as f:
        for r in results:
            if not r["valid"]:
                issues_str = ",".join(r["issues"])
                f.write(f"{r['path']}\t{issues_str}\n")
                bad_count += 1
    logger.info(f"问题文件清单: {bad_path} ({bad_count} 条)")

    good_path = output_dir / "good_files.txt"
    good_count = 0
    with open(good_path, 'w', encoding='utf-8') as f:
        for r in results:
            if r["valid"]:
                f.write(f"{r['path']}\n")
                good_count += 1
    logger.info(f"合格文件清单: {good_path} ({good_count} 条)")


def generate_report(results: list, output_dir: Path, logger):
    """生成统计报告"""
    total = len(results)
    valid = [r for r in results if r["valid"]]
    invalid = [r for r in results if not r["valid"]]

    # 问题类型统计
    issue_counts = defaultdict(int)
    for r in results:
        for iss in r["issues"]:
            tag = iss.split(":")[0]
            issue_counts[tag] += 1

    # 时长统计（仅合格文件）
    durations = [r["duration"] for r in valid if r["duration"] > 0]
    dur_buckets = defaultdict(int)
    for d in durations:
        if d < 1:
            dur_buckets["<1s"] += 1
        elif d < 2:
            dur_buckets["1-2s"] += 1
        elif d < 3:
            dur_buckets["2-3s"] += 1
        elif d < 5:
            dur_buckets["3-5s"] += 1
        elif d < 10:
            dur_buckets["5-10s"] += 1
        elif d < 20:
            dur_buckets["10-20s"] += 1
        else:
            dur_buckets["20s+"] += 1

    total_dur_hours = sum(durations) / 3600 if durations else 0
    avg_dur = sum(durations) / len(durations) if durations else 0
    min_dur = min(durations) if durations else 0
    max_dur = max(durations) if durations else 0

    # 采样率统计
    sr_counts = defaultdict(int)
    for r in results:
        if r["sample_rate"] > 0:
            sr_counts[r["sample_rate"]] += 1

    # 声道统计
    ch_counts = defaultdict(int)
    for r in results:
        if r["channels"] > 0:
            ch_counts[r["channels"]] += 1

    # RMS 分布（合格文件）
    rms_values = [r["rms"] for r in valid if r["rms"] > 0]
    rms_buckets = defaultdict(int)
    for v in rms_values:
        if v < 100:
            rms_buckets["<100 (很安静)"] += 1
        elif v < 500:
            rms_buckets["100-500 (偏安静)"] += 1
        elif v < 2000:
            rms_buckets["500-2000 (正常)"] += 1
        elif v < 5000:
            rms_buckets["2000-5000 (偏响)"] += 1
        else:
            rms_buckets["5000+ (很响)"] += 1

    # SNR 分布（合格文件）
    snr_values = [r["snr_db"] for r in valid if r["snr_db"] > 0]
    snr_buckets = defaultdict(int)
    for v in snr_values:
        if v < 5:
            snr_buckets["<5dB (极差)"] += 1
        elif v < 10:
            snr_buckets["5-10dB (较差)"] += 1
        elif v < 20:
            snr_buckets["10-20dB (一般)"] += 1
        elif v < 35:
            snr_buckets["20-35dB (良好)"] += 1
        else:
            snr_buckets["35dB+ (优秀)"] += 1

    # DC偏移分布
    dc_values = [r["dc_offset"] for r in results if r["dc_offset"] != 0.0]
    dc_buckets = defaultdict(int)
    for v in dc_values:
        av = abs(v)
        if av < 50:
            dc_buckets["<50 (正常)"] += 1
        elif av < 200:
            dc_buckets["50-200 (轻微)"] += 1
        elif av < 500:
            dc_buckets["200-500 (偏高)"] += 1
        else:
            dc_buckets["500+ (异常)"] += 1

    # 文件大小统计
    sizes = [r["file_size"] for r in results if r["file_size"] > 0]
    total_size_gb = sum(sizes) / (1024 ** 3) if sizes else 0

    # 按来源目录统计
    dir_stats = defaultdict(lambda: {"total": 0, "valid": 0, "invalid": 0})
    for r in results:
        p = Path(r["path"])
        parts = p.parts
        group = "other"
        for i, part in enumerate(parts):
            if part == "source" and i + 1 < len(parts):
                group = f"source/{parts[i + 1]}"
                break
            elif part == "labeled":
                group = "labeled"
                break
            elif part == "unlabeled":
                group = "unlabeled"
                break
        dir_stats[group]["total"] += 1
        if r["valid"]:
            dir_stats[group]["valid"] += 1
        else:
            dir_stats[group]["invalid"] += 1

    # 写文本报告
    report_path = output_dir / "quality_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        def w(s=""):
            f.write(s + "\n")
            logger.info(s)

        w("=" * 65)
        w("S10 ASR 音频数据质量报告")
        w(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        w("=" * 65)

        w()
        w("--- 总览 ---")
        w(f"文件总数:     {total:>10,}")
        w(f"合格:         {len(valid):>10,} ({len(valid)/total*100:.1f}%)")
        w(f"有问题:       {len(invalid):>10,} ({len(invalid)/total*100:.1f}%)")
        w(f"总大小:       {total_size_gb:>10.2f} GB")
        w(f"总时长:       {total_dur_hours:>10.1f} 小时")

        w()
        w("--- 时长统计 (合格文件) ---")
        w(f"平均: {avg_dur:.2f}s | 最短: {min_dur:.3f}s | 最长: {max_dur:.2f}s")
        dur_order = ["<1s", "1-2s", "2-3s", "3-5s", "5-10s", "10-20s", "20s+"]
        for bucket in dur_order:
            cnt = dur_buckets.get(bucket, 0)
            pct = cnt / len(durations) * 100 if durations else 0
            bar = "█" * int(pct / 2)
            w(f"  {bucket:>8s}: {cnt:>8,} ({pct:5.1f}%) {bar}")

        w()
        w("--- 问题类型统计 ---")
        for iss_type, cnt in sorted(issue_counts.items(), key=lambda x: -x[1]):
            pct = cnt / total * 100
            w(f"  {iss_type:<30s}: {cnt:>8,} ({pct:5.2f}%)")

        w()
        w("--- 采样率分布 ---")
        for sr, cnt in sorted(sr_counts.items()):
            pct = cnt / total * 100
            marker = " ✓" if sr == EXPECTED_SAMPLE_RATE else " ✗"
            w(f"  {sr:>6d} Hz: {cnt:>8,} ({pct:5.1f}%){marker}")

        w()
        w("--- 声道分布 ---")
        for ch, cnt in sorted(ch_counts.items()):
            pct = cnt / total * 100
            w(f"  {ch}ch: {cnt:>8,} ({pct:5.1f}%)")

        w()
        w("--- RMS 能量分布 (合格文件) ---")
        rms_order = ["<100 (很安静)", "100-500 (偏安静)", "500-2000 (正常)", "2000-5000 (偏响)", "5000+ (很响)"]
        for bucket in rms_order:
            cnt = rms_buckets.get(bucket, 0)
            pct = cnt / len(rms_values) * 100 if rms_values else 0
            bar = "█" * int(pct / 2)
            w(f"  {bucket:<22s}: {cnt:>8,} ({pct:5.1f}%) {bar}")

        w()
        w("--- SNR 信噪比分布 (合格文件) ---")
        snr_order = ["<5dB (极差)", "5-10dB (较差)", "10-20dB (一般)", "20-35dB (良好)", "35dB+ (优秀)"]
        for bucket in snr_order:
            cnt = snr_buckets.get(bucket, 0)
            pct = cnt / len(snr_values) * 100 if snr_values else 0
            bar = "█" * int(pct / 2)
            w(f"  {bucket:<22s}: {cnt:>8,} ({pct:5.1f}%) {bar}")

        w()
        w("--- DC偏移分布 ---")
        dc_order = ["<50 (正常)", "50-200 (轻微)", "200-500 (偏高)", "500+ (异常)"]
        for bucket in dc_order:
            cnt = dc_buckets.get(bucket, 0)
            pct = cnt / len(dc_values) * 100 if dc_values else 0
            bar = "█" * int(pct / 2)
            w(f"  {bucket:<22s}: {cnt:>8,} ({pct:5.1f}%) {bar}")

        w()
        w("--- 按目录统计 ---")
        for group in sorted(dir_stats.keys()):
            s = dir_stats[group]
            inv_pct = s["invalid"] / s["total"] * 100 if s["total"] > 0 else 0
            w(f"  {group:<35s}: 总{s['total']:>7,} | 合格{s['valid']:>7,} | 问题{s['invalid']:>6,} ({inv_pct:.1f}%)")

    logger.info(f"报告已保存: {report_path}")

    # JSON 汇总
    summary = {
        "total_files": total,
        "valid_files": len(valid),
        "invalid_files": len(invalid),
        "total_size_gb": round(total_size_gb, 2),
        "total_duration_hours": round(total_dur_hours, 1),
        "avg_duration_s": round(avg_dur, 2),
        "issue_counts": dict(issue_counts),
        "sample_rate_distribution": {str(k): v for k, v in sr_counts.items()},
        "channel_distribution": {str(k): v for k, v in ch_counts.items()},
        "duration_distribution": dict(dur_buckets),
        "rms_distribution": dict(rms_buckets),
        "snr_distribution": dict(snr_buckets),
        "dc_offset_distribution": dict(dc_buckets),
        "dir_stats": {k: dict(v) for k, v in dir_stats.items()},
    }
    summary_path = output_dir / "quality_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"汇总JSON: {summary_path}")


def run_report(args):
    output_dir = OUTPUT_DIR
    logger = setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("S10 ASR 音频数据质量报告生成")
    logger.info("=" * 60)

    results = load_results(Path(args.input), logger)
    if not results:
        return

    apply_thresholds(results, args, logger)
    write_file_lists(results, output_dir, logger)
    generate_report(results, output_dir, logger)


def parse_args():
    parser = argparse.ArgumentParser(description="S10 ASR 音频数据质量报告生成")
    parser.add_argument(
        '--input', default=str(DEFAULT_INPUT),
        help=f'输入的 quality_detail.jsonl 路径 (default: {DEFAULT_INPUT})'
    )
    parser.add_argument(
        '--min-dur', type=float, default=DEFAULT_MIN_DURATION,
        help=f'最短时长秒 (default: {DEFAULT_MIN_DURATION})'
    )
    parser.add_argument(
        '--max-dur', type=float, default=DEFAULT_MAX_DURATION,
        help=f'最长时长秒 (default: {DEFAULT_MAX_DURATION})'
    )
    parser.add_argument(
        '--rms-threshold', type=float, default=DEFAULT_RMS_THRESHOLD,
        help=f'静音RMS阈值 (default: {DEFAULT_RMS_THRESHOLD})'
    )
    parser.add_argument(
        '--clipping-ratio', type=float, default=DEFAULT_CLIPPING_RATIO,
        help=f'削波比例阈值 (default: {DEFAULT_CLIPPING_RATIO})'
    )
    parser.add_argument(
        '--dc-offset', type=float, default=DEFAULT_DC_OFFSET,
        help=f'直流偏移阈值 (default: {DEFAULT_DC_OFFSET})'
    )
    parser.add_argument(
        '--snr-threshold', type=float, default=DEFAULT_SNR_THRESHOLD,
        help=f'最低SNR阈值dB (default: {DEFAULT_SNR_THRESHOLD})'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_report(args)
