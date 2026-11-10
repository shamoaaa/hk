#!/usr/bin/env python3
"""
基于能量检测校验唤醒词时间戳边界的合理性

原理:
  唤醒词和正文之间通常有一个短暂的能量低谷（微停顿/换气）。
  用短时能量分析验证 trim_log.jsonl 中的 head_boundary 是否落在合理位置。

校验逻辑:
  1. 读取 trim_log.jsonl 中检测到开头唤醒词的条目
  2. 对每条音频计算短时能量曲线
  3. 在 head_boundary[1]（唤醒词结束时间）附近寻找能量低谷
  4. 如果低谷与 boundary 偏差较大，标记为可疑

输出:
  - verify_result.jsonl: 每条校验结果
  - 统计报告: 通过/可疑/失败的比例

用法:
    python verify_trim_energy.py
    python verify_trim_energy.py --trim-log path/to/trim_log.jsonl
    python verify_trim_energy.py --sample 50
    python verify_trim_energy.py --plot  # 生成可视化图片（前20条）
"""

import os
import json
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path


# ============ 默认配置 ============
DEFAULT_TRIM_LOG = r"E:\8.音频\03 s10\12_test\output\trim_log.jsonl"
DEFAULT_WAV_DIR = r"E:\8.音频\03 s10\12_test\wavs2"
DEFAULT_OUTPUT_DIR = r"E:\8.音频\03 s10\12_test\output"

# 能量分析参数
FRAME_MS = 20          # 帧长（毫秒）
HOP_MS = 10           # 帧移（毫秒）
SEARCH_WINDOW_MS = 300  # 在 boundary 附近搜索能量低谷的窗口（±ms）
TOLERANCE_MS = 150     # 低谷与 boundary 的容许偏差（ms）
# ==================================


def compute_short_time_energy(audio: np.ndarray, sr: int,
                              frame_ms: int = 20, hop_ms: int = 10) -> tuple:
    """
    计算短时能量（dB）。

    Returns:
        (energy_db, time_axis_ms)
        energy_db: 每帧的能量(dB)
        time_axis_ms: 每帧对应的时间(ms)
    """
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)

    n_frames = (len(audio) - frame_len) // hop_len + 1
    if n_frames <= 0:
        return np.array([]), np.array([])

    energy = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_len
        frame = audio[start:start + frame_len]
        energy[i] = np.sum(frame ** 2) / frame_len

    # 转 dB，避免 log(0)
    energy_db = 10 * np.log10(energy + 1e-10)
    time_axis = np.arange(n_frames) * hop_ms + frame_ms / 2

    return energy_db, time_axis


def find_energy_valley(energy_db: np.ndarray, time_axis: np.ndarray,
                       center_ms: float, window_ms: float = 300) -> tuple:
    """
    在 center_ms 附近 ±window_ms 范围内寻找能量最低点。

    Returns:
        (valley_ms, valley_db) 或 (None, None)
    """
    mask = (time_axis >= center_ms - window_ms) & (time_axis <= center_ms + window_ms)
    if not np.any(mask):
        return None, None

    local_energy = energy_db[mask]
    local_time = time_axis[mask]

    min_idx = np.argmin(local_energy)
    return float(local_time[min_idx]), float(local_energy[min_idx])


def verify_boundary(wav_path: str, head_boundary: list,
                    frame_ms: int = FRAME_MS, hop_ms: int = HOP_MS,
                    search_window_ms: int = SEARCH_WINDOW_MS,
                    tolerance_ms: int = TOLERANCE_MS) -> dict:
    """
    校验单条音频的唤醒词边界。

    Returns:
        {
            "status": "pass" | "suspect" | "fail",
            "boundary_end_ms": int,
            "valley_ms": float or None,
            "offset_ms": float,  # valley 与 boundary 的偏差
            "avg_energy_before": float,  # 唤醒词段平均能量
            "avg_energy_after": float,   # 正文段前500ms平均能量
            "energy_drop_db": float,     # boundary 处的能量下降
        }
    """
    data, sr = sf.read(wav_path)
    if len(data.shape) > 1:
        data = data[:, 0]  # 取第一通道

    energy_db, time_axis = compute_short_time_energy(data, sr, frame_ms, hop_ms)
    if len(energy_db) == 0:
        return {"status": "fail", "reason": "audio_too_short"}

    boundary_end_ms = head_boundary[1]

    # 1. 在 boundary 附近找能量低谷
    valley_ms, valley_db = find_energy_valley(
        energy_db, time_axis, boundary_end_ms, search_window_ms
    )

    if valley_ms is None:
        return {
            "status": "fail",
            "reason": "no_valley_found",
            "boundary_end_ms": boundary_end_ms,
        }

    offset_ms = abs(valley_ms - boundary_end_ms)

    # 2. 计算唤醒词段和正文段的平均能量
    wk_mask = (time_axis >= head_boundary[0]) & (time_axis <= head_boundary[1])
    after_start = boundary_end_ms + 50  # 跳过过渡区
    after_end = min(boundary_end_ms + 500, time_axis[-1])
    after_mask = (time_axis >= after_start) & (time_axis <= after_end)

    avg_before = float(np.mean(energy_db[wk_mask])) if np.any(wk_mask) else 0
    avg_after = float(np.mean(energy_db[after_mask])) if np.any(after_mask) else 0

    # 3. boundary 处的能量（取 boundary 附近 ±50ms 的最小值）
    boundary_mask = (time_axis >= boundary_end_ms - 50) & (time_axis <= boundary_end_ms + 50)
    boundary_energy = float(np.min(energy_db[boundary_mask])) if np.any(boundary_mask) else 0
    energy_drop = avg_before - boundary_energy

    # 4. 判定
    if offset_ms <= tolerance_ms and energy_drop > 3:
        status = "pass"  # 低谷在容许范围内，且有明显能量下降
    elif offset_ms <= tolerance_ms * 2:
        status = "suspect"  # 偏差稍大但可接受
    else:
        status = "suspect"  # 低谷偏离较远

    return {
        "status": status,
        "boundary_end_ms": boundary_end_ms,
        "valley_ms": round(valley_ms, 1),
        "offset_ms": round(offset_ms, 1),
        "avg_energy_before_db": round(avg_before, 1),
        "avg_energy_after_db": round(avg_after, 1),
        "energy_drop_db": round(energy_drop, 1),
    }


def plot_energy_with_boundary(wav_path: str, head_boundary: list,
                              verify_result: dict, output_path: str):
    """生成能量曲线 + 边界标注的可视化图"""
    import matplotlib.pyplot as plt

    data, sr = sf.read(wav_path)
    if len(data.shape) > 1:
        data = data[:, 0]

    energy_db, time_axis = compute_short_time_energy(data, sr)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # 波形
    t_wav = np.arange(len(data)) / sr * 1000
    ax1.plot(t_wav, data, linewidth=0.3, color='steelblue')
    ax1.axvline(head_boundary[1], color='red', linestyle='--', label=f'boundary: {head_boundary[1]}ms')
    if verify_result.get("valley_ms"):
        ax1.axvline(verify_result["valley_ms"], color='green', linestyle=':', label=f'valley: {verify_result["valley_ms"]}ms')
    ax1.axvspan(head_boundary[0], head_boundary[1], alpha=0.2, color='orange', label='wakeword')
    ax1.set_ylabel('Amplitude')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_title(f'{Path(wav_path).stem} [{verify_result["status"]}]')

    # 能量
    ax2.plot(time_axis, energy_db, linewidth=0.8, color='darkgreen')
    ax2.axvline(head_boundary[1], color='red', linestyle='--')
    if verify_result.get("valley_ms"):
        ax2.axvline(verify_result["valley_ms"], color='green', linestyle=':')
    ax2.axvspan(head_boundary[0], head_boundary[1], alpha=0.2, color='orange')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Energy (dB)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="基于能量检测校验唤醒词边界",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--trim-log", default=DEFAULT_TRIM_LOG,
                        help=f"trim_log.jsonl 路径 (default: {DEFAULT_TRIM_LOG})")
    parser.add_argument("--wav-dir", default=DEFAULT_WAV_DIR,
                        help=f"本地音频目录 (default: {DEFAULT_WAV_DIR})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"输出目录 (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--sample", type=int, default=None,
                        help="只校验前 N 条")
    parser.add_argument("--plot", action="store_true",
                        help="生成可视化图片（前20条）")
    parser.add_argument("--tolerance", type=int, default=TOLERANCE_MS,
                        help=f"容许偏差(ms) (default: {TOLERANCE_MS})")

    args = parser.parse_args()

    trim_log_path = Path(args.trim_log)
    wav_dir = Path(args.wav_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not trim_log_path.exists():
        print(f"[错误] trim_log.jsonl 不存在: {trim_log_path}")
        return

    # 1. 读取有 head_boundary 的条目
    entries = []
    with open(trim_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("head_boundary"):
                entries.append(entry)

    print(f"读取 trim_log: {len(entries)} 条（有开头唤醒词）")

    if args.sample and args.sample < len(entries):
        entries = entries[:args.sample]
        print(f"只校验前 {args.sample} 条")

    # 2. 构建 uid -> 本地 wav 路径的索引（递归扫描）
    print(f"扫描本地音频目录: {wav_dir}")
    uid_to_local = {}
    for root, dirs, files in os.walk(wav_dir):
        for f in files:
            if f.lower().endswith('.wav'):
                uid = Path(f).stem
                uid_to_local[uid] = os.path.join(root, f)
    print(f"找到本地音频: {len(uid_to_local)} 个")

    # 3. 逐条校验
    results = []
    stats = {"pass": 0, "suspect": 0, "fail": 0, "file_not_found": 0}

    plot_dir = output_dir / "energy_plots"
    if args.plot:
        plot_dir.mkdir(parents=True, exist_ok=True)

    for idx, entry in enumerate(entries):
        uid = entry["uid"]
        head_boundary = entry["head_boundary"]

        # 在本地索引中查找音频
        wav_path_str = uid_to_local.get(uid)
        if not wav_path_str:
            stats["file_not_found"] += 1
            continue

        try:
            result = verify_boundary(
                wav_path_str, head_boundary,
                tolerance_ms=args.tolerance,
            )
            result["uid"] = uid
            result["recognized_text"] = entry.get("recognized_text", "")
            results.append(result)
            stats[result["status"]] += 1

            # 生成可视化
            if args.plot and idx < 20:
                plot_path = plot_dir / f"{uid[:60]}.png"
                plot_energy_with_boundary(wav_path_str, head_boundary, result, str(plot_path))

        except Exception as e:
            stats["fail"] += 1
            results.append({"uid": uid, "status": "fail", "reason": str(e)})

        if (idx + 1) % 200 == 0 or idx + 1 == len(entries):
            print(f"  进度: {idx + 1}/{len(entries)}")

    # 3. 保存结果
    result_path = output_dir / "verify_result.jsonl"
    with open(result_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 4. 统计报告
    total_verified = stats["pass"] + stats["suspect"] + stats["fail"]
    print("\n" + "=" * 60)
    print("能量校验结果")
    print("=" * 60)
    print(f"总校验条数:     {total_verified}")
    print(f"通过 (pass):    {stats['pass']} ({stats['pass']/max(total_verified,1)*100:.1f}%)")
    print(f"可疑 (suspect): {stats['suspect']} ({stats['suspect']/max(total_verified,1)*100:.1f}%)")
    print(f"失败 (fail):    {stats['fail']} ({stats['fail']/max(total_verified,1)*100:.1f}%)")
    print(f"文件未找到:     {stats['file_not_found']}")

    # 偏差统计
    offsets = [r["offset_ms"] for r in results if "offset_ms" in r]
    if offsets:
        print(f"\n边界偏差统计 (ms):")
        print(f"  平均: {np.mean(offsets):.1f}")
        print(f"  中位数: {np.median(offsets):.1f}")
        print(f"  90分位: {np.percentile(offsets, 90):.1f}")
        print(f"  最大: {max(offsets):.1f}")

    # 能量下降统计
    drops = [r["energy_drop_db"] for r in results if "energy_drop_db" in r]
    if drops:
        print(f"\n能量下降统计 (dB):")
        print(f"  平均: {np.mean(drops):.1f}")
        print(f"  中位数: {np.median(drops):.1f}")
        print(f"  <3dB (可能不准): {sum(1 for d in drops if d < 3)}")

    print(f"\n校验结果: {result_path}")
    if args.plot:
        print(f"可视化图片: {plot_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
