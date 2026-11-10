"""CLI script for audio metadata preprocessing.

从增强后的 FunASR jsonl 读取每条数据的 source wav，计算音频元数据（真实时长、
SNR、能量特征），写入一个新的带元数据的 jsonl 文件。下游 dataset_analyzer
可以基于这些元数据做 SNR 过滤、场景分布分析、语速分析。

注意：scene_type 已在增强阶段由 NoiseAugmentor 自动标注（babble→background_voice
等），此脚本仅在 scene_type 缺失时做基于 SNR 的粗略推断。

Usage:
    # 预处理增强后的 train 数据
    python scripts/preprocess_metadata.py --input data/augmented/train_aug.jsonl --output data/augmented/train_aug_meta.jsonl

    # 预处理 train/val/test 全部（默认读取增强后数据）
    python scripts/preprocess_metadata.py --all

    # 自定义 SNR 计算方法 (默认 vad)
    python scripts/preprocess_metadata.py --input data/augmented/train_aug.jsonl --output data/augmented/train_aug_meta.jsonl --snr-method vad

输出 jsonl 格式在原始字段基础上增加：
    - duration: float 真实音频时长（秒）
    - snr: float 信噪比估计（dB，基于 VAD 或静音段检测）
    - energy_mean: float 平均能量（RMS）
    - energy_std: float 能量标准差
    - zero_crossing_rate: float 过零率（粗略语速代理）
    - scene_type: str 保留增强时标注的值，仅缺失时做推断

Requirements: 数据前置优化 — SNR 过滤、场景分布分析的基础
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from scripts.core.config_utils import load_config, read_jsonl

logger = logging.getLogger(__name__)

_FRAME_LEN_MS = 25  # 25ms 帧长
_FRAME_SHIFT_MS = 10  # 10ms 帧移


def _load_audio(wav_path: str):
    """加载音频，返回 (samples, sample_rate) 或 (None, None) 失败时。"""
    try:
        import soundfile as sf
        import numpy as np
        samples, sr = sf.read(wav_path)
        if len(samples.shape) > 1:
            samples = samples[:, 0]
        return samples.astype(np.float32), int(sr)
    except Exception as e:
        logger.debug("Failed to load %s: %s", wav_path, e)
        return None, None


def _estimate_snr_vad(samples, sr: int) -> float:
    """基于 VAD 的 SNR 估计。

    将音频切成短帧，按能量分布把前 30% 低能量帧视为"噪声段"，后 30% 高能量
    帧视为"语音段"，用两段的 RMS 比值计算 SNR。对大多数语音数据（语音占优）
    这个启发式足够用了。

    返回 dB 值；失败返回 NaN。
    """
    import numpy as np

    if samples is None or len(samples) == 0:
        return float("nan")

    frame_len = int(sr * _FRAME_LEN_MS / 1000)
    frame_shift = int(sr * _FRAME_SHIFT_MS / 1000)
    if frame_len <= 0 or frame_shift <= 0:
        return float("nan")

    # 切帧，计算每帧能量
    energies = []
    for start in range(0, len(samples) - frame_len + 1, frame_shift):
        frame = samples[start : start + frame_len]
        energy = float(np.sqrt(np.mean(frame ** 2) + 1e-12))
        energies.append(energy)
    if not energies:
        return float("nan")

    energies = np.array(energies)
    # 排序后：低 30% 视为噪声帧，高 30% 视为语音帧
    sorted_e = np.sort(energies)
    n = len(sorted_e)
    noise_rms = float(np.mean(sorted_e[: max(1, n // 3)]))
    speech_rms = float(np.mean(sorted_e[-max(1, n // 3) :]))

    if noise_rms <= 1e-10:
        return 60.0  # 几乎无噪声，上限
    snr = 20.0 * float(np.log10(speech_rms / noise_rms))
    return round(snr, 2)


def _compute_energy_stats(samples, sr: int) -> tuple[float, float, float]:
    """返回 (energy_mean, energy_std, zero_crossing_rate)。"""
    import numpy as np

    if samples is None or len(samples) == 0:
        return (float("nan"), float("nan"), float("nan"))

    rms = float(np.sqrt(np.mean(samples ** 2) + 1e-12))
    std = float(np.std(samples))
    # zero crossing rate = 符号变化次数 / 总样本数
    zcr = float(np.mean(np.diff(np.sign(samples)) != 0))
    return (round(rms, 5), round(std, 5), round(zcr, 5))


def _classify_scene(snr: float, zcr: float) -> str:
    """基于 SNR + 过零率的简单场景分类启发式。

    这个分类器精度一般，但比没有强。高精度需要训练专门的场景分类器
    （MFCC + GMM 或深度模型），超出当前范围。

    规则：
    - SNR > 20 dB 且 zcr 正常（0.04-0.12）→ clean
    - SNR 10-20 dB → background_noise
    - SNR < 10 dB 且 zcr 较高（> 0.10，背景人声的高频特征）→ background_voice
    - SNR < 10 dB 且 zcr 正常/偏低 → background_noise
    - 其他 → unknown
    """
    import math

    if math.isnan(snr) or math.isnan(zcr):
        return "unknown"

    if snr > 20 and 0.04 <= zcr <= 0.12:
        return "clean"
    if snr < 10:
        if zcr > 0.10:
            return "background_voice"
        return "background_noise"
    return "background_noise"


def _process_entry(entry: dict, snr_method: str) -> dict:
    """处理单条 entry，追加元数据。保留增强时已标注的 scene_type。"""
    wav_path = entry.get("source", "")
    out = dict(entry)

    samples, sr = _load_audio(wav_path)
    if samples is None:
        out.update({
            "duration": None,
            "snr": None,
            "energy_mean": None,
            "energy_std": None,
            "zero_crossing_rate": None,
            "metadata_error": "audio_load_failed",
        })
        out.setdefault("scene_type", "unknown")
        return out

    duration = round(len(samples) / sr, 3)
    if snr_method == "vad":
        snr = _estimate_snr_vad(samples, sr)
    else:
        # energy_based (default) — same as VAD in current impl
        snr = _estimate_snr_vad(samples, sr)
    emean, estd, zcr = _compute_energy_stats(samples, sr)

    out.update({
        "duration": duration,
        "snr": snr,
        "energy_mean": emean,
        "energy_std": estd,
        "zero_crossing_rate": zcr,
    })
    # 保留增强时已标注的 scene_type；仅在缺失时做粗略推断
    if "scene_type" not in out or not out["scene_type"]:
        out["scene_type"] = _classify_scene(snr, zcr)
    return out


def _process_file(
    input_path: str, output_path: str, snr_method: str
) -> dict:
    """处理一个 jsonl 文件，返回统计信息。"""
    entries = read_jsonl(input_path)
    logger.info("Processing %d entries from %s", len(entries), input_path)

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_counts = {"clean": 0, "background_voice": 0, "background_noise": 0, "unknown": 0}
    snr_buckets = {"<0dB": 0, "0-10dB": 0, "10-20dB": 0, "20-30dB": 0, "30+dB": 0, "unknown": 0}
    failed = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for i, entry in enumerate(entries):
            processed = _process_entry(entry, snr_method)
            f.write(json.dumps(processed, ensure_ascii=False) + "\n")

            scene = processed.get("scene_type", "unknown")
            scene_counts[scene] = scene_counts.get(scene, 0) + 1

            snr = processed.get("snr")
            if snr is None:
                snr_buckets["unknown"] += 1
                failed += 1
            elif snr < 0:
                snr_buckets["<0dB"] += 1
            elif snr < 10:
                snr_buckets["0-10dB"] += 1
            elif snr < 20:
                snr_buckets["10-20dB"] += 1
            elif snr < 30:
                snr_buckets["20-30dB"] += 1
            else:
                snr_buckets["30+dB"] += 1

            if (i + 1) % 100 == 0:
                logger.info("  progress %d/%d", i + 1, len(entries))

    return {
        "input_path": input_path,
        "output_path": output_path,
        "total_entries": len(entries),
        "failed_load": failed,
        "scene_distribution": scene_counts,
        "snr_distribution": snr_buckets,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Precompute audio metadata (duration/SNR/scene_type) for FunASR jsonl."
    )
    parser.add_argument("--input", default=None, help="Path to input jsonl.")
    parser.add_argument("--output", default=None, help="Path to output jsonl with metadata.")
    parser.add_argument(
        "--all", action="store_true",
        help="Process train/val/test using config.data paths. Output to *_meta.jsonl.",
    )
    parser.add_argument(
        "--snr-method", choices=["vad", "energy_based"], default="vad",
        help="SNR estimation method (default: vad).",
    )
    parser.add_argument("--config", default=None, help="Path to config YAML.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.all:
        config = load_config(args.config)
        data_cfg = config.get("data", {})
        # 分析对象是增强后的数据（训练/推理实际使用的）
        splits = [
            ("train", data_cfg.get("augmented_train_jsonl", data_cfg.get("train_jsonl"))),
            ("val", data_cfg.get("augmented_val_jsonl", data_cfg.get("val_jsonl"))),
            ("test", data_cfg.get("augmented_test_jsonl", data_cfg.get("test_jsonl"))),
        ]
        all_results = {}
        for name, inp in splits:
            if not inp or not Path(inp).exists():
                all_results[name] = {"status": "skipped", "reason": "missing input"}
                continue
            out = str(Path(inp).parent / f"{Path(inp).stem}_meta.jsonl")
            all_results[name] = _process_file(inp, out, args.snr_method)
        json.dump({"mode": "all", "results": all_results}, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return

    if not args.input or not args.output:
        parser.error("--input and --output required (or use --all)")

    result = _process_file(args.input, args.output, args.snr_method)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
