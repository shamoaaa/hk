#!/usr/bin/env python3
"""
短音频嵌入会议背景增强脚本。

场景：
  主说话人是短促语音（≤2秒，如唤醒词、短指令），
  背景是会议场景的远场多人语音（AliMeeting 数据集）。
  将主说话人音频随机嵌入到背景音频的某个位置，模拟真实会议干扰场景。

数据特点：
  - 主说话人：≤2秒的清晰短音频（超过2秒或不清晰的不参与增强）
  - 背景说话人：AliMeeting Train_Ali_far，8声道远场录音，每条14-42分钟
    · 16kHz 采样率，209个文件，209个说话人，12个房间
    · 活跃语音比例低（6%-31%），大量静音段
    · 需要截取4-8秒片段，优先选取含活跃语音的段落

增强流程：
  1. 扫描主说话人目录，过滤 ≤2秒 且清晰的音频（RMS 足够、无严重削波）
  2. 预处理背景音频：随机选声道，检测活跃语音段，截取3-10秒片段
  3. 将主说话人音频按指定 SNR 混入背景片段的随机位置
  4. 可选：对混合结果施加额外增强（混响、噪声等）

用法：
    python augment_short_in_meeting.py \
        --main_dir  E:/path/to/short_wavs \
        --bg_dir    F:/asr_datasets/alimeeting/Train_Ali_far/audio_dir \
        --output_dir E:/path/to/output \
        --factor 3 \
        --snr_min 5 --snr_max 20
"""

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import resample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  背景音频管理器
# ──────────────────────────────────────────────
class MeetingBgProvider:
    """从 AliMeeting 远场录音中按需截取背景片段。

    特点：
      - 懒加载：不一次性读入全部音频，按需读取
      - 活跃段优先：检测含语音的区域，优先从中截取
      - 声道随机：每次随机选一个声道，增加多样性
    """

    def __init__(self, bg_dir: str, crop_range: tuple = (4.0, 8.0),
                 target_sr: int = 16000, seed: int = 42):
        self.bg_dir = bg_dir
        self.crop_min, self.crop_max = crop_range
        self.target_sr = target_sr
        self.rng = np.random.default_rng(seed)

        # 加载或构建索引缓存
        self.bg_files, self._info_cache = self._load_or_build_index(bg_dir)

        if not self.bg_files:
            raise FileNotFoundError(f"背景目录无 wav 文件: {bg_dir}")

        logger.info("背景音频: %d 个文件 (来自 %s)", len(self.bg_files), bg_dir)

    @staticmethod
    def _dir_fingerprint(bg_dir: str) -> str:
        """根据目录下 wav 文件名列表生成指纹，用于判断缓存是否过期。"""
        wav_names = sorted(
            p.name for p in Path(bg_dir).rglob("*.wav")
        )
        content = "\n".join(wav_names).encode("utf-8")
        return hashlib.md5(content).hexdigest()

    @staticmethod
    def _index_path(bg_dir: str) -> str:
        """索引文件路径：放在背景目录同级，名为 .bg_index.json。"""
        return os.path.join(bg_dir, ".bg_index.json")

    def _load_or_build_index(self, bg_dir: str):
        """加载已有索引或扫描目录构建新索引。

        索引文件 (.bg_index.json) 包含：
          - fingerprint: 文件列表的 MD5，用于检测目录变化
          - files: [{path, frames, samplerate, channels}, ...]

        Returns:
            (bg_files, info_cache): 文件路径列表和信息缓存字典。
        """
        index_file = self._index_path(bg_dir)
        current_fp = self._dir_fingerprint(bg_dir)

        # 尝试加载已有索引
        if os.path.isfile(index_file):
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    index = json.load(f)
                if index.get("fingerprint") == current_fp:
                    bg_files = []
                    info_cache = {}
                    for item in index["files"]:
                        p = item["path"]
                        bg_files.append(p)
                        info_cache[p] = (item["frames"], item["samplerate"], item["channels"])
                    logger.info("从缓存加载背景索引: %d 个文件 (%s)", len(bg_files), index_file)
                    return bg_files, info_cache
                else:
                    logger.info("背景目录已变化，重新扫描...")
            except Exception as e:
                logger.warning("索引文件损坏，重新扫描: %s", e)

        # 扫描并构建索引
        logger.info("首次扫描背景目录，构建索引...")
        t0 = time.time()
        bg_files = sorted(str(p) for p in Path(bg_dir).rglob("*.wav"))
        info_cache = {}
        file_entries = []

        for wp in bg_files:
            try:
                info = sf.info(wp)
                info_cache[wp] = (info.frames, info.samplerate, info.channels)
                file_entries.append({
                    "path": wp,
                    "frames": info.frames,
                    "samplerate": info.samplerate,
                    "channels": info.channels,
                })
            except Exception as e:
                logger.warning("读取文件信息失败，跳过: %s (%s)", wp, e)

        # 保存索引
        index = {"fingerprint": current_fp, "files": file_entries}
        try:
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
            logger.info("索引已保存: %s (%.1f 秒)", index_file, time.time() - t0)
        except Exception as e:
            logger.warning("保存索引失败（不影响运行）: %s", e)

        bg_files = [item["path"] for item in file_entries]
        return bg_files, info_cache

    def _get_info(self, path: str):
        """获取文件基本信息（从索引缓存读取，未命中时回退到 sf.info）。"""
        if path not in self._info_cache:
            info = sf.info(path)
            self._info_cache[path] = (info.frames, info.samplerate, info.channels)
        return self._info_cache[path]

    def get_crop(self, prefer_active: bool = True) -> tuple:
        """截取一段背景音频。

        Args:
            prefer_active: 是否优先选取含活跃语音的段落。

        Returns:
            (audio_mono, sr): 单声道音频和采样率。
        """
        # 随机选一个背景文件
        bg_path = self.rng.choice(self.bg_files)
        n_frames, sr, n_channels = self._get_info(bg_path)

        # 随机截取时长（3-10秒，均匀分布）
        crop_sec = self.rng.uniform(self.crop_min, self.crop_max)
        crop_samples = int(crop_sec * sr)

        if crop_samples >= n_frames:
            # 文件太短（不太可能，AliMeeting 都很长），直接读全部
            audio, sr = sf.read(bg_path, dtype="float64")
        else:
            if prefer_active:
                # 尝试找活跃语音段：随机采样几个候选位置，选能量最高的
                candidates = []
                n_tries = 5
                for _ in range(n_tries):
                    start = int(self.rng.integers(0, n_frames - crop_samples))
                    audio_chunk, _ = sf.read(
                        bg_path, start=start, frames=crop_samples, dtype="float64"
                    )
                    # 随机选一个声道计算能量
                    ch = int(self.rng.integers(0, n_channels))
                    if audio_chunk.ndim > 1:
                        mono = audio_chunk[:, ch]
                    else:
                        mono = audio_chunk
                    energy = np.sqrt(np.mean(mono ** 2))
                    candidates.append((start, ch, energy))

                # 选能量最高的候选（更可能包含语音）
                best = max(candidates, key=lambda x: x[2])
                start, ch = best[0], best[1]

                audio, _ = sf.read(
                    bg_path, start=start, frames=crop_samples, dtype="float64"
                )
                if audio.ndim > 1:
                    audio = audio[:, ch]
            else:
                # 纯随机截取
                start = int(self.rng.integers(0, n_frames - crop_samples))
                audio, _ = sf.read(
                    bg_path, start=start, frames=crop_samples, dtype="float64"
                )
                if audio.ndim > 1:
                    ch = int(self.rng.integers(0, n_channels))
                    audio = audio[:, ch]

        # 重采样到目标采样率（如果不一致）
        if sr != self.target_sr:
            new_len = int(len(audio) * self.target_sr / sr)
            audio = resample(audio, new_len).astype(np.float64)
            sr = self.target_sr

        return audio, sr


# ──────────────────────────────────────────────
#  核心混合函数
# ──────────────────────────────────────────────
def mix_main_into_bg(
    main_audio: np.ndarray,
    bg_audio: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple:
    """将主说话人音频混入背景音频的随机位置。

    Args:
        main_audio: 主说话人音频（短，≤2秒）。
        bg_audio: 背景音频（4-8秒）。
        snr_db: 主说话人相对背景的信噪比（dB）。
                正值 = 主说话人更响（推荐 ≥3dB 以保证主说话人突出）。
        rng: 随机数生成器。

    Returns:
        (mixed, offset, main_len): 混合音频、主说话人插入位置、主说话人长度。
    """
    bg_len = len(bg_audio)
    main_len = len(main_audio)

    if main_len >= bg_len:
        main_audio = main_audio[:bg_len]
        main_len = bg_len

    # 随机插入位置：确保主音频完全在背景范围内
    max_offset = bg_len - main_len
    offset = int(rng.integers(0, max_offset + 1))

    # 计算增益：以背景为参考，按 SNR 调整主说话人音量
    bg_power = np.mean(bg_audio ** 2)
    main_power = np.mean(main_audio ** 2)

    if main_power == 0 or bg_power == 0:
        mixed = bg_audio.copy()
        mixed[offset:offset + main_len] += main_audio
        return mixed, offset, main_len

    # SNR = 10 * log10(main_power_target / bg_power)
    target_main_power = bg_power * (10 ** (snr_db / 10))
    gain = np.sqrt(target_main_power / main_power)

    mixed = bg_audio.copy()
    mixed[offset:offset + main_len] += main_audio * gain

    return mixed, offset, main_len


# ──────────────────────────────────────────────
#  主流程
# ──────────────────────────────────────────────
def scan_short_wavs(input_dir: str, max_duration: float = 2.0,
                    min_rms: float = 0.02, max_clip_ratio: float = 0.05) -> list:
    """扫描目录下所有 ≤ max_duration 秒且清晰的 wav 文件（带索引缓存）。

    清晰度判定：
      - RMS ≥ min_rms：排除近似静音的录音
      - 削波比例 ≤ max_clip_ratio：排除严重削波（peak ≥ 0.99 的采样点占比）

    首次扫描后将文件列表和时长/清晰度信息保存到 input_dir/.main_index.json，
    后续运行如果目录未变化则直接加载，跳过逐文件读取。

    Returns:
        [(path, duration_sec), ...]
    """
    index_file = os.path.join(input_dir, ".main_index.json")

    # 生成目录指纹（基于 wav 文件名列表）
    all_wavs = sorted(str(p) for p in Path(input_dir).rglob("*.wav"))
    wav_names = sorted(Path(p).name for p in all_wavs)
    current_fp = hashlib.md5("\n".join(wav_names).encode("utf-8")).hexdigest()

    # 尝试加载已有索引
    if os.path.isfile(index_file):
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                index = json.load(f)
            if index.get("fingerprint") == current_fp:
                entries = index["files"]  # [{path, duration, rms, clip_ratio}, ...]
                short_wavs = []
                skipped_dur = 0
                skipped_clarity = 0
                for item in entries:
                    if item["duration"] > max_duration:
                        skipped_dur += 1
                    elif item.get("rms", 1.0) < min_rms:
                        skipped_clarity += 1
                    elif item.get("clip_ratio", 0.0) > max_clip_ratio:
                        skipped_clarity += 1
                    else:
                        short_wavs.append((item["path"], item["duration"]))
                logger.info(
                    "从缓存加载主说话人索引: %d 个 wav → %d 个合格，"
                    "跳过 %d 超时长 + %d 不清晰",
                    len(entries), len(short_wavs), skipped_dur, skipped_clarity,
                )
                if short_wavs:
                    durs = [d for _, d in short_wavs]
                    logger.info(
                        "  时长分布: min=%.3fs, max=%.3fs, mean=%.3fs",
                        min(durs), max(durs), sum(durs) / len(durs),
                    )
                return short_wavs
            else:
                logger.info("主说话人目录已变化，重新扫描...")
        except Exception as e:
            logger.warning("主说话人索引文件损坏，重新扫描: %s", e)

    # 扫描并构建索引（含清晰度检测）
    logger.info("首次扫描主说话人目录，构建索引（含清晰度检测）...")
    t0 = time.time()
    file_entries = []
    short_wavs = []
    skipped_dur = 0
    skipped_clarity = 0

    for wp in all_wavs:
        try:
            info = sf.info(wp)
            dur = info.duration

            # 读取音频计算清晰度指标
            audio_tmp, _ = sf.read(wp, dtype="float64")
            if audio_tmp.ndim > 1:
                audio_tmp = audio_tmp[:, 0]
            rms = float(np.sqrt(np.mean(audio_tmp ** 2)))
            clip_ratio = float(np.mean(np.abs(audio_tmp) >= 0.99))

            file_entries.append({
                "path": wp, "duration": dur,
                "rms": rms, "clip_ratio": clip_ratio,
            })

            if dur > max_duration:
                skipped_dur += 1
            elif rms < min_rms:
                skipped_clarity += 1
            elif clip_ratio > max_clip_ratio:
                skipped_clarity += 1
            else:
                short_wavs.append((wp, dur))
        except Exception as e:
            logger.warning("读取失败，跳过: %s (%s)", wp, e)

    # 保存索引
    index = {"fingerprint": current_fp, "files": file_entries}
    try:
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        logger.info("主说话人索引已保存: %s (%.1f 秒)", index_file, time.time() - t0)
    except Exception as e:
        logger.warning("保存索引失败（不影响运行）: %s", e)

    logger.info(
        "主说话人: %d 个 wav → %d 个合格，跳过 %d 超时长 + %d 不清晰",
        len(all_wavs), len(short_wavs), skipped_dur, skipped_clarity,
    )

    if short_wavs:
        durs = [d for _, d in short_wavs]
        logger.info(
            "  时长分布: min=%.3fs, max=%.3fs, mean=%.3fs",
            min(durs), max(durs), sum(durs) / len(durs),
        )

    return short_wavs


def run_augment(
    main_dir: str,
    bg_dir: str,
    output_dir: str,
    factor: int = 3,
    snr_min: float = 5.0,
    snr_max: float = 20.0,
    main_gain_db: float = 0.0,
    bg_gain_db: float = 0.0,
    max_main_duration: float = 2.0,
    min_rms: float = 0.02,
    max_clip_ratio: float = 0.05,
    crop_range: tuple = (4.0, 8.0),
    seed: int = 42,
    add_reverb: bool = False,
    reverb_rt60_range: tuple = (0.1, 0.4),
    add_noise: bool = False,
    noise_snr_range: tuple = (30, 45),
    normalize: bool = True,
    normalize_mode: str = "main_region",
    target_peak: float = 0.6,
    text_file: str = None,
    scp_file: str = None,
    resume: bool = True,
):
    """执行增强主流程。

    Args:
        main_dir: 主说话人 wav 目录。
        bg_dir: 背景说话人 wav 目录（AliMeeting）。
        output_dir: 输出目录。
        factor: 每条主音频生成几条增强样本。
        snr_min: 主说话人 SNR 下限（dB）。
        snr_max: 主说话人 SNR 上限（dB）。
        max_main_duration: 主说话人最大时长（秒），超过则跳过。
        crop_range: 背景截取时长范围（秒）。
        seed: 随机种子。
        add_reverb: 是否对混合结果加混响。
        reverb_rt60_range: 混响 RT60 范围。
        add_noise: 是否加高斯底噪。
        noise_snr_range: 底噪 SNR 范围。
        normalize: 是否峰值归一化。
        text_file: 原始标签文件路径（格式: key 文本），提供后生成增强标签。
        scp_file: 原始 wav.scp 文件路径（格式: key /path/to/wav），
                  用于将 wav 路径映射到 key。如不提供则用文件名 stem 作为 key。
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # SNR 校验：正值保证主说话人比背景响
    if snr_min < 0:
        logger.warning(
            "snr_min=%.1f < 0，背景可能比主说话人响。"
            "如需保证主说话人突出，请设置 snr_min >= 0（推荐 ≥3）。",
            snr_min,
        )

    # 加载标签映射（如果提供了 text_file）
    label_map = {}       # {key: text_label}
    path_to_key = {}     # {wav_abs_path: key}
    has_labels = False

    if text_file:
        if not os.path.isfile(text_file):
            logger.error("标签文件不存在: %s", text_file)
            return
        with open(text_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                key = parts[0]
                txt = parts[1] if len(parts) > 1 else ""
                label_map[key] = txt
        logger.info("加载标签: %d 条 (from %s)", len(label_map), text_file)

        # 从 scp 文件建立 wav_path → key 的映射
        if scp_file and os.path.isfile(scp_file):
            with open(scp_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        key, wav = parts
                        # 统一用绝对路径做 key
                        path_to_key[os.path.abspath(wav)] = key
            logger.info("加载 wav.scp: %d 条 (from %s)", len(path_to_key), scp_file)

        has_labels = len(label_map) > 0

    # 1. 扫描主说话人
    short_wavs = scan_short_wavs(main_dir, max_duration=max_main_duration,
                                  min_rms=min_rms, max_clip_ratio=max_clip_ratio)
    if not short_wavs:
        logger.error("没有符合条件的主说话人音频，退出。")
        return

    # 2. 初始化背景提供器
    bg_provider = MeetingBgProvider(
        bg_dir=bg_dir,
        crop_range=crop_range,
        target_sr=16000,
        seed=seed,
    )

    # 3. 增强循环
    total = len(short_wavs) * factor
    processed = 0
    skipped = 0
    log_interval = max(1, total // 20)
    t_start = time.time()

    # wav 文件输出到 output_dir/wav/ 子目录
    wav_out_dir = os.path.join(output_dir, "wav")
    os.makedirs(wav_out_dir, exist_ok=True)

    # 检查已有文件（断点续跑）
    existing_files = set()
    if resume and os.path.isdir(wav_out_dir):
        existing_files = {f for f in os.listdir(wav_out_dir) if f.endswith(".wav")}
    if existing_files:
        logger.info("检测到输出目录已有 %d 个文件，将跳过（断点续跑）", len(existing_files))

    skipped_exist = 0

    # 打开标签输出文件（如果有标签）
    f_text = None
    f_scp = None
    out_text_path = os.path.join(output_dir, "text")
    out_scp_path = os.path.join(output_dir, "wav.scp")
    if has_labels:
        f_text = open(out_text_path, "w", encoding="utf-8")
        f_scp = open(out_scp_path, "w", encoding="utf-8")
    label_count = 0
    label_miss = 0

    for file_idx, (wav_path, dur) in enumerate(short_wavs):
        stem = Path(wav_path).stem

        # 查找该文件对应的 key 和标签
        orig_key = None
        orig_label = None
        if has_labels:
            # 优先从 scp 映射查找
            abs_path = os.path.abspath(wav_path)
            if abs_path in path_to_key:
                orig_key = path_to_key[abs_path]
            else:
                # 回退：用文件名 stem 作为 key
                orig_key = stem
            orig_label = label_map.get(orig_key)

        # 读取主说话人音频
        try:
            main_audio, main_sr = sf.read(wav_path, dtype="float64")
        except Exception as e:
            logger.warning("读取主音频失败: %s (%s)", wav_path, e)
            skipped += 1
            continue

        # 单声道处理
        if main_audio.ndim > 1:
            main_audio = main_audio[:, 0]

        for aug_idx in range(factor):
            out_name = f"{stem}_meetbg{aug_idx}.wav"

            # 跳过已存在
            if out_name in existing_files:
                skipped_exist += 1
                continue

            # 为每条增强生成确定性种子
            aug_rng = np.random.default_rng(seed + file_idx * factor + aug_idx)

            # 截取背景片段
            bg_audio, bg_sr = bg_provider.get_crop(prefer_active=True)

            # 独立增益调整（在 SNR 混合之前施加）
            if bg_gain_db != 0.0:
                bg_audio = bg_audio * (10 ** (bg_gain_db / 20))

            # 重采样主音频到背景采样率（如果不一致）
            if main_sr != bg_sr:
                new_len = int(len(main_audio) * bg_sr / main_sr)
                main_resampled = resample(main_audio, new_len).astype(np.float64)
            else:
                main_resampled = main_audio.copy()

            if main_gain_db != 0.0:
                main_resampled = main_resampled * (10 ** (main_gain_db / 20))

            # 随机 SNR（均匀分布）
            snr_db = aug_rng.uniform(snr_min, snr_max)

            # 混合
            mixed, main_offset, main_len = mix_main_into_bg(
                main_resampled, bg_audio, snr_db, aug_rng,
            )

            # 可选后处理：混响
            if add_reverb:
                rt60 = aug_rng.uniform(reverb_rt60_range[0], reverb_rt60_range[1])
                mixed = _apply_simple_reverb(mixed, bg_sr, rt60, aug_rng)

            # 可选后处理：高斯底噪
            if add_noise:
                noise_snr = aug_rng.uniform(noise_snr_range[0], noise_snr_range[1])
                mixed = _add_gaussian_noise(mixed, noise_snr, aug_rng)

            # 峰值归一化：以主说话人所在区域的峰值为参考
            # 这样背景部分不会被拉高，主说话人始终清晰突出
            if normalize:
                if normalize_mode == "main_region":
                    main_region = mixed[main_offset:main_offset + main_len]
                    peak = np.max(np.abs(main_region))
                else:
                    peak = np.max(np.abs(mixed))
                if peak > 0:
                    mixed = mixed / peak * target_peak

            # 检查有效性
            if np.any(np.isnan(mixed)) or np.any(np.isinf(mixed)):
                logger.warning("增强结果无效（NaN/Inf），跳过: %s aug_%d", stem, aug_idx)
                skipped += 1
                continue

            # 写入
            out_path = os.path.join(wav_out_dir, out_name)
            sf.write(out_path, mixed, bg_sr)

            # 写入标签
            if has_labels and orig_label is not None:
                aug_key = f"{stem}_meetbg{aug_idx}"
                abs_out = os.path.abspath(out_path)
                f_text.write(f"{aug_key} {orig_label}\n")
                f_scp.write(f"{aug_key} {abs_out}\n")
                label_count += 1
            elif has_labels and orig_label is None:
                label_miss += 1

            processed += 1
            done = processed + skipped_exist
            if processed % log_interval == 0:
                elapsed = time.time() - t_start
                speed = processed / elapsed if elapsed > 0 else 0
                remaining = total - done
                eta = remaining / speed if speed > 0 else 0
                logger.info(
                    "  %d/%d (%.1f%%) | 跳过 %d 已有 | %.1f 条/秒 | ETA %.0f 秒",
                    done, total, 100 * done / total, skipped_exist, speed, eta,
                )

    elapsed = time.time() - t_start
    logger.info(
        "增强完成: %d 条主音频 × %d 倍 → 新增 %d 条 + 跳过 %d 已有 + %d 失败，耗时 %.1f 秒",
        len(short_wavs), factor, processed, skipped_exist, skipped, elapsed,
    )

    # 关闭标签文件并输出统计
    if f_text:
        f_text.close()
        f_scp.close()
        logger.info(
            "标签输出: %d 条写入 text + wav.scp（%d 条未匹配到标签）",
            label_count, label_miss,
        )
        logger.info("  text: %s", out_text_path)
        logger.info("  wav.scp: %s", out_scp_path)


# ──────────────────────────────────────────────
#  辅助增强函数
# ──────────────────────────────────────────────
def _apply_simple_reverb(
    audio: np.ndarray, sr: int, rt60: float, rng: np.random.Generator
) -> np.ndarray:
    """简单合成混响（直达声 + 早期反射 + 晚期尾巴）。"""
    from scipy.signal import fftconvolve

    ir_length = int(sr * rt60)
    if ir_length <= 0:
        return audio.copy()

    ir = np.zeros(ir_length)
    ir[0] = 1.0

    # 早期反射
    early_end = min(int(0.05 * sr), ir_length)
    if early_end > int(0.005 * sr):
        n_ref = int(rng.integers(3, 8))
        for pos in rng.integers(int(0.005 * sr), early_end, size=n_ref):
            ir[pos] += rng.choice([-1, 1]) * rng.uniform(0.1, 0.4)

    # 晚期混响
    late_start = early_end
    if late_start < ir_length:
        late_len = ir_length - late_start
        t = np.arange(late_len) / sr
        decay = 0.08 * np.exp(-6.9 * t / rt60)
        ir[late_start:] = rng.normal(0, 1, late_len) * decay

    ir /= np.max(np.abs(ir)) + 1e-12

    wet = fftconvolve(audio, ir, mode="full")[:len(audio)]
    audio_peak = np.max(np.abs(audio))
    wet_peak = np.max(np.abs(wet))
    if wet_peak > 0 and audio_peak > 0:
        wet = wet / wet_peak * audio_peak
    return wet.astype(audio.dtype)


def _add_gaussian_noise(
    audio: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """添加高斯白噪声。"""
    signal_power = np.mean(audio ** 2)
    if signal_power == 0:
        return audio.copy()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), len(audio))
    return audio + noise


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────
def main():
    import yaml

    parser = argparse.ArgumentParser(
        description="短音频嵌入会议背景增强",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 使用 YAML 配置文件（推荐）
  python augment_short_in_meeting.py --config augment_config_meeting_bg.yaml

  # 命令行参数覆盖配置文件中的值
  python augment_short_in_meeting.py --config augment_config_meeting_bg.yaml \\
      --main_dir E:/other/dir --factor 5

  # 纯命令行模式（不使用配置文件）
  python augment_short_in_meeting.py \\
      --main_dir  E:/path/to/short_wavs \\
      --bg_dir    F:/asr_datasets/alimeeting/Train_Ali_far/audio_dir \\
      --output_dir E:/path/to/output
        """,
    )

    parser.add_argument("--config", type=str, default=None,
                        help="YAML 配置文件路径（命令行参数可覆盖配置文件中的值）")

    # 输入输出
    parser.add_argument("--main_dir", type=str, default=None,
                        help="主说话人 wav 目录（短音频，≤2秒）")
    parser.add_argument("--bg_dir", type=str, default=None,
                        help="背景说话人 wav 目录（AliMeeting 远场录音）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出增强音频目录")

    # 增强参数
    parser.add_argument("--factor", type=int, default=None,
                        help="每条主音频生成几条增强样本")
    parser.add_argument("--snr_min", type=float, default=None)
    parser.add_argument("--snr_max", type=float, default=None)
    parser.add_argument("--max_main_duration", type=float, default=None)
    parser.add_argument("--crop_min", type=float, default=None)
    parser.add_argument("--crop_max", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--target_peak", type=float, default=None,
                        help="归一化目标峰值")

    # 标签
    parser.add_argument("--text_file", type=str, default=None)
    parser.add_argument("--scp_file", type=str, default=None)

    args = parser.parse_args()

    # 加载 YAML 配置
    cfg = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        logger.info("加载配置: %s", args.config)

    # 合并：命令行参数 > YAML 配置 > 默认值
    def get(cli_val, *yaml_keys, default=None):
        """优先取命令行值，其次从 YAML 嵌套 key 中取，最后用默认值。"""
        if cli_val is not None:
            return cli_val
        obj = cfg
        for k in yaml_keys:
            if isinstance(obj, dict) and k in obj:
                obj = obj[k]
            else:
                return default
        return obj if obj is not None else default

    main_dir = get(args.main_dir, "main_dir")
    bg_dir = get(args.bg_dir, "bg_dir",
                 default=r"F:\asr_datasets\alimeeting\Train_Ali_far\audio_dir")
    output_dir = get(args.output_dir, "output_dir")

    if not main_dir or not output_dir:
        parser.error("必须指定 --main_dir 和 --output_dir（通过命令行或配置文件）")

    factor = get(args.factor, "augment_factor", default=3)
    seed = get(args.seed, "seed", default=42)
    resume = cfg.get("resume", True)

    # 主说话人过滤
    max_main_duration = get(args.max_main_duration, "main_speaker", "max_duration", default=2.0)
    min_rms = get(None, "main_speaker", "min_rms", default=0.02)
    max_clip_ratio = get(None, "main_speaker", "max_clip_ratio", default=0.05)

    # 背景截取
    crop_min = get(args.crop_min, "background", "crop_min", default=4.0)
    crop_max = get(args.crop_max, "background", "crop_max", default=8.0)

    # 混合
    snr_min = get(args.snr_min, "mix", "snr_min", default=5.0)
    snr_max = get(args.snr_max, "mix", "snr_max", default=20.0)
    main_gain_db = get(None, "mix", "main_gain_db", default=0.0)
    bg_gain_db = get(None, "mix", "bg_gain_db", default=0.0)

    # 归一化
    norm_cfg = cfg.get("normalize", {})
    normalize = norm_cfg.get("enabled", True)
    normalize_mode = norm_cfg.get("mode", "main_region")
    target_peak = get(args.target_peak, "normalize", "target_peak", default=0.6)

    # 后处理
    post = cfg.get("post_processing", {})
    reverb_cfg = post.get("reverb", {})
    add_reverb = reverb_cfg.get("enabled", False)
    reverb_rt60_range = (reverb_cfg.get("rt60_min", 0.1), reverb_cfg.get("rt60_max", 0.4))

    noise_cfg = post.get("gaussian_noise", {})
    add_noise = noise_cfg.get("enabled", False)
    noise_snr_range = (noise_cfg.get("snr_min", 30.0), noise_cfg.get("snr_max", 45.0))

    # 标签
    text_file = get(args.text_file, "text_file", default="") or None
    scp_file = get(args.scp_file, "scp_file", default="") or None

    run_augment(
        main_dir=main_dir,
        bg_dir=bg_dir,
        output_dir=output_dir,
        factor=factor,
        snr_min=snr_min,
        snr_max=snr_max,
        main_gain_db=main_gain_db,
        bg_gain_db=bg_gain_db,
        max_main_duration=max_main_duration,
        min_rms=min_rms,
        max_clip_ratio=max_clip_ratio,
        crop_range=(crop_min, crop_max),
        seed=seed,
        add_reverb=add_reverb,
        reverb_rt60_range=reverb_rt60_range,
        add_noise=add_noise,
        noise_snr_range=noise_snr_range,
        normalize=normalize,
        normalize_mode=normalize_mode,
        target_peak=target_peak,
        text_file=text_file,
        scp_file=scp_file,
        resume=resume,
    )


if __name__ == "__main__":
    main()
