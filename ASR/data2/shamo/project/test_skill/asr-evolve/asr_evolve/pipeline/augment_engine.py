#!/usr/bin/env python3
"""
音频数据增强引擎。

提供 AudioAugmentor 类，包含所有增强策略的实现。
由调用脚本（augment_manifest.py / augment_dir.py）导入使用。

已实现的增强策略：
  - gaussian_noise:    高斯白噪声叠加
  - speaker_mix:       多说话人混合（近场+远场）
  - reverb:            合成 RIR 房间混响
  - real_rir:          真实 RIR 卷积（需提供 RIR 文件目录）
  - speed_perturb:     速度扰动（重采样）
  - volume_perturb:    音量扰动
  - noise_from_file:   环境噪声叠加（需提供噪声文件目录）
  - pitch_shift:       音调扰动（保持时长不变）
  - eq_random:         随机均衡器（频响随机化）
  - random_crop_pad:   随机裁剪/填充
  - time_stretch:      时间拉伸（phase vocoder，保持音调不变）
  - random_silence:    随机静音插入
  - codec_simulate:    编解码失真模拟（G.711 μ-law / A-law）
  - bandpass_filter:   带通滤波（模拟电话信道等）
  - clipping:          削波失真模拟
  - concatenate:       多条短音频拼接
"""

import logging
import os

import numpy as np
import yaml
from scipy.signal import fftconvolve, resample, butter, sosfilt

logger = logging.getLogger(__name__)


class AudioAugmentor:
    """音频数据增强器，支持多种增强策略和链式组合。"""

    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.seed = self.config.get("seed", 42)
        self.num_workers = self.config.get("num_workers", 4)
        self.augment_factor = self.config.get("augment_factor", 2)
        self.strategies = self.config.get("strategies", {})
        self.chain = self.config.get("chain", [])
        self.folder_chains = self.config.get("folder_chains", {})
        self.resume = self.config.get("resume", True)
        self.rng = np.random.default_rng(self.seed)

    @staticmethod
    def _sample_normal_clipped(rng: np.random.Generator, low: float, high: float) -> float:
        """正态分布采样并 clip 到 [low, high]。

        以 (low+high)/2 为均值，(high-low)/4 为标准差（约 95% 落在范围内），
        clip 防止越界。比均匀分布更贴近真实场景（典型值集中，极端值少见）。
        """
        mu = (low + high) / 2.0
        sigma = (high - low) / 4.0
        if sigma <= 0:
            return mu
        return float(np.clip(rng.normal(mu, sigma), low, high))

    def add_gaussian_noise(self, audio: np.ndarray, sr: int, snr_db: float) -> np.ndarray:
        """添加高斯噪声。"""
        signal_power = np.mean(audio ** 2)
        if signal_power == 0:
            return audio.copy()
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = self.rng.normal(0, np.sqrt(noise_power), len(audio)).astype(audio.dtype)
        return audio + noise

    def mix_speakers(
        self, audio: np.ndarray, sr: int, bg_audios: list, snr_db: float,
        random_insert: bool = True, rng: np.random.Generator = None,
        overlap_mode: str = "random", min_overlap: float = 0.8,
    ) -> np.ndarray:
        """多说话人混合，支持多种时间对齐模式。

        Args:
            audio: 主音频。
            sr: 采样率。
            bg_audios: 背景音频列表。
            snr_db: 信噪比。
            random_insert: 兼容旧参数，overlap_mode 优先。
            rng: 随机数生成器。
            overlap_mode: 时间对齐模式：
                "full"       — 背景音全程覆盖主音频（循环填充）
                "random"     — 随机位置插入，保证 min_overlap 重叠比例
                "mid_insert" — 主说话人说到一半插入背景声
                "bg_first"   — 先出现背景声，再是主说话人
                "bg_tail"    — 主说话人说完后出现背景声
                "bg_longer"  — 背景声比主音频长，主音频结束后背景声还在继续
                "auto"       — 随机选择以上模式之一
            min_overlap: 背景声与主音频的最小重叠比例 (0~1)，默认 0.8。
        """
        rng = rng or self.rng
        mixed = audio.copy()
        n = len(audio)

        # auto 模式：随机选一种
        if overlap_mode == "auto":
            modes = ["full", "random", "mid_insert", "bg_first", "bg_tail", "bg_longer"]
            overlap_mode = rng.choice(modes)

        for bg in bg_audios:
            bg_audio = bg.copy()
            bg_len = len(bg_audio)

            # 确保背景音频足够长以满足 min_overlap 要求
            min_bg_len = max(int(n * min_overlap), 1)
            if bg_len < min_bg_len:
                repeats = int(np.ceil(min_bg_len / bg_len))
                bg_audio = np.tile(bg_audio, repeats)
                bg_len = len(bg_audio)

            bg_placed = np.zeros(n)

            if overlap_mode == "full":
                # 全程覆盖：循环填充到主音频长度
                if bg_len < n:
                    repeats = int(np.ceil(n / bg_len))
                    bg_audio = np.tile(bg_audio, repeats)
                bg_placed[:] = bg_audio[:n]

            elif overlap_mode == "mid_insert":
                # 主说话人说到一半时插入背景声，背景声持续到结尾
                # 插入点在 [0, 20%] 之间，保证 ≥80% 重叠
                max_start = max(int(n * (1.0 - min_overlap)), 0)
                start = int(rng.integers(0, max_start + 1))
                seg_len = n - start
                if bg_len > seg_len:
                    bg_audio = bg_audio[:seg_len]
                else:
                    repeats = int(np.ceil(seg_len / bg_len))
                    bg_audio = np.tile(bg_audio, repeats)[:seg_len]
                bg_placed[start:start + len(bg_audio)] = bg_audio

            elif overlap_mode == "bg_first":
                # 背景声先出现，主说话人稍后加入
                # 背景声从头开始，长度覆盖 [min_overlap, 100%] 的主音频
                min_len = int(n * min_overlap)
                cover_len = int(rng.integers(min_len, n + 1))
                if bg_len > cover_len:
                    bg_audio = bg_audio[:cover_len]
                else:
                    repeats = int(np.ceil(cover_len / bg_len))
                    bg_audio = np.tile(bg_audio, repeats)[:cover_len]
                bg_placed[:len(bg_audio)] = bg_audio

            elif overlap_mode == "bg_tail":
                # 主说话人说完后出现背景声
                # 背景声从 [0, 20%] 处开始，持续到结尾或更长
                max_start = max(int(n * (1.0 - min_overlap)), 0)
                start = int(rng.integers(0, max_start + 1))
                seg_len = n - start
                if bg_len > seg_len:
                    bg_audio = bg_audio[:seg_len]
                else:
                    repeats = int(np.ceil(seg_len / bg_len))
                    bg_audio = np.tile(bg_audio, repeats)[:seg_len]
                bg_placed[start:start + len(bg_audio)] = bg_audio

            elif overlap_mode == "bg_longer":
                # 背景声比主音频长，从主音频之前就开始
                # 背景声从负偏移开始，主音频结束时背景声还在继续
                # 实际效果：背景声全程覆盖主音频
                if bg_len < n:
                    repeats = int(np.ceil((n * 1.3) / bg_len))
                    bg_audio = np.tile(bg_audio, repeats)
                    bg_len = len(bg_audio)
                # 随机截取一段覆盖主音频
                if bg_len > n:
                    max_offset = bg_len - n
                    offset = int(rng.integers(0, max_offset + 1))
                    bg_placed[:] = bg_audio[offset:offset + n]
                else:
                    bg_placed[:] = bg_audio[:n]

            else:
                # "random" 模式（默认）：随机位置插入，保证 min_overlap
                use_len = max(min_bg_len, min(bg_len, n))
                if bg_len > use_len:
                    # 从背景音频中随机截取
                    crop_start = int(rng.integers(0, bg_len - use_len + 1))
                    bg_audio = bg_audio[crop_start:crop_start + use_len]
                elif bg_len < use_len:
                    repeats = int(np.ceil(use_len / bg_len))
                    bg_audio = np.tile(bg_audio, repeats)[:use_len]

                # 计算允许的偏移范围，保证重叠 ≥ min_overlap
                max_offset = n - int(n * min_overlap)
                max_offset = min(max_offset, n - len(bg_audio)) if len(bg_audio) <= n else 0
                max_offset = max(max_offset, 0)
                offset = int(rng.integers(0, max_offset + 1))
                end = min(offset + len(bg_audio), n)
                bg_placed[offset:end] = bg_audio[:end - offset]

            # 按 SNR 计算增益
            signal_power = np.mean(audio ** 2)
            bg_power = np.mean(bg_placed ** 2)
            if bg_power == 0 or signal_power == 0:
                continue
            target_bg_power = signal_power / (10 ** (snr_db / 10))
            gain = np.sqrt(target_bg_power / bg_power)
            mixed = mixed + bg_placed * gain

        return mixed[:n]

    def apply_reverb(self, audio: np.ndarray, sr: int, rt60: float,
                    rng: np.random.Generator = None) -> np.ndarray:
        """房间混响模拟（改进合成 RIR：直达声 + 早期反射 + 晚期混响尾巴）。

        RIR 结构：
          [0ms]        直达声脉冲（归一化为 1.0）
          [5-50ms]     稀疏早期反射（3-8 个随机位置的衰减脉冲）
          [50ms-RT60]  密集晚期混响尾巴（指数衰减的高斯噪声）
        """
        rng = rng or self.rng
        ir_length = int(sr * rt60)
        if ir_length <= 0:
            return audio.copy()

        ir = np.zeros(ir_length)

        # 1. 直达声脉冲
        ir[0] = 1.0

        # 2. 早期反射（5-50ms，稀疏脉冲）
        early_start = int(0.005 * sr)   # 5ms
        early_end = min(int(0.05 * sr), ir_length)  # 50ms
        if early_end > early_start:
            n_reflections = int(rng.integers(3, 9))  # 3-8 个反射
            positions = rng.integers(early_start, early_end, size=n_reflections)
            for pos in positions:
                # 反射能量随时间衰减，幅度 0.1-0.5
                t_ratio = (pos - early_start) / max(early_end - early_start, 1)
                amplitude = rng.uniform(0.1, 0.5) * (1.0 - 0.7 * t_ratio)
                sign = rng.choice([-1.0, 1.0])
                ir[pos] += sign * amplitude

        # 3. 晚期混响尾巴（50ms 之后，密集高斯噪声 + 指数衰减）
        late_start = early_end
        if late_start < ir_length:
            late_len = ir_length - late_start
            t_late = np.arange(late_len) / sr
            # 衰减从早期反射末尾的能量水平开始，比直达声弱很多
            late_amplitude = 0.1
            decay = late_amplitude * np.exp(-6.9 * t_late / rt60)
            ir[late_start:] = rng.normal(0, 1, late_len) * decay

        # 归一化：保持直达声为最大值
        ir_peak = np.max(np.abs(ir))
        if ir_peak > 0:
            ir = ir / ir_peak

        # FFT 卷积
        wet = fftconvolve(audio, ir, mode="full")[:len(audio)]

        # 保持原始音量水平
        audio_peak = np.max(np.abs(audio))
        wet_peak = np.max(np.abs(wet))
        if wet_peak > 0 and audio_peak > 0:
            wet = wet / wet_peak * audio_peak
        return wet.astype(audio.dtype)

    def speed_perturb(self, audio: np.ndarray, sr: int, factor: float) -> np.ndarray:
        """速度扰动。"""
        if factor == 1.0 or len(audio) == 0:
            return audio.copy()
        new_length = round(len(audio) / factor)
        if new_length <= 0:
            return audio.copy()
        resampled = resample(audio, new_length)
        return resampled.astype(audio.dtype)

    def volume_perturb(self, audio: np.ndarray, gain_db: float) -> np.ndarray:
        """音量扰动。"""
        gain_linear = 10 ** (gain_db / 20)
        return audio * gain_linear

    def add_noise_from_file(self, audio: np.ndarray, sr: int, noise_audio: np.ndarray, snr_db: float) -> np.ndarray:
        """环境噪声叠加。"""
        noise = noise_audio.copy()
        if len(noise) < len(audio):
            repeats = int(np.ceil(len(audio) / len(noise)))
            noise = np.tile(noise, repeats)
        noise = noise[:len(audio)]

        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power == 0 or signal_power == 0:
            return audio.copy()
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        gain = np.sqrt(target_noise_power / noise_power)
        return audio + noise * gain

    def apply_real_rir(self, audio: np.ndarray, sr: int, rir: np.ndarray) -> np.ndarray:
        """真实 RIR 卷积混响。"""
        rir = rir / (np.max(np.abs(rir)) + 1e-12)
        wet = fftconvolve(audio, rir, mode="full")[:len(audio)]
        audio_peak = np.max(np.abs(audio))
        wet_peak = np.max(np.abs(wet))
        if wet_peak > 0 and audio_peak > 0:
            wet = wet / wet_peak * audio_peak
        return wet.astype(audio.dtype)

    def pitch_shift(self, audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """音调扰动（保持时长不变）。"""
        if semitones == 0 or len(audio) == 0:
            return audio.copy()
        ratio = 2 ** (semitones / 12.0)
        shifted_len = round(len(audio) / ratio)
        if shifted_len <= 0:
            return audio.copy()
        shifted = resample(audio, shifted_len)
        result = resample(shifted, len(audio))
        return result.astype(audio.dtype)

    def eq_random(self, audio: np.ndarray, sr: int, num_bands: int = 5,
                  gain_range: list = None, rng: np.random.Generator = None) -> np.ndarray:
        """随机均衡器（频响随机化）。"""
        rng = rng or self.rng
        if gain_range is None:
            gain_range = [-6, 6]
        if len(audio) == 0:
            return audio.copy()

        nyquist = sr / 2.0
        band_edges = np.linspace(80, min(nyquist - 100, 8000), num_bands + 1)
        result = np.zeros_like(audio)

        for i in range(num_bands):
            low = band_edges[i] / nyquist
            high = band_edges[i + 1] / nyquist
            low = np.clip(low, 0.001, 0.999)
            high = np.clip(high, low + 0.001, 0.999)
            try:
                sos = butter(4, [low, high], btype="band", output="sos")
                band_signal = sosfilt(sos, audio)
                gain_db = rng.uniform(gain_range[0], gain_range[1])
                gain_linear = 10 ** (gain_db / 20)
                result += band_signal * gain_linear
            except Exception:
                result += audio / num_bands
        return result.astype(audio.dtype)

    def random_crop_pad(self, audio: np.ndarray, sr: int, target_length: int = None,
                        pad_mode: str = "zero", rng: np.random.Generator = None) -> np.ndarray:
        """随机裁剪/填充。"""
        rng = rng or self.rng
        if target_length is None:
            target_length = len(audio)
        if len(audio) == 0:
            return np.zeros(target_length, dtype=audio.dtype)

        if len(audio) > target_length:
            max_start = len(audio) - target_length
            start = int(rng.integers(0, max_start + 1))
            return audio[start:start + target_length].copy()
        elif len(audio) < target_length:
            pad_total = target_length - len(audio)
            pad_left = int(rng.integers(0, pad_total + 1))
            pad_right = pad_total - pad_left
            if pad_mode == "noise":
                energy = np.mean(audio ** 2) * 0.001
                left = rng.normal(0, np.sqrt(energy), pad_left).astype(audio.dtype)
                right = rng.normal(0, np.sqrt(energy), pad_right).astype(audio.dtype)
            else:
                left = np.zeros(pad_left, dtype=audio.dtype)
                right = np.zeros(pad_right, dtype=audio.dtype)
            return np.concatenate([left, audio, right])
        else:
            return audio.copy()

    def time_stretch(self, audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
        """时间拉伸（Phase Vocoder，保持音调不变）。"""
        if rate == 1.0 or len(audio) == 0:
            return audio.copy()

        n_fft = 2048
        hop = n_fft // 4
        n_frames = 1 + (len(audio) - n_fft) // hop
        if n_frames <= 1:
            return audio.copy()

        window = np.hanning(n_fft)
        stft = np.array([
            np.fft.rfft(window * audio[i * hop:i * hop + n_fft])
            for i in range(n_frames)
        ])

        n_out_frames = int(n_frames / rate)
        if n_out_frames <= 1:
            return audio.copy()

        phase_advance = np.linspace(0, np.pi * hop, stft.shape[1], endpoint=False)
        result_stft = np.zeros((n_out_frames, stft.shape[1]), dtype=complex)
        phase_acc = np.angle(stft[0])

        for i in range(n_out_frames):
            src_idx = i * rate
            idx0 = int(src_idx)
            frac = src_idx - idx0
            idx1 = min(idx0 + 1, n_frames - 1)
            idx0 = min(idx0, n_frames - 1)

            mag = (1 - frac) * np.abs(stft[idx0]) + frac * np.abs(stft[idx1])
            result_stft[i] = mag * np.exp(1j * phase_acc)

            if i < n_out_frames - 1:
                dphi = np.angle(stft[idx1]) - np.angle(stft[idx0])
                dphi -= phase_advance
                dphi = dphi - 2 * np.pi * np.round(dphi / (2 * np.pi))
                phase_acc += phase_advance + dphi

        out_length = (n_out_frames - 1) * hop + n_fft
        output = np.zeros(out_length)
        window_sum = np.zeros(out_length)
        for i in range(n_out_frames):
            frame = np.fft.irfft(result_stft[i], n=n_fft) * window
            start = i * hop
            output[start:start + n_fft] += frame
            window_sum[start:start + n_fft] += window ** 2

        nonzero = window_sum > 1e-8
        output[nonzero] /= window_sum[nonzero]
        return output.astype(audio.dtype)

    def random_silence_insert(self, audio: np.ndarray, sr: int,
                              max_inserts: int = 3, duration_range: list = None,
                              rng: np.random.Generator = None) -> np.ndarray:
        """随机静音插入。"""
        rng = rng or self.rng
        if duration_range is None:
            duration_range = [0.05, 0.3]
        if len(audio) == 0:
            return audio.copy()

        n_inserts = int(rng.integers(1, max_inserts + 1))
        positions = sorted(rng.integers(0, len(audio), size=n_inserts), reverse=True)

        result = audio.copy()
        for pos in positions:
            dur = rng.uniform(duration_range[0], duration_range[1])
            silence_len = int(dur * sr)
            silence = np.zeros(silence_len, dtype=audio.dtype)
            result = np.concatenate([result[:pos], silence, result[pos:]])
        return result

    def codec_simulate(self, audio: np.ndarray, sr: int, codec: str = "ulaw") -> np.ndarray:
        """编解码失真模拟（G.711 μ-law / A-law）。"""
        if len(audio) == 0:
            return audio.copy()

        clipped = np.clip(audio, -1.0, 1.0)

        if codec == "ulaw":
            mu = 255.0
            sign = np.sign(clipped)
            compressed = sign * np.log1p(mu * np.abs(clipped)) / np.log1p(mu)
            quantized = np.round(compressed * 127) / 127
            decoded = sign * (1.0 / mu) * ((1 + mu) ** np.abs(quantized) - 1)
        elif codec == "alaw":
            A = 87.6
            sign = np.sign(clipped)
            abs_x = np.abs(clipped)
            compressed = np.where(
                abs_x < 1.0 / A,
                sign * A * abs_x / (1 + np.log(A)),
                sign * (1 + np.log(A * abs_x)) / (1 + np.log(A)),
            )
            quantized = np.round(compressed * 127) / 127
            abs_q = np.abs(quantized)
            decoded = np.where(
                abs_q < 1.0 / (1 + np.log(A)),
                np.sign(quantized) * abs_q * (1 + np.log(A)) / A,
                np.sign(quantized) * np.exp(abs_q * (1 + np.log(A)) - 1) / A,
            )
        else:
            return audio.copy()
        return decoded.astype(audio.dtype)

    def bandpass_filter(self, audio: np.ndarray, sr: int,
                        low_freq: float = 300, high_freq: float = 3400,
                        order: int = 5) -> np.ndarray:
        """带通滤波。"""
        if len(audio) == 0:
            return audio.copy()
        nyquist = sr / 2.0
        low = np.clip(low_freq / nyquist, 0.001, 0.999)
        high = np.clip(high_freq / nyquist, low + 0.001, 0.999)
        sos = butter(order, [low, high], btype="band", output="sos")
        filtered = sosfilt(sos, audio)
        return filtered.astype(audio.dtype)

    def apply_clipping(self, audio: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """削波失真模拟。"""
        if len(audio) == 0:
            return audio.copy()
        peak = np.max(np.abs(audio))
        if peak == 0:
            return audio.copy()
        clip_level = peak * threshold
        return np.clip(audio, -clip_level, clip_level)

    def concatenate_audios(self, audios: list, sr: int,
                           gap_range: list = None,
                           rng: np.random.Generator = None) -> np.ndarray:
        """多条短音频拼接。"""
        rng = rng or self.rng
        if gap_range is None:
            gap_range = [0.1, 0.5]
        if not audios:
            return np.array([], dtype=np.float64)

        segments = []
        for i, a in enumerate(audios):
            segments.append(a)
            if i < len(audios) - 1:
                gap_dur = rng.uniform(gap_range[0], gap_range[1])
                gap = np.zeros(int(gap_dur * sr), dtype=a.dtype)
                segments.append(gap)
        return np.concatenate(segments)

    def normalize_peak(self, audio: np.ndarray) -> np.ndarray:
        """峰值归一化，防止削波。"""
        peak = np.max(np.abs(audio))
        if peak == 0:
            return audio.copy()
        return audio / peak

    def _resolve_chain(self, wav_path: str) -> list:
        """根据音频文件路径匹配对应的增强链。"""
        path_normalized = wav_path.replace("\\", "/")
        for pattern, chain_cfg in self.folder_chains.items():
            if pattern.replace("\\", "/") in path_normalized:
                chain = chain_cfg if isinstance(chain_cfg, list) else chain_cfg.get("chain", [])
                return chain
        return self.chain

    def _detect_split(self, wav_path: str) -> str:
        """从音频路径中检测所属的数据集划分（train/dev/test）。"""
        path_lower = wav_path.replace("\\", "/")
        for split in ("train", "dev", "test"):
            if f"/{split}/" in path_lower:
                return split
        return "unknown"

    def augment_single(self, audio: np.ndarray, sr: int, wav_path: str,
                       aug_idx: int, file_idx: int,
                       audio_cache: dict = None,
                       candidate_paths: list = None) -> np.ndarray:
        """对单条音频执行完整增强链。

        统一的增强入口，供 augment_manifest.py 和 augment_dir.py 共用。

        Args:
            audio: 输入音频。
            sr: 采样率。
            wav_path: 音频文件路径（用于 _resolve_chain 匹配）。
            aug_idx: 增强副本索引。
            file_idx: 文件在列表中的索引（用于生成确定性随机种子）。
            audio_cache: 音频缓存字典 {path: (audio, sr)}，用于 speaker_mix/concatenate。
            candidate_paths: 可用于混合的候选音频路径列表。

        Returns:
            增强后的音频，已做峰值归一化。None 表示结果无效（含 NaN/Inf）。
        """
        import soundfile as sf

        active_chain = self._resolve_chain(wav_path)
        aug_rng = np.random.default_rng(self.seed + file_idx * self.augment_factor + aug_idx)
        aug_audio = audio.copy()

        for strategy_name in active_chain:
            strategy_config = self.strategies.get(strategy_name, {})
            if not strategy_config.get("enabled", False):
                continue

            if strategy_name == "gaussian_noise":
                snr_range = strategy_config.get("snr_range", [5, 20])
                snr_db = self._sample_normal_clipped(aug_rng, snr_range[0], snr_range[1])
                aug_audio = self.add_gaussian_noise(aug_audio, sr, snr_db)

            elif strategy_name == "speaker_mix":
                if not candidate_paths or not audio_cache:
                    continue
                snr_range = strategy_config.get("snr_range", [0, 15])
                max_speakers = strategy_config.get("max_speakers", 2)
                random_insert = strategy_config.get("random_insert", True)
                overlap_mode = strategy_config.get("overlap_mode", "random" if random_insert else "full")
                min_overlap = strategy_config.get("min_overlap", 0.8)
                bg_reverb = strategy_config.get("bg_reverb", False)
                bg_rt60_range = strategy_config.get("bg_rt60_range", [0.4, 1.2])
                bg_lowpass_range = strategy_config.get("bg_lowpass_range", [0, 0])
                bg_lowpass_freq = strategy_config.get("bg_lowpass_freq", 0)
                bg_pitch_shift = strategy_config.get("bg_pitch_shift", False)
                bg_semitone_range = strategy_config.get("bg_semitone_range", [-3, 3])
                bg_distance_correlated = strategy_config.get("bg_distance_correlated", False)
                candidates = [p for p in candidate_paths if p != wav_path]
                if not candidates:
                    continue
                n_bg = aug_rng.integers(1, max_speakers + 1)
                bg_paths = aug_rng.choice(candidates, size=min(n_bg, len(candidates)), replace=False)
                bg_audios = []
                bg_snrs = []
                for bp in bg_paths:
                    if bp in audio_cache:
                        bg_audio_tmp = audio_cache[bp][0].copy()

                        # 距离因子关联采样：t=0 最近，t=1 最远
                        # SNR、RT60、lowpass 由同一个 t 驱动，物理一致
                        t = self._sample_normal_clipped(aug_rng, 0.0, 1.0)

                        # 每条背景人声独立 pitch shift（与距离无关，是音色差异）
                        if bg_pitch_shift:
                            semitones = aug_rng.uniform(bg_semitone_range[0], bg_semitone_range[1])
                            bg_audio_tmp = self.pitch_shift(bg_audio_tmp, sr, semitones)

                        # 混响：t=0 → rt60_min（近），t=1 → rt60_max（远）
                        if bg_reverb:
                            if bg_distance_correlated:
                                bg_rt60 = bg_rt60_range[0] + t * (bg_rt60_range[1] - bg_rt60_range[0])
                            else:
                                bg_rt60 = self._sample_normal_clipped(aug_rng, bg_rt60_range[0], bg_rt60_range[1])
                            bg_audio_tmp = self.apply_reverb(bg_audio_tmp, sr, bg_rt60)

                        # 低通：t=0 → lp_max（近，高频保留），t=1 → lp_min（远，高频衰减）
                        has_lp_range = bg_lowpass_range[0] > 0 and bg_lowpass_range[1] > 0
                        if has_lp_range:
                            if bg_distance_correlated:
                                lp_freq = bg_lowpass_range[1] - t * (bg_lowpass_range[1] - bg_lowpass_range[0])
                            else:
                                lp_freq = aug_rng.uniform(bg_lowpass_range[0], bg_lowpass_range[1])
                            bg_audio_tmp = self.bandpass_filter(
                                bg_audio_tmp, sr, low_freq=60, high_freq=lp_freq, order=3,
                            )
                        elif bg_lowpass_freq > 0:
                            bg_audio_tmp = self.bandpass_filter(
                                bg_audio_tmp, sr, low_freq=60, high_freq=bg_lowpass_freq, order=3,
                            )

                        bg_audios.append(bg_audio_tmp)

                        # SNR：t=0 → snr_min（近，声音响），t=1 → snr_max（远，声音弱）
                        if bg_distance_correlated:
                            bg_snrs.append(snr_range[0] + t * (snr_range[1] - snr_range[0]))
                        else:
                            bg_snrs.append(self._sample_normal_clipped(aug_rng, snr_range[0], snr_range[1]))

                if bg_audios:
                    for bg_a, bg_s in zip(bg_audios, bg_snrs):
                        aug_audio = self.mix_speakers(
                            aug_audio, sr, [bg_a], bg_s,
                            overlap_mode=overlap_mode, min_overlap=min_overlap, rng=aug_rng,
                        )

            elif strategy_name == "reverb":
                rt60_range = strategy_config.get("rt60_range", [0.2, 0.8])
                rt60 = self._sample_normal_clipped(aug_rng, rt60_range[0], rt60_range[1])
                aug_audio = self.apply_reverb(aug_audio, sr, rt60)

            elif strategy_name == "speed_perturb":
                factor_range = strategy_config.get("factor_range", [0.9, 1.1])
                factor = aug_rng.uniform(factor_range[0], factor_range[1])
                aug_audio = self.speed_perturb(aug_audio, sr, factor)

            elif strategy_name == "volume_perturb":
                gain_range = strategy_config.get("gain_range", [-6, 6])
                gain_db = self._sample_normal_clipped(aug_rng, gain_range[0], gain_range[1])
                aug_audio = self.volume_perturb(aug_audio, gain_db)

            elif strategy_name == "noise_from_file":
                noise_dir = strategy_config.get("noise_dir", "")
                if not noise_dir or not os.path.isdir(noise_dir):
                    continue
                noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith(".wav")]
                if not noise_files:
                    continue
                noise_path = aug_rng.choice(noise_files)
                try:
                    noise_audio, _ = sf.read(noise_path, dtype="float64")
                    snr_range = strategy_config.get("snr_range", [5, 20])
                    snr_db = self._sample_normal_clipped(aug_rng, snr_range[0], snr_range[1])
                    aug_audio = self.add_noise_from_file(aug_audio, sr, noise_audio, snr_db)
                except Exception as e:
                    logger.error("读取噪声文件失败 %s: %s", noise_path, e)

            elif strategy_name == "real_rir":
                rir_dir = strategy_config.get("rir_dir", "")
                if not rir_dir or not os.path.isdir(rir_dir):
                    continue
                rir_files = [os.path.join(rir_dir, f) for f in os.listdir(rir_dir) if f.endswith(".wav")]
                if not rir_files:
                    continue
                rir_path = aug_rng.choice(rir_files)
                try:
                    rir_audio, _ = sf.read(rir_path, dtype="float64")
                    aug_audio = self.apply_real_rir(aug_audio, sr, rir_audio)
                except Exception as e:
                    logger.error("读取 RIR 文件失败 %s: %s", rir_path, e)

            elif strategy_name == "pitch_shift":
                semitone_range = strategy_config.get("semitone_range", [-2, 2])
                semitones = aug_rng.uniform(semitone_range[0], semitone_range[1])
                aug_audio = self.pitch_shift(aug_audio, sr, semitones)

            elif strategy_name == "eq_random":
                num_bands = strategy_config.get("num_bands", 5)
                gain_range = strategy_config.get("gain_range", [-6, 6])
                aug_audio = self.eq_random(aug_audio, sr, num_bands=num_bands,
                                           gain_range=gain_range, rng=aug_rng)

            elif strategy_name == "random_crop_pad":
                pad_mode = strategy_config.get("pad_mode", "zero")
                target_sec = strategy_config.get("target_sec", None)
                target_len = int(target_sec * sr) if target_sec else len(aug_audio)
                aug_audio = self.random_crop_pad(aug_audio, sr, target_length=target_len,
                                                 pad_mode=pad_mode, rng=aug_rng)

            elif strategy_name == "time_stretch":
                rate_range = strategy_config.get("rate_range", [0.9, 1.1])
                rate = aug_rng.uniform(rate_range[0], rate_range[1])
                aug_audio = self.time_stretch(aug_audio, sr, rate)

            elif strategy_name == "random_silence":
                max_inserts = strategy_config.get("max_inserts", 3)
                duration_range = strategy_config.get("duration_range", [0.05, 0.3])
                aug_audio = self.random_silence_insert(aug_audio, sr, max_inserts=max_inserts,
                                                       duration_range=duration_range, rng=aug_rng)

            elif strategy_name == "codec_simulate":
                codecs = strategy_config.get("codecs", ["ulaw", "alaw"])
                codec = aug_rng.choice(codecs)
                aug_audio = self.codec_simulate(aug_audio, sr, codec=codec)

            elif strategy_name == "bandpass_filter":
                low_freq = strategy_config.get("low_freq", 300)
                high_freq = strategy_config.get("high_freq", 3400)
                order = strategy_config.get("order", 5)
                aug_audio = self.bandpass_filter(aug_audio, sr, low_freq=low_freq,
                                                 high_freq=high_freq, order=order)

            elif strategy_name == "clipping":
                threshold_range = strategy_config.get("threshold_range", [0.3, 0.8])
                threshold = aug_rng.uniform(threshold_range[0], threshold_range[1])
                aug_audio = self.apply_clipping(aug_audio, threshold=threshold)

            elif strategy_name == "concatenate":
                if not candidate_paths or not audio_cache:
                    continue
                max_concat = strategy_config.get("max_concat", 3)
                gap_range = strategy_config.get("gap_range", [0.1, 0.5])
                candidates = [p for p in candidate_paths if p != wav_path]
                if candidates:
                    n_concat = int(aug_rng.integers(1, max_concat + 1))
                    concat_paths = aug_rng.choice(candidates, size=min(n_concat, len(candidates)), replace=False)
                    concat_audios = [aug_audio]
                    for cp in concat_paths:
                        if cp in audio_cache:
                            concat_audios.append(audio_cache[cp][0])
                    aug_audio = self.concatenate_audios(concat_audios, sr, gap_range=gap_range, rng=aug_rng)

        if not np.isfinite(aug_audio).all():
            return None

        return self.normalize_peak(aug_audio)
