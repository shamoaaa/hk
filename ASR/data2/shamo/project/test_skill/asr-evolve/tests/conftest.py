"""Pytest fixtures: 临时数据目录、生成假 wav.scp/text 的辅助函数.

提供：
- tmp_pipeline_dir: 带有标准目录结构的临时流水线目录
- make_wav_scp: 生成 Kaldi wav.scp 文件的工厂函数
- make_text: 生成 Kaldi text 文件的工厂函数
- make_wav_file: 生成合法 WAV 音频文件的工厂函数
- sample_config: 标准测试配置
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest


@pytest.fixture
def tmp_pipeline_dir(tmp_path: Path) -> Path:
    """创建标准流水线目录结构。"""
    stages_root = tmp_path / "data" / "pipeline"
    stages_root.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def sample_config() -> dict:
    """标准测试配置。"""
    return {
        "pipeline": {
            "mode": "aishell",
            "datasets": {
                "aishell": ["test_ds"],
                "business": ["biz_ds"],
            },
            "stages_root": "data/pipeline",
            "quality_thresholds": {
                "min_duration_s": 0.5,
                "max_duration_s": 20.0,
                "expected_sample_rate": 16000,
                "expected_channels": 1,
                "min_rms": 0.001,
            },
            "split": {
                "ratios": [0.90, 0.05, 0.05],
                "seed": 42,
            },
        },
        "recipe": {
            "max_epoch": 5,
            "learning_rate": 2e-5,
            "batch_size": 60000,
        },
        "evaluation": {
            "wakewords": ["你好萤石"],
        },
    }


def make_wav_scp(
    directory: Path, entries: list[tuple[str, str]], filename: str = "wav.scp"
) -> Path:
    """生成 Kaldi wav.scp 文件.

    Args:
        directory: 输出目录
        entries: [(utt_id, audio_path), ...]
        filename: 文件名，默认 "wav.scp"

    Returns:
        生成的文件路径
    """
    directory.mkdir(parents=True, exist_ok=True)
    lines = [f"{uid} {path}" for uid, path in entries]
    filepath = directory / filename
    filepath.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return filepath


def make_text(
    directory: Path, entries: list[tuple[str, str]], filename: str = "text"
) -> Path:
    """生成 Kaldi text 文件.

    Args:
        directory: 输出目录
        entries: [(utt_id, text_content), ...]
        filename: 文件名，默认 "text"

    Returns:
        生成的文件路径
    """
    directory.mkdir(parents=True, exist_ok=True)
    lines = [f"{uid} {txt}" for uid, txt in entries]
    filepath = directory / filename
    filepath.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return filepath


def make_wav_file(
    filepath: Path,
    duration_s: float = 1.0,
    sample_rate: int = 16000,
    channels: int = 1,
    amplitude: float = 0.1,
) -> Path:
    """生成合法的 16-bit PCM WAV 文件.

    Args:
        filepath: 输出文件路径
        duration_s: 时长（秒）
        sample_rate: 采样率
        channels: 声道数
        amplitude: 振幅 (0.0~1.0)

    Returns:
        生成的文件路径
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    n_frames = int(sample_rate * duration_s)
    # 生成简单正弦波近似（用三角波代替，避免 math 依赖）
    max_val = int(32767 * amplitude)
    samples = []
    period = sample_rate // 440  # 440Hz
    for i in range(n_frames * channels):
        # 三角波
        pos = (i % period) / period
        if pos < 0.5:
            val = int(max_val * (4 * pos - 1))
        else:
            val = int(max_val * (3 - 4 * pos))
        samples.append(val)

    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        raw = struct.pack(f"<{len(samples)}h", *samples)
        wf.writeframes(raw)

    return filepath


@pytest.fixture
def make_dataset(tmp_pipeline_dir: Path, sample_config: dict):
    """工厂 fixture：创建一个包含 labeled 数据的测试数据集。"""

    def _make(
        dataset_name: str = "test_ds",
        num_devices: int = 5,
        utts_per_device: int = 10,
        with_wav_files: bool = False,
    ) -> Path:
        """创建测试数据集.

        生成 num_devices 个设备，每个设备 utts_per_device 条语音。
        utt_id 格式: "DEV{i:03d}_{j:04d}"
        """
        stages_root = tmp_pipeline_dir / "data" / "pipeline"
        labeled_dir = stages_root / dataset_name / "labeled"
        labeled_dir.mkdir(parents=True, exist_ok=True)

        scp_entries: list[tuple[str, str]] = []
        text_entries: list[tuple[str, str]] = []

        for dev_idx in range(num_devices):
            dev_id = f"DEV{dev_idx:03d}"
            for utt_idx in range(utts_per_device):
                utt_id = f"{dev_id}_{utt_idx:04d}"
                if with_wav_files:
                    wav_path = stages_root / dataset_name / "wavs" / f"{utt_id}.wav"
                    make_wav_file(wav_path)
                    audio_path = str(wav_path)
                else:
                    audio_path = f"/fake/audio/{utt_id}.wav"
                scp_entries.append((utt_id, audio_path))
                text_entries.append((utt_id, f"这是测试语音{utt_id}的文本内容"))

        make_wav_scp(labeled_dir, scp_entries)
        make_text(labeled_dir, text_entries)

        return stages_root

    return _make
