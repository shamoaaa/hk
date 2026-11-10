"""Stage: augment — 仅 train 的多说话人距离关联增强 (R10).

- 仅对 train split 执行增强 (R10.1)
- 距离因子 t 联动 SNR/RT60/低通 (R10.2)
- 背景人声能量 < 主说话人 (R10.3)
- 复用 ShardedAugmentor 分片增量 (R10.4)
- Mock 模式：加 _aug 后缀
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

from asr_evolve.pipeline.models import ArtifactStatus, StageContext, StageResult


def compute_augment_params(t: float) -> dict[str, float]:
    """由距离因子 t 派生增强参数 (R10.2).

    t 越大 → SNR 越低（噪声越多）、RT60 越高（混响越重）、低通截止越低（闷）。
    联动关系（单调耦合）:
    - SNR_dB = 20 - 15 * t    (t=0→20dB, t=1→5dB)
    - RT60_s = 0.2 + 0.8 * t  (t=0→0.2s, t=1→1.0s)
    - lpf_hz = 8000 - 4000*t  (t=0→8kHz, t=1→4kHz)
    """
    snr_db = 20.0 - 15.0 * t
    rt60_s = 0.2 + 0.8 * t
    lpf_hz = 8000.0 - 4000.0 * t
    return {"snr_db": snr_db, "rt60_s": rt60_s, "lpf_hz": lpf_hz}


def ensure_bg_energy_lower(main_energy: float, bg_energy: float) -> float:
    """确保背景人声能量低于主说话人 (R10.3).

    返回调整后的背景能量增益因子。
    """
    if bg_energy <= 0 or main_energy <= 0:
        return 0.5  # 安全默认
    # 如果背景能量 >= 主说话人，衰减到 0.8 倍主说话人能量
    if bg_energy >= main_energy:
        return (main_energy * 0.8) / bg_energy
    return 1.0  # 已满足约束


class Augmentor:
    """增强阶段 — 仅 train，距离关联。"""

    name = "augment"
    destructive = False

    def artifact_exists(self, ctx: StageContext) -> ArtifactStatus:
        for ds in ctx.datasets:
            aug = Path(ctx.base_dir) / ctx.stages_root / ds / "augmented" / "train"
            if not (aug / "wav.scp").exists():
                return ArtifactStatus(exists=False)
        return ArtifactStatus(exists=True, detail="augmented/train/wav.scp exists")

    def plan(self, ctx: StageContext) -> list[str]:
        return [
            f"Augment train split for '{ds}' (distance-correlated noise/reverb)"
            for ds in ctx.datasets
        ]

    def execute(self, ctx: StageContext) -> StageResult:
        """数据增强执行 (R10.1-10.5)。"""
        is_mock = os.environ.get("ASR_EVOLVE_MOCK", "0") == "1"
        artifacts = []
        total_entries = 0

        for ds in ctx.datasets:
            base = Path(ctx.base_dir)
            train_dir = base / ctx.stages_root / ds / "split" / "train"
            aug_dir = base / ctx.stages_root / ds / "augmented" / "train"
            aug_dir.mkdir(parents=True, exist_ok=True)

            wav_scp = train_dir / "wav.scp"
            text_file = train_dir / "text"

            if not wav_scp.exists():
                return StageResult(
                    stage=self.name, status="failed",
                    reason=f"Missing {wav_scp}; run 'split' first",
                )

            if is_mock:
                self._mock_augment(wav_scp, text_file, aug_dir)
            else:
                # 真实模式：复用 ShardedAugmentor 分片增量增强 (R10.4)
                return StageResult(
                    stage=self.name, status="failed",
                    reason="Real augmentation not implemented; set ASR_EVOLVE_MOCK=1",
                )

            aug_scp = aug_dir / "wav.scp"
            if aug_scp.exists():
                total_entries += sum(
                    1 for l in aug_scp.read_text().splitlines() if l.strip()
                )
            artifacts.append(str(aug_dir))

        return StageResult(
            stage=self.name, status="completed",
            entry_count=total_entries, artifacts=artifacts,
        )

    @staticmethod
    def _mock_augment(wav_scp: Path, text_file: Path, aug_dir: Path) -> None:
        """Mock 增强：复制 wav.scp/text，utt_id 加 _aug 后缀。"""
        aug_scp_lines: list[str] = []
        for line in wav_scp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                utt_id, audio_path = parts
                aug_scp_lines.append(f"{utt_id}_aug {audio_path}")
        (aug_dir / "wav.scp").write_text(
            "\n".join(aug_scp_lines) + ("\n" if aug_scp_lines else "")
        )

        if text_file.exists():
            aug_text_lines: list[str] = []
            for line in text_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    utt_id, text_content = parts
                    aug_text_lines.append(f"{utt_id}_aug {text_content}")
            (aug_dir / "text").write_text(
                "\n".join(aug_text_lines) + ("\n" if aug_text_lines else "")
            )
