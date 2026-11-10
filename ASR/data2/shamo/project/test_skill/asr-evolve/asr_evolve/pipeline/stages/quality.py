"""Stage: quality — 音频质量扫描 (R5.1).

用 wave 标准库采集每条音频的 duration/sample_rate/channels/rms/clip_rate。
扫描结果写入 metrics.jsonl，慢扫描只跑一次。
"""

from __future__ import annotations

import json
import struct
import wave
from pathlib import Path

from asr_evolve.pipeline.models import ArtifactStatus, StageContext, StageResult


class QualityScanner:
    """音频质量扫描器 — 采集原始指标。"""

    name = "quality"
    destructive = False

    def artifact_exists(self, ctx: StageContext) -> ArtifactStatus:
        """metrics.jsonl 与 bad.lst 都存在视为已完成。"""
        for ds in ctx.datasets:
            quality_dir = Path(ctx.base_dir) / ctx.stages_root / ds / "quality"
            if not (quality_dir / "bad.lst").exists():
                return ArtifactStatus(exists=False)
        return ArtifactStatus(exists=True, detail="quality bad.lst exists for all datasets")

    def plan(self, ctx: StageContext) -> list[str]:
        return [f"Scan quality for '{ds}' → metrics.jsonl + good/bad lists" for ds in ctx.datasets]

    def execute(self, ctx: StageContext) -> StageResult:
        """扫描音频质量并调用 QualityJudge 输出 good/bad 列表。"""
        from asr_evolve.pipeline.stages.quality_judge import QualityJudge

        thresholds = ctx.config.get("pipeline", {}).get("quality_thresholds", {})
        judge = QualityJudge(thresholds)
        artifacts = []
        total_scanned = 0

        for ds in ctx.datasets:
            base = Path(ctx.base_dir)
            labeled_dir = base / ctx.stages_root / ds / "labeled"
            quality_dir = base / ctx.stages_root / ds / "quality"
            quality_dir.mkdir(parents=True, exist_ok=True)

            wav_scp = labeled_dir / "wav.scp"
            if not wav_scp.exists():
                return StageResult(
                    stage=self.name, status="failed",
                    reason=f"Missing {wav_scp}; run 'prepare' first",
                )

            entries = self._parse_wav_scp(wav_scp)
            metrics_lines: list[str] = []
            good_ids: list[str] = []
            bad_entries: list[str] = []

            for utt_id, audio_path in entries:
                # 扫描 (R5.1)
                metrics = self.scan(audio_path)
                metrics["utt_id"] = utt_id
                metrics_lines.append(json.dumps(metrics, ensure_ascii=False))

                # 判定 (R5.2-5.3)
                is_good, problems = judge.judge(metrics)
                if is_good:
                    good_ids.append(utt_id)
                else:
                    # bad 条目记录违反的具体问题类型 (R5.3)
                    bad_entries.append(f"{utt_id}\t{';'.join(problems)}")

            # 写入产物
            (quality_dir / "metrics.jsonl").write_text(
                "\n".join(metrics_lines) + ("\n" if metrics_lines else ""),
                encoding="utf-8",
            )
            (quality_dir / "good.lst").write_text(
                "\n".join(good_ids) + ("\n" if good_ids else "")
            )
            (quality_dir / "bad.lst").write_text(
                "\n".join(bad_entries) + ("\n" if bad_entries else "")
            )

            total_scanned += len(entries)
            artifacts.append(str(quality_dir / "metrics.jsonl"))

        return StageResult(
            stage=self.name, status="completed",
            entry_count=total_scanned, artifacts=artifacts,
        )

    def scan(self, audio_path: str) -> dict:
        """用 wave 标准库读取音频属性并计算 RMS/削波率 (R5.1)。"""
        metrics: dict = {"path": audio_path}
        p = Path(audio_path)

        if not p.exists():
            metrics["error"] = "file_not_found"
            return metrics

        try:
            with wave.open(str(p), "rb") as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                duration = n_frames / sample_rate if sample_rate > 0 else 0.0

                metrics["duration_s"] = round(duration, 3)
                metrics["sample_rate"] = sample_rate
                metrics["channels"] = n_channels

                # 计算 RMS 和削波率
                raw_data = wf.readframes(n_frames)
                if sample_width == 2 and len(raw_data) > 0:
                    n_samples = len(raw_data) // 2
                    samples = struct.unpack(f"<{n_samples}h", raw_data)
                    sum_sq = sum(s * s for s in samples)
                    rms = (sum_sq / n_samples) ** 0.5 / 32768.0 if n_samples > 0 else 0.0
                    # 削波率：超过 ±32000 的样本占比
                    clip_count = sum(1 for s in samples if abs(s) > 32000)
                    clip_rate = clip_count / n_samples if n_samples > 0 else 0.0
                    metrics["rms"] = round(rms, 6)
                    metrics["clip_rate"] = round(clip_rate, 6)
                else:
                    metrics["rms"] = 0.0
                    metrics["clip_rate"] = 0.0

        except (wave.Error, EOFError, struct.error) as e:
            metrics["error"] = str(e)

        return metrics

    @staticmethod
    def _parse_wav_scp(path: Path) -> list[tuple[str, str]]:
        """解析 wav.scp: 每行 'utt_id /path/to/audio.wav'。"""
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                entries.append((parts[0], parts[1]))
        return entries
