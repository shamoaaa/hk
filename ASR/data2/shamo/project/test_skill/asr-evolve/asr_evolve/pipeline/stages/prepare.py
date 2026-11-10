"""Stage: prepare — 原始数据整理为 Kaldi 格式 (R4).

产物: labeled/(wav.scp + text) + unlabeled/(wav.scp)
跳过: labeled/ 已存在时跳过并报告条目数
"""

from __future__ import annotations

from pathlib import Path

from asr_evolve.pipeline.models import ArtifactStatus, StageContext, StageResult


class PrepareStage:
    name = "prepare"
    destructive = False

    def artifact_exists(self, ctx: StageContext) -> ArtifactStatus:
        """检查所有数据集的 labeled/wav.scp 是否存在 (R4.1)。"""
        total_entries = 0
        for ds in ctx.datasets:
            labeled = Path(ctx.base_dir) / ctx.stages_root / ds / "labeled"
            wav_scp = labeled / "wav.scp"
            if not wav_scp.exists():
                return ArtifactStatus(exists=False)
            total_entries += sum(1 for l in wav_scp.read_text().splitlines() if l.strip())
        return ArtifactStatus(
            exists=True, entry_count=total_entries,
            detail="All datasets have labeled/wav.scp",
        )

    def plan(self, ctx: StageContext) -> list[str]:
        return [f"Prepare dataset '{ds}' → labeled/ + unlabeled/" for ds in ctx.datasets]

    def execute(self, ctx: StageContext) -> StageResult:
        """扫描原始目录，按文件名规则分类输出 Kaldi 格式文件 (R4.3)。

        数据来源:
        - raw_data/{ds}/ 下的 .wav + 同名 .txt
        - raw_data/{ds}/label/ 下的 transcript 文件（每行 "utt_id text"）
        """
        artifacts = []
        total_entries = 0

        for ds in ctx.datasets:
            base = Path(ctx.base_dir)
            raw_dir = base / "raw_data" / ds
            labeled_dir = base / ctx.stages_root / ds / "labeled"
            unlabeled_dir = base / ctx.stages_root / ds / "unlabeled"
            labeled_dir.mkdir(parents=True, exist_ok=True)
            unlabeled_dir.mkdir(parents=True, exist_ok=True)

            labeled_scp_lines: list[str] = []
            labeled_text_lines: list[str] = []
            unlabeled_scp_lines: list[str] = []

            # 兼容：如果已有 labeled 数据，视为已准备好（R4.2）
            if not raw_dir.exists():
                existing_scp = labeled_dir / "wav.scp"
                if existing_scp.exists():
                    artifacts.append(str(labeled_dir))
                    continue
                # 无原始数据也无已有产物 → 报告需先运行 prepare 子脚本
                return StageResult(
                    stage=self.name, status="failed",
                    reason=f"Dataset '{ds}': 未检测到 labeled/，需先运行对应的 prepare 子脚本",
                )

            # 优先加载 label/ 目录下的 transcript 文件（每行 "utt_id text"）
            transcript_labels: dict[str, str] = {}
            label_dir = raw_dir / "label"
            if label_dir.exists():
                for tf in sorted(label_dir.iterdir()):
                    if tf.is_file() and tf.suffix in (".txt", ".text", ""):
                        for line in tf.read_text(encoding="utf-8").splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split(maxsplit=1)
                            if len(parts) == 2:
                                transcript_labels[parts[0]] = parts[1]

            # 扫描所有 .wav 文件
            wav_files = sorted(raw_dir.rglob("*.wav"))
            for wav_path in wav_files:
                utt_id = wav_path.stem
                audio_path = str(wav_path.resolve())

                # 标签来源优先级: transcript 文件 > 同名 .txt
                if utt_id in transcript_labels:
                    text_content = transcript_labels[utt_id]
                    labeled_scp_lines.append(f"{utt_id} {audio_path}")
                    labeled_text_lines.append(f"{utt_id} {text_content}")
                else:
                    txt_path = wav_path.with_suffix(".txt")
                    if txt_path.exists():
                        text_content = txt_path.read_text(encoding="utf-8").strip()
                        if text_content:
                            first_line = text_content.splitlines()[0].strip()
                            labeled_scp_lines.append(f"{utt_id} {audio_path}")
                            labeled_text_lines.append(f"{utt_id} {first_line}")
                        else:
                            unlabeled_scp_lines.append(f"{utt_id} {audio_path}")
                    else:
                        unlabeled_scp_lines.append(f"{utt_id} {audio_path}")

            # 写入 Kaldi 格式文件
            (labeled_dir / "wav.scp").write_text(
                "\n".join(labeled_scp_lines) + ("\n" if labeled_scp_lines else ""),
                encoding="utf-8",
            )
            (labeled_dir / "text").write_text(
                "\n".join(labeled_text_lines) + ("\n" if labeled_text_lines else ""),
                encoding="utf-8",
            )
            (unlabeled_dir / "wav.scp").write_text(
                "\n".join(unlabeled_scp_lines) + ("\n" if unlabeled_scp_lines else ""),
                encoding="utf-8",
            )

            total_entries += len(labeled_scp_lines)
            artifacts.append(str(labeled_dir))

        return StageResult(
            stage=self.name, status="completed",
            entry_count=total_entries, artifacts=artifacts,
        )
