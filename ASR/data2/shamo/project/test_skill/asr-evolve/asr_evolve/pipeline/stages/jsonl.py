"""Stage: jsonl — Kaldi scp+text → FunASR manifest.jsonl (R12).

- 键交集 (R3.4, R12.1)
- 保留音频路径与对应文本 (R12.3)
- 缺 wav.scp 跳过 (R12.2)
- 单 split 可独立转换 (R12.1)
"""

from __future__ import annotations

import json
from pathlib import Path

from asr_evolve.pipeline.models import ArtifactStatus, StageContext, StageResult


class ManifestGenerator:
    """JSONL manifest 生成阶段。"""

    name = "jsonl"
    destructive = False

    def artifact_exists(self, ctx: StageContext) -> ArtifactStatus:
        train_jsonl = Path(ctx.base_dir) / ctx.stages_root / "train.jsonl"
        if not train_jsonl.exists():
            return ArtifactStatus(exists=False)
        count = sum(1 for l in train_jsonl.read_text().splitlines() if l.strip())
        return ArtifactStatus(exists=True, entry_count=count, detail="train.jsonl exists")

    def plan(self, ctx: StageContext) -> list[str]:
        return ["Convert all_label/{train,dev,test}/wav.scp+text → {train,dev,test}.jsonl"]

    def execute(self, ctx: StageContext) -> StageResult:
        """将 Kaldi 文件转换为 JSONL 训练清单 (R12.1-12.3)。"""
        base = Path(ctx.base_dir)
        all_label_dir = base / ctx.stages_root / "all_label"
        artifacts = []
        total_entries = 0

        for split_name in ("train", "dev", "test"):
            split_dir = all_label_dir / split_name
            wav_scp_path = split_dir / "wav.scp"
            text_path = split_dir / "text"

            # R12.2: 缺 wav.scp 跳过该 split
            if not wav_scp_path.exists():
                continue

            scp_dict = self._parse_to_dict(wav_scp_path)
            text_dict = self._parse_to_dict(text_path) if text_path.exists() else {}

            # R3.4: 取键交集
            common_keys = sorted(set(scp_dict.keys()) & set(text_dict.keys()))

            # R12.3: 保留音频路径与对应文本
            jsonl_lines: list[str] = []
            for key in common_keys:
                record = {
                    "key": key,
                    "source": scp_dict[key],
                    "target": text_dict[key],
                }
                jsonl_lines.append(json.dumps(record, ensure_ascii=False))

            output_path = base / ctx.stages_root / f"{split_name}.jsonl"
            output_path.write_text(
                "\n".join(jsonl_lines) + ("\n" if jsonl_lines else ""),
                encoding="utf-8",
            )

            total_entries += len(jsonl_lines)
            artifacts.append(str(output_path))

        if total_entries == 0:
            return StageResult(
                stage=self.name, status="failed",
                reason="No entries produced; check all_label/ contains valid wav.scp+text with common keys",
            )

        return StageResult(
            stage=self.name, status="completed",
            entry_count=total_entries, artifacts=artifacts,
        )

    @staticmethod
    def _parse_to_dict(path: Path) -> dict[str, str]:
        """解析 Kaldi 文件为 {key: value} 字典。"""
        result: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                result[parts[0]] = parts[1]
            elif len(parts) == 1:
                result[parts[0]] = ""
        return result
