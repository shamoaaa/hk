"""Stage: merge — 多源按 split 合并去重 (R11).

- 按 split 合并 reflow/augmented/public → all_label (R11.1)
- 增强数据仅并入 train (R11.2)
- 去重保留首现 (R11.3)
- 仅纳入 wav.scp ∩ text (R11.4)
- 保持 split 间隔离 (R11.5)
"""

from __future__ import annotations

from pathlib import Path

from asr_evolve.pipeline.models import ArtifactStatus, StageContext, StageResult


class Merger:
    """多源数据合并去重阶段。"""

    name = "merge"
    destructive = False

    def artifact_exists(self, ctx: StageContext) -> ArtifactStatus:
        all_label = Path(ctx.base_dir) / ctx.stages_root / "all_label" / "train"
        if not (all_label / "wav.scp").exists():
            return ArtifactStatus(exists=False)
        count = sum(1 for l in (all_label / "wav.scp").read_text().splitlines() if l.strip())
        return ArtifactStatus(exists=True, entry_count=count, detail="all_label/train/wav.scp exists")

    def plan(self, ctx: StageContext) -> list[str]:
        return ["Merge all sources by split → all_label/ (dedup, train-only augment)"]

    def execute(self, ctx: StageContext) -> StageResult:
        """按 split 合并多源数据并去重 (R11.1-11.5)。"""
        base = Path(ctx.base_dir)
        all_label_dir = base / ctx.stages_root / "all_label"
        artifacts = []
        total_entries = 0

        for split_name in ("train", "dev", "test"):
            split_out = all_label_dir / split_name
            split_out.mkdir(parents=True, exist_ok=True)

            all_scp: list[tuple[str, str]] = []
            all_text: list[tuple[str, str]] = []

            for ds in ctx.datasets:
                # 原始 split 数据
                ds_split = base / ctx.stages_root / ds / "split" / split_name
                if (ds_split / "wav.scp").exists():
                    all_scp.extend(self._parse_kaldi_file(ds_split / "wav.scp"))
                if (ds_split / "text").exists():
                    all_text.extend(self._parse_kaldi_file(ds_split / "text"))

                # R11.2: 增强数据只并入 train
                if split_name == "train":
                    aug_dir = base / ctx.stages_root / ds / "augmented" / "train"
                    if (aug_dir / "wav.scp").exists():
                        all_scp.extend(self._parse_kaldi_file(aug_dir / "wav.scp"))
                    if (aug_dir / "text").exists():
                        all_text.extend(self._parse_kaldi_file(aug_dir / "text"))

            # R11.3: 去重保留首现
            dedup_scp = self._dedup(all_scp)
            dedup_text = self._dedup(all_text)

            # R11.4: 仅纳入 wav.scp ∩ text
            scp_keys = {uid for uid, _ in dedup_scp}
            text_keys = {uid for uid, _ in dedup_text}
            common_keys = scp_keys & text_keys

            final_scp = [(uid, val) for uid, val in dedup_scp if uid in common_keys]
            final_text = [(uid, val) for uid, val in dedup_text if uid in common_keys]

            # 写入合并结果
            scp_lines = [f"{uid} {val}" for uid, val in final_scp]
            text_lines = [f"{uid} {val}" for uid, val in final_text]

            (split_out / "wav.scp").write_text(
                "\n".join(scp_lines) + ("\n" if scp_lines else "")
            )
            (split_out / "text").write_text(
                "\n".join(text_lines) + ("\n" if text_lines else "")
            )

            total_entries += len(final_scp)
            artifacts.append(str(split_out / "wav.scp"))

        return StageResult(
            stage=self.name, status="completed",
            entry_count=total_entries, artifacts=artifacts,
        )

    @staticmethod
    def _parse_kaldi_file(path: Path) -> list[tuple[str, str]]:
        """解析 Kaldi 格式文件。"""
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                entries.append((parts[0], parts[1]))
            elif len(parts) == 1:
                entries.append((parts[0], ""))
        return entries

    @staticmethod
    def _dedup(entries: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """按 key 去重，保留首次出现 (R11.3)。"""
        seen: set[str] = set()
        result: list[tuple[str, str]] = []
        for key, value in entries:
            if key not in seen:
                seen.add(key)
                result.append((key, value))
        return result
