"""Stage: clean — 按 bad 清单删除问题音频（破坏性，默认关闭）(R6).

安全约束:
- destructive = True
- 未启用时保留全部 (R6.1)
- dry-run 输出将删除数量 (R6.3)
- 缺 bad 清单跳过 (R6.4)
"""

from __future__ import annotations

import time
from pathlib import Path

from asr_evolve.pipeline.models import ArtifactStatus, StageContext, StageResult


class Cleaner:
    """破坏性清理阶段 — 默认关闭。"""

    name = "clean"
    destructive = True  # 破坏性阶段

    def artifact_exists(self, ctx: StageContext) -> ArtifactStatus:
        """通过 .done 标记文件判断是否已执行。"""
        for ds in ctx.datasets:
            done = Path(ctx.base_dir) / ctx.stages_root / ds / "clean" / ".done"
            if not done.exists():
                return ArtifactStatus(exists=False)
        return ArtifactStatus(exists=True, detail="clean/.done marker exists")

    def plan(self, ctx: StageContext) -> list[str]:
        actions = []
        for ds in ctx.datasets:
            bad_lst = Path(ctx.base_dir) / ctx.stages_root / ds / "quality" / "bad.lst"
            if bad_lst.exists():
                count = sum(1 for l in bad_lst.read_text().splitlines() if l.strip())
                actions.append(f"Remove {count} bad entries from '{ds}/labeled/' (DESTRUCTIVE)")
            else:
                actions.append(f"No bad.lst found for '{ds}' → skip")
        return actions

    def execute(self, ctx: StageContext) -> StageResult:
        """从 labeled/ 中移除 bad.lst 中的条目 (R6.2)。

        未启用清理时保留全部 (R6.1): 通过 config 中 clean_enabled 控制。
        安全限制：bad 比例 > 30% 时拒绝执行（可能是阈值设错了）。
        注意：不删除物理文件，只修改 wav.scp 和 text。
        """
        # R6.1: 未显式启用清理选项时保留全部
        clean_enabled = ctx.config.get("pipeline", {}).get("clean_enabled", False)
        if not clean_enabled:
            # 标记为跳过，保留全部
            for ds in ctx.datasets:
                clean_dir = Path(ctx.base_dir) / ctx.stages_root / ds / "clean"
                clean_dir.mkdir(parents=True, exist_ok=True)
                (clean_dir / ".done").write_text(f"disabled\n{time.time()}\n")
            return StageResult(
                stage=self.name, status="completed",
                reason="Clean disabled — all audio preserved (R6.1)",
            )

        artifacts = []
        total_removed = 0

        for ds in ctx.datasets:
            base = Path(ctx.base_dir)
            quality_dir = base / ctx.stages_root / ds / "quality"
            labeled_dir = base / ctx.stages_root / ds / "labeled"
            clean_dir = base / ctx.stages_root / ds / "clean"
            clean_dir.mkdir(parents=True, exist_ok=True)

            bad_lst = quality_dir / "bad.lst"
            # R6.4: 缺 bad 清单则跳过
            if not bad_lst.exists():
                (clean_dir / ".done").write_text(f"no_bad_list\n{time.time()}\n")
                continue

            # 解析 bad.lst
            bad_ids: set[str] = set()
            for line in bad_lst.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                bad_ids.add(line.split("\t")[0])

            if not bad_ids:
                (clean_dir / ".done").write_text(f"empty_bad_list\n{time.time()}\n")
                continue

            # R6.3: dry-run 输出将删除条目数
            if ctx.dry_run:
                return StageResult(
                    stage=self.name, status="planned",
                    reason=f"Would remove {len(bad_ids)} entries from '{ds}'",
                    entry_count=len(bad_ids),
                )

            # 读取 wav.scp
            wav_scp = labeled_dir / "wav.scp"
            if not wav_scp.exists():
                return StageResult(
                    stage=self.name, status="failed",
                    reason=f"Missing {wav_scp}; run 'prepare' first",
                )

            scp_lines = [l for l in wav_scp.read_text(encoding="utf-8").splitlines() if l.strip()]

            # 安全限制：bad 比例 > 30% 时拒绝执行
            if scp_lines and len(bad_ids) / len(scp_lines) > 0.30:
                return StageResult(
                    stage=self.name, status="failed",
                    reason=(
                        f"安全限制: bad 比例 {len(bad_ids)}/{len(scp_lines)} "
                        f"= {len(bad_ids)/len(scp_lines):.1%} > 30%，"
                        f"可能是阈值设错了，拒绝执行"
                    ),
                )

            new_scp = [l for l in scp_lines if l.split(maxsplit=1)[0] not in bad_ids]

            # 修改 wav.scp（不删除物理文件）
            wav_scp.write_text("\n".join(new_scp) + ("\n" if new_scp else ""))

            text_file = labeled_dir / "text"
            if text_file.exists():
                text_lines = [l for l in text_file.read_text(encoding="utf-8").splitlines() if l.strip()]
                new_text = [l for l in text_lines if l.split(maxsplit=1)[0] not in bad_ids]
                text_file.write_text("\n".join(new_text) + ("\n" if new_text else ""))

            removed_count = len(scp_lines) - len(new_scp)
            total_removed += removed_count

            (clean_dir / ".done").write_text(
                f"removed={removed_count}\ntimestamp={int(time.time())}\n"
            )
            artifacts.append(str(clean_dir / ".done"))

        return StageResult(
            stage=self.name, status="completed",
            entry_count=total_removed, artifacts=artifacts,
            reason=f"Removed {total_removed} entries",
        )
