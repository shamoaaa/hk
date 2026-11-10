"""Stage: split — 按 Device_ID 隔离的 90/5/5 切分 (R9) + 路径重映射 (R3.3).

- extract_device_id: 取第一个 '_' 前的部分
- split_by_device: 按设备分组，seed=42，90/5/5
- remap_paths: 失效路径重映射，丢弃不存在的
- 多数据集独立切分 (R9.4)
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from asr_evolve.pipeline.models import ArtifactStatus, StageContext, StageResult


def extract_device_id(utt_id: str) -> str:
    """从 utt_id 提取 device_id：第一个 '_' 前的部分。"""
    idx = utt_id.find("_")
    if idx > 0:
        return utt_id[:idx]
    return utt_id


def split_by_device(
    entries: list[tuple[str, str]],
    ratios: tuple[float, float, float] = (0.90, 0.05, 0.05),
    seed: int = 42,
) -> dict[str, list[tuple[str, str]]]:
    """按设备分组并随机分配到 train/dev/test (R9.1-9.3).

    Args:
        entries: [(utt_id, value), ...]
        ratios: (train_ratio, dev_ratio, test_ratio)
        seed: 随机种子 (R9.2: 固定 42)

    Returns:
        {"train": [...], "dev": [...], "test": [...]}
    """
    # 按 device_id 分组
    device_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for utt_id, value in entries:
        dev_id = extract_device_id(utt_id)
        device_groups[dev_id].append((utt_id, value))

    # 排序后随机打乱（确定性）
    device_ids = sorted(device_groups.keys())
    rng = random.Random(seed)
    rng.shuffle(device_ids)

    # 按比例分配设备
    n = len(device_ids)
    train_end = int(n * ratios[0])
    dev_end = train_end + int(n * ratios[1])

    # 确保至少 1 个设备进 dev 和 test（如果有足够设备）
    if n >= 3:
        train_end = max(train_end, 1)
        dev_end = max(dev_end, train_end + 1)

    train_devs = set(device_ids[:train_end])
    dev_devs = set(device_ids[train_end:dev_end])
    test_devs = set(device_ids[dev_end:])

    result: dict[str, list[tuple[str, str]]] = {"train": [], "dev": [], "test": []}
    for utt_id, value in entries:
        dev_id = extract_device_id(utt_id)
        if dev_id in train_devs:
            result["train"].append((utt_id, value))
        elif dev_id in dev_devs:
            result["dev"].append((utt_id, value))
        else:
            result["test"].append((utt_id, value))

    return result


def remap_paths(
    wav_scp_entries: list[tuple[str, str]],
    audio_root: str,
) -> list[tuple[str, str]]:
    """失效路径重映射 (R3.3).

    对 wav.scp 中的绝对路径，尝试用 audio_root + 文件名重映射。
    重映射后仍不存在的条目丢弃。
    """
    root = Path(audio_root)
    result: list[tuple[str, str]] = []

    for utt_id, audio_path in wav_scp_entries:
        p = Path(audio_path)
        # 如果原路径有效，直接保留
        if p.exists():
            result.append((utt_id, audio_path))
            continue
        # 尝试重映射：audio_root / 文件名
        remapped = root / p.name
        if remapped.exists():
            result.append((utt_id, str(remapped)))
        # 重映射后仍不存在 → 丢弃

    return result


class DataSplitter:
    """按设备隔离的数据切分阶段。"""

    name = "split"
    destructive = False

    def artifact_exists(self, ctx: StageContext) -> ArtifactStatus:
        for ds in ctx.datasets:
            split_dir = Path(ctx.base_dir) / ctx.stages_root / ds / "split" / "train"
            if not (split_dir / "wav.scp").exists():
                return ArtifactStatus(exists=False)
        return ArtifactStatus(exists=True, detail="split/train/wav.scp exists")

    def plan(self, ctx: StageContext) -> list[str]:
        ratios = ctx.config.get("pipeline", {}).get("split", {}).get("ratios", [0.90, 0.05, 0.05])
        seed = ctx.config.get("pipeline", {}).get("split", {}).get("seed", 42)
        return [
            f"Split '{ds}' by Device_ID → train/dev/test "
            f"({ratios[0]:.0%}/{ratios[1]:.0%}/{ratios[2]:.0%}, seed={seed})"
            for ds in ctx.datasets
        ]

    def execute(self, ctx: StageContext) -> StageResult:
        """按设备 ID 分组，随机分配到 train/dev/test (R9.1-9.5)。"""
        split_cfg = ctx.config.get("pipeline", {}).get("split", {})
        ratios = tuple(split_cfg.get("ratios", [0.90, 0.05, 0.05]))
        seed = split_cfg.get("seed", 42)

        artifacts = []
        total_entries = 0

        for ds in ctx.datasets:
            base = Path(ctx.base_dir)
            labeled_dir = base / ctx.stages_root / ds / "labeled"

            wav_scp_path = labeled_dir / "wav.scp"
            text_path = labeled_dir / "text"

            if not wav_scp_path.exists():
                return StageResult(
                    stage=self.name, status="failed",
                    reason=f"Missing {wav_scp_path}; run 'prepare' first",
                )

            scp_entries = self._parse_kaldi_file(wav_scp_path)
            text_entries = self._parse_kaldi_file(text_path) if text_path.exists() else []
            text_dict = dict(text_entries)

            # R9.4: 每个数据集独立切分
            splits = split_by_device(scp_entries, ratios, seed)

            for split_name, split_scp in splits.items():
                split_dir = base / ctx.stages_root / ds / "split" / split_name
                split_dir.mkdir(parents=True, exist_ok=True)

                scp_lines = [f"{uid} {val}" for uid, val in split_scp]
                (split_dir / "wav.scp").write_text(
                    "\n".join(scp_lines) + ("\n" if scp_lines else "")
                )

                text_lines = []
                for uid, _ in split_scp:
                    if uid in text_dict:
                        text_lines.append(f"{uid} {text_dict[uid]}")
                (split_dir / "text").write_text(
                    "\n".join(text_lines) + ("\n" if text_lines else "")
                )

                total_entries += len(split_scp)
                artifacts.append(str(split_dir / "wav.scp"))

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
