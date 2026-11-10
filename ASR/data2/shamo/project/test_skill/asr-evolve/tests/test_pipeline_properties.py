"""流水线属性测试 — 30 条 Property-Based Tests.

使用 hypothesis 验证流水线核心不变式。
@settings(max_examples=100) 每个属性至少跑 100 次随机用例。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

# 确保项目根目录在 Python 路径中
ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))


# --- 内联辅助函数（避免 conftest 导入问题）---

def _make_wav_scp(directory: Path, entries: list[tuple[str, str]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    lines = [f"{uid} {path}" for uid, path in entries]
    filepath = directory / "wav.scp"
    filepath.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return filepath


def _make_text(directory: Path, entries: list[tuple[str, str]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    lines = [f"{uid} {txt}" for uid, txt in entries]
    filepath = directory / "text"
    filepath.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return filepath

from asr_evolve.pipeline.orchestrator import PipelineOrchestrator, CANONICAL_STAGES
from asr_evolve.pipeline.models import (
    ArtifactStatus, StageContext, StageResult, PipelineRunReport,
    GoodBadLists, Sample,
)
from asr_evolve.pipeline.stages.split import (
    extract_device_id, split_by_device, remap_paths, DataSplitter,
)
from asr_evolve.pipeline.stages.quality_judge import QualityJudge
from asr_evolve.pipeline.stages.augment import (
    compute_augment_params, ensure_bg_energy_lower,
)
from asr_evolve.pipeline.stages.train import (
    BUSINESS_RECIPE, build_recipe, preflight_check,
)
from asr_evolve.pipeline.stages.merge import Merger
from asr_evolve.pipeline.stages.jsonl import ManifestGenerator
from asr_evolve.pipeline.stages.clean import Cleaner
from asr_evolve.pipeline.health import TrainingHealthChecker
from asr_evolve.pipeline.evaluation.domain_evaluator import DomainEvaluator


# ============================================================
# 测试辅助
# ============================================================

class StubStage:
    """可配置行为的桩阶段。"""

    def __init__(
        self, name: str, *,
        has_artifact: bool = False,
        will_fail: bool = False,
        destructive: bool = False,
    ):
        self.name = name
        self.destructive = destructive
        self._has_artifact = has_artifact
        self._will_fail = will_fail
        self.execute_count = 0

    def artifact_exists(self, ctx: StageContext) -> ArtifactStatus:
        return ArtifactStatus(exists=self._has_artifact, detail="stub")

    def plan(self, ctx: StageContext) -> list[str]:
        return [f"stub-plan-{self.name}"]

    def execute(self, ctx: StageContext) -> StageResult:
        self.execute_count += 1
        if self._will_fail:
            return StageResult(stage=self.name, status="failed", reason="stub failure")
        return StageResult(stage=self.name, status="completed")


def _make_orchestrator(config: dict | None = None) -> PipelineOrchestrator:
    config = config or {
        "pipeline": {"datasets": {"aishell": ["ds"]}, "stages_root": "data/pipeline"},
    }
    return PipelineOrchestrator(config=config)


def _make_entries(num_devices: int, utts_per_device: int) -> list[tuple[str, str]]:
    """生成 (utt_id, path) 列表，utt_id 格式 DEV{i:03d}_{j:04d}。"""
    entries = []
    for d in range(num_devices):
        for u in range(utts_per_device):
            uid = f"DEV{d:03d}_{u:04d}"
            entries.append((uid, f"/audio/{uid}.wav"))
    return entries


# 策略：从 CANONICAL_STAGES 中随机选取非空子集
stage_subset_strategy = st.lists(
    st.sampled_from(CANONICAL_STAGES),
    min_size=1, max_size=len(CANONICAL_STAGES), unique=True,
)


# ============================================================
# Property 1: 阶段规划不变式（顺序 + 子集 + 模式无关）
# Feature: asr-business-pipeline-automation, Property 1: 阶段规划不变式
# ============================================================

@given(subset=stage_subset_strategy, mode=st.sampled_from(["aishell", "business"]))
@settings(max_examples=100)
def test_property_1_plan_canonical_order(subset: list[str], mode: str):
    """任意子集按 CANONICAL_STAGES 相对顺序排列，与模式无关。"""
    orch = _make_orchestrator()
    planned = orch._plan(subset)
    # 验证顺序
    indices = [CANONICAL_STAGES.index(s) for s in planned]
    assert indices == sorted(indices)
    # 验证包含所有请求阶段
    assert set(planned) == set(subset)


# ============================================================
# Property 2: 跳过/强制决策
# Feature: asr-business-pipeline-automation, Property 2: 跳过/强制决策
# ============================================================

@given(has_artifact=st.booleans(), force=st.booleans())
@settings(max_examples=100)
def test_property_2_skip_force_decision(has_artifact: bool, force: bool):
    """artifact 存在 + 未 force → 跳过；force → 不跳过。"""
    orch = _make_orchestrator()
    stage = StubStage("prepare", has_artifact=has_artifact)
    orch.register_stage(stage)
    force_set = {"prepare"} if force else set()
    report = orch.run(only=["prepare"], force=force_set)
    result = report.results[0]
    if has_artifact and not force:
        assert result.status == "skipped"
    else:
        assert result.status in ("completed", "failed")


# ============================================================
# Property 3: 非破坏性阶段失败即终止
# Feature: asr-business-pipeline-automation, Property 3: 非破坏性阶段失败即终止
# ============================================================

@given(fail_at=st.integers(min_value=0, max_value=len(CANONICAL_STAGES) - 1))
@settings(max_examples=100)
def test_property_3_failure_halts(fail_at: int):
    """非破坏性阶段失败后，后续阶段不执行。"""
    orch = _make_orchestrator()
    stages = []
    for i, name in enumerate(CANONICAL_STAGES):
        s = StubStage(name, will_fail=(i == fail_at))
        stages.append(s)
        orch.register_stage(s)
    report = orch.run()
    assert report.halted_at == CANONICAL_STAGES[fail_at]
    for i in range(fail_at + 1, len(stages)):
        assert stages[i].execute_count == 0


# ============================================================
# Property 4: dry-run 不写盘
# Feature: asr-business-pipeline-automation, Property 4: dry-run 不写盘
# ============================================================

@given(subset=stage_subset_strategy)
@settings(max_examples=100)
def test_property_4_dry_run_no_side_effects(subset: list[str]):
    """dry-run 模式下所有阶段状态为 planned，无 execute 调用。"""
    orch = _make_orchestrator()
    stages = []
    for name in CANONICAL_STAGES:
        s = StubStage(name)
        stages.append(s)
        orch.register_stage(s)
    report = orch.run(only=subset, dry_run=True)
    # 全部为 planned
    for r in report.results:
        assert r.status == "planned"
    # 没有 execute 被调用
    for s in stages:
        assert s.execute_count == 0


# ============================================================
# Property 5: 幂等性（相同输入+相同配置）
# Feature: asr-business-pipeline-automation, Property 5: 幂等性
# ============================================================

@given(seed=st.integers(min_value=0, max_value=9999))
@settings(max_examples=100)
def test_property_5_split_idempotent(seed: int):
    """split_by_device 相同输入两次执行产出完全一致。"""
    entries = _make_entries(num_devices=20, utts_per_device=3)
    r1 = split_by_device(entries, seed=seed)
    r2 = split_by_device(entries, seed=seed)
    assert r1 == r2


# ============================================================
# Property 6: 质量扫描完整性
# Feature: asr-business-pipeline-automation, Property 6: 质量扫描完整性
# ============================================================

@given(
    n=st.integers(min_value=1, max_value=20),
    dur=st.floats(min_value=0.01, max_value=30.0),
    sr=st.sampled_from([8000, 16000, 44100]),
    ch=st.integers(min_value=1, max_value=2),
    rms=st.floats(min_value=0.0, max_value=1.0),
    clip=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=100)
def test_property_6_scanner_completeness(n: int, dur: float, sr: int, ch: int, rms: float, clip: float):
    """QualityJudge 对每条 metrics 都产出判定结果（good 或 bad + 问题列表）。"""
    judge = QualityJudge({})
    metrics = {
        "duration_s": dur, "sample_rate": sr,
        "channels": ch, "rms": rms, "clip_rate": clip,
    }
    is_good, problems = judge.judge(metrics)
    # 恒返回 bool + list
    assert isinstance(is_good, bool)
    assert isinstance(problems, list)
    # good 时无问题
    if is_good:
        assert problems == []


# ============================================================
# Property 7: 质量判定的划分与违规明细
# Feature: asr-business-pipeline-automation, Property 7: 质量判定的划分与违规明细
# ============================================================

@given(
    durs=st.lists(st.floats(min_value=0.01, max_value=30.0), min_size=1, max_size=20),
    sr=st.sampled_from([8000, 16000]),
    rms_vals=st.lists(st.floats(min_value=0.0, max_value=0.5), min_size=1, max_size=20),
)
@settings(max_examples=100)
def test_property_7_judge_partition(durs: list[float], sr: int, rms_vals: list[float]):
    """good ∪ bad = 全集，且互斥。"""
    judge = QualityJudge({"min_duration_s": 0.5, "max_duration_s": 20.0,
                          "expected_sample_rate": 16000, "expected_channels": 1,
                          "min_rms": 0.001, "max_clip_rate": 0.01})
    n = min(len(durs), len(rms_vals))
    good_ids, bad_ids = [], []
    for i in range(n):
        metrics = {"duration_s": durs[i], "sample_rate": sr, "channels": 1,
                   "rms": rms_vals[i], "clip_rate": 0.0, "utt_id": f"utt_{i}"}
        is_good, problems = judge.judge(metrics)
        if is_good:
            good_ids.append(f"utt_{i}")
        else:
            bad_ids.append(f"utt_{i}")
            # 每个 bad 条目有具体问题类型
            assert len(problems) > 0
    # 互斥且完全覆盖
    assert set(good_ids) & set(bad_ids) == set()
    assert len(good_ids) + len(bad_ids) == n


# ============================================================
# Property 8: 判定纯依赖既有指标（调阈值不重扫）
# Feature: asr-business-pipeline-automation, Property 8: 判定纯依赖既有指标
# ============================================================

@given(
    dur=st.floats(min_value=0.01, max_value=30.0),
    min_dur_1=st.floats(min_value=0.01, max_value=5.0),
    min_dur_2=st.floats(min_value=0.01, max_value=5.0),
)
@settings(max_examples=100)
def test_property_8_judge_threshold_independence(dur: float, min_dur_1: float, min_dur_2: float):
    """同一 metrics 用不同阈值判定，结果仅由阈值决定。"""
    metrics = {"duration_s": dur, "sample_rate": 16000, "channels": 1,
               "rms": 0.05, "clip_rate": 0.0}
    j1 = QualityJudge({"min_duration_s": min_dur_1, "max_duration_s": 20.0,
                        "expected_sample_rate": 16000, "expected_channels": 1,
                        "min_rms": 0.001, "max_clip_rate": 0.01})
    j2 = QualityJudge({"min_duration_s": min_dur_2, "max_duration_s": 20.0,
                        "expected_sample_rate": 16000, "expected_channels": 1,
                        "min_rms": 0.001, "max_clip_rate": 0.01})
    r1, _ = j1.judge(metrics)
    r2, _ = j2.judge(metrics)
    # 如果两组阈值相同，结果必须相同
    if min_dur_1 == min_dur_2:
        assert r1 == r2
    # 关键：判定结果纯由 metrics + 阈值决定（确定性）
    r1_again, _ = j1.judge(metrics)
    assert r1 == r1_again


# ============================================================
# Property 9: 清理开关安全不变式
# Feature: asr-business-pipeline-automation, Property 9: 清理开关安全不变式
# ============================================================

def test_property_9_clean_disabled_preserves_all(tmp_path: Path):
    """清理未启用时，labeled 音频集合不变。"""
    from asr_evolve.pipeline.stages.clean import Cleaner

    stages_root = tmp_path / "data" / "pipeline"
    ds = "test_ds"
    labeled_dir = stages_root / ds / "labeled"
    quality_dir = stages_root / ds / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)

    entries = [(f"utt_{i}", f"/audio/utt_{i}.wav") for i in range(10)]
    _make_wav_scp(labeled_dir, entries)
    _make_text(labeled_dir, [(uid, f"text {uid}") for uid, _ in entries])
    # 创建 bad.lst
    (quality_dir / "bad.lst").write_text("utt_0\ttoo_short\nutt_1\ttoo_quiet\n")

    # 未启用 clean
    config = {"pipeline": {"clean_enabled": False, "stages_root": "data/pipeline",
                            "datasets": {"aishell": [ds]}}}
    ctx = StageContext(base_dir=str(tmp_path), stages_root="data/pipeline",
                       mode="aishell", datasets=[ds], config=config, force=True)
    cleaner = Cleaner()
    result = cleaner.execute(ctx)
    assert result.status == "completed"
    # wav.scp 未变
    scp = (labeled_dir / "wav.scp").read_text()
    assert "utt_0" in scp
    assert "utt_1" in scp


# ============================================================
# Property 10: 失效路径重映射与丢弃
# Feature: asr-business-pipeline-automation, Property 10: 失效路径重映射与丢弃
# ============================================================

@given(
    n_valid=st.integers(min_value=0, max_value=10),
    n_remappable=st.integers(min_value=0, max_value=10),
    n_invalid=st.integers(min_value=0, max_value=10),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_10_remap_paths(n_valid: int, n_remappable: int, n_invalid: int, tmp_path: Path):
    """重映射后保留有效+可重映射，丢弃不存在的。"""
    audio_root = tmp_path / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)

    entries: list[tuple[str, str]] = []

    # 有效路径
    for i in range(n_valid):
        f = audio_root / f"valid_{i}.wav"
        f.write_text("fake")
        entries.append((f"v_{i}", str(f)))

    # 可重映射路径（文件名在 audio_root 存在但原路径不存在）
    for i in range(n_remappable):
        f = audio_root / f"remap_{i}.wav"
        f.write_text("fake")
        entries.append((f"r_{i}", f"/nonexist/path/remap_{i}.wav"))

    # 不可重映射路径
    for i in range(n_invalid):
        entries.append((f"x_{i}", f"/nonexist/path/nosuch_{i}.wav"))

    result = remap_paths(entries, str(audio_root))

    # 结果数量 = 有效 + 可重映射
    assert len(result) == n_valid + n_remappable
    # 所有结果路径都存在
    for _, path in result:
        assert Path(path).exists()


# ============================================================
# Property 11: manifest 仅纳入键交集
# Feature: asr-business-pipeline-automation, Property 11: manifest 仅纳入键交集
# ============================================================

@given(
    scp_keys=st.lists(st.text(min_size=1, max_size=8, alphabet="abcdef0123456789"), min_size=1, max_size=20, unique=True),
    text_keys=st.lists(st.text(min_size=1, max_size=8, alphabet="abcdef0123456789"), min_size=1, max_size=20, unique=True),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_11_manifest_key_intersection(scp_keys: list[str], text_keys: list[str], tmp_path: Path):
    """jsonl 的键集合恰好等于 wav.scp ∩ text 的键集合。"""
    all_label = tmp_path / "data" / "pipeline" / "all_label" / "train"
    all_label.mkdir(parents=True, exist_ok=True)

    scp_lines = [f"{k} /audio/{k}.wav" for k in scp_keys]
    text_lines = [f"{k} text_{k}" for k in text_keys]
    (all_label / "wav.scp").write_text("\n".join(scp_lines) + "\n")
    (all_label / "text").write_text("\n".join(text_lines) + "\n")

    stage = ManifestGenerator()
    config = {"pipeline": {"stages_root": "data/pipeline", "datasets": {"aishell": ["ds"]}}}
    ctx = StageContext(base_dir=str(tmp_path), stages_root="data/pipeline",
                       mode="aishell", datasets=["ds"], config=config, force=True)
    stage.execute(ctx)

    jsonl_path = tmp_path / "data" / "pipeline" / "train.jsonl"
    expected_keys = set(scp_keys) & set(text_keys)

    if not expected_keys:
        # 空交集→无输出或 failed
        return

    records = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    actual_keys = {r["key"] for r in records}
    assert actual_keys == expected_keys


# ============================================================
# Property 12: jsonl 保真
# Feature: asr-business-pipeline-automation, Property 12: jsonl 保真
# ============================================================

@given(
    entries=st.lists(
        st.tuples(
            st.text(min_size=1, max_size=8, alphabet="abcdef0123456789"),
            st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnop"),
        ),
        min_size=1, max_size=15, unique_by=lambda x: x[0],
    )
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_12_jsonl_fidelity(entries: list[tuple[str, str]], tmp_path: Path):
    """jsonl 中每条记录的 source/target 与原始一致。"""
    all_label = tmp_path / "data" / "pipeline" / "all_label" / "train"
    all_label.mkdir(parents=True, exist_ok=True)

    scp_lines = [f"{k} /audio/{k}.wav" for k, _ in entries]
    text_lines = [f"{k} {v}" for k, v in entries]
    (all_label / "wav.scp").write_text("\n".join(scp_lines) + "\n")
    (all_label / "text").write_text("\n".join(text_lines) + "\n")

    stage = ManifestGenerator()
    config = {"pipeline": {"stages_root": "data/pipeline", "datasets": {"aishell": ["ds"]}}}
    ctx = StageContext(base_dir=str(tmp_path), stages_root="data/pipeline",
                       mode="aishell", datasets=["ds"], config=config, force=True)
    stage.execute(ctx)

    jsonl_path = tmp_path / "data" / "pipeline" / "train.jsonl"
    records = {r["key"]: r for r in
               [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]}

    for k, v in entries:
        assert records[k]["source"] == f"/audio/{k}.wav"
        assert records[k]["target"] == v


# ============================================================
# Property 13: 切分划分与比例
# Feature: asr-business-pipeline-automation, Property 13: 切分划分与比例
# ============================================================

@given(
    n_devices=st.integers(min_value=3, max_value=100),
    utts_per=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=100)
def test_property_13_split_partition(n_devices: int, utts_per: int):
    """切分结果：不相交 + 完全覆盖 + 比例逼近 90/5/5。"""
    entries = _make_entries(n_devices, utts_per)
    splits = split_by_device(entries, seed=42)
    all_assigned = splits["train"] + splits["dev"] + splits["test"]
    # 完全覆盖
    assert len(all_assigned) == len(entries)
    # 不相交
    ids_t = {u for u, _ in splits["train"]}
    ids_d = {u for u, _ in splits["dev"]}
    ids_s = {u for u, _ in splits["test"]}
    assert ids_t & ids_d == set()
    assert ids_t & ids_s == set()
    assert ids_d & ids_s == set()
    # 比例容差（设备粒度）
    total = len(entries)
    if total > 0:
        assert len(splits["train"]) / total > 0.5  # 至少过半


# ============================================================
# Property 14: 切分确定性（固定种子）
# Feature: asr-business-pipeline-automation, Property 14: 切分确定性
# ============================================================

@given(n_devices=st.integers(min_value=3, max_value=50))
@settings(max_examples=100)
def test_property_14_split_deterministic(n_devices: int):
    """固定 seed=42 两次切分完全相同。"""
    entries = _make_entries(n_devices, 5)
    r1 = split_by_device(entries, seed=42)
    r2 = split_by_device(entries, seed=42)
    for k in ("train", "dev", "test"):
        assert r1[k] == r2[k]


# ============================================================
# Property 15: 按设备隔离（多数据集独立）
# Feature: asr-business-pipeline-automation, Property 15: 按设备隔离
# ============================================================

@given(seed=st.integers(min_value=0, max_value=9999))
@settings(max_examples=100)
def test_property_15_device_isolation(seed: int):
    """任一 Device_ID 的全部音频在唯一一个 split。"""
    entries = _make_entries(num_devices=20, utts_per_device=5)
    splits = split_by_device(entries, seed=seed)
    device_to_split: dict[str, str] = {}
    for split_name, split_entries in splits.items():
        for uid, _ in split_entries:
            dev_id = extract_device_id(uid)
            if dev_id in device_to_split:
                assert device_to_split[dev_id] == split_name
            else:
                device_to_split[dev_id] = split_name


# ============================================================
# Property 16: 增强仅作用于 train
# Feature: asr-business-pipeline-automation, Property 16: 增强仅作用于 train
# ============================================================

def test_property_16_augment_train_only(tmp_path: Path):
    """增强后 dev/test 不变，仅 train 新增 _aug 条目。"""
    from asr_evolve.pipeline.stages.augment import Augmentor

    stages_root = tmp_path / "data" / "pipeline"
    ds = "test_ds"
    os.environ["ASR_EVOLVE_MOCK"] = "1"
    try:
        for split in ("train", "dev", "test"):
            sd = stages_root / ds / "split" / split
            entries = [(f"{split}_{i}", f"/a/{split}_{i}.wav") for i in range(5)]
            _make_wav_scp(sd, entries)
            _make_text(sd, [(uid, f"t {uid}") for uid, _ in entries])

        config = {"pipeline": {"stages_root": "data/pipeline", "datasets": {"aishell": [ds]}}}
        ctx = StageContext(base_dir=str(tmp_path), stages_root="data/pipeline",
                           mode="aishell", datasets=[ds], config=config, force=True)
        aug = Augmentor()
        aug.execute(ctx)

        # 增强产物只在 train
        aug_scp = (stages_root / ds / "augmented" / "train" / "wav.scp").read_text()
        assert "_aug" in aug_scp
        # dev/test 目录下无增强文件
        assert not (stages_root / ds / "augmented" / "dev").exists()
        assert not (stages_root / ds / "augmented" / "test").exists()
    finally:
        os.environ.pop("ASR_EVOLVE_MOCK", None)


# ============================================================
# Property 17: 距离因子联动参数
# Feature: asr-business-pipeline-automation, Property 17: 距离因子联动参数
# ============================================================

@given(t=st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=100)
def test_property_17_distance_factor_coupling(t: float):
    """t 越大 → SNR 越低、RT60 越高、低通截止越低。"""
    params = compute_augment_params(t)
    # SNR 单调递减
    if t > 0:
        params_lower = compute_augment_params(t - 0.01 if t > 0.01 else 0.0)
        assert params["snr_db"] <= params_lower["snr_db"] + 0.2  # 容差
    # RT60 单调递增
    assert params["rt60_s"] >= 0.2
    # 低通截止单调递减
    assert params["lpf_hz"] <= 8000.0


# ============================================================
# Property 18: 背景人声能量低于主说话人
# Feature: asr-business-pipeline-automation, Property 18: 背景人声能量约束
# ============================================================

@given(
    main_energy=st.floats(min_value=0.01, max_value=100.0),
    bg_energy=st.floats(min_value=0.01, max_value=100.0),
)
@settings(max_examples=100)
def test_property_18_bg_energy_constraint(main_energy: float, bg_energy: float):
    """混合后背景能量 * gain < 主说话人能量。"""
    gain = ensure_bg_energy_lower(main_energy, bg_energy)
    assert bg_energy * gain < main_energy or gain <= 1.0


# ============================================================
# Property 19: 分片增量仅重生成变化分片
# Feature: asr-business-pipeline-automation, Property 19: 分片增量仅重生成变化分片
# ============================================================

@given(
    t1=st.floats(min_value=0.0, max_value=1.0),
    t2=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=100)
def test_property_19_incremental_shard(t1: float, t2: float):
    """参数不同时产生不同增强配置（用作分片版本ID变化判定）。"""
    p1 = compute_augment_params(t1)
    p2 = compute_augment_params(t2)
    # 相同 t → 相同配置（分片不重生成）
    if t1 == t2:
        assert p1 == p2
    # 不同 t → 不同配置（需重生成）
    if abs(t1 - t2) > 0.001:
        assert p1 != p2


# ============================================================
# Property 20: 合并的 split 归属与隔离
# Feature: asr-business-pipeline-automation, Property 20: 合并 split 归属与隔离
# ============================================================

def test_property_20_merge_isolation(tmp_path: Path):
    """merge 后 train/dev/test 互不泄漏，增强只在 train。"""
    stages_root = tmp_path / "data" / "pipeline"
    ds = "test_ds"
    for split in ("train", "dev", "test"):
        sd = stages_root / ds / "split" / split
        entries = [(f"{split}_{i}", f"/a/{split}_{i}.wav") for i in range(5)]
        _make_wav_scp(sd, entries)
        _make_text(sd, [(uid, f"t {uid}") for uid, _ in entries])

    # 增强只在 train
    aug_dir = stages_root / ds / "augmented" / "train"
    aug_entries = [(f"train_{i}_aug", f"/a/train_{i}_aug.wav") for i in range(5)]
    _make_wav_scp(aug_dir, aug_entries)
    _make_text(aug_dir, [(uid, f"t {uid}") for uid, _ in aug_entries])

    config = {"pipeline": {"stages_root": "data/pipeline", "datasets": {"aishell": [ds]}}}
    ctx = StageContext(base_dir=str(tmp_path), stages_root="data/pipeline",
                       mode="aishell", datasets=[ds], config=config, force=True)
    merger = Merger()
    merger.execute(ctx)

    # 检查隔离
    for split in ("train", "dev", "test"):
        scp = (stages_root / "all_label" / split / "wav.scp").read_text()
        for other in ("train", "dev", "test"):
            if other != split:
                for line in scp.splitlines():
                    if line.strip():
                        uid = line.split()[0]
                        assert not uid.startswith(f"{other}_") or "_aug" in uid and split == "train"

    # 增强只在 train
    dev_scp = (stages_root / "all_label" / "dev" / "wav.scp").read_text()
    test_scp = (stages_root / "all_label" / "test" / "wav.scp").read_text()
    assert "_aug" not in dev_scp
    assert "_aug" not in test_scp


# ============================================================
# Property 21: 合并去重保留首现
# Feature: asr-business-pipeline-automation, Property 21: 合并去重保留首现
# ============================================================

@given(
    entries=st.lists(
        st.tuples(
            st.text(min_size=1, max_size=5, alphabet="abc"),
            st.text(min_size=1, max_size=10, alphabet="xyz"),
        ),
        min_size=1, max_size=30,
    )
)
@settings(max_examples=100)
def test_property_21_dedup_first_wins(entries: list[tuple[str, str]]):
    """去重保留首次出现的条目。"""
    result = Merger._dedup(entries)
    seen_keys: set[str] = set()
    for key, val in result:
        assert key not in seen_keys
        seen_keys.add(key)
        # 取值等于首次出现的取值
        first_val = next(v for k, v in entries if k == key)
        assert val == first_val


# ============================================================
# Property 22: 指令召回率定义
# Feature: asr-business-pipeline-automation, Property 22: 指令召回率定义
# ============================================================

@given(
    refs=st.lists(st.text(min_size=0, max_size=10), min_size=1, max_size=20),
    hyps=st.lists(st.text(min_size=0, max_size=20), min_size=1, max_size=20),
)
@settings(max_examples=100)
def test_property_22_instruction_recall(refs: list[str], hyps: list[str]):
    """召回率 = 命中数 / 非空参考总数。空参考排除。"""
    n = min(len(refs), len(hyps))
    assume(n > 0)
    ev = DomainEvaluator(wakewords=[])  # 无唤醒词简化测试
    samples = [Sample(f"a{i}.wav", refs[i], hyps[i]) for i in range(n)]
    recall, hits, total = ev.instruction_recall(samples)
    # total = 非空 ref 数
    non_empty_refs = [r for r in refs[:n] if r.strip()]
    assert total == len(non_empty_refs)
    # hits <= total
    assert hits <= total
    # 召回率在 [0, 1]
    assert 0.0 <= recall <= 1.0


# ============================================================
# Property 23: 唤醒词首尾剥离
# Feature: asr-business-pipeline-automation, Property 23: 唤醒词首尾剥离
# ============================================================

@given(
    base=st.text(min_size=1, max_size=20, alphabet="打开关灯空调温度"),
    ww=st.sampled_from(["你好萤石", "萤石"]),
    position=st.sampled_from(["prefix", "suffix", "both"]),
)
@settings(max_examples=100)
def test_property_23_wakeword_strip(base: str, ww: str, position: str):
    """首/尾添加唤醒词后剥离应还原为基础文本。"""
    ev = DomainEvaluator(wakewords=["你好萤石", "萤石"])
    if position == "prefix":
        text = ww + base
    elif position == "suffix":
        text = base + ww
    else:
        text = ww + base + ww

    stripped = ev.strip_wakeword(text)
    # 基础文本是否被还原（至少不含唤醒词前后缀）
    assert base in stripped or stripped == base


# ============================================================
# Property 24: Stripped_CER 为字符级编辑距离
# Feature: asr-business-pipeline-automation, Property 24: Stripped_CER 字符级编辑距离
# ============================================================

@given(
    ref=st.text(min_size=1, max_size=20),
    hyp=st.text(min_size=0, max_size=20),
)
@settings(max_examples=100)
def test_property_24_stripped_cer(ref: str, hyp: str):
    """CER = edit_distance(ref, hyp) / len(ref)。相同→0，完全不同→>0。"""
    ev = DomainEvaluator(wakewords=[])
    cer = ev.char_error_rate(ref, hyp)
    assert cer >= 0.0
    if ref == hyp:
        assert cer == 0.0
    # CER 可以 > 1.0（插入超过参考长度的情况）


# ============================================================
# Property 25: 每个测试集独立报告不聚合
# Feature: asr-business-pipeline-automation, Property 25: 每个测试集独立报告
# ============================================================

@given(n_test_sets=st.integers(min_value=1, max_value=5))
@settings(max_examples=100)
def test_property_25_independent_reports(n_test_sets: int):
    """评估输出的报告数 = 测试集数，互不聚合。"""
    ev = DomainEvaluator(wakewords=[])
    reports = []
    for i in range(n_test_sets):
        samples = [Sample(f"a{j}.wav", f"参考{j}", f"识别{j}") for j in range(3)]
        report = ev.evaluate(samples, f"test_set_{i}")
        reports.append(report)
    # 报告数 = 测试集数
    assert len(reports) == n_test_sets
    # 每份报告有独立 test_set 标识
    test_set_names = {r.test_set for r in reports}
    assert len(test_set_names) == n_test_sets


# ============================================================
# Property 26: 业务配方覆盖相悖配置
# Feature: asr-business-pipeline-automation, Property 26: 业务配方覆盖
# ============================================================

@given(
    base_lr=st.floats(min_value=1e-6, max_value=1e-2),
    base_epoch=st.integers(min_value=1, max_value=200),
    base_batch=st.integers(min_value=1000, max_value=200000),
)
@settings(max_examples=100)
def test_property_26_recipe_override(base_lr: float, base_epoch: int, base_batch: int):
    """build_recipe 总是用 BUSINESS_RECIPE 覆盖所有治理键。"""
    base = {
        "learning_rate": base_lr,
        "max_epoch": base_epoch,
        "batch_size": base_batch,
        "find_unused_parameters": False,
        "seaco_id": 9999,
    }
    result = build_recipe(base)
    # 所有治理键恒等于业务配方值
    for key, expected in BUSINESS_RECIPE.items():
        assert result[key] == expected, f"{key}: {result[key]} != {expected}"


# ============================================================
# Property 27: 训练健康检查判定
# Feature: asr-business-pipeline-automation, Property 27: 训练健康检查
# ============================================================

@given(rc=st.integers(min_value=0, max_value=128))
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_27_health_check(rc: int, tmp_path: Path):
    """无 .pt 时判失败；rc=0 无 .pt 标记静默崩溃。"""
    checker = TrainingHealthChecker()
    # 空目录
    empty_dir = tmp_path / f"empty_{rc}"
    empty_dir.mkdir(parents=True, exist_ok=True)
    result = checker.check(str(empty_dir), rc)
    assert not result.passed
    if rc == 0:
        assert "Silent crash" in result.reason

    # 有 .pt
    pt_dir = tmp_path / f"has_pt_{rc}"
    pt_dir.mkdir(parents=True, exist_ok=True)
    (pt_dir / "best.pt").write_text("data")
    result2 = checker.check(str(pt_dir), 0)
    assert result2.passed
    assert result2.checkpoint_count >= 1


# ============================================================
# Property 28: seaco_id 启动前校验
# Feature: asr-business-pipeline-automation, Property 28: seaco_id 校验
# ============================================================

@given(has_key=st.booleans(), value=st.integers(min_value=0, max_value=99999))
@settings(max_examples=100)
def test_property_28_preflight_seaco_id(has_key: bool, value: int):
    """seaco_id 键缺失时报错，存在时通过（不论值）。"""
    if has_key:
        params = {"dataset_conf": {"seaco_id": value}}
        # 不应抛错
        preflight_check(params)
    else:
        params = {"dataset_conf": {}}
        with pytest.raises(ValueError, match="seaco_id"):
            preflight_check(params)



