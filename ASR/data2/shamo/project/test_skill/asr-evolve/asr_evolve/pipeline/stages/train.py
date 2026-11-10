# -*- coding: utf-8 -*-
"""Stage: train — 远程单卡训练 (nohup + funasr.bin.train).

经实战验证的训练路径：
1. preflight: seaco_id 校验
2. 生成远程 bash 训练脚本
3. nohup 后台提交
4. 轮询等待完成
5. 健康检查: .pt 存在性
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from asr_evolve.pipeline.models import ArtifactStatus, StageContext, StageResult


def preflight_check(config: dict) -> None:
    """校验 seaco_id。缺失会导致训练静默崩溃。"""
    model_cfg = config.get("model", {})
    seaco_id = model_cfg.get("seaco", {}).get("seaco_id")
    if not seaco_id:
        raise ValueError(
            "model.seaco.seaco_id 未设置 — 训练会静默崩溃。"
            "请在 configs/model.yaml 中设置 seaco.seaco_id。"
        )


class RecipeTrainer:
    """远程单卡训练阶段（nohup 后台 + 轮询）。"""

    name = "train"
    destructive = False

    def artifact_exists(self, ctx: StageContext) -> ArtifactStatus:
        """训练阶段不做本地产物检测——每次需显式触发。"""
        return ArtifactStatus(exists=False)

    def plan(self, ctx: StageContext) -> list[str]:
        try:
            import yaml
            with open("configs/training.yaml", encoding="utf-8") as f:
                tcfg = yaml.safe_load(f).get("training", {})
        except (FileNotFoundError, Exception):
            tcfg = {}
        return [
            f"Remote funasr.bin.train (single GPU, nohup background): "
            f"bs={tcfg.get('batch_size','?')}({tcfg.get('batch_type','?')}), "
            f"accum_grad={tcfg.get('accum_grad','?')}, "
            f"epochs={tcfg.get('max_epoch','?')}, "
            f"lr={tcfg.get('learning_rate','?')}"
        ]

    def execute(self, ctx: StageContext) -> StageResult:
        """提交训练并等待完成。"""
        is_mock = os.environ.get("ASR_EVOLVE_MOCK", "0") == "1"
        if is_mock:
            return self._mock_train(ctx)

        # 调用 train 命令的核心逻辑
        from asr_evolve.commands.train import (
            _load_configs,
            _get_executor,
            _build_train_script,
            _save_state,
        )
        from datetime import datetime
        import json

        try:
            remote_cfg, training_cfg, model_cfg = _load_configs()
        except FileNotFoundError as e:
            return StageResult(stage=self.name, status="failed", reason=str(e))

        # preflight
        try:
            preflight_check(model_cfg)
        except ValueError as e:
            return StageResult(stage=self.name, status="failed", reason=str(e))

        tcfg = training_cfg["training"]
        r = remote_cfg["remote"]
        ws = r["workspace"]
        tag = f"exp_{int(time.time())}"
        output_dir = f"{ws}/output/{tag}"

        # 生成远程脚本
        script_content = _build_train_script(remote_cfg, training_cfg, model_cfg, output_dir)

        try:
            executor = _get_executor(remote_cfg)
        except Exception as e:
            return StageResult(stage=self.name, status="failed", reason=f"SSH failed: {e}")

        try:
            # 写远程脚本
            remote_script = f"{ws}/run_train_current.sh"
            rc, _, err = executor.run_command(
                f"cat > {remote_script} << 'ASR_EVOLVE_EOF'\n{script_content}\nASR_EVOLVE_EOF\nchmod +x {remote_script}",
                timeout=10,
            )
            if rc != 0:
                return StageResult(stage=self.name, status="failed", reason=f"Script write: {err}")

            # 创建目录
            executor.run_command(f"mkdir -p {output_dir}", timeout=10)

            # 后台提交
            rc, out, err = executor.run_command(
                f"nohup bash {remote_script} > {output_dir}/train_stdout.log 2>&1 & echo $!",
                timeout=15,
            )
            pid = out.strip()
            if rc != 0 or not pid:
                return StageResult(stage=self.name, status="failed", reason=f"Submit failed: {err}")

            # 保存状态
            _save_state({
                "tag": tag, "pid": pid, "output_dir": output_dir,
                "gpu": tcfg["distributed"]["cuda_visible_devices"],
                "submitted_at": datetime.now().isoformat(),
                "status": "running",
            })

            print(f"  [train] Submitted: tag={tag}, pid={pid}")

            # 轮询等待（pipeline 模式下同步等待完成）
            poll_interval = 60
            max_polls = 480  # 8 小时
            for i in range(max_polls):
                time.sleep(poll_interval)
                rc, alive, _ = executor.run_command(
                    f"kill -0 {pid} 2>/dev/null && echo RUNNING || echo DONE",
                    timeout=10,
                )
                if alive.strip() == "DONE":
                    break
                if i % 5 == 4:
                    _, tail, _ = executor.run_command(
                        f"tail -1 {output_dir}/train_stdout.log 2>/dev/null", timeout=10
                    )
                    print(f"  [train] {(i+1)}min | {tail.strip()[:80]}")
            else:
                return StageResult(
                    stage=self.name, status="failed",
                    reason="Training exceeded 8h poll timeout",
                )

            # 健康检查
            _, pts, _ = executor.run_command(
                f"find {output_dir} -name '*.pt' | wc -l", timeout=10
            )
            n_pts = int(pts.strip()) if pts.strip().isdigit() else 0

            if n_pts == 0:
                return StageResult(
                    stage=self.name, status="failed",
                    reason="Silent crash: rc=0 but no .pt checkpoints",
                )

            return StageResult(
                stage=self.name, status="completed",
                entry_count=n_pts,
                artifacts=[output_dir],
                reason=f"Training completed: {tag} ({n_pts} checkpoints)",
            )

        finally:
            executor.disconnect()

    def _mock_train(self, ctx: StageContext) -> StageResult:
        """Mock 训练（测试用）。"""
        base = Path(ctx.base_dir)
        output_dir = base / ctx.stages_root / "output" / f"mock_{int(time.time())}"
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("model.pt.best", "model.pt.ep0.500", "model.pt.ep1.1000"):
            (output_dir / name).write_text("mock\n")
        return StageResult(
            stage=self.name, status="completed",
            entry_count=3,
            artifacts=[str(output_dir)],
            reason="Mock training completed",
        )
