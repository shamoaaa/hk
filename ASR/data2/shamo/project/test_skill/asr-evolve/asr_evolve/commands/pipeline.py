"""pipeline 命令 — 流水线编排."""
import yaml
from pathlib import Path
from asr_evolve.pipeline import PipelineOrchestrator, CANONICAL_STAGES
from asr_evolve.pipeline.stages import ALL_STAGES


def cmd_pipeline(args):
    cfg_path = Path("configs/asr_evolve.yaml")
    config = {}
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    orch = PipelineOrchestrator(config=config, base_dir=".")
    for StageClass in ALL_STAGES:
        orch.register_stage(StageClass())

    if args.pipe_action == "run":
        force = set(args.force) if args.force else None
        report = orch.run(
            only=None, dry_run=args.dry_run,
            force=force, mode=args.mode,
        )
    elif args.pipe_action == "stage":
        only = [s.strip() for s in args.only.split(",")]
        force = set(args.force) if args.force else None
        report = orch.run(
            only=only, dry_run=False,
            force=force, mode=args.mode,
        )
    else:
        print("用法: asr-evolve pipeline {run|stage}")
        return

    # 打印报告
    print(f"\n{'='*40}")
    print(f"Pipeline Report (mode={report.mode})")
    print(f"{'='*40}")
    print(f"Planned stages: {report.planned}")
    for r in report.results:
        icon = {"completed": "✅", "skipped": "⏭️", "failed": "❌", "planned": "📋"}.get(r.status, "?")
        print(f"  {icon} {r.stage}: {r.status} {('— ' + r.reason) if r.reason else ''}")
    if report.halted_at:
        print(f"\n⚠️  Halted at: {report.halted_at}")
    print(f"{'='*40}")
