#!/usr/bin/env python3
"""
S10 ASR 数据处理 Pipeline
按顺序执行各阶段脚本，支持从指定步骤开始、跳过步骤、dry-run 预览。

流水线步骤:
  1. 整理数据        01_prepare_funasr_data.py
  2. 音频质量扫描    02_audio_quality_check.py
  3. 生成质量报告    03_quality_report.py
  4. 分类问题音频    04_sort_bad_files.py
  5. 删除问题音频    05_remove_bad_files.py       (可选，需 --remove-bad)
  6. FireRedASR 预标注 06_eval_and_pretag/FireRedASR_pretag.py

用法:
    python run_pipeline.py                    # 执行全部步骤（跳过可选步骤）
    python run_pipeline.py --start 3          # 从第3步开始
    python run_pipeline.py --start 3 --end 5  # 只执行第3~5步
    python run_pipeline.py --skip 4           # 跳过第4步
    python run_pipeline.py --remove-bad       # 包含删除问题音频步骤
    python run_pipeline.py --dry-run          # 预览，不实际执行
    python run_pipeline.py --list             # 列出所有步骤
"""

import sys
import time
import argparse
import subprocess
from datetime import timedelta

# 流水线定义: (步骤号, 名称, 命令)
PIPELINE = [
    (1, "整理数据 (labeled/unlabeled)",
     [sys.executable, "01_prepare_funasr_data.py"]),

    (2, "音频质量扫描 → quality_detail.jsonl",
     [sys.executable, "02_audio_quality_check.py", "--workers", "8"]),

    (3, "生成质量报告 → bad_files/good_files",
     [sys.executable, "03_quality_report.py"]),

    (4, "分类问题音频 → bad_sorted/",
     [sys.executable, "04_sort_bad_files.py"]),

    (5, "删除问题音频 (可选，需 --remove-bad)",
     [sys.executable, "05_remove_bad_files.py"]),

    (6, "FireRedASR 预标注 → firered_pretag_output/",
     [sys.executable, "06_eval_and_pretag/FireRedASR_pretag.py"]),
]

# 需要用户确认才执行的步骤
CONFIRM_STEPS = {5}

# 默认跳过的可选步骤（需要显式启用）
OPTIONAL_STEPS = {5}


def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def list_steps():
    print("S10 ASR Pipeline 步骤列表:")
    print("-" * 50)
    for step_id, name, cmd in PIPELINE:
        optional = " (可选)" if step_id in OPTIONAL_STEPS else ""
        print(f"  [{step_id}] {name}{optional}")
    print("-" * 50)


def run_pipeline(args):
    skip_set = set()
    if args.skip:
        skip_set = {int(x.strip()) for x in args.skip.split(",")}

    steps = [
        (sid, name, cmd) for sid, name, cmd in PIPELINE
        if sid >= args.start and sid <= args.end and sid not in skip_set
        and (sid not in OPTIONAL_STEPS or args.remove_bad)
    ]

    if not steps:
        print("没有需要执行的步骤")
        return

    print("=" * 60)
    print("S10 ASR 数据处理 Pipeline")
    print("=" * 60)
    print(f"执行范围: 步骤 {args.start} ~ {args.end}")
    if skip_set:
        print(f"跳过步骤: {sorted(skip_set)}")
    if args.dry_run:
        print("模式: 预览 (dry-run)")
    print()

    total_steps = len(steps)
    passed = 0
    failed = 0
    skipped = 0
    pipeline_start = time.time()

    for idx, (step_id, name, cmd) in enumerate(steps, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{total_steps}] 步骤 {step_id}: {name}")
        print(f"命令: {' '.join(cmd)}")
        print("=" * 60)

        if args.dry_run:
            print("  (dry-run, 跳过执行)")
            skipped += 1
            continue

        # 需要确认的步骤
        if step_id in CONFIRM_STEPS and not args.yes:
            answer = input(f"\n⚠️  步骤 {step_id} 会修改/删除文件，确认执行? [y/N] ").strip().lower()
            if answer != 'y':
                print("  已跳过")
                skipped += 1
                continue

        step_start = time.time()
        try:
            result = subprocess.run(cmd, check=True)
            step_time = time.time() - step_start
            print(f"\n✓ 步骤 {step_id} 完成，耗时: {format_duration(step_time)}")
            passed += 1
        except subprocess.CalledProcessError as e:
            step_time = time.time() - step_start
            print(f"\n✗ 步骤 {step_id} 失败 (返回码: {e.returncode})，耗时: {format_duration(step_time)}")
            failed += 1
            if not args.continue_on_error:
                print("Pipeline 中断。使用 --continue-on-error 忽略错误继续。")
                break
        except KeyboardInterrupt:
            print("\n\n用户中断 Pipeline")
            break

    # 汇总
    total_time = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print("Pipeline 执行完毕")
    print(f"  成功: {passed}  失败: {failed}  跳过: {skipped}")
    print(f"  总耗时: {format_duration(total_time)}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="S10 ASR 数据处理 Pipeline")
    parser.add_argument(
        '--start', type=int, default=1,
        help='从第几步开始 (default: 1)'
    )
    parser.add_argument(
        '--end', type=int, default=99,
        help='执行到第几步 (default: 全部)'
    )
    parser.add_argument(
        '--skip', type=str, default=None,
        help='跳过的步骤，逗号分隔，如 --skip 5,6'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='预览模式，不实际执行'
    )
    parser.add_argument(
        '--yes', '-y', action='store_true',
        help='跳过确认提示，自动执行所有步骤'
    )
    parser.add_argument(
        '--remove-bad', action='store_true',
        help='启用可选的删除问题音频步骤 (步骤5)'
    )
    parser.add_argument(
        '--continue-on-error', action='store_true',
        help='某步失败后继续执行后续步骤'
    )
    parser.add_argument(
        '--list', action='store_true',
        help='列出所有步骤'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.list:
        list_steps()
    else:
        run_pipeline(args)
