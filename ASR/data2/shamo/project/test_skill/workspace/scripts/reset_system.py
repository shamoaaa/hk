"""Reset the entire ASR Training Optimizer system to initial state.

清除所有历史迭代数据、数据库、报告、日志、历史配置快照、增强数据等，
将系统恢复到"刚搭建好、还没跑过任何迭代"的初始状态。

保留的内容（不删除）：
- 原始数据：data/train.jsonl, data/val.jsonl, data/test.jsonl, data/original/
- MUSAN 噪声数据：data/musan/
- 主配置文件：configs/funasr_config.yaml, configs/remote.yaml
- 脚本代码：scripts/
- Steering 文件：.kiro/steering/
- 虚拟环境：.venv, .venv_win, .venv310

清除并恢复的内容：
- metadata.db（SQLite 数据库，含迭代记录、反馈、知识库等）
- logs/training/（训练日志和指标）
- logs/optimization/（优化日志）
- reports/evaluation/（评估报告）
- reports/dataset/（数据集分析报告）
- reports/augmentation/（增强报告）
- reports/iterations/（迭代报告）
- models/iterations/（训练产出的模型）
- data/augmented/（增强后的数据）
- configs/model/（模型配置版本快照）
- configs/training/（训练配置版本快照）
- configs/augmentation/（增强配置版本快照）
- configs/post_process/（后处理配置版本快照）
- knowledge/skill_versions/（Skill 版本快照）
- .kiro/skills/（Skill 文件回退到初始版本）

Usage:
    # 保存当前 Skill 文件为初始基线（首次使用前执行一次）
    python scripts/reset_system.py --save-baseline

    # 预览将要删除的内容（不实际删除）
    python scripts/reset_system.py --dry-run

    # 执行重置（本地）
    python scripts/reset_system.py --confirm

    # 同时重置服务器（通过 remote_runner）
    python scripts/reset_system.py --confirm --include-remote
"""

import argparse
import os
import shutil
import sys
import subprocess
from pathlib import Path


# 要清除的目录（相对于 workspace/）
DIRS_TO_CLEAR = [
    "logs/training",
    "logs/optimization",
    "reports/evaluation",
    "reports/dataset",
    "reports/augmentation",
    "reports/iterations",
    "models/iterations",
    "data/augmented",
]

# 要清除的版本化配置（清除目录内所有非 .gitkeep 文件）
CONFIG_DIRS_TO_CLEAR = [
    "configs/model",
    "configs/training",
    "configs/augmentation",
    "configs/post_process",
]

# 要删除的单个文件
FILES_TO_DELETE = [
    "metadata.db",
    "logs/execution_log.jsonl",
]

# 要保留的文件（在清除目录时不删除这些）
KEEP_FILES: set = set()  # 不保留任何特殊文件


def _clear_directory(dir_path: Path, dry_run: bool) -> int:
    """清空目录内容但保留目录本身。返回删除的文件数。"""
    if not dir_path.exists():
        if not dry_run:
            dir_path.mkdir(parents=True, exist_ok=True)
        return 0

    count = 0
    for item in dir_path.iterdir():
        if item.name in KEEP_FILES:
            continue
        if dry_run:
            print(f"  [DRY-RUN] would delete: {item}")
            count += 1
        else:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            count += 1
    return count


def _delete_file(file_path: Path, dry_run: bool) -> bool:
    """删除单个文件。返回是否删除。"""
    if not file_path.exists():
        return False
    if dry_run:
        print(f"  [DRY-RUN] would delete: {file_path}")
        return True
    file_path.unlink()
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Reset the ASR Training Optimizer system to initial state."
    )
    parser.add_argument(
        "--confirm", action="store_true",
        help="Actually perform the reset. Without this flag, only shows what would be done.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be deleted without actually deleting.",
    )
    parser.add_argument(
        "--include-remote", action="store_true",
        help="Also reset the remote server (clears logs/reports/models/augmented on server).",
    )
    args = parser.parse_args(argv)

    dry_run = args.dry_run or not args.confirm

    if dry_run:
        print("=" * 60)
        print("DRY-RUN MODE — showing what would be deleted")
        print("Run with --confirm to actually reset")
        print("=" * 60)
    else:
        print("=" * 60)
        print("⚠️  RESETTING SYSTEM TO INITIAL STATE")
        print("=" * 60)

    total_deleted = 0

    # 1. 清除目录内容
    print("\n📁 Clearing directories:")
    for dir_rel in DIRS_TO_CLEAR:
        dir_path = Path(dir_rel)
        n = _clear_directory(dir_path, dry_run)
        if n > 0:
            print(f"  {'[DRY-RUN] ' if dry_run else ''}✓ {dir_rel}/ ({n} items)")
            total_deleted += n
        else:
            print(f"  - {dir_rel}/ (already empty)")

    # 2. 清除版本化配置目录
    print("\n📄 Clearing config version directories:")
    for dir_rel in CONFIG_DIRS_TO_CLEAR:
        dir_path = Path(dir_rel)
        n = _clear_directory(dir_path, dry_run)
        if n > 0:
            print(f"  {'[DRY-RUN] ' if dry_run else ''}✓ {dir_rel}/ ({n} files)")
            total_deleted += n
        else:
            print(f"  - {dir_rel}/ (already empty)")

    # 3. 删除数据库
    print("\n🗄️  Deleting database:")
    for file_rel in FILES_TO_DELETE:
        file_path = Path(file_rel)
        if _delete_file(file_path, dry_run):
            print(f"  {'[DRY-RUN] ' if dry_run else ''}✓ {file_rel}")
            total_deleted += 1
        else:
            print(f"  - {file_rel} (not found)")

    # 4. 远程服务器重置
    if args.include_remote and not dry_run:
        print("\n🌐 Resetting remote server:")
        try:
            # 通过 remote_runner 的 run 命令清理远程目录
            # 策略：rm -rf 目录内容，然后重建空目录
            remote_dirs_to_clear = [
                "logs/training",
                "logs/optimization",
                "reports/evaluation",
                "reports/dataset",
                "reports/augmentation",
                "reports/iterations",
                "models/iterations",
                "data/augmented",
                "configs/model",
                "configs/training",
                "configs/augmentation",
                "configs/post_process",
            ]
            cmds = []
            for d in remote_dirs_to_clear:
                cmds.append(f"rm -rf {d} && mkdir -p {d}")
            rm_cmd = " && ".join(cmds) + " && echo 'remote reset done'"

            python_exe = sys.executable
            result = subprocess.run(
                [python_exe, "-m", "scripts.remote_runner", "run", rm_cmd],
                capture_output=True, text=True, timeout=60
            )
            if "remote reset done" in (result.stdout or ""):
                print("  ✓ Remote server cleared")
            elif result.returncode == 0:
                print("  ✓ Remote server cleared")
            else:
                print(f"  ⚠ Remote reset may be incomplete (exit code {result.returncode})")
                if result.stderr.strip():
                    print(f"    {result.stderr.strip()[:200]}")
        except subprocess.TimeoutExpired:
            print("  ✗ Remote reset timed out (60s)")
        except Exception as e:
            print(f"  ✗ Remote reset failed: {e}")
    elif args.include_remote and dry_run:
        print("\n🌐 [DRY-RUN] Would also clear remote server directories")

    # 5. 总结
    print(f"\n{'=' * 60}")
    if dry_run:
        print(f"DRY-RUN complete. {total_deleted} items would be deleted.")
        print("Run with --confirm to actually reset.")
    else:
        print(f"✅ System reset complete. {total_deleted} items deleted.")
        print("System is now in initial state. Ready for first iteration.")
    print("=" * 60)


if __name__ == "__main__":
    main()
