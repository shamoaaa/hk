"""自动诊断脚本：基于当前状态输出推荐的下一步动作。

读取最近的评估报告、训练指标、迭代历史，自动判断：
- 训练是否充分
- 是否停滞
- 应该优化哪个维度
- 具体推荐动作

Usage:
    python scripts/auto_diagnose.py
    python scripts/auto_diagnose.py --eval-report reports/evaluation/eval_report_iter_001.json
    python scripts/auto_diagnose.py --verbose
"""

import argparse
import json
import sys
from pathlib import Path

from scripts.core.database import Database
from scripts.core.config_utils import load_config, read_json, read_jsonl


_DEFAULT_DB_PATH = "metadata.db"


def _find_latest_eval_report() -> str | None:
    """Find the most recent eval_report JSON file."""
    report_dir = Path("reports/evaluation")
    if not report_dir.exists():
        return None
    reports = sorted(report_dir.glob("eval_report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(reports[0]) if reports else None


def _find_latest_training_metrics() -> str | None:
    """Find training_metrics.jsonl."""
    p = Path("logs/training/training_metrics.jsonl")
    return str(p) if p.exists() else None


def _load_iteration_history(db: Database) -> list[dict]:
    """Load completed iterations ordered by time."""
    rows = db.get_all("iterations", order_by="created_at ASC")
    return [dict(r) for r in rows]


def _analyze_training(metrics_path: str) -> dict:
    """Analyze training metrics for convergence, NaN, and overfitting."""
    records = read_jsonl(metrics_path)
    if not records:
        return {"status": "no_data"}

    losses = [r["train_loss"] for r in records if r.get("train_loss") is not None]
    if len(losses) < 2:
        return {"status": "insufficient_data", "steps": len(losses)}

    first_loss = losses[0]
    last_loss = losses[-1]
    loss_reduction = (first_loss - last_loss) / first_loss if first_loss > 0 else 0

    # Check if loss is still decreasing in last 20% of training
    tail_start = int(len(losses) * 0.8)
    tail_losses = losses[tail_start:]
    still_decreasing = len(tail_losses) > 1 and tail_losses[-1] < tail_losses[0]

    # Check for NaN/Inf
    has_nan = any(r.get("train_loss") is not None and (r["train_loss"] != r["train_loss"]) for r in records)

    # Overfitting detection: compare train_loss vs val_loss if available
    val_losses = [r["val_loss"] for r in records if r.get("val_loss") is not None]
    overfitting = False
    train_val_gap = None
    if val_losses and losses:
        last_train = losses[-1]
        last_val = val_losses[-1]
        if last_train > 0:
            train_val_gap = (last_val - last_train) / last_train
            # If val_loss is more than 2x train_loss, likely overfitting
            overfitting = train_val_gap > 1.0

    return {
        "status": "ok",
        "total_steps": len(losses),
        "first_loss": first_loss,
        "last_loss": last_loss,
        "loss_reduction_pct": loss_reduction * 100,
        "still_decreasing": still_decreasing,
        "has_nan": has_nan,
        "overfitting": overfitting,
        "train_val_gap": train_val_gap,
    }


def _check_stagnation(iterations: list[dict]) -> dict:
    """Check if optimization is stagnating."""
    completed = [it for it in iterations if it.get("status") == "completed" and it.get("cer_after")]
    if len(completed) < 2:
        return {"stagnating": False, "reason": "not_enough_iterations"}

    # Check last 3 iterations
    recent = completed[-3:] if len(completed) >= 3 else completed
    cer_values = [it["cer_after"] for it in recent]

    # Stagnating if CER hasn't improved by more than 0.001 across recent iterations
    if len(cer_values) >= 3:
        improvement = cer_values[0] - cer_values[-1]
        if improvement < 0.001:
            return {"stagnating": True, "reason": f"CER unchanged across {len(recent)} iterations", "cer_values": cer_values}

    return {"stagnating": False, "cer_values": cer_values}


def diagnose(eval_report_path: str | None = None, verbose: bool = False) -> dict:
    """Run full diagnosis and return recommended actions."""

    recommendations: list[dict] = []
    diagnosis: dict = {}

    # 1. Load eval report
    if not eval_report_path:
        eval_report_path = _find_latest_eval_report()

    if eval_report_path and Path(eval_report_path).exists():
        eval_data = read_json(eval_report_path)
        diagnosis["eval_report"] = eval_report_path
        diagnosis["cer"] = eval_data.get("cer", eval_data.get("overall_cer", 0))
        diagnosis["wer"] = eval_data.get("wer", eval_data.get("overall_wer", 0))
        diagnosis["error_distribution"] = eval_data.get("error_type_distribution", {})
        diagnosis["badcase_count"] = len(eval_data.get("badcases", []))

        # Strategy matching
        from scripts.decision.badcase_strategy_map import match_strategies_dynamic
        class _Report:
            def __init__(self, data):
                self.overall_cer = data.get("cer", data.get("overall_cer", 0))
                self.overall_wer = data.get("wer", data.get("overall_wer", 0))
                self.error_type_distribution = data.get("error_type_distribution", {})
                self.scene_metrics = []
                for sm in data.get("scene_metrics", []):
                    self.scene_metrics.append(type("SM", (), sm)())

        matched = match_strategies_dynamic(_Report(eval_data))
        diagnosis["matched_strategies"] = [s.get("name") for s in matched]

        for s in matched[:3]:  # Top 3 strategies
            recommendations.append({
                "action": s.get("name"),
                "priority": s.get("priority", 0),
                "description": s.get("description", ""),
                "commands": s.get("commands", []),
            })

        # Error pattern analysis (Ch21 FAQ)
        error_dist = eval_data.get("error_type_distribution", {})
        total_errors = sum(error_dist.values()) if error_dist else 0
        if total_errors > 0:
            sub_ratio = error_dist.get("substitution", 0) / total_errors
            del_ratio = error_dist.get("deletion", 0) / total_errors
            ins_ratio = error_dist.get("insertion", 0) / total_errors
            diagnosis["error_ratios"] = {
                "substitution": sub_ratio,
                "deletion": del_ratio,
                "insertion": ins_ratio,
            }

            # Rule: deletion dominant → model is "eating" characters
            if del_ratio > 0.4:
                recommendations.append({
                    "action": "fix_deletion",
                    "priority": 4,
                    "description": f"Deletion 占比 {del_ratio:.0%}，模型吃字严重。检查 max_token_length 或增加训练 epoch",
                    "commands": ["# 检查 max_token_length", "# 增加 epoch"],
                })

            # Rule: substitution dominant → hotword / confusion issue
            if sub_ratio > 0.7:
                recommendations.append({
                    "action": "fix_substitution",
                    "priority": 4,
                    "description": f"Substitution 占比 {sub_ratio:.0%}，同音混淆为主。优先提取热词",
                    "commands": ["# extract_hotwords --merge"],
                })

        # Repetition detection in badcases
        badcases = eval_data.get("badcases", [])
        if badcases:
            repeat_count = 0
            for bc in badcases:
                hyp = bc.get("hypothesis_text", "")
                # Detect repeated 3+ char sequences
                if len(hyp) >= 6:
                    for k in range(3, len(hyp) // 2 + 1):
                        if hyp[:k] * 2 in hyp or any(
                            hyp[i:i+k] == hyp[i+k:i+2*k]
                            for i in range(len(hyp) - 2*k + 1)
                        ):
                            repeat_count += 1
                            break
            repeat_ratio = repeat_count / len(badcases) if badcases else 0
            diagnosis["repetition_ratio"] = repeat_ratio

            if repeat_ratio > 0.1:
                recommendations.append({
                    "action": "fix_repetition",
                    "priority": 4,
                    "description": f"重复率 {repeat_ratio:.0%}，Attention 对齐问题。建议增加 CTC 权重或 Length Penalty",
                    "commands": ["# 增加 CTC loss 权重"],
                })
    else:
        diagnosis["eval_report"] = None

    # 2. Analyze training
    metrics_path = _find_latest_training_metrics()
    if metrics_path:
        training_analysis = _analyze_training(metrics_path)
        diagnosis["training"] = training_analysis

        if training_analysis.get("status") == "ok":
            # Rule: if loss reduction < 5%, training is insufficient
            if training_analysis["loss_reduction_pct"] < 5:
                recommendations.append({
                    "action": "increase_training",
                    "priority": 5,
                    "description": f"训练不充分（loss 仅下降 {training_analysis['loss_reduction_pct']:.1f}%），建议增加 epoch 或提高学习率",
                    "commands": ["# 增加 max_epoch 或提高 learning_rate"],
                })

            # Rule: if still decreasing, more epochs would help
            if training_analysis["still_decreasing"]:
                recommendations.append({
                    "action": "more_epochs",
                    "priority": 3,
                    "description": "Loss 仍在下降，增加 epoch 可能继续改善",
                    "commands": ["# 增加 max_epoch"],
                })

            # Rule: NaN detected
            if training_analysis.get("has_nan"):
                recommendations.append({
                    "action": "reduce_lr_for_nan",
                    "priority": 5,
                    "description": "训练出现 NaN，建议降低学习率至当前值的 1/10",
                    "commands": ["# 降低 learning_rate"],
                })

            # Rule: overfitting detected (Ch21 Q2)
            if training_analysis.get("overfitting"):
                gap = training_analysis.get("train_val_gap", 0)
                recommendations.append({
                    "action": "fix_overfitting",
                    "priority": 4,
                    "description": f"过拟合（val/train loss gap={gap:.1%}），建议增加 SpecAugment / 数据增强 / 减小模型",
                    "commands": [
                        "# 确认 spec_aug.enabled: true",
                        "# 增加数据增强概率",
                        "# 或减少 max_epoch",
                    ],
                })

    # 3. Check stagnation
    try:
        db = Database(_DEFAULT_DB_PATH)
        db.initialize()
        iterations = _load_iteration_history(db)
        stagnation = _check_stagnation(iterations)
        diagnosis["stagnation"] = stagnation

        if stagnation["stagnating"]:
            recommendations.append({
                "action": "switch_dimension",
                "priority": 4,
                "description": "优化停滞，建议切换优化维度（如从训练参数切到数据增强或模型结构）",
                "commands": ["# 切换到未尝试过的优化维度"],
            })
    except Exception:
        diagnosis["stagnation"] = {"error": "database not available"}

    # 4. Sort recommendations by priority
    recommendations.sort(key=lambda r: -r["priority"])

    return {
        "status": "success",
        "diagnosis": diagnosis,
        "recommendations": recommendations,
        "top_action": recommendations[0] if recommendations else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Auto-diagnose and recommend next optimization action.")
    parser.add_argument("--eval-report", default=None, help="Path to eval report JSON (auto-detect if omitted).")
    parser.add_argument("--verbose", action="store_true", help="Include full diagnosis details.")
    parser.add_argument("--db-path", default=_DEFAULT_DB_PATH)

    args = parser.parse_args(argv)
    result = diagnose(eval_report_path=args.eval_report, verbose=args.verbose)

    if not args.verbose:
        # Compact output: just recommendations
        output = {
            "status": result["status"],
            "top_action": result["top_action"],
            "all_recommendations": result["recommendations"],
        }
    else:
        output = result

    json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")

    from scripts.core.execution_logger import log_execution
    log_execution("auto_diagnose", output)


if __name__ == "__main__":
    main()
