"""生成单次迭代的详细执行报告。

在每轮优化迭代完成后调用，汇总本次迭代的所有细节：
- 训练指标（loss/lr/cer/wer 趋势）
- 评估结果（整体 + 场景级 CER/WER）
- Badcase 分析（错误模式、top badcase）
- 执行的优化操作（配置变更）
- 前后对比（CER/WER delta）
- 知识库更新（新增/修改的条目）
- Skill 进化（权重变化、规则更新）

输出 markdown 报告到 reports/iterations/<iter_id>_report.md

Usage:
    PYTHONPATH=. .venv/bin/python scripts/generate_iteration_report.py --iteration-id <iter_id>
    PYTHONPATH=. .venv/bin/python scripts/generate_iteration_report.py --latest
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from scripts.core.config_utils import load_config, read_jsonl
from scripts.core.database import Database

logger = logging.getLogger(__name__)


def _get_latest_iteration(db: Database) -> dict | None:
    """获取最近一次迭代记录。"""
    conn = db.get_connection()
    cur = conn.execute(
        "SELECT * FROM iterations ORDER BY created_at DESC LIMIT 1"
    )
    row = cur.fetchone()
    return dict(row) if row else None


def _get_iteration(db: Database, iter_id: str) -> dict | None:
    """获取指定迭代记录。"""
    return db.get_by_id("iterations", "iteration_id", iter_id)


def _get_feedbacks(db: Database, iter_id: str) -> list[dict]:
    """获取迭代相关的反馈。"""
    conn = db.get_connection()
    cur = conn.execute(
        "SELECT * FROM effect_feedbacks WHERE iteration_id = ?", (iter_id,)
    )
    return [dict(r) for r in cur.fetchall()]


def _get_recent_knowledge(db: Database, limit: int = 10) -> list[dict]:
    """获取最近更新的知识条目。"""
    conn = db.get_connection()
    cur = conn.execute(
        "SELECT * FROM knowledge_entries ORDER BY updated_at DESC LIMIT ?",
        (limit,),
    )
    return [dict(r) for r in cur.fetchall()]


def _get_strategy_weights(db: Database) -> list[dict]:
    """获取当前策略权重。"""
    conn = db.get_connection()
    cur = conn.execute(
        "SELECT * FROM strategy_weights ORDER BY weight DESC"
    )
    return [dict(r) for r in cur.fetchall()]


def _get_training_metrics(log_dir: str, tag: str = "") -> list[dict]:
    """读取训练指标。"""
    metrics_file = Path(log_dir) / "training_metrics.jsonl"
    if not metrics_file.exists():
        return []
    metrics = read_jsonl(str(metrics_file))
    if tag:
        # 过滤属于本次 tag 的指标（如果有 tag 字段）
        tagged = [m for m in metrics if m.get("tag") == tag]
        if tagged:
            return tagged
    return metrics


def _get_eval_report(reports_dir: str, tag: str) -> dict | None:
    """读取评估报告。"""
    report_path = Path(reports_dir) / f"eval_report_{tag}.json"
    if report_path.exists():
        with open(report_path) as f:
            return json.load(f)
    # 尝试找最新的报告
    reports = sorted(Path(reports_dir).glob("eval_report_*.json"), reverse=True)
    if reports:
        with open(reports[0]) as f:
            return json.load(f)
    return None


def _get_config_versions(configs_dir: str) -> dict:
    """获取各维度的最新配置版本号。"""
    result = {}
    for dim in ["model", "training", "augmentation", "post_process"]:
        dim_dir = Path(configs_dir) / dim
        if dim_dir.exists():
            files = sorted(dim_dir.glob("v*_*.yaml"), reverse=True)
            result[dim] = files[0].name if files else "无快照"
        else:
            result[dim] = "目录不存在"
    return result


def generate_report(iter_id: str, base_dir: str = ".") -> str:
    """生成迭代报告 markdown。"""
    base = Path(base_dir)
    db = Database(str(base / "metadata.db"))

    iteration = _get_iteration(db, iter_id) if iter_id != "__latest__" else _get_latest_iteration(db)

    lines = []
    lines.append(f"# 迭代执行报告")
    lines.append(f"")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # ── 1. 迭代基本信息 ──
    lines.append("## 1. 迭代信息")
    lines.append("")
    if iteration:
        lines.append(f"| 字段 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 迭代 ID | `{iteration.get('iteration_id', 'N/A')}` |")
        lines.append(f"| 描述 | {iteration.get('description', 'N/A')} |")
        lines.append(f"| 状态 | {iteration.get('status', 'N/A')} |")
        lines.append(f"| 创建时间 | {iteration.get('created_at', 'N/A')} |")
        lines.append(f"| 完成时间 | {iteration.get('completed_at', 'N/A')} |")
        actual_id = iteration.get("iteration_id", iter_id)
    else:
        lines.append("*未找到迭代记录（可能未通过 manage_iterations.py 创建）*")
        actual_id = iter_id
    lines.append("")

    # ── 2. 训练指标 ──
    lines.append("## 2. 训练指标")
    lines.append("")
    metrics = _get_training_metrics(str(base / "logs" / "training"), tag=actual_id)
    if metrics:
        last = metrics[-1]
        first = metrics[0]
        lines.append(f"| 指标 | 起始值 | 最终值 | 变化 |")
        lines.append(f"|------|--------|--------|------|")
        for key in ["train_loss", "val_loss", "learning_rate", "cer", "wer"]:
            v0 = first.get(key)
            v1 = last.get(key)
            if v0 is not None and v1 is not None:
                delta = v1 - v0
                lines.append(f"| {key} | {v0:.6f} | {v1:.6f} | {delta:+.6f} |")
        lines.append(f"")
        lines.append(f"总步数: {last.get('step', 'N/A')}, 总 epoch: {last.get('epoch', 'N/A')}")
    else:
        lines.append("*未找到训练指标数据*")
    lines.append("")

    # ── 3. 评估结果 ──
    lines.append("## 3. 评估结果")
    lines.append("")
    eval_report = _get_eval_report(str(base / "reports" / "evaluation"), actual_id)
    if eval_report:
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 整体 CER | {eval_report.get('overall_cer', 'N/A')} |")
        lines.append(f"| 整体 WER | {eval_report.get('overall_wer', 'N/A')} |")
        lines.append("")

        scene_metrics = eval_report.get("scene_metrics", {})
        if scene_metrics:
            lines.append("### 场景级指标")
            lines.append("")
            lines.append("| 场景 | CER | WER | 样本数 |")
            lines.append("|------|-----|-----|--------|")
            if isinstance(scene_metrics, list):
                for sm in scene_metrics:
                    lines.append(f"| {sm.get('scene_type', '?')} | {sm.get('cer', '?')} | {sm.get('wer', '?')} | {sm.get('sample_count', '?')} |")
            elif isinstance(scene_metrics, dict):
                for scene, data in scene_metrics.items():
                    if isinstance(data, dict):
                        lines.append(f"| {scene} | {data.get('cer', '?')} | {data.get('wer', '?')} | {data.get('sample_count', '?')} |")
        lines.append("")

        # Badcase 摘要
        badcases = eval_report.get("badcases", [])
        error_dist = eval_report.get("error_type_distribution", eval_report.get("error_distribution", {}))
        if error_dist:
            lines.append("### 错误类型分布")
            lines.append("")
            lines.append("| 类型 | 数量 |")
            lines.append("|------|------|")
            for etype, count in error_dist.items():
                lines.append(f"| {etype} | {count} |")
            lines.append("")

        if badcases:
            lines.append(f"### Top Badcase（共 {len(badcases)} 条，展示前 5）")
            lines.append("")
            lines.append("| 音频 | 参考文本 | 识别结果 | CER |")
            lines.append("|------|---------|---------|-----|")
            for bc in badcases[:5]:
                ref = bc.get("reference_text", bc.get("ref", ""))[:20]
                hyp = bc.get("hypothesis_text", bc.get("hyp", ""))[:20]
                cer = bc.get("cer", "?")
                audio = Path(bc.get("audio_path", "?")).name
                lines.append(f"| {audio} | {ref}... | {hyp}... | {cer} |")
            lines.append("")
    else:
        lines.append("*未找到评估报告*")
    lines.append("")

    # ── 4. 前后对比 ──
    lines.append("## 4. 前后对比")
    lines.append("")
    if iteration:
        cer_before = iteration.get("cer_before")
        cer_after = iteration.get("cer_after")
        wer_before = iteration.get("wer_before")
        wer_after = iteration.get("wer_after")
        if cer_before is not None and cer_after is not None:
            lines.append(f"| 指标 | 优化前 | 优化后 | 变化 | 结论 |")
            lines.append(f"|------|--------|--------|------|------|")
            cer_delta = cer_after - cer_before
            wer_delta = (wer_after - wer_before) if wer_after and wer_before else None
            cer_verdict = "✓ 改善" if cer_delta < 0 else "✗ 退化" if cer_delta > 0 else "— 持平"
            lines.append(f"| CER | {cer_before:.4f} | {cer_after:.4f} | {cer_delta:+.4f} | {cer_verdict} |")
            if wer_delta is not None:
                wer_verdict = "✓ 改善" if wer_delta < 0 else "✗ 退化" if wer_delta > 0 else "— 持平"
                lines.append(f"| WER | {wer_before:.4f} | {wer_after:.4f} | {wer_delta:+.4f} | {wer_verdict} |")
        else:
            lines.append("*迭代记录中无 CER before/after 数据*")
    else:
        lines.append("*无迭代记录*")
    lines.append("")

    # ── 5. 执行的优化操作 ──
    lines.append("## 5. 执行的优化操作")
    lines.append("")
    if iteration:
        for field, label in [
            ("param_changes_json", "训练参数变更"),
            ("model_changes_json", "模型结构变更"),
            ("data_changes_json", "数据/增强变更"),
        ]:
            raw = iteration.get(field)
            if raw:
                try:
                    changes = json.loads(raw) if isinstance(raw, str) else raw
                    if changes:
                        lines.append(f"### {label}")
                        lines.append(f"```json")
                        lines.append(json.dumps(changes, ensure_ascii=False, indent=2))
                        lines.append(f"```")
                        lines.append("")
                except (json.JSONDecodeError, TypeError):
                    pass
    config_versions = _get_config_versions(str(base / "configs"))
    lines.append("### 当前配置版本")
    lines.append("")
    lines.append("| 维度 | 最新快照 |")
    lines.append("|------|---------|")
    for dim, ver in config_versions.items():
        lines.append(f"| {dim} | `{ver}` |")
    lines.append("")

    # ── 6. 知识库更新 ──
    lines.append("## 6. 知识库状态")
    lines.append("")
    recent_knowledge = _get_recent_knowledge(db, limit=5)
    if recent_knowledge:
        lines.append("### 最近更新的知识条目（Top 5）")
        lines.append("")
        lines.append("| ID | 类型 | 置信度 | 状态 | 来源迭代 | 更新时间 |")
        lines.append("|-----|------|--------|------|---------|---------|")
        for entry in recent_knowledge:
            lines.append(
                f"| {entry.get('entry_id', '?')[:12]} | {entry.get('knowledge_type', '?')} "
                f"| {entry.get('confidence', '?')} | {entry.get('state', '?')} "
                f"| {entry.get('source_iteration_id', '?')} | {entry.get('updated_at', '?')[:16]} |"
            )
    else:
        lines.append("*知识库为空*")
    lines.append("")

    # 知识库统计
    conn = db.get_connection()
    cur = conn.execute(
        "SELECT state, COUNT(*) as cnt FROM knowledge_entries GROUP BY state"
    )
    stats = [dict(r) for r in cur.fetchall()]
    if stats:
        lines.append("### 知识条目统计")
        lines.append("")
        lines.append("| 状态 | 数量 |")
        lines.append("|------|------|")
        for row in stats:
            lines.append(f"| {row['state']} | {row['cnt']} |")
        lines.append("")

    # ── 7. Skill 进化状态 ──
    lines.append("## 7. Skill 进化状态")
    lines.append("")
    weights = _get_strategy_weights(db)
    if weights:
        lines.append("### 策略权重排名")
        lines.append("")
        lines.append("| 策略 | 权重 | 成功率 | 样本数 |")
        lines.append("|------|------|--------|--------|")
        for w in weights[:10]:
            lines.append(
                f"| {w.get('strategy_name', '?')} | {w.get('weight', '?'):.3f} "
                f"| {w.get('success_rate', '?'):.2%} | {w.get('sample_count', '?')} |"
            )
    else:
        lines.append("*尚无策略权重数据（需要至少完成一轮反馈后触发学习）*")
    lines.append("")

    # 反馈统计
    feedbacks = _get_feedbacks(db, actual_id) if actual_id else []
    conn = db.get_connection()
    cur = conn.execute("SELECT COUNT(*) as total, SUM(is_positive) as positive FROM effect_feedbacks")
    all_feedbacks_row = cur.fetchone()
    if all_feedbacks_row and (all_feedbacks_row["total"] or 0) > 0:
        total = all_feedbacks_row["total"]
        positive = all_feedbacks_row["positive"] or 0
        lines.append(f"### 反馈统计")
        lines.append(f"")
        lines.append(f"- 总反馈数: {total}")
        lines.append(f"- 正面反馈: {positive} ({positive/total:.0%})")
        lines.append(f"- 负面反馈: {total - positive} ({(total-positive)/total:.0%})")
        lines.append(f"- 本次迭代反馈: {len(feedbacks)} 条")
    lines.append("")

    lines.append("---")
    lines.append(f"*报告由 `generate_iteration_report.py` 自动生成*")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="生成迭代执行详细报告")
    parser.add_argument("--iteration-id", default=None, help="指定迭代 ID")
    parser.add_argument("--latest", action="store_true", help="使用最近一次迭代")
    parser.add_argument("--output", default=None, help="输出路径（默认 reports/iterations/）")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    if args.latest:
        iter_id = "__latest__"
    elif args.iteration_id:
        iter_id = args.iteration_id
    else:
        iter_id = "__latest__"

    report = generate_report(iter_id)

    # 输出
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path("reports/iterations")
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = args.iteration_id or "latest"
        out_path = out_dir / f"{tag}_report.md"

    out_path.write_text(report, encoding="utf-8")
    print(f"报告已生成: {out_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()
