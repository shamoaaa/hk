"""analyse 命令 — 错误归因分析.

从评估结果中提取深层错误模式，生成结构化报告供 Agent 决策。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from asr_evolve.pipeline.evaluation.error_analyser import ErrorAnalyser
from asr_evolve.pipeline.models import Sample


def _load_samples(results_path: str) -> list[Sample]:
    """从 JSONL 结果文件加载样本。"""
    samples: list[Sample] = []
    p = Path(results_path)
    if p.is_file():
        files = [p]
    elif p.is_dir():
        files = sorted(p.rglob("*.jsonl"))
    else:
        return []

    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            samples.append(Sample(
                audio_path=record.get("key", ""),
                reference_text=record.get("ref", ""),
                hypothesis_text=record.get("hyp", ""),
            ))
    return samples


def cmd_analyse(args):
    """错误分析入口。"""
    results_path = args.results or "results"
    test_set = args.test_set or "default"
    threshold = args.threshold or 0.1
    top_n = args.top_n or 20

    # 扩展唤醒词列表（对齐背景人声项目）
    from asr_evolve.pipeline.evaluation.domain_evaluator import WAKEWORDS
    analyser = ErrorAnalyser(
        bad_case_threshold=threshold,
        top_n_confusion=top_n,
        wakewords=WAKEWORDS,
    )

    # 如果指定了 test_set 且 results 是默认值，尝试从 results/{test_set}/ 读
    p = Path(results_path)
    if not p.exists() and test_set != "default":
        alt = Path("results") / test_set / "eval_detail.jsonl"
        if alt.exists():
            p = alt

    samples = _load_samples(str(p))
    if not samples:
        # 尝试从最新评估结果中读取
        results_dir = Path("results")
        if results_dir.exists():
            versions = sorted(results_dir.iterdir(), reverse=True)
            for v_dir in versions:
                detail = v_dir / "eval_detail.jsonl"
                if detail.exists():
                    samples = _load_samples(str(detail))
                    if samples:
                        test_set = v_dir.name
                        print(f"自动发现评估结果: results/{test_set}/eval_detail.jsonl")
                        break

    if not samples:
        print(f"❌ 未找到结果文件: {results_path}")
        print("   请先跑 'asr-evolve eval' 生成评估结果")
        return

    report = analyser.analyse(samples, test_set=test_set)

    # 输出报告
    print(f"\n{'='*50}")
    print(f"错误分析报告: {report.test_set} ({report.total_samples} 条)")
    print(f"{'='*50}")
    print(f"  平均 CER: {report.avg_cer:.4f}")
    print(f"  替换: {report.total_sub} | 插入: {report.total_ins} | 删除: {report.total_del}")
    if report.error_type_distribution:
        dist = report.error_type_distribution
        print(f"  分布: Sub={dist.get('substitution',0):.0%} | "
              f"Ins={dist.get('insertion',0):.0%} | Del={dist.get('deletion',0):.0%}")

    print(f"\n--- 混淆对 Top-{min(10, len(report.top_confusion_pairs))} ---")
    for p in report.top_confusion_pairs[:10]:
        print(f"  '{p['ref']}' → '{p['hyp']}' ({p['count']}次)")

    print(f"\n--- 薄弱环节 ---")
    for w in report.weak_spots:
        print(f"  {w['device']}: CER={w['cer']:.4f} ({w['samples']}条)")

    print(f"\n--- Bad Cases ({len(report.bad_cases)}条, CER>{threshold}) ---")
    for bc in report.bad_cases[:5]:
        print(f"  [{bc['cer']:.2f}] ref='{bc['ref'][:30]}' | hyp='{bc['hyp'][:30]}'")
    if len(report.bad_cases) > 5:
        print(f"  ... 还有 {len(report.bad_cases)-5} 条")

    print(f"\n--- 建议 ---")
    for s in report.suggestions:
        print(f"  💡 {s}")
    print(f"{'='*50}")

    # 保存完整报告（供 Agent 消费）
    report_path = Path("reports")
    report_path.mkdir(exist_ok=True)
    out_file = report_path / f"analysis_{test_set}.json"
    out_file.write_text(
        json.dumps({
            "test_set": report.test_set,
            "total_samples": report.total_samples,
            "avg_cer": report.avg_cer,
            "error_distribution": report.error_type_distribution,
            "top_confusion_pairs": report.top_confusion_pairs,
            "per_device_cer": report.per_device_cer,
            "weak_spots": report.weak_spots,
            "bad_cases": report.bad_cases,
            "suggestions": report.suggestions,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n📄 完整报告已保存: {out_file}")
    print("   → Agent 可消费 bad_cases 做 LLM 归因")
    print("   → Agent 可消费 weak_spots 决定下一步方向")
