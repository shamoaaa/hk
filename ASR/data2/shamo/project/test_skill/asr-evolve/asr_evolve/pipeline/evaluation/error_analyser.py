"""ErrorAnalyser — 错误归因分析.

原项目"背景人声"中的 analyse 阶段。从评估结果中提取深层错误模式，
生成结构化分析报告供 Agent 决策下一步优化方向。

两层分析：
1. 规则型分析（本文件实现）：
   - CER 分解（substitution / insertion / deletion）
   - 混淆对提取（哪些字经常被认错，top-N）
   - 按场景/设备分组统计（哪个 device/场景 CER 最高）
   - 错误类型分布
   - Bad cases 结构化输出

2. LLM 型归因（由 Agent 在 REFLECT 阶段消费 bad_cases 报告）：
   - 代码不内置归因策略
   - 只把 bad cases 结构化输出，Agent 做语义归因
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from asr_evolve.pipeline.models import Sample


# ─── 编辑距离操作类型 ───

def edit_distance_ops(ref: str, hyp: str) -> list[str]:
    """计算字符级编辑距离并返回操作序列.

    返回: ["C", "S", "I", "D", ...] 其中
    C = correct（匹配）
    S = substitution（替换）
    I = insertion（插入，hyp 多余）
    D = deletion（删除，ref 丢失）
    """
    m, n = len(ref), len(hyp)
    # dp[i][j] = (distance, ops)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    ops = [[""] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i
        ops[i][0] = ops[i - 1][0] + "D"
    for j in range(1, n + 1):
        dp[0][j] = j
        ops[0][j] = ops[0][j - 1] + "I"

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                ops[i][j] = ops[i - 1][j - 1] + "C"
            else:
                candidates = [
                    (dp[i - 1][j - 1] + 1, ops[i - 1][j - 1] + "S"),  # substitution
                    (dp[i - 1][j] + 1, ops[i - 1][j] + "D"),          # deletion
                    (dp[i][j - 1] + 1, ops[i][j - 1] + "I"),          # insertion
                ]
                best = min(candidates, key=lambda x: x[0])
                dp[i][j] = best[0]
                ops[i][j] = best[1]

    return list(ops[m][n])


def extract_confusion_pairs(ref: str, hyp: str) -> list[tuple[str, str]]:
    """提取替换混淆对 (ref_char → hyp_char)."""
    ops = edit_distance_ops(ref, hyp)
    pairs: list[tuple[str, str]] = []
    ri, hi = 0, 0
    for op in ops:
        if op == "C":
            ri += 1
            hi += 1
        elif op == "S":
            pairs.append((ref[ri], hyp[hi]))
            ri += 1
            hi += 1
        elif op == "D":
            pairs.append((ref[ri], "∅"))  # 删除
            ri += 1
        elif op == "I":
            pairs.append(("∅", hyp[hi]))  # 插入
            hi += 1
    return pairs


# ─── 数据结构 ───

@dataclass
class ErrorStats:
    """单条样本的错误统计。"""
    utt_id: str
    ref: str
    hyp: str
    cer: float
    n_sub: int = 0
    n_ins: int = 0
    n_del: int = 0
    confusion_pairs: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class AnalysisReport:
    """完整分析报告——供 Agent 消费。"""
    test_set: str
    total_samples: int
    # 全局统计
    avg_cer: float
    total_sub: int
    total_ins: int
    total_del: int
    error_type_distribution: dict[str, float]  # {"sub": 0.6, "ins": 0.2, "del": 0.2}
    # 混淆对 top-N
    top_confusion_pairs: list[dict[str, Any]]  # [{"ref":"开", "hyp":"关", "count":5}, ...]
    # 按设备/场景分组
    per_device_cer: dict[str, float]  # {"DEV001": 0.012, "DEV002": 0.008}
    # 薄弱环节（CER 最高的设备/场景）
    weak_spots: list[dict[str, Any]]  # [{"device":"DEV001", "cer":0.025, "samples":10}, ...]
    # Bad cases（CER > 阈值的样本，供 Agent 做 LLM 归因）
    bad_cases: list[dict[str, Any]]  # [{"utt_id":"...", "ref":"...", "hyp":"...", "cer":0.5}, ...]
    # 建议（规则型，基于统计模式）
    suggestions: list[str]


# ─── 分析器 ───

class ErrorAnalyser:
    """规则型错误归因分析器。

    使用方式：
        analyser = ErrorAnalyser(bad_case_threshold=0.1)
        report = analyser.analyse(samples, test_set="s10_test")
        # Agent 消费 report.bad_cases 做 LLM 归因
        # Agent 消费 report.weak_spots 决定下一步方向
    """

    def __init__(
        self,
        bad_case_threshold: float = 0.1,
        top_n_confusion: int = 20,
        wakewords: list[str] | None = None,
    ):
        self.bad_case_threshold = bad_case_threshold
        self.top_n = top_n_confusion
        self.wakewords = wakewords or []

    def analyse(self, samples: list[Sample], test_set: str = "") -> AnalysisReport:
        """对评估结果做完整错误分析。"""
        if not samples:
            return AnalysisReport(
                test_set=test_set, total_samples=0, avg_cer=0,
                total_sub=0, total_ins=0, total_del=0,
                error_type_distribution={}, top_confusion_pairs=[],
                per_device_cer={}, weak_spots=[], bad_cases=[], suggestions=[],
            )

        # 逐条分析
        error_stats: list[ErrorStats] = []
        all_confusion: Counter = Counter()
        total_sub, total_ins, total_del = 0, 0, 0
        cer_sum = 0.0

        for s in samples:
            ref = self._strip_wakeword(s.reference_text)
            hyp = self._strip_wakeword(s.hypothesis_text)
            if not ref:
                continue

            # 编辑距离操作
            ops = edit_distance_ops(ref, hyp)
            n_sub = ops.count("S")
            n_ins = ops.count("I")
            n_del = ops.count("D")
            cer = (n_sub + n_ins + n_del) / len(ref) if ref else 0.0

            # 混淆对
            pairs = extract_confusion_pairs(ref, hyp)

            total_sub += n_sub
            total_ins += n_ins
            total_del += n_del
            cer_sum += cer
            all_confusion.update(pairs)

            error_stats.append(ErrorStats(
                utt_id=s.audio_path, ref=ref, hyp=hyp, cer=cer,
                n_sub=n_sub, n_ins=n_ins, n_del=n_del,
                confusion_pairs=pairs,
            ))

        total = len(error_stats)
        avg_cer = cer_sum / total if total else 0.0

        # 错误类型分布
        total_errors = total_sub + total_ins + total_del
        error_dist = {}
        if total_errors > 0:
            error_dist = {
                "substitution": round(total_sub / total_errors, 3),
                "insertion": round(total_ins / total_errors, 3),
                "deletion": round(total_del / total_errors, 3),
            }

        # 混淆对 top-N
        top_pairs = [
            {"ref": p[0], "hyp": p[1], "count": c}
            for p, c in all_confusion.most_common(self.top_n)
        ]

        # 按设备分组 CER
        device_cers: dict[str, list[float]] = defaultdict(list)
        for es in error_stats:
            dev_id = self._extract_device(es.utt_id)
            device_cers[dev_id].append(es.cer)

        per_device_cer = {
            dev: round(sum(cers) / len(cers), 4)
            for dev, cers in device_cers.items()
        }

        # 薄弱环节（CER 最高的 top-5 设备）
        sorted_devices = sorted(per_device_cer.items(), key=lambda x: x[1], reverse=True)
        weak_spots = [
            {"device": dev, "cer": cer, "samples": len(device_cers[dev])}
            for dev, cer in sorted_devices[:5]
        ]

        # Bad cases（CER > 阈值）
        bad_cases = sorted(
            [
                {"utt_id": es.utt_id, "ref": es.ref, "hyp": es.hyp, "cer": round(es.cer, 4)}
                for es in error_stats if es.cer > self.bad_case_threshold
            ],
            key=lambda x: x["cer"], reverse=True,
        )[:50]  # 最多 50 条供 Agent 归因

        # 自动建议（基于统计模式）
        suggestions = self._generate_suggestions(
            avg_cer, error_dist, weak_spots, top_pairs, total,
        )

        return AnalysisReport(
            test_set=test_set,
            total_samples=total,
            avg_cer=round(avg_cer, 4),
            total_sub=total_sub,
            total_ins=total_ins,
            total_del=total_del,
            error_type_distribution=error_dist,
            top_confusion_pairs=top_pairs,
            per_device_cer=per_device_cer,
            weak_spots=weak_spots,
            bad_cases=bad_cases,
            suggestions=suggestions,
        )

    def _generate_suggestions(
        self, avg_cer: float, error_dist: dict,
        weak_spots: list, top_pairs: list, total: int,
    ) -> list[str]:
        """基于统计模式生成建议（规则型，不是 LLM）。"""
        suggestions: list[str] = []

        # 替换占比高 → 可能是声学模型混淆
        if error_dist.get("substitution", 0) > 0.6:
            suggestions.append("替换错误占比 >60%，建议：增加易混淆字的训练数据或使用热词偏置")

        # 删除占比高 → 可能是语速快或噪声
        if error_dist.get("deletion", 0) > 0.3:
            suggestions.append("删除错误占比 >30%，建议：增加速度扰动增强或降噪增强")

        # 插入占比高 → 可能是误检
        if error_dist.get("insertion", 0) > 0.3:
            suggestions.append("插入错误占比 >30%，建议：检查是否有噪声被误认为语音（VAD 问题）")

        # 某设备 CER 异常高
        if weak_spots and weak_spots[0]["cer"] > avg_cer * 2:
            dev = weak_spots[0]["device"]
            suggestions.append(f"设备 {dev} CER 是均值的 2 倍以上，建议：针对该设备增加训练数据")

        # 特定混淆对频繁出现
        if top_pairs and top_pairs[0]["count"] > total * 0.1:
            p = top_pairs[0]
            suggestions.append(f"混淆 '{p['ref']}→{p['hyp']}' 出现频率极高，建议：添加到热词或增加对比训练")

        if not suggestions:
            suggestions.append("错误分布均匀，无明显单一瓶颈，建议：继续增加数据量或尝试模型结构改进")

        return suggestions

    def _strip_wakeword(self, text: str) -> str:
        """多轮首尾剥离唤醒词（对齐 domain_evaluator.strip_wakewords）。"""
        from asr_evolve.pipeline.evaluation.domain_evaluator import normalize_text, strip_wakewords
        return strip_wakewords(normalize_text(text))

    @staticmethod
    def _extract_device(utt_id: str) -> str:
        """从 utt_id 提取设备 ID。"""
        idx = utt_id.find("_")
        return utt_id[:idx] if idx > 0 else utt_id
