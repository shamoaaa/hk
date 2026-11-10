# -*- coding: utf-8 -*-
"""DomainEvaluator — 领域 KPI 评估.

与背景人声项目 eval_utils.py 对齐：
- normalize_text: 去标点→小写→去空格
- compute_cer: 字符级编辑距离（normalize 后）
- strip_wakewords: 多轮首尾剥离（按长度降序）
- instruction_recall: strip(ref) in strip(hyp)
"""
from __future__ import annotations

import re

from asr_evolve.pipeline.models import (
    ComparisonReport,
    DomainReport,
    RecallReport,
    Sample,
)


# 唤醒词变体列表（与背景人声项目 calc_recall_and_stripped_cer.py 对齐）
WAKEWORDS: list[str] = [
    "你好萤萤石", "营养萤石", "你好萤石", "你好伊思", "你好游石", "你好萤乐",
    "你好萤哲", "你好李石", "你好刘石", "控好萤石", "控楼萤石", "平好萤石",
    "你要萤石", "你很萤石", "怎样萤石", "你奥伊石",
    "杨萤石", "样萤石", "萤萤石", "好萤石", "你好萤", "你好石", "萤石",
]
WAKEWORDS.sort(key=len, reverse=True)


def normalize_text(text: str) -> str:
    """文本规范化（与背景人声 eval_utils.normalize_text 对齐）。

    去除标点 → 转小写 → 去除所有空格。
    """
    text = re.sub(r'[，。！？、；：""''（）《》【】\[\]{},.!?;:\'"()\-—…·]', '', text)
    text = text.lower().strip()
    text = re.sub(r'\s+', '', text)
    return text


def strip_wakewords(text: str, max_rounds: int = 3) -> str:
    """多轮首尾剥离唤醒词（与背景人声 calc_recall_and_stripped_cer.py 对齐）。"""
    for _ in range(max_rounds):
        stripped = False
        for ww in WAKEWORDS:
            if text.startswith(ww):
                text = text[len(ww):]
                stripped = True
                break
        for ww in WAKEWORDS:
            if text.endswith(ww):
                text = text[:-len(ww)]
                stripped = True
                break
        if not stripped:
            break
    return text


def edit_distance(ref: str, hyp: str) -> tuple[int, int, int]:
    """编辑距离，返回 (substitutions, insertions, deletions)。

    与背景人声 eval_utils.edit_distance 对齐。
    """
    ref_list = list(ref)
    hyp_list = list(hyp)
    n, m = len(ref_list), len(hyp_list)
    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, 0, i)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, j, 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_list[i - 1] == hyp_list[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub = dp[i - 1][j - 1]
                ins = dp[i][j - 1]
                dele = dp[i - 1][j]
                candidates = [
                    (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                    (ins[0] + 1, ins[1], ins[2] + 1, ins[3]),
                    (dele[0] + 1, dele[1], dele[2], dele[3] + 1),
                ]
                dp[i][j] = min(candidates, key=lambda x: x[0])
    _, s, i, d = dp[n][m]
    return s, i, d


def compute_cer(reference: str, hypothesis: str) -> tuple[float, int, int, int, int]:
    """计算 CER（与背景人声 eval_utils.compute_cer 对齐）。

    先 normalize_text，再算编辑距离。
    返回 (cer, sub, ins, del, ref_len)。
    """
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    if len(ref_norm) == 0:
        return (1.0 if len(hyp_norm) > 0 else 0.0), 0, len(hyp_norm), 0, 0

    s, i, d = edit_distance(ref_norm, hyp_norm)
    cer = (s + i + d) / len(ref_norm)
    return cer, s, i, d, len(ref_norm)


class DomainEvaluator:
    """领域 KPI 评估器（对齐背景人声项目）。"""

    def __init__(self, wakewords: list[str] | None = None):
        self.wakewords = WAKEWORDS if wakewords is None else wakewords

    def instruction_recall(self, samples: list[Sample]) -> tuple[float, int, int]:
        """指令召回率（对齐背景人声 calc_recall）。

        strip(ref) in strip(hyp)，空 ref 排除分子分母。
        """
        hits, total = 0, 0
        for s in samples:
            ref_norm = normalize_text(s.reference_text)
            hyp_norm = normalize_text(s.hypothesis_text)
            ref_stripped = strip_wakewords(ref_norm)
            hyp_stripped = strip_wakewords(hyp_norm)
            if not ref_stripped:
                continue
            total += 1
            if ref_stripped in hyp_stripped:
                hits += 1
        recall = hits / total if total > 0 else 0.0
        return recall, hits, total

    def evaluate(self, samples: list[Sample], test_set: str) -> DomainReport:
        """评估单个测试集。

        计算：CER（normalize 后）、Stripped CER（去唤醒词后）、指令召回率。
        """
        total_sub, total_ins, total_del, total_ref_len = 0, 0, 0, 0
        stripped_sub, stripped_ins, stripped_del, stripped_ref_len = 0, 0, 0, 0

        for s in samples:
            # 原始 CER（normalize 后）
            ref_norm = normalize_text(s.reference_text)
            hyp_norm = normalize_text(s.hypothesis_text)
            if ref_norm:
                sub, ins, d = edit_distance(ref_norm, hyp_norm)
                total_sub += sub
                total_ins += ins
                total_del += d
                total_ref_len += len(ref_norm)

            # Stripped CER（去唤醒词后）
            ref_stripped = strip_wakewords(ref_norm)
            hyp_stripped = strip_wakewords(hyp_norm)
            if ref_stripped:
                sub, ins, d = edit_distance(ref_stripped, hyp_stripped)
                stripped_sub += sub
                stripped_ins += ins
                stripped_del += d
                stripped_ref_len += len(ref_stripped)

        cer = (total_sub + total_ins + total_del) / total_ref_len if total_ref_len > 0 else 0.0
        stripped_cer = (stripped_sub + stripped_ins + stripped_del) / stripped_ref_len if stripped_ref_len > 0 else 0.0

        recall, hits, total = self.instruction_recall(samples)

        return DomainReport(
            test_set=test_set,
            cer=cer,
            stripped_cer=stripped_cer,
            recall=RecallReport(
                test_set=test_set, recall=recall,
                numerator=hits, denominator=total,
            ),
        )
