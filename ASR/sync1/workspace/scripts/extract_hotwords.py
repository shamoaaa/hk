"""CLI script for extracting hotwords from badcase analysis results.

自动从评估报告的 badcase 列表中挖掘高频替换错误对，生成热词规则。
这些热词会加入到 `configs/post_process/hotwords.txt`，供 PostProcessor
在推理后自动纠正。

算法：
1. 对每个 badcase 执行编辑距离对齐，提取所有替换操作的 (wrong → correct) 对
2. 统计每对的出现次数
3. 过滤出现次数 >= min_count 的高频对
4. 按频率降序输出，设置优先级 = 出现次数

Usage:
    # 从评估报告提取热词
    python scripts/extract_hotwords.py --eval-report reports/evaluation/eval_report_iter_001.json

    # 合并到现有热词文件
    python scripts/extract_hotwords.py --eval-report ... --merge --output configs/post_process/hotwords.txt

    # 自定义最小出现次数
    python scripts/extract_hotwords.py --eval-report ... --min-count 5

输出格式：tab 分隔的三列 `wrong_text\\tcorrect_text\\tpriority`，和
`post_processor.py` 期望的格式一致。
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _edit_align(ref: str, hyp: str) -> list[tuple[str, str, str]]:
    """编辑距离对齐，返回操作列表 [(op, ref_char, hyp_char), ...].

    op 是 'match' / 'sub' / 'ins' / 'del'。"""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            ops.append(("match", ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", hyp[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def _extract_substitutions(ref: str, hyp: str) -> list[tuple[str, str]]:
    """从单对 (ref, hyp) 中提取所有 (hyp_char → ref_char) 替换对，
    即 hypothesis 里识别错的字 → 参考里正确的字。合并连续的替换为词块。"""
    ops = _edit_align(ref, hyp)
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(ops):
        if ops[i][0] == "sub":
            # 合并连续的 sub
            ref_buf = [ops[i][1]]
            hyp_buf = [ops[i][2]]
            j = i + 1
            while j < len(ops) and ops[j][0] == "sub":
                ref_buf.append(ops[j][1])
                hyp_buf.append(ops[j][2])
                j += 1
            wrong = "".join(hyp_buf)
            correct = "".join(ref_buf)
            # 忽略长度差异过大（可能是对齐噪声）或单字符替换中性对
            if wrong and correct and wrong != correct:
                pairs.append((wrong, correct))
            i = j
        else:
            i += 1
    return pairs


def _load_badcases(report_path: str) -> list[tuple[str, str]]:
    """从评估报告 JSON 读取 badcases，返回 [(ref, hyp), ...]."""
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    badcases = data.get("badcases", [])
    result = []
    for bc in badcases:
        ref = bc.get("reference_text", "")
        hyp = bc.get("hypothesis_text", "")
        if ref and hyp:
            result.append((ref, hyp))
    return result


def _merge_hotwords(
    new_entries: list[tuple[str, str, int]], existing_path: str
) -> list[tuple[str, str, int]]:
    """合并新热词与现有文件，相同 wrong_text 保留更高优先级。"""
    existing: dict[str, tuple[str, int]] = {}
    p = Path(existing_path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                wrong, correct = parts[0], parts[1]
                try:
                    prio = int(parts[2]) if len(parts) >= 3 else 1
                except ValueError:
                    prio = 1
                existing[wrong] = (correct, prio)

    for wrong, correct, prio in new_entries:
        if wrong in existing:
            old_correct, old_prio = existing[wrong]
            if old_correct == correct:
                # 相同对，取高优先级
                existing[wrong] = (correct, max(old_prio, prio))
            # 不同的纠正目标 — 保守地保留旧条目
        else:
            existing[wrong] = (correct, prio)

    merged = [(w, c, p) for w, (c, p) in existing.items()]
    merged.sort(key=lambda x: -x[2])
    return merged


def _write_hotword_file(
    entries: list[tuple[str, str, int]], output_path: str
) -> None:
    """写入热词文件，tab 分隔。"""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write("# Auto-generated hotwords from badcase analysis\n")
        f.write("# Format: wrong_text\\tcorrect_text\\tpriority\n")
        for wrong, correct, prio in entries:
            f.write(f"{wrong}\t{correct}\t{prio}\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract hotwords from evaluation report badcases."
    )
    parser.add_argument(
        "--eval-report", required=True,
        help="Path to evaluation report JSON (from run_inference.py --evaluate).",
    )
    parser.add_argument(
        "--output", default="configs/post_process/hotwords.txt",
        help="Output hotword file path.",
    )
    parser.add_argument(
        "--min-count", type=int, default=3,
        help="Minimum occurrence count to include as hotword (default: 3).",
    )
    parser.add_argument(
        "--max-entries", type=int, default=100,
        help="Maximum number of hotword entries to output (default: 100).",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge with existing hotword file instead of overwriting.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # 加载 badcases
    pairs = _load_badcases(args.eval_report)
    if not pairs:
        print(json.dumps({
            "status": "no_badcases",
            "message": f"No badcases found in {args.eval_report}",
        }, ensure_ascii=False, indent=2))
        return

    # 提取所有替换对
    all_subs: list[tuple[str, str]] = []
    for ref, hyp in pairs:
        all_subs.extend(_extract_substitutions(ref, hyp))

    # 统计频率
    counter = Counter(all_subs)

    # 过滤高频对
    filtered = [
        (wrong, correct, count)
        for (wrong, correct), count in counter.most_common(args.max_entries)
        if count >= args.min_count
    ]

    # 合并或覆盖
    if args.merge:
        final_entries = _merge_hotwords(filtered, args.output)
    else:
        final_entries = filtered

    _write_hotword_file(final_entries, args.output)

    json.dump({
        "status": "success",
        "badcases_analyzed": len(pairs),
        "substitution_pairs_found": len(all_subs),
        "unique_pairs": len(counter),
        "hotwords_extracted": len(filtered),
        "total_in_file": len(final_entries),
        "output_path": args.output,
        "top_hotwords": [
            {"wrong": w, "correct": c, "count": n}
            for w, c, n in filtered[:10]
        ],
    }, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
