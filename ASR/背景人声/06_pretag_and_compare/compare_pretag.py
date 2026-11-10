#!/usr/bin/env python3
"""
比对两个预标注模型的标签结果，找出不一致的条目。

读取两份 pretag.jsonl（或 text.txt），按 uid 匹配，
计算 CER 并输出不一致清单和统计报告。

用法:
    python 08_compare_pretag.py --a firered_pretag_output/text.txt --b qwen_pretag_output/text.txt
    python 08_compare_pretag.py --a firered_pretag_output/pretag.jsonl --b qwen_pretag_output/pretag.jsonl
    python 08_compare_pretag.py --a ... --b ... --threshold 0.1   # CER>10% 视为不一致
    python 08_compare_pretag.py --a ... --b ... --output compare_output/
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

# ============ 配置 ============
BASE_DIR = Path(r'E:\8.音频\03 s10\s10_data_reflow')
DEFAULT_A = BASE_DIR / "firered_pretag_output" / "text.txt"
DEFAULT_B = BASE_DIR / "qwen_pretag_output" / "text.txt"
DEFAULT_OUTPUT = BASE_DIR / "compare_output"
# ==============================


def normalize_text(text: str) -> str:
    text = re.sub(r'[，。！？、；：""''（）《》【】\[\]{},.!?;:\'"()\-—…·]', '', text)
    text = text.lower().strip()
    text = re.sub(r'\s+', '', text)
    return text


def edit_distance(ref: list, hyp: list):
    n, m = len(ref), len(hyp)
    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, 0, i)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, j, 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
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


def compute_cer(text_a: str, text_b: str):
    chars_a = list(text_a)
    chars_b = list(text_b)
    if len(chars_a) == 0 and len(chars_b) == 0:
        return 0.0, 0, 0, 0, 0
    ref_len = max(len(chars_a), len(chars_b))
    if ref_len == 0:
        return 0.0, 0, 0, 0, 0
    s, i, d = edit_distance(chars_a, chars_b)
    cer = (s + i + d) / ref_len
    return cer, s, i, d, ref_len


def load_labels(path: Path) -> dict:
    """加载标签，支持 text.txt 和 jsonl 格式，返回 {uid: text}"""
    labels = {}
    suffix = path.suffix.lower()

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if suffix in ('.jsonl', '.json'):
                try:
                    obj = json.loads(line)
                    if "messages" in obj:
                        msgs = obj["messages"]
                        user_content = msgs[1]["content"]
                        wav_part = user_content.split("!")[-1].split("<|endofspeech|>")[0]
                        uid = Path(wav_part).stem
                        text = msgs[-1]["content"]
                        labels[uid] = text
                    elif "uid" in obj and "text" in obj:
                        labels[obj["uid"]] = obj["text"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
            else:
                parts = line.split(None, 1)
                if len(parts) >= 1:
                    uid = parts[0]
                    text = parts[1] if len(parts) >= 2 else ""
                    labels[uid] = text

    return labels


def run_compare(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"加载 A: {args.a}")
    labels_a = load_labels(Path(args.a))
    print(f"  A 条目数: {len(labels_a)}")

    print(f"加载 B: {args.b}")
    labels_b = load_labels(Path(args.b))
    print(f"  B 条目数: {len(labels_b)}")

    # 匹配
    common_uids = set(labels_a.keys()) & set(labels_b.keys())
    only_a = set(labels_a.keys()) - set(labels_b.keys())
    only_b = set(labels_b.keys()) - set(labels_a.keys())

    print(f"\n共同 uid: {len(common_uids)}")
    print(f"仅在 A 中: {len(only_a)}")
    print(f"仅在 B 中: {len(only_b)}")

    if not common_uids:
        print("没有共同条目可比对")
        return

    # 逐条比对
    results = []
    exact_match = 0
    threshold = args.threshold

    for uid in sorted(common_uids):
        text_a = labels_a[uid]
        text_b = labels_b[uid]
        norm_a = normalize_text(text_a)
        norm_b = normalize_text(text_b)

        if norm_a == norm_b:
            exact_match += 1
            cer = 0.0
            s = i = d = ref_len = 0
        else:
            cer, s, i, d, ref_len = compute_cer(norm_a, norm_b)

        results.append({
            "uid": uid,
            "text_a": text_a,
            "text_b": text_b,
            "norm_a": norm_a,
            "norm_b": norm_b,
            "cer": round(cer, 4),
            "sub": s, "ins": i, "del": d,
            "ref_len": ref_len,
            "match": norm_a == norm_b,
        })

    # 不一致条目
    mismatches = [r for r in results if not r["match"]]
    above_threshold = [r for r in mismatches if r["cer"] >= threshold]

    # 输出逐条详情
    detail_path = output_dir / "compare_detail.jsonl"
    with open(detail_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 输出不一致清单
    diff_path = output_dir / "diff_list.txt"
    with open(diff_path, 'w', encoding='utf-8') as f:
        for r in sorted(above_threshold, key=lambda x: -x["cer"]):
            f.write(f"[{r['uid']}] CER={r['cer']:.0%}\n")
            f.write(f"  A: {r['text_a']}\n")
            f.write(f"  B: {r['text_b']}\n\n")

    # CER 分布
    cer_buckets = defaultdict(int)
    for r in results:
        c = r["cer"]
        if c == 0:
            cer_buckets["0% (完全一致)"] += 1
        elif c <= 0.05:
            cer_buckets["0-5%"] += 1
        elif c <= 0.10:
            cer_buckets["5-10%"] += 1
        elif c <= 0.20:
            cer_buckets["10-20%"] += 1
        elif c <= 0.50:
            cer_buckets["20-50%"] += 1
        else:
            cer_buckets[">50%"] += 1

    all_cers = [r["cer"] for r in results]
    avg_cer = sum(all_cers) / len(all_cers)

    # 报告
    report_path = output_dir / "compare_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        def w(s=""):
            f.write(s + "\n")
            print(s)

        w("=" * 60)
        w("预标注结果比对报告")
        w("=" * 60)
        w(f"A: {args.a}")
        w(f"B: {args.b}")
        w(f"不一致阈值: CER >= {threshold:.0%}")
        w()
        w("--- 总览 ---")
        w(f"共同条目:     {len(common_uids):>8,}")
        w(f"完全一致:     {exact_match:>8,} ({exact_match/len(common_uids)*100:.1f}%)")
        w(f"有差异:       {len(mismatches):>8,} ({len(mismatches)/len(common_uids)*100:.1f}%)")
        w(f"超阈值差异:   {len(above_threshold):>8,} ({len(above_threshold)/len(common_uids)*100:.1f}%)")
        w(f"平均 CER:     {avg_cer:.2%}")
        w(f"仅在 A 中:    {len(only_a):>8,}")
        w(f"仅在 B 中:    {len(only_b):>8,}")
        w()
        w("--- CER 分布 ---")
        bucket_order = ["0% (完全一致)", "0-5%", "5-10%", "10-20%", "20-50%", ">50%"]
        for bucket in bucket_order:
            cnt = cer_buckets.get(bucket, 0)
            pct = cnt / len(results) * 100
            bar = "█" * int(pct / 2)
            w(f"  {bucket:>16s}: {cnt:>6d} ({pct:5.1f}%) {bar}")
        w()
        w(f"--- 差异最大的条目 (前 30) ---")
        for r in sorted(mismatches, key=lambda x: -x["cer"])[:30]:
            w(f"  [{r['uid']}] CER={r['cer']:.0%}")
            w(f"    A: {r['text_a']}")
            w(f"    B: {r['text_b']}")
            w()

    print(f"\n输出文件:")
    print(f"  {detail_path}  — 逐条详情")
    print(f"  {diff_path}  — 不一致清单 (CER>={threshold:.0%})")
    print(f"  {report_path}  — 统计报告")

    # JSON 汇总
    summary = {
        "common_count": len(common_uids),
        "exact_match": exact_match,
        "mismatch_count": len(mismatches),
        "above_threshold": len(above_threshold),
        "avg_cer": round(avg_cer, 4),
        "only_a": len(only_a),
        "only_b": len(only_b),
        "threshold": threshold,
        "cer_distribution": {k: cer_buckets.get(k, 0) for k in bucket_order},
    }
    summary_path = output_dir / "compare_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="比对两个预标注模型的标签结果")
    parser.add_argument('--a', default=str(DEFAULT_A), help='模型A的标签文件 (text.txt 或 jsonl)')
    parser.add_argument('--b', default=str(DEFAULT_B), help='模型B的标签文件 (text.txt 或 jsonl)')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT), help='输出目录')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='CER 不一致阈值，超过此值列入差异清单 (default: 0.1)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_compare(args)
