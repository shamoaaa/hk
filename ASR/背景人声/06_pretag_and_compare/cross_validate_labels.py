#!/usr/bin/env python3
"""
三方交叉验证标签：人工标注 vs FireRedASR vs Qwen3-ASR
找出不一致条目，重点发现人工标注可能出错的地方。

判定逻辑:
  - 两个模型一致但与人工不同 → 人工大概率错了
  - 人工与其中一个模型一致，另一个不同 → 该模型可能错了
  - 三方都不同 → 需要人工复核

用法:
    python 09_cross_validate_labels.py
    python 09_cross_validate_labels.py --human labeled/text.txt --firered firered_pretag_output/text.txt --qwen qwen_pretag_output/text.txt
    python 09_cross_validate_labels.py --threshold 0.05
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

# ============ 配置 ============
BASE_DIR = Path(r'E:\8.音频\03 s10\s10_data_reflow')
DEFAULT_HUMAN = BASE_DIR / "labeled" / "text.txt"
DEFAULT_FIRERED = BASE_DIR / "firered_pretag_output" / "text.txt"
DEFAULT_QWEN = BASE_DIR / "qwen_pretag_output" / "text.txt"
DEFAULT_OUTPUT = BASE_DIR / "cross_validate_output"
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


def compute_cer(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    ref_len = max(len(a), len(b))
    if ref_len == 0:
        return 0.0
    s, i, d = edit_distance(list(a), list(b))
    return (s + i + d) / ref_len


def load_labels(path: Path) -> dict:
    """加载标签，支持 text.txt 和 jsonl"""
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
                        labels[uid] = msgs[-1]["content"]
                    elif "uid" in obj and "text" in obj:
                        labels[obj["uid"]] = obj["text"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
            else:
                parts = line.split(None, 1)
                if len(parts) >= 2:
                    labels[parts[0]] = parts[1]
                elif len(parts) == 1:
                    # 空音频：只有 uid 没有文本，记为空字符串
                    labels[parts[0]] = ""
    return labels


def classify_disagreement(h_norm, f_norm, q_norm, threshold):
    """
    三方比对分类:
      all_agree     — 三方一致
      models_agree  — 两模型一致但与人工不同（人工可能错）
      human_firered — 人工与FireRedASR一致，Qwen不同
      human_qwen    — 人工与Qwen一致，FireRedASR不同
      all_differ    — 三方都不同，需复核
    """
    hf = compute_cer(h_norm, f_norm)
    hq = compute_cer(h_norm, q_norm)
    fq = compute_cer(f_norm, q_norm)

    h_eq_f = hf <= threshold
    h_eq_q = hq <= threshold
    f_eq_q = fq <= threshold

    if h_eq_f and h_eq_q:
        return "all_agree", hf, hq, fq
    elif f_eq_q and not h_eq_f:
        return "models_agree", hf, hq, fq  # 人工可能错
    elif h_eq_f and not h_eq_q:
        return "human_firered", hf, hq, fq
    elif h_eq_q and not h_eq_f:
        return "human_qwen", hf, hq, fq
    else:
        return "all_differ", hf, hq, fq


def run_validate(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold = args.threshold

    print(f"加载人工标注: {args.human}")
    human = load_labels(Path(args.human))
    print(f"  条目数: {len(human)}")

    print(f"加载 FireRedASR: {args.firered}")
    firered = load_labels(Path(args.firered))
    print(f"  条目数: {len(firered)}")

    print(f"加载 Qwen3-ASR: {args.qwen}")
    qwen = load_labels(Path(args.qwen))
    print(f"  条目数: {len(qwen)}")

    # 三方共有
    common = set(human.keys()) & set(firered.keys()) & set(qwen.keys())
    print(f"\n三方共有 uid: {len(common)}")

    if not common:
        print("没有共同条目")
        return

    # 逐条比对
    results = []
    category_counts = defaultdict(int)

    for uid in sorted(common):
        h_raw, f_raw, q_raw = human[uid], firered[uid], qwen[uid]
        h_norm = normalize_text(h_raw)
        f_norm = normalize_text(f_raw)
        q_norm = normalize_text(q_raw)

        cat, cer_hf, cer_hq, cer_fq = classify_disagreement(h_norm, f_norm, q_norm, threshold)
        category_counts[cat] += 1

        results.append({
            "uid": uid,
            "category": cat,
            "human": h_raw, "firered": f_raw, "qwen": q_raw,
            "human_norm": h_norm, "firered_norm": f_norm, "qwen_norm": q_norm,
            "cer_human_firered": round(cer_hf, 4),
            "cer_human_qwen": round(cer_hq, 4),
            "cer_firered_qwen": round(cer_fq, 4),
        })

    # 输出逐条详情
    detail_path = output_dir / "validate_detail.jsonl"
    with open(detail_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 按类别输出清单
    for cat in ["models_agree", "all_differ", "human_firered", "human_qwen"]:
        items = [r for r in results if r["category"] == cat]
        if not items:
            continue
        # 按最大 CER 降序
        items.sort(key=lambda r: max(r["cer_human_firered"], r["cer_human_qwen"]), reverse=True)
        cat_path = output_dir / f"{cat}.txt"
        with open(cat_path, 'w', encoding='utf-8') as f:
            for r in items:
                f.write(f"[{r['uid']}] H-F={r['cer_human_firered']:.0%} H-Q={r['cer_human_qwen']:.0%} F-Q={r['cer_firered_qwen']:.0%}\n")
                f.write(f"  人工:     {r['human']}\n")
                f.write(f"  FireRed:  {r['firered']}\n")
                f.write(f"  Qwen:     {r['qwen']}\n\n")

    # 报告
    total = len(common)
    report_path = output_dir / "validate_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        def w(s=""):
            f.write(s + "\n")
            print(s)

        w("=" * 65)
        w("三方交叉验证报告 (人工 vs FireRedASR vs Qwen3-ASR)")
        w("=" * 65)
        w(f"一致性阈值: CER <= {threshold:.0%} 视为一致")
        w()
        w("--- 分类统计 ---")

        cat_labels = {
            "all_agree":     "三方一致",
            "models_agree":  "两模型一致 人工不同 ← 人工可能错",
            "human_firered": "人工=FireRed Qwen不同",
            "human_qwen":    "人工=Qwen   FireRed不同",
            "all_differ":    "三方都不同  ← 需复核",
        }
        for cat, label in cat_labels.items():
            cnt = category_counts.get(cat, 0)
            pct = cnt / total * 100
            bar = "█" * int(pct / 2)
            w(f"  {label:<35s}: {cnt:>6d} ({pct:5.1f}%) {bar}")

        w()
        w(f"--- 人工可能出错的条目 (models_agree, 共 {category_counts.get('models_agree', 0)} 条, 展示前 30) ---")
        suspect = [r for r in results if r["category"] == "models_agree"]
        suspect.sort(key=lambda r: max(r["cer_human_firered"], r["cer_human_qwen"]), reverse=True)
        for r in suspect[:30]:
            w(f"  [{r['uid']}] H-F={r['cer_human_firered']:.0%} H-Q={r['cer_human_qwen']:.0%}")
            w(f"    人工:     {r['human']}")
            w(f"    FireRed:  {r['firered']}")
            w(f"    Qwen:     {r['qwen']}")
            w()

        w(f"--- 三方都不同的条目 (all_differ, 共 {category_counts.get('all_differ', 0)} 条, 展示前 20) ---")
        all_diff = [r for r in results if r["category"] == "all_differ"]
        all_diff.sort(key=lambda r: max(r["cer_human_firered"], r["cer_human_qwen"]), reverse=True)
        for r in all_diff[:20]:
            w(f"  [{r['uid']}] H-F={r['cer_human_firered']:.0%} H-Q={r['cer_human_qwen']:.0%} F-Q={r['cer_firered_qwen']:.0%}")
            w(f"    人工:     {r['human']}")
            w(f"    FireRed:  {r['firered']}")
            w(f"    Qwen:     {r['qwen']}")
            w()

    print(f"\n输出文件:")
    print(f"  {detail_path}  — 逐条详情")
    print(f"  {report_path}  — 统计报告")
    for cat in ["models_agree", "all_differ", "human_firered", "human_qwen"]:
        p = output_dir / f"{cat}.txt"
        if p.exists():
            print(f"  {p}  — {cat_labels[cat]}")

    # JSON 汇总
    summary = {
        "total": total,
        "threshold": threshold,
        "categories": {cat: category_counts.get(cat, 0) for cat in cat_labels},
    }
    with open(output_dir / "validate_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="三方交叉验证标签")
    parser.add_argument('--human', default=str(DEFAULT_HUMAN), help='人工标注文件')
    parser.add_argument('--firered', default=str(DEFAULT_FIRERED), help='FireRedASR 标签文件')
    parser.add_argument('--qwen', default=str(DEFAULT_QWEN), help='Qwen3-ASR 标签文件')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT), help='输出目录')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='CER 一致性阈值，低于此视为一致 (default: 0.05)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_validate(args)
