"""
生成v0.3.0 vs v0.5.0/ep23的完整对比报告：
1. 去除开头唤醒词后的错误对比（类似errors_v3_v4_xxx_stripped）
2. 指令召回率专项对比（类似recall_comparison_v3_v4）
"""
import json
import os
from collections import Counter

WAKEWORDS = sorted([
    "你好萤石", "你好萤", "你好石", "你好刘石", "你好伊思",
    "你好游石", "你好萤乐", "你好萤哲", "你好李石",
    "控好萤石", "控楼萤石", "萤萤石", "你好萤萤石",
    "好萤石", "平好萤石", "你要萤石", "你很萤石",
    "杨萤石", "样萤石", "营养萤石", "你奥伊石",
    "怎样萤石", "萤石",
], key=len, reverse=True)


def strip_wakeword_head(text):
    if not text:
        return text
    text = text.strip()
    for _ in range(3):
        changed = False
        for ww in WAKEWORDS:
            if text.startswith(ww):
                text = text[len(ww):].strip()
                changed = True
                break
        if not changed:
            break
    return text


def compute_cer(ref, hyp):
    n, m = len(ref), len(hyp)
    if n == 0 and m == 0:
        return 0, 0, 0, 0
    if n == 0:
        return m, 0, m, 0
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1]+1, dp[i][j-1]+1, dp[i-1][j]+1)
    i, j = n, m
    sub, ins, dele = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1]+1:
            sub += 1; i -= 1; j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1]+1:
            ins += 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j]+1:
            dele += 1; i -= 1
        else:
            break
    return dp[n][m], sub, ins, dele


def load_and_strip(jsonl_path):
    results = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            uid = item["uid"]
            ref = strip_wakeword_head(item["ref_norm"])
            hyp = strip_wakeword_head(item["hyp_norm"])
            if not ref:
                continue
            if ref == hyp:
                results[uid] = {"uid": uid, "ref": ref, "hyp": hyp, "cer": 0,
                                "sub": 0, "ins": 0, "del": 0, "correct": True,
                                "recalled": True, "ref_len": len(ref)}
            else:
                _, sub, ins, dele = compute_cer(ref, hyp)
                cer = (sub + ins + dele) / len(ref) * 100
                results[uid] = {"uid": uid, "ref": ref, "hyp": hyp, "cer": cer,
                                "sub": sub, "ins": ins, "del": dele, "correct": False,
                                "recalled": ref in hyp, "ref_len": len(ref)}
    return results


def generate_error_report(v3, v5, ds_name, output_path):
    """生成错误对比报告"""
    common = set(v3.keys()) & set(v5.keys())
    total = len(common)

    v3_correct = sum(1 for uid in common if v3[uid]["correct"])
    v5_correct = sum(1 for uid in common if v5[uid]["correct"])
    v3_recalled = sum(1 for uid in common if v3[uid]["recalled"])
    v5_recalled = sum(1 for uid in common if v5[uid]["recalled"])
    v3_sub = sum(v3[uid]["sub"] for uid in common)
    v3_ins = sum(v3[uid]["ins"] for uid in common)
    v3_del = sum(v3[uid]["del"] for uid in common)
    v5_sub = sum(v5[uid]["sub"] for uid in common)
    v5_ins = sum(v5[uid]["ins"] for uid in common)
    v5_del = sum(v5[uid]["del"] for uid in common)
    v3_ref = sum(v3[uid]["ref_len"] for uid in common)
    v5_ref = sum(v5[uid]["ref_len"] for uid in common)
    v3_cer = (v3_sub + v3_ins + v3_del) / v3_ref * 100
    v5_cer = (v5_sub + v5_ins + v5_del) / v5_ref * 100

    error_uids = [uid for uid in common if not v3[uid]["correct"] or not v5[uid]["correct"]]
    both_err = [uid for uid in error_uids if not v3[uid]["correct"] and not v5[uid]["correct"]]
    v3_only_err = [uid for uid in error_uids if not v3[uid]["correct"] and v5[uid]["correct"]]
    v5_only_err = [uid for uid in error_uids if not v5[uid]["correct"] and v3[uid]["correct"]]

    def sort_key(uid):
        return max(v3[uid]["cer"], v5[uid]["cer"])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*90}\n")
        f.write(f"{ds_name} 去除开头唤醒词后错误对比 (v0.3.0 vs v0.5.0/ep23)\n")
        f.write(f"{'='*90}\n")
        f.write(f"总样本: {total}\n\n")
        f.write(f"{'指标':<12} {'v0.3.0':>10} {'v0.5.0/ep23':>12} {'差异':>10}\n")
        f.write(f"{'-'*50}\n")
        f.write(f"{'准确率':<12} {v3_correct/total*100:>9.2f}% {v5_correct/total*100:>11.2f}% {(v5_correct-v3_correct)/total*100:>+9.2f}%\n")
        f.write(f"{'召回率':<12} {v3_recalled/total*100:>9.2f}% {v5_recalled/total*100:>11.2f}% {(v5_recalled-v3_recalled)/total*100:>+9.2f}%\n")
        f.write(f"{'CER':<12} {v3_cer:>9.2f}% {v5_cer:>11.2f}% {v5_cer-v3_cer:>+9.2f}%\n")
        f.write(f"{'Sub':<12} {v3_sub:>10} {v5_sub:>12} {v5_sub-v3_sub:>+10}\n")
        f.write(f"{'Ins':<12} {v3_ins:>10} {v5_ins:>12} {v5_ins-v3_ins:>+10}\n")
        f.write(f"{'Del':<12} {v3_del:>10} {v5_del:>12} {v5_del-v3_del:>+10}\n")
        f.write(f"\n错误分布:\n")
        f.write(f"  两者都错: {len(both_err)}\n")
        f.write(f"  仅v0.3.0错 (v0.5.0正确): {len(v3_only_err)}\n")
        f.write(f"  仅v0.5.0错 (v0.3.0正确): {len(v5_only_err)}\n")
        f.write(f"{'='*90}\n")

        sections = [
            ("两者都错", both_err),
            ("仅v0.3.0错 (v0.5.0/ep23正确)", v3_only_err),
            ("仅v0.5.0/ep23错 (v0.3.0正确)", v5_only_err),
        ]

        for section_name, uids in sections:
            uids_sorted = sorted(uids, key=sort_key, reverse=True)
            f.write(f"\n{'─'*90}\n")
            f.write(f"【{section_name}】共 {len(uids_sorted)} 条\n")
            f.write(f"{'─'*90}\n\n")
            for uid in uids_sorted:
                d3 = v3[uid]
                d5 = v5[uid]
                r3 = "✓" if d3["recalled"] else "✗"
                r5 = "✓" if d5["recalled"] else "✗"
                f.write(f"  [{uid}]\n")
                f.write(f"  标签:       {d3['ref']}\n")
                f.write(f"  v0.3.0:     {d3['hyp']}  (CER={d3['cer']:.0f}%, 召回{r3})\n")
                f.write(f"  v0.5.0/ep23:{d5['hyp']}  (CER={d5['cer']:.0f}%, 召回{r5})\n")
                f.write(f"\n")

    return {
        "total": total, "v3_correct": v3_correct, "v5_correct": v5_correct,
        "v3_recalled": v3_recalled, "v5_recalled": v5_recalled,
        "v3_cer": v3_cer, "v5_cer": v5_cer,
        "both_err": len(both_err), "v3_only": len(v3_only_err), "v5_only": len(v5_only_err),
    }


def generate_recall_report(v3, v5, ds_name, f):
    """生成召回率对比部分"""
    common = set(v3.keys()) & set(v5.keys())
    total = len(common)
    v3_recalled = sum(1 for uid in common if v3[uid]["recalled"])
    v5_recalled = sum(1 for uid in common if v5[uid]["recalled"])

    both_recalled = [uid for uid in common if v3[uid]["recalled"] and v5[uid]["recalled"]]
    both_missed = [uid for uid in common if not v3[uid]["recalled"] and not v5[uid]["recalled"]]
    v3_only = [uid for uid in common if v3[uid]["recalled"] and not v5[uid]["recalled"]]
    v5_only = [uid for uid in common if not v3[uid]["recalled"] and v5[uid]["recalled"]]

    f.write(f"\n{'═'*90}\n")
    f.write(f"【{ds_name}】\n")
    f.write(f"{'═'*90}\n\n")
    f.write(f"总样本: {total}\n")
    f.write(f"v0.3.0 召回: {v3_recalled} ({v3_recalled/total*100:.2f}%)\n")
    f.write(f"v0.5.0/ep23 召回: {v5_recalled} ({v5_recalled/total*100:.2f}%)\n")
    f.write(f"两者都召回: {len(both_recalled)}\n")
    f.write(f"两者都未召回: {len(both_missed)}\n")
    f.write(f"v0.3.0召回 但 v0.5.0未召回: {len(v3_only)} (v0.3.0优势)\n")
    f.write(f"v0.5.0召回 但 v0.3.0未召回: {len(v5_only)} (v0.5.0优势)\n\n")

    # v0.3.0独有召回
    f.write(f"{'─'*90}\n")
    f.write(f"v0.3.0能召回 但 v0.5.0/ep23不能 ({len(v3_only)} 条)\n")
    f.write(f"{'─'*90}\n\n")
    v3_only_sorted = sorted(v3_only, key=lambda uid: len(v3[uid]["ref"]))
    for uid in v3_only_sorted:
        f.write(f"  [{uid}]\n")
        f.write(f"  标签: {v3[uid]['ref']}\n")
        f.write(f"  v0.3.0: {v3[uid]['hyp']}\n")
        f.write(f"  v0.5.0: {v5[uid]['hyp']}\n\n")

    # v0.5.0独有召回
    f.write(f"\n{'─'*90}\n")
    f.write(f"v0.5.0/ep23能召回 但 v0.3.0不能 ({len(v5_only)} 条)\n")
    f.write(f"{'─'*90}\n\n")
    v5_only_sorted = sorted(v5_only, key=lambda uid: len(v5[uid]["ref"]))
    for uid in v5_only_sorted:
        f.write(f"  [{uid}]\n")
        f.write(f"  标签: {v3[uid]['ref']}\n")
        f.write(f"  v0.3.0: {v3[uid]['hyp']}\n")
        f.write(f"  v0.5.0: {v5[uid]['hyp']}\n\n")

    # 两者都未召回
    f.write(f"\n{'─'*90}\n")
    f.write(f"两者都未召回 ({len(both_missed)} 条)\n")
    f.write(f"{'─'*90}\n\n")
    both_missed_sorted = sorted(both_missed, key=lambda uid: len(v3[uid]["ref"]))
    for uid in both_missed_sorted:
        f.write(f"  [{uid}]\n")
        f.write(f"  标签: {v3[uid]['ref']}\n")
        f.write(f"  v0.3.0: {v3[uid]['hyp']}\n")
        f.write(f"  v0.5.0: {v5[uid]['hyp']}\n\n")

    return {"v3_only": len(v3_only), "v5_only": len(v5_only), "both_missed": len(both_missed)}


def main():
    base = r"e:\8.音频\03 s10\eval_output"
    output_dir = r"e:\8.音频\03 s10\11_analyse"

    datasets = {
        "s10_reflow/test": "reflow_label/s10_reflow/test/eval_detail.jsonl",
        "sp4_reflow/test": "reflow_label/sp4_reflow/test/eval_detail.jsonl",
    }

    report_dir = os.path.join(output_dir, "reports", "v3_vs_v5")
    os.makedirs(report_dir, exist_ok=True)

    # === Part 1: 错误对比报告 ===
    print("=" * 80)
    print("生成错误对比报告")
    print("=" * 80)

    for ds_name, ds_path in datasets.items():
        v3 = load_and_strip(os.path.join(base, "v0.3.0/model.pt.best", ds_path))
        v5 = load_and_strip(os.path.join(base, "v0.5.0/model.pt.ep23", ds_path))

        safe_name = ds_name.replace("/", "_")
        output_path = os.path.join(report_dir, f"errors_v3_v5_{safe_name}_stripped.txt")
        stats = generate_error_report(v3, v5, ds_name, output_path)

        print(f"\n【{ds_name}】")
        print(f"  v0.3.0: 准确率={stats['v3_correct']/stats['total']*100:.2f}% 召回率={stats['v3_recalled']/stats['total']*100:.2f}% CER={stats['v3_cer']:.2f}%")
        print(f"  v0.5.0: 准确率={stats['v5_correct']/stats['total']*100:.2f}% 召回率={stats['v5_recalled']/stats['total']*100:.2f}% CER={stats['v5_cer']:.2f}%")
        print(f"  两者都错: {stats['both_err']}, 仅v0.3.0错: {stats['v3_only']}, 仅v0.5.0错: {stats['v5_only']}")
        print(f"  已导出: {output_path}")

    # === Part 2: 召回率专项报告 ===
    print(f"\n\n{'='*80}")
    print("生成召回率专项报告")
    print("=" * 80)

    recall_path = os.path.join(report_dir, "recall_comparison_v3_v5.txt")
    with open(recall_path, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("v0.3.0 vs v0.5.0/ep23 指令召回率对比报告\n")
        f.write("（去除开头唤醒词后，标签是否作为子串出现在识别结果中）\n")
        f.write("=" * 90 + "\n")

        for ds_name, ds_path in datasets.items():
            v3 = load_and_strip(os.path.join(base, "v0.3.0/model.pt.best", ds_path))
            v5 = load_and_strip(os.path.join(base, "v0.5.0/model.pt.ep23", ds_path))
            stats = generate_recall_report(v3, v5, ds_name, f)
            print(f"\n【{ds_name}】")
            print(f"  v0.3.0独有召回: {stats['v3_only']} 条")
            print(f"  v0.5.0独有召回: {stats['v5_only']} 条")
            print(f"  两者都未召回: {stats['both_missed']} 条")

    print(f"\n  已导出: {recall_path}")


if __name__ == "__main__":
    main()
