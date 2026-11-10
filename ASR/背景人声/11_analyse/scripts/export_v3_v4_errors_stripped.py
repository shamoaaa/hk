"""
去除标签和识别结果开头的唤醒词后，重新对比v0.3.0和v0.4.0在业务数据集上的错误
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
    """只去除开头的唤醒词"""
    if not text:
        return text
    text = text.strip()
    changed = True
    max_iter = 3
    iteration = 0
    while changed and iteration < max_iter:
        changed = False
        iteration += 1
        for ww in WAKEWORDS:
            if text.startswith(ww):
                text = text[len(ww):]
                text = text.strip()
                changed = True
                break
    return text


def compute_cer(ref, hyp):
    """计算CER"""
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
    """加载并去除开头唤醒词"""
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


def main():
    base = r"e:\8.音频\03 s10\eval_output"
    output_dir = r"e:\8.音频\03 s10\11_analyse"

    datasets = {
        "s10_reflow": "reflow_label/s10_reflow/test/eval_detail.jsonl",
        "sp4_reflow": "reflow_label/sp4_reflow/test/eval_detail.jsonl",
    }

    for ds_name, ds_path in datasets.items():
        v3 = load_and_strip(os.path.join(base, "v0.3.0/model.pt.best", ds_path))
        v4 = load_and_strip(os.path.join(base, "v0.4.0/model.pt.best", ds_path))

        common_uids = set(v3.keys()) & set(v4.keys())

        # 统计
        total = len(common_uids)
        v3_correct = sum(1 for uid in common_uids if v3[uid]["correct"])
        v4_correct = sum(1 for uid in common_uids if v4[uid]["correct"])
        v3_recalled = sum(1 for uid in common_uids if v3[uid]["recalled"])
        v4_recalled = sum(1 for uid in common_uids if v4[uid]["recalled"])
        v3_sub = sum(v3[uid]["sub"] for uid in common_uids)
        v3_ins = sum(v3[uid]["ins"] for uid in common_uids)
        v3_del = sum(v3[uid]["del"] for uid in common_uids)
        v4_sub = sum(v4[uid]["sub"] for uid in common_uids)
        v4_ins = sum(v4[uid]["ins"] for uid in common_uids)
        v4_del = sum(v4[uid]["del"] for uid in common_uids)
        v3_ref = sum(v3[uid]["ref_len"] for uid in common_uids)
        v4_ref = sum(v4[uid]["ref_len"] for uid in common_uids)
        v3_cer = (v3_sub + v3_ins + v3_del) / v3_ref * 100
        v4_cer = (v4_sub + v4_ins + v4_del) / v4_ref * 100

        # 找差异样本
        error_uids = [uid for uid in common_uids
                      if not v3[uid]["correct"] or not v4[uid]["correct"]]
        both_err = [uid for uid in error_uids
                    if not v3[uid]["correct"] and not v4[uid]["correct"]]
        v3_only_err = [uid for uid in error_uids
                       if not v3[uid]["correct"] and v4[uid]["correct"]]
        v4_only_err = [uid for uid in error_uids
                       if not v4[uid]["correct"] and v3[uid]["correct"]]

        # 打印汇总
        print(f"\n{'='*90}")
        print(f"【{ds_name}/test】去除开头唤醒词后对比")
        print(f"{'='*90}")
        print(f"  总样本: {total}")
        print(f"  {'指标':<12} {'v0.3.0':>10} {'v0.4.0':>10} {'差异':>10}")
        print(f"  {'-'*50}")
        print(f"  {'准确率':<12} {v3_correct/total*100:>9.2f}% {v4_correct/total*100:>9.2f}% {(v4_correct-v3_correct)/total*100:>+9.2f}%")
        print(f"  {'召回率':<12} {v3_recalled/total*100:>9.2f}% {v4_recalled/total*100:>9.2f}% {(v4_recalled-v3_recalled)/total*100:>+9.2f}%")
        print(f"  {'CER':<12} {v3_cer:>9.2f}% {v4_cer:>9.2f}% {v4_cer-v3_cer:>+9.2f}%")
        print(f"  {'Sub':<12} {v3_sub:>10} {v4_sub:>10} {v4_sub-v3_sub:>+10}")
        print(f"  {'Ins':<12} {v3_ins:>10} {v4_ins:>10} {v4_ins-v3_ins:>+10}")
        print(f"  {'Del':<12} {v3_del:>10} {v4_del:>10} {v4_del-v3_del:>+10}")
        print(f"\n  错误分布:")
        print(f"    两者都错: {len(both_err)}")
        print(f"    仅v0.3.0错: {len(v3_only_err)} (v0.4.0优势)")
        print(f"    仅v0.4.0错: {len(v4_only_err)} (v0.3.0优势)")

        # 导出错误文件
        output_path = os.path.join(output_dir, f"errors_v3_v4_{ds_name}_stripped.txt")

        def sort_key(uid):
            return max(v3[uid]["cer"], v4[uid]["cer"])

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"{'='*90}\n")
            f.write(f"{ds_name}/test 去除开头唤醒词后错误对比 (v0.3.0 vs v0.4.0)\n")
            f.write(f"{'='*90}\n")
            f.write(f"总样本: {total}\n")
            f.write(f"v0.3.0: 准确率={v3_correct/total*100:.2f}% 召回率={v3_recalled/total*100:.2f}% CER={v3_cer:.2f}%\n")
            f.write(f"v0.4.0: 准确率={v4_correct/total*100:.2f}% 召回率={v4_recalled/total*100:.2f}% CER={v4_cer:.2f}%\n")
            f.write(f"两者都错: {len(both_err)}, 仅v0.3.0错: {len(v3_only_err)}, 仅v0.4.0错: {len(v4_only_err)}\n")
            f.write(f"{'='*90}\n\n")

            sections = [
                ("两者都错", both_err),
                ("仅v0.3.0错 (v0.4.0正确)", v3_only_err),
                ("仅v0.4.0错 (v0.3.0正确)", v4_only_err),
            ]

            for section_name, uids in sections:
                uids_sorted = sorted(uids, key=sort_key, reverse=True)
                f.write(f"\n{'─'*90}\n")
                f.write(f"【{section_name}】共 {len(uids_sorted)} 条\n")
                f.write(f"{'─'*90}\n\n")

                for uid in uids_sorted:
                    d3 = v3[uid]
                    d4 = v4[uid]
                    r3 = "✓" if d3["recalled"] else "✗"
                    r4 = "✓" if d4["recalled"] else "✗"
                    f.write(f"[{uid}]\n")
                    f.write(f"  标签:   {d3['ref']}\n")
                    f.write(f"  v0.3.0: {d3['hyp']}  (CER={d3['cer']:.0f}%, 召回{r3})\n")
                    f.write(f"  v0.4.0: {d4['hyp']}  (CER={d4['cer']:.0f}%, 召回{r4})\n")
                    f.write(f"\n")

        print(f"\n  已导出: {output_path}")


if __name__ == "__main__":
    main()
