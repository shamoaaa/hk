"""
去除识别结果开头/结尾的唤醒词"你好萤石"及其变体后，重新计算CER和指令召回率
"""
import json
import os
import re
from collections import Counter


def strip_wakeword(text):
    """去除开头和结尾的唤醒词及变体"""
    # 常见唤醒词变体
    wakewords = [
        "你好萤石", "你好萤", "你好石", "你好刘石", "你好伊思",
        "你好游石", "你好萤乐", "你好萤哲", "萤石", "石",
        "控好萤石", "控楼萤石",
    ]

    text = text.strip()
    changed = True
    while changed:
        changed = False
        for ww in wakewords:
            # 去除开头
            if text.startswith(ww):
                text = text[len(ww):]
                changed = True
            # 去除结尾
            if text.endswith(ww):
                text = text[:-len(ww)]
                changed = True
        text = text.strip()

    return text


def compute_cer(ref, hyp):
    """计算字符错误率（编辑距离）"""
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    n = len(ref_chars)
    m = len(hyp_chars)

    if n == 0:
        return m, 0, m, 0  # 全是插入

    # 动态规划计算编辑距离
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    ops = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                sub = dp[i-1][j-1] + 1
                ins = dp[i][j-1] + 1
                dele = dp[i-1][j] + 1
                dp[i][j] = min(sub, ins, dele)

    # 回溯获取 sub/ins/del
    i, j = n, m
    sub_count = ins_count = del_count = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_chars[i-1] == hyp_chars[j-1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            sub_count += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            ins_count += 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            del_count += 1
            i -= 1
        else:
            break

    return dp[n][m], sub_count, ins_count, del_count


def load_and_process(jsonl_path):
    """加载数据并去除唤醒词后重新计算"""
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    results_original = []
    results_stripped = []

    for item in items:
        ref = item["ref_norm"]
        hyp_orig = item["hyp_norm"]
        hyp_stripped = strip_wakeword(hyp_orig)

        # 原始指标
        results_original.append({
            "uid": item["uid"],
            "ref": ref,
            "hyp": hyp_orig,
            "cer": item["cer"],
            "sub": item["sub"],
            "ins": item["ins"],
            "del": item["del"],
            "ref_len": item["ref_len"],
            "correct": item["correct"],
            "recalled": ref in hyp_orig,
        })

        # 去除唤醒词后重新计算
        if hyp_stripped == hyp_orig:
            # 没变化，直接用原始值
            results_stripped.append(results_original[-1].copy())
        else:
            # 重新计算CER
            ref_len = len(ref)
            if ref_len == 0:
                new_cer = len(hyp_stripped)
                new_sub, new_ins, new_del = 0, len(hyp_stripped), 0
            else:
                _, new_sub, new_ins, new_del = compute_cer(ref, hyp_stripped)
                new_cer = (new_sub + new_ins + new_del) / ref_len

            results_stripped.append({
                "uid": item["uid"],
                "ref": ref,
                "hyp": hyp_stripped,
                "cer": new_cer,
                "sub": new_sub,
                "ins": new_ins,
                "del": new_del,
                "ref_len": ref_len,
                "correct": new_cer == 0,
                "recalled": ref in hyp_stripped,
            })

    return results_original, results_stripped


def summarize(results, label):
    """汇总统计"""
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    recalled = sum(1 for r in results if r["recalled"])
    total_sub = sum(r["sub"] for r in results)
    total_ins = sum(r["ins"] for r in results)
    total_del = sum(r["del"] for r in results)
    total_ref = sum(r["ref_len"] for r in results)
    cer = (total_sub + total_ins + total_del) / total_ref * 100 if total_ref > 0 else 0

    return {
        "label": label,
        "total": total,
        "correct": correct,
        "accuracy": correct / total * 100,
        "recalled": recalled,
        "recall_rate": recalled / total * 100,
        "sub": total_sub,
        "ins": total_ins,
        "del": total_del,
        "cer": cer,
    }


def main():
    base = r"e:\8.音频\03 s10\eval_output\testset_results"

    models = {
        "v0.1.0/best": "v0.1.0_model.pt.best",
        "v0.1.0/ep16": "v0.1.0_model.pt.ep16.4000",
        "v0.2.0/best": "v0.2.0_model.pt.best",
        "v0.2.0/ep14": "v0.2.0_model.pt.ep14.4000",
        "v0.3.0/best": "v0.3.0_model.pt.best",
        "v0.3.0/ep27": "v0.3.0_model.pt.ep27.4000",
    }

    print("=" * 95)
    print("去除开头/结尾唤醒词后的评测结果对比")
    print("=" * 95)

    print(f"\n{'─' * 95}")
    print(f"{'模型':<16} {'条件':<8} {'指令召回率':>10} {'CER':>8} {'句准确率':>10} {'Sub':>6} {'Ins':>6} {'Del':>6}")
    print(f"{'─' * 95}")

    all_orig = {}
    all_strip = {}

    for name, path in models.items():
        jsonl_path = os.path.join(base, path, "eval_detail.jsonl")
        if not os.path.exists(jsonl_path):
            continue

        orig, stripped = load_and_process(jsonl_path)
        s_orig = summarize(orig, "原始")
        s_strip = summarize(stripped, "去唤醒词")

        all_orig[name] = s_orig
        all_strip[name] = s_strip

        print(f"{name:<16} {'原始':<8} {s_orig['recall_rate']:>9.2f}% {s_orig['cer']:>7.2f}% {s_orig['accuracy']:>9.2f}% {s_orig['sub']:>6} {s_orig['ins']:>6} {s_orig['del']:>6}")
        print(f"{'':<16} {'去唤醒词':<8} {s_strip['recall_rate']:>9.2f}% {s_strip['cer']:>7.2f}% {s_strip['accuracy']:>9.2f}% {s_strip['sub']:>6} {s_strip['ins']:>6} {s_strip['del']:>6}")
        print(f"{'─' * 95}")

    # 改善幅度对比
    print(f"\n\n{'=' * 95}")
    print("去除唤醒词后的改善幅度")
    print(f"{'=' * 95}")

    print(f"\n{'模型':<16} {'召回率提升':>10} {'CER降低':>10} {'准确率提升':>10} {'Ins减少':>10}")
    print(f"{'─' * 95}")
    for name in models:
        if name in all_orig and name in all_strip:
            o = all_orig[name]
            s = all_strip[name]
            print(f"{name:<16} {s['recall_rate']-o['recall_rate']:>+9.2f}% {o['cer']-s['cer']:>+9.2f}% {s['accuracy']-o['accuracy']:>+9.2f}% {o['ins']-s['ins']:>+10}")

    # 去唤醒词后的最终排名
    print(f"\n\n{'=' * 95}")
    print("去除唤醒词后最终排名")
    print(f"{'=' * 95}")

    print(f"\n{'模型':<16} {'指令召回率':>10} {'CER':>8} {'句准确率':>10}")
    print(f"{'─' * 60}")
    ranked = sorted(all_strip.items(), key=lambda x: (-x[1]['recall_rate'], x[1]['cer']))
    for name, s in ranked:
        print(f"{name:<16} {s['recall_rate']:>9.2f}% {s['cer']:>7.2f}% {s['accuracy']:>9.2f}%")


if __name__ == "__main__":
    main()
