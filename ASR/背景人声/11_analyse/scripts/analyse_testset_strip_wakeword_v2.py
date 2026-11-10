"""
去除标签和识别结果开头/结尾的唤醒词"你好萤石"及变体后，重新计算CER和指令召回率。
对标签和识别结果都做同样的处理。
"""
import json
import os
import re


# 唤醒词变体列表（按长度降序排列，优先匹配长的）
WAKEWORDS = sorted([
    "你好萤石",
    "你好萤",
    "你好石",
    "你好刘石",
    "你好伊思",
    "你好游石",
    "你好萤乐",
    "你好萤哲",
    "你好李石",
    "控好萤石",
    "控楼萤石",
    "萤萤石",
    "你好萤萤石",
    "好萤石",
    "平好萤石",
    "你要萤石",
    "你很萤石",
    "杨萤石",
    "样萤石",
    "营养萤石",
    "你好萤石",
], key=len, reverse=True)

# 结尾变体（更宽松，因为结尾经常被截断）
WAKEWORDS_END = sorted([
    "你好萤石",
    "你好萤",
    "你好石",
    "你好刘石",
    "你好伊思",
    "你好游石",
    "你好萤乐",
    "你好萤哲",
    "你好李石",
    "控好萤石",
    "控楼萤石",
    "萤萤石",
    "你好萤萤石",
    "好萤石",
    "平好萤石",
    "你要萤石",
    "你很萤石",
    "杨萤石",
    "样萤石",
    "营养萤石",
    "怎样萤石",
    "你好萤石",
    "你奥伊石",
], key=len, reverse=True)


def strip_wakeword(text):
    """去除开头和结尾的唤醒词及变体"""
    if not text:
        return text

    text = text.strip()
    changed = True
    max_iter = 5  # 防止无限循环
    iteration = 0

    while changed and iteration < max_iter:
        changed = False
        iteration += 1

        # 去除开头
        for ww in WAKEWORDS:
            if text.startswith(ww):
                text = text[len(ww):]
                changed = True
                break

        # 去除结尾
        for ww in WAKEWORDS_END:
            if text.endswith(ww):
                text = text[:-len(ww)]
                changed = True
                break

        text = text.strip()

    return text


def compute_cer_details(ref, hyp):
    """计算CER并返回 sub/ins/del"""
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    n = len(ref_chars)
    m = len(hyp_chars)

    if n == 0 and m == 0:
        return 0, 0, 0, 0
    if n == 0:
        return m, 0, m, 0

    # DP
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1] + 1, dp[i][j-1] + 1, dp[i-1][j] + 1)

    # 回溯
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


def process_model(jsonl_path):
    """处理一个模型的数据"""
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    orig_stats = {"sub": 0, "ins": 0, "del": 0, "ref_chars": 0, "correct": 0, "recalled": 0, "total": len(items)}
    strip_stats = {"sub": 0, "ins": 0, "del": 0, "ref_chars": 0, "correct": 0, "recalled": 0, "total": len(items)}

    changed_count = 0

    for item in items:
        ref_orig = item["ref_norm"]
        hyp_orig = item["hyp_norm"]

        # 原始统计
        orig_stats["sub"] += item["sub"]
        orig_stats["ins"] += item["ins"]
        orig_stats["del"] += item["del"]
        orig_stats["ref_chars"] += item["ref_len"]
        orig_stats["correct"] += 1 if item["correct"] else 0
        orig_stats["recalled"] += 1 if ref_orig in hyp_orig else 0

        # 去唤醒词
        ref_stripped = strip_wakeword(ref_orig)
        hyp_stripped = strip_wakeword(hyp_orig)

        if ref_stripped != ref_orig or hyp_stripped != hyp_orig:
            changed_count += 1

        # 重新计算
        ref_len = len(ref_stripped)
        if ref_len == 0:
            # 标签去完唤醒词后为空，跳过这条
            strip_stats["total"] -= 1
            continue

        if ref_stripped == hyp_stripped:
            strip_stats["ref_chars"] += ref_len
            strip_stats["correct"] += 1
            strip_stats["recalled"] += 1
        else:
            _, sub, ins, dele = compute_cer_details(ref_stripped, hyp_stripped)
            strip_stats["sub"] += sub
            strip_stats["ins"] += ins
            strip_stats["del"] += dele
            strip_stats["ref_chars"] += ref_len
            strip_stats["correct"] += 1 if (sub + ins + dele) == 0 else 0
            strip_stats["recalled"] += 1 if ref_stripped in hyp_stripped else 0

    # 计算汇总指标
    for stats in [orig_stats, strip_stats]:
        total_err = stats["sub"] + stats["ins"] + stats["del"]
        stats["cer"] = total_err / stats["ref_chars"] * 100 if stats["ref_chars"] > 0 else 0
        stats["accuracy"] = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        stats["recall_rate"] = stats["recalled"] / stats["total"] * 100 if stats["total"] > 0 else 0

    return orig_stats, strip_stats, changed_count


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

    print("=" * 100)
    print("去除标签和识别结果开头/结尾唤醒词后的评测结果对比")
    print("（标签和识别结果都做同样的唤醒词去除处理）")
    print("=" * 100)

    all_orig = {}
    all_strip = {}

    print(f"\n{'─' * 100}")
    print(f"{'模型':<16} {'条件':<10} {'召回率':>8} {'CER':>8} {'准确率':>8} {'Sub':>6} {'Ins':>6} {'Del':>6} {'有效条数':>8} {'变更条数':>8}")
    print(f"{'─' * 100}")

    for name, path in models.items():
        jsonl_path = os.path.join(base, path, "eval_detail.jsonl")
        if not os.path.exists(jsonl_path):
            continue

        orig, strip, changed = process_model(jsonl_path)
        all_orig[name] = orig
        all_strip[name] = strip

        print(f"{name:<16} {'原始':<10} {orig['recall_rate']:>7.2f}% {orig['cer']:>7.2f}% {orig['accuracy']:>7.2f}% {orig['sub']:>6} {orig['ins']:>6} {orig['del']:>6} {orig['total']:>8}")
        print(f"{'':<16} {'去唤醒词':<10} {strip['recall_rate']:>7.2f}% {strip['cer']:>7.2f}% {strip['accuracy']:>7.2f}% {strip['sub']:>6} {strip['ins']:>6} {strip['del']:>6} {strip['total']:>8} {changed:>8}")
        print(f"{'─' * 100}")

    # 改善幅度
    print(f"\n\n{'=' * 100}")
    print("改善幅度对比")
    print(f"{'=' * 100}")

    print(f"\n{'模型':<16} {'召回率变化':>10} {'CER降低':>10} {'准确率提升':>10} {'Ins减少':>10} {'Sub变化':>10}")
    print(f"{'─' * 80}")
    for name in models:
        if name in all_orig and name in all_strip:
            o = all_orig[name]
            s = all_strip[name]
            print(f"{name:<16} {s['recall_rate']-o['recall_rate']:>+9.2f}% {o['cer']-s['cer']:>+9.2f}% {s['accuracy']-o['accuracy']:>+9.2f}% {o['ins']-s['ins']:>+10} {o['sub']-s['sub']:>+10}")

    # 最终排名
    print(f"\n\n{'=' * 100}")
    print("去除唤醒词后最终排名（按指令召回率降序）")
    print(f"{'=' * 100}")

    print(f"\n{'排名':<4} {'模型':<16} {'指令召回率':>10} {'CER':>8} {'句准确率':>10}")
    print(f"{'─' * 60}")
    ranked = sorted(all_strip.items(), key=lambda x: (-x[1]['recall_rate'], x[1]['cer']))
    for i, (name, s) in enumerate(ranked, 1):
        print(f"{i:<4} {name:<16} {s['recall_rate']:>9.2f}% {s['cer']:>7.2f}% {s['accuracy']:>9.2f}%")

    # 按CER排名
    print(f"\n去除唤醒词后最终排名（按CER升序）")
    print(f"\n{'排名':<4} {'模型':<16} {'CER':>8} {'指令召回率':>10} {'句准确率':>10}")
    print(f"{'─' * 60}")
    ranked_cer = sorted(all_strip.items(), key=lambda x: x[1]['cer'])
    for i, (name, s) in enumerate(ranked_cer, 1):
        print(f"{i:<4} {name:<16} {s['cer']:>7.2f}% {s['recall_rate']:>9.2f}% {s['accuracy']:>9.2f}%")


if __name__ == "__main__":
    main()
