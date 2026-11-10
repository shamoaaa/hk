"""
去除标签+识别结果开头/结尾唤醒词后，导出v0.2.0和v0.3.0在testset_results上的所有错误样本
分析错误特点
"""
import json
import os
from collections import Counter, defaultdict

# 唤醒词变体
WAKEWORDS = sorted([
    "你好萤石", "你好萤", "你好石", "你好刘石", "你好伊思",
    "你好游石", "你好萤乐", "你好萤哲", "你好李石",
    "控好萤石", "控楼萤石", "萤萤石", "你好萤萤石",
    "好萤石", "平好萤石", "你要萤石", "你很萤石",
    "杨萤石", "样萤石", "营养萤石", "你奥伊石",
    "怎样萤石",
], key=len, reverse=True)


def strip_wakeword(text):
    if not text:
        return text
    text = text.strip()
    changed = True
    max_iter = 5
    iteration = 0
    while changed and iteration < max_iter:
        changed = False
        iteration += 1
        for ww in WAKEWORDS:
            if text.startswith(ww):
                text = text[len(ww):]
                changed = True
                break
        for ww in WAKEWORDS:
            if text.endswith(ww):
                text = text[:-len(ww)]
                changed = True
                break
        text = text.strip()
    return text


def compute_cer_details(ref, hyp):
    """计算CER"""
    n, m = len(ref), len(hyp)
    if n == 0 and m == 0:
        return 0, 0, 0, 0
    if n == 0:
        return m, 0, m, 0

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1] + 1, dp[i][j-1] + 1, dp[i-1][j] + 1)

    i, j = n, m
    sub_count = ins_count = del_count = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            sub_count += 1; i -= 1; j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            ins_count += 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            del_count += 1; i -= 1
        else:
            break
    return dp[n][m], sub_count, ins_count, del_count


def process_model(jsonl_path):
    """处理一个模型，返回去唤醒词后的错误列表"""
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    errors = []
    correct_count = 0

    for item in items:
        ref_orig = item["ref_norm"]
        hyp_orig = item["hyp_norm"]
        ref = strip_wakeword(ref_orig)
        hyp = strip_wakeword(hyp_orig)

        if not ref:
            continue

        if ref == hyp:
            correct_count += 1
            continue

        _, sub, ins, dele = compute_cer_details(ref, hyp)
        cer = (sub + ins + dele) / len(ref) * 100

        errors.append({
            "uid": item["uid"],
            "ref": ref,
            "hyp": hyp,
            "ref_orig": ref_orig,
            "hyp_orig": hyp_orig,
            "cer": cer,
            "sub": sub,
            "ins": ins,
            "del": dele,
            "ref_len": len(ref),
            "recalled": ref in hyp,
        })

    return errors, correct_count, len(items)


def classify_error(e):
    """错误分类"""
    if e["recalled"]:
        if e["ins"] > 0 and e["sub"] == 0 and e["del"] == 0:
            return "纯插入(指令正确+多余内容)"
        else:
            return "指令在但有其他错误"
    else:
        if e["sub"] > 0 and e["ins"] == 0 and e["del"] == 0:
            return "纯替换(字识别错)"
        elif e["del"] > 0 and e["sub"] == 0 and e["ins"] == 0:
            return "纯删除(字丢失)"
        elif e["sub"] > 0 and e["del"] > 0 and e["ins"] == 0:
            return "替换+删除"
        elif e["ins"] > e["ref_len"]:
            return "指令淹没(大量插入覆盖)"
        elif e["sub"] > 0 and e["ins"] > 0:
            return "混合错误(替换+插入)"
        else:
            return "其他"


def main():
    base = r"e:\8.音频\03 s10\eval_output\testset_results"

    models = {
        "v0.2.0/best": os.path.join(base, "v0.2.0_model.pt.best", "eval_detail.jsonl"),
        "v0.3.0/best": os.path.join(base, "v0.3.0_model.pt.best", "eval_detail.jsonl"),
    }

    all_errors = {}

    for model_name, path in models.items():
        errors, correct, total = process_model(path)
        errors.sort(key=lambda x: x["cer"], reverse=True)
        all_errors[model_name] = errors

        print(f"\n{'=' * 100}")
        print(f"【{model_name}】去唤醒词后错误分析")
        print(f"  总样本: {total}, 正确: {correct}, 错误: {len(errors)}")
        print(f"  准确率: {correct/total*100:.2f}%")
        print(f"{'=' * 100}")

        # 错误分类统计
        categories = Counter()
        for e in errors:
            categories[classify_error(e)] += 1

        print(f"\n  错误分类分布:")
        for cat, count in categories.most_common():
            print(f"    {cat:<25} {count:>5} ({count/len(errors)*100:>5.1f}%)")

        # 指令召回统计
        recalled = sum(1 for e in errors if e["recalled"])
        not_recalled = len(errors) - recalled
        print(f"\n  错误样本中指令召回: {recalled} ({recalled/len(errors)*100:.1f}%)")
        print(f"  错误样本中指令未召回: {not_recalled} ({not_recalled/len(errors)*100:.1f}%)")

    # 导出所有错误到文件
    for model_name, errors in all_errors.items():
        safe_name = model_name.replace("/", "_")
        output_path = os.path.join(
            r"e:\8.音频\03 s10\11_analyse",
            f"testset_errors_{safe_name}_stripped.txt"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"{'=' * 80}\n")
            f.write(f"{model_name} 去唤醒词后错误样本 (共 {len(errors)} 条，按CER降序)\n")
            f.write(f"{'=' * 80}\n\n")

            for e in errors:
                cer_str = f"{e['cer']:.0f}%"
                recalled_str = "✓召回" if e["recalled"] else "✗未召回"
                category = classify_error(e)
                f.write(f"[{e['uid']}] CER={cer_str} {recalled_str} [{category}]\n")
                f.write(f"  标签: {e['ref']}\n")
                f.write(f"  识别: {e['hyp']}\n")
                if e['ref'] != e['ref_orig'] or e['hyp'] != e['hyp_orig']:
                    f.write(f"  (原标签: {e['ref_orig']})\n")
                    f.write(f"  (原识别: {e['hyp_orig']})\n")
                f.write(f"\n")

        print(f"\n已导出: {output_path}")

    # 对比两个模型的差异
    print(f"\n\n{'=' * 100}")
    print("两模型错误差异分析")
    print(f"{'=' * 100}")

    v2_uids = {e["uid"] for e in all_errors["v0.2.0/best"]}
    v3_uids = {e["uid"] for e in all_errors["v0.3.0/best"]}

    v2_only = v2_uids - v3_uids  # v0.2.0错但v0.3.0对
    v3_only = v3_uids - v2_uids  # v0.3.0错但v0.2.0对
    both_err = v2_uids & v3_uids  # 两个都错

    print(f"\n  v0.2.0错 但 v0.3.0对: {len(v2_only)} 条 (v0.3.0优势)")
    print(f"  v0.3.0错 但 v0.2.0对: {len(v3_only)} 条 (v0.2.0优势)")
    print(f"  两个都错: {len(both_err)} 条")

    # v0.3.0的独有优势样本
    v2_map = {e["uid"]: e for e in all_errors["v0.2.0/best"]}
    v3_map = {e["uid"]: e for e in all_errors["v0.3.0/best"]}

    print(f"\n\n--- v0.2.0错但v0.3.0对的样本（前20条）---")
    v2_only_items = sorted([v2_map[uid] for uid in v2_only], key=lambda x: x["cer"], reverse=True)
    for e in v2_only_items[:20]:
        print(f"  CER={e['cer']:.0f}% 标签: {e['ref']}")
        print(f"         识别: {e['hyp'][:60]}")
        print()

    print(f"\n--- v0.3.0错但v0.2.0对的样本（前20条）---")
    v3_only_items = sorted([v3_map[uid] for uid in v3_only], key=lambda x: x["cer"], reverse=True)
    for e in v3_only_items[:20]:
        print(f"  CER={e['cer']:.0f}% 标签: {e['ref']}")
        print(f"         识别: {e['hyp'][:60]}")
        print()


if __name__ == "__main__":
    main()
