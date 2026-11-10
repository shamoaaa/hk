"""
分析 testset_results 下6个模型的错误特征
重点：指令召回率、错误类型分布、典型错误模式
"""
import json
import os
from collections import defaultdict, Counter


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def analyse_model(items, model_name):
    """全面分析一个模型的错误"""
    total = len(items)
    correct = sum(1 for i in items if i["correct"])
    errors = [i for i in items if not i["correct"]]

    # 指令召回率
    recalled = sum(1 for i in items if i["ref_norm"] in i["hyp_norm"])
    not_recalled = [i for i in items if i["ref_norm"] not in i["hyp_norm"]]

    # 错误类型统计
    total_sub = sum(i["sub"] for i in items)
    total_ins = sum(i["ins"] for i in items)
    total_del = sum(i["del"] for i in items)
    total_ref = sum(i["ref_len"] for i in items)

    return {
        "model": model_name,
        "total": total,
        "correct": correct,
        "accuracy": correct / total * 100,
        "errors": len(errors),
        "recalled": recalled,
        "recall_rate": recalled / total * 100,
        "not_recalled": len(not_recalled),
        "not_recalled_items": not_recalled,
        "sub": total_sub,
        "ins": total_ins,
        "del": total_del,
        "ref_chars": total_ref,
        "cer": (total_sub + total_ins + total_del) / total_ref * 100,
        "error_items": errors,
    }


def classify_error(item):
    """对错误进行分类"""
    ref = item["ref_norm"]
    hyp = item["hyp_norm"]

    # 指令是否被召回
    if ref in hyp:
        # 指令在，但有多余内容
        extra = hyp.replace(ref, "", 1)
        if "你好萤石" in extra or "你好萤" in extra:
            return "唤醒词残留"
        elif len(hyp) > len(ref) * 1.5:
            return "多识别(环境音)"
        else:
            return "多识别(轻微)"
    else:
        # 指令未被召回
        if item["del"] > 0 and item["sub"] == 0 and item["ins"] == 0:
            return "漏识别(纯删除)"
        elif item["sub"] > 0 and item["ins"] == 0:
            return "替换错误"
        elif item["ins"] > item["ref_len"]:
            return "指令淹没(大量插入)"
        elif item["sub"] > 0 and item["ins"] > 0:
            return "混合错误"
        else:
            return "其他"


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

    all_results = {}
    for name, path in models.items():
        jsonl_path = os.path.join(base, path, "eval_detail.jsonl")
        if os.path.exists(jsonl_path):
            items = load_jsonl(jsonl_path)
            all_results[name] = analyse_model(items, name)

    # ========== 总览 ==========
    print("=" * 90)
    print("testset_results 6个模型对比（业务数据抽样测试集）")
    print("=" * 90)

    print(f"\n{'模型':<16} {'指令召回率':>10} {'CER':>8} {'句准确率':>10} {'Sub':>6} {'Ins':>6} {'Del':>6} {'未召回':>8}")
    print("─" * 90)
    for name in models:
        if name in all_results:
            r = all_results[name]
            print(f"{name:<16} {r['recall_rate']:>9.2f}% {r['cer']:>7.2f}% {r['accuracy']:>9.2f}% {r['sub']:>6} {r['ins']:>6} {r['del']:>6} {r['not_recalled']:>8}")

    # ========== 错误分类分析 ==========
    print(f"\n\n{'=' * 90}")
    print("错误分类分布")
    print("=" * 90)

    for name in models:
        if name not in all_results:
            continue
        r = all_results[name]
        error_types = Counter()
        for item in r["error_items"]:
            error_types[classify_error(item)] += 1

        print(f"\n【{name}】错误总数: {r['errors']}")
        for etype, count in error_types.most_common():
            pct = count / r["errors"] * 100
            print(f"    {etype:<20} {count:>5} ({pct:>5.1f}%)")

    # ========== 未召回样本深度分析 ==========
    print(f"\n\n{'=' * 90}")
    print("未召回样本对比（指令完全丢失的情况）")
    print("=" * 90)

    # 找出各模型共同未召回和独有未召回
    not_recalled_sets = {}
    not_recalled_maps = {}
    for name in models:
        if name in all_results:
            nr_items = all_results[name]["not_recalled_items"]
            not_recalled_sets[name] = {i["uid"] for i in nr_items}
            not_recalled_maps[name] = {i["uid"]: i for i in nr_items}

    # 所有模型都未召回的（硬骨头）
    if not_recalled_sets:
        common_missed = set.intersection(*not_recalled_sets.values())
        print(f"\n所有模型都未召回的样本: {len(common_missed)} 条")

        # 展示这些硬骨头
        ref_model = list(not_recalled_maps.keys())[0]
        hard_cases = [not_recalled_maps[ref_model][uid] for uid in common_missed if uid in not_recalled_maps[ref_model]]
        hard_cases.sort(key=lambda x: x["cer"], reverse=True)

        print(f"\n前30条（按CER降序）:")
        print("-" * 90)
        for item in hard_cases[:30]:
            print(f"  CER={item['cer']*100:.0f}% 标签: {item['ref_norm']}")
            print(f"         识别: {item['hyp_norm'][:70]}")
            print()

    # v0.3.0 独有的未召回（v0.3.0的弱点）
    print(f"\n\n{'=' * 90}")
    print("v0.3.0/best 独有的未召回（其他模型能召回但v0.3.0不能）")
    print("=" * 90)

    if "v0.3.0/best" in not_recalled_sets and "v0.2.0/best" in not_recalled_sets:
        v3_only = not_recalled_sets["v0.3.0/best"] - not_recalled_sets["v0.2.0/best"]
        v3_items = [not_recalled_maps["v0.3.0/best"][uid] for uid in v3_only if uid in not_recalled_maps["v0.3.0/best"]]
        v3_items.sort(key=lambda x: x["cer"], reverse=True)
        print(f"\nv0.3.0未召回但v0.2.0召回: {len(v3_only)} 条")
        for item in v3_items[:20]:
            print(f"  CER={item['cer']*100:.0f}% 标签: {item['ref_norm']}")
            print(f"         识别: {item['hyp_norm'][:70]}")
            print()

    # v0.2.0 独有的未召回（v0.2.0的弱点）
    print(f"\n{'=' * 90}")
    print("v0.2.0/best 独有的未召回（v0.3.0能召回但v0.2.0不能）")
    print("=" * 90)

    if "v0.3.0/best" in not_recalled_sets and "v0.2.0/best" in not_recalled_sets:
        v2_only = not_recalled_sets["v0.2.0/best"] - not_recalled_sets["v0.3.0/best"]
        v2_items = [not_recalled_maps["v0.2.0/best"][uid] for uid in v2_only if uid in not_recalled_maps["v0.2.0/best"]]
        v2_items.sort(key=lambda x: x["cer"], reverse=True)
        print(f"\nv0.2.0未召回但v0.3.0召回: {len(v2_only)} 条")
        for item in v2_items[:20]:
            print(f"  CER={item['cer']*100:.0f}% 标签: {item['ref_norm']}")
            print(f"         识别: {item['hyp_norm'][:70]}")
            print()

    # ========== 高频错误模式 ==========
    print(f"\n\n{'=' * 90}")
    print("v0.3.0/best 高频替换错误模式（ref→hyp 的常见混淆对）")
    print("=" * 90)

    if "v0.3.0/best" in all_results:
        confusion_pairs = Counter()
        for item in all_results["v0.3.0/best"]["error_items"]:
            ref = item["ref_norm"]
            hyp = item["hyp_norm"]
            if item["sub"] > 0 and len(ref) <= 10 and len(hyp) <= 15:
                confusion_pairs[(ref, hyp)] += 1

        print(f"\n出现2次以上的混淆对:")
        for (ref, hyp), count in confusion_pairs.most_common(30):
            if count >= 2:
                print(f"  [{count:>3}次] {ref} → {hyp}")


if __name__ == "__main__":
    main()
