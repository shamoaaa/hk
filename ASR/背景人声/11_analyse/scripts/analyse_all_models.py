"""
分析 eval_output 下 3个版本 x 2个权重 = 6个模型的错误特征
重点关注：
1. 各数据集类别的错误分布
2. 指令召回率对比
3. 错误类型（替换/插入/删除）分布差异
"""
import json
import os
from pathlib import Path
from collections import defaultdict


def load_eval_details(base_dir):
    """递归加载某个模型目录下所有 eval_detail.jsonl"""
    results = {}
    base = Path(base_dir)
    for jsonl_file in base.rglob("eval_detail.jsonl"):
        rel_path = str(jsonl_file.relative_to(base).parent).replace("\\", "/")
        items = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        results[rel_path] = items
    return results


def calc_command_recall(items):
    """计算指令召回率"""
    total = len(items)
    recalled = sum(1 for item in items if item["ref_norm"] in item["hyp_norm"])
    return recalled, total


def analyse_error_types(items):
    """分析错误类型分布"""
    total_sub = sum(item["sub"] for item in items)
    total_ins = sum(item["ins"] for item in items)
    total_del = sum(item["del"] for item in items)
    total_ref_chars = sum(item["ref_len"] for item in items)
    error_items = [item for item in items if not item["correct"]]
    return {
        "total": len(items),
        "errors": len(error_items),
        "accuracy": (len(items) - len(error_items)) / len(items) * 100,
        "sub": total_sub,
        "ins": total_ins,
        "del": total_del,
        "ref_chars": total_ref_chars,
        "cer": (total_sub + total_ins + total_del) / total_ref_chars * 100 if total_ref_chars > 0 else 0,
    }


def main():
    eval_base = r"e:\8.音频\03 s10\eval_output"

    # 定义6个模型
    models = {
        "v0.1.0/best": "v0.1.0/model.pt.best",
        "v0.1.0/ep16": "v0.1.0/model.pt.ep16.4000",
        "v0.2.0/best": "v0.2.0/model.pt.best",
        "v0.2.0/ep14": "v0.2.0/model.pt.ep14.4000",
        "v0.3.0/best": "v0.3.0/model.pt.best",
        "v0.3.0/ep27": "v0.3.0/model.pt.ep27.4000",
    }

    # 关注的数据集分类
    categories = {
        "s10_reflow": "reflow_label/s10_reflow/test",
        "sp4_reflow": "reflow_label/sp4_reflow/test",
        "s10_augment": "reflow_label/s10_augment/test",
    }

    print("=" * 80)
    print("6个模型在业务测试集上的指令召回率和错误特征对比")
    print("=" * 80)

    # 收集所有模型在关键数据集上的指标
    all_results = {}

    for model_name, model_path in models.items():
        full_path = os.path.join(eval_base, model_path)
        if not os.path.exists(full_path):
            print(f"  [跳过] {model_name}: 路径不存在")
            continue

        data = load_eval_details(full_path)
        all_results[model_name] = {}

        for cat_name, cat_path in categories.items():
            if cat_path in data:
                items = data[cat_path]
                recalled, total = calc_command_recall(items)
                stats = analyse_error_types(items)
                recall_rate = recalled / total * 100 if total > 0 else 0

                all_results[model_name][cat_name] = {
                    "recall_rate": recall_rate,
                    "recalled": recalled,
                    "total": total,
                    **stats,
                }

    # 打印对比表格
    print(f"\n{'─' * 80}")
    print(f"{'数据集':<15} {'模型':<18} {'指令召回率':>10} {'CER':>8} {'句准确率':>10} {'替换':>6} {'插入':>6} {'删除':>6}")
    print(f"{'─' * 80}")

    for cat_name in categories:
        for model_name in models:
            if model_name in all_results and cat_name in all_results[model_name]:
                r = all_results[model_name][cat_name]
                print(f"{cat_name:<15} {model_name:<18} {r['recall_rate']:>9.2f}% {r['cer']:>7.2f}% {r['accuracy']:>9.2f}% {r['sub']:>6} {r['ins']:>6} {r['del']:>6}")
        print(f"{'─' * 80}")

    # 重点分析 sp4_reflow（你之前分析的业务数据集）
    print(f"\n\n{'=' * 80}")
    print("sp4_reflow/test 详细对比（你的核心业务数据集）")
    print(f"{'=' * 80}")

    print(f"\n{'模型':<20} {'指令召回率':>12} {'CER':>8} {'句准确率':>10} {'Sub':>6} {'Ins':>8} {'Del':>6} {'未召回':>8}")
    print(f"{'─' * 80}")
    for model_name in models:
        if model_name in all_results and "sp4_reflow" in all_results[model_name]:
            r = all_results[model_name]["sp4_reflow"]
            not_recalled = r["total"] - r["recalled"]
            print(f"{model_name:<20} {r['recall_rate']:>11.2f}% {r['cer']:>7.2f}% {r['accuracy']:>9.2f}% {r['sub']:>6} {r['ins']:>8} {r['del']:>6} {not_recalled:>8}")

    # 分析 s10_augment 的极端错误
    print(f"\n\n{'=' * 80}")
    print("s10_augment/test 分析（CER极高的数据集）")
    print(f"{'=' * 80}")

    print(f"\n{'模型':<20} {'指令召回率':>12} {'CER':>10} {'句准确率':>10} {'Sub':>6} {'Ins':>8} {'Del':>6}")
    print(f"{'─' * 80}")
    for model_name in models:
        if model_name in all_results and "s10_augment" in all_results[model_name]:
            r = all_results[model_name]["s10_augment"]
            print(f"{model_name:<20} {r['recall_rate']:>11.2f}% {r['cer']:>9.2f}% {r['accuracy']:>9.2f}% {r['sub']:>6} {r['ins']:>8} {r['del']:>6}")

    # 分析 s10_reflow
    print(f"\n\n{'=' * 80}")
    print("s10_reflow/test 分析")
    print(f"{'=' * 80}")

    print(f"\n{'模型':<20} {'指令召回率':>12} {'CER':>8} {'句准确率':>10} {'Sub':>6} {'Ins':>8} {'Del':>6}")
    print(f"{'─' * 80}")
    for model_name in models:
        if model_name in all_results and "s10_reflow" in all_results[model_name]:
            r = all_results[model_name]["s10_reflow"]
            print(f"{model_name:<20} {r['recall_rate']:>11.2f}% {r['cer']:>7.2f}% {r['accuracy']:>9.2f}% {r['sub']:>6} {r['ins']:>8} {r['del']:>6}")

    # 12_test 测试集对比
    print(f"\n\n{'=' * 80}")
    print("12_test 测试集对比（testset_results）")
    print(f"{'=' * 80}")

    testset_models = {
        "v0.1.0/best": "testset_results/v0.1.0_model.pt.best",
        "v0.1.0/ep16": "testset_results/v0.1.0_model.pt.ep16.4000",
        "v0.2.0/best": "testset_results/v0.2.0_model.pt.best",
        "v0.2.0/ep14": "testset_results/v0.2.0_model.pt.ep14.4000",
        "v0.3.0/best": "testset_results/v0.3.0_model.pt.best",
        "v0.3.0/ep27": "testset_results/v0.3.0_model.pt.ep27.4000",
    }

    print(f"\n{'模型':<20} {'指令召回率':>12} {'CER':>8} {'句准确率':>10} {'Sub':>6} {'Ins':>8} {'Del':>6} {'未召回':>8}")
    print(f"{'─' * 80}")

    for model_name, model_path in testset_models.items():
        jsonl_path = os.path.join(eval_base, model_path, "eval_detail.jsonl")
        if not os.path.exists(jsonl_path):
            continue
        items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))

        recalled, total = calc_command_recall(items)
        stats = analyse_error_types(items)
        recall_rate = recalled / total * 100 if total > 0 else 0
        not_recalled = total - recalled

        print(f"{model_name:<20} {recall_rate:>11.2f}% {stats['cer']:>7.2f}% {stats['accuracy']:>9.2f}% {stats['sub']:>6} {stats['ins']:>8} {stats['del']:>6} {not_recalled:>8}")


if __name__ == "__main__":
    main()
