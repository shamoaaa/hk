"""
对比4个模型（seaco基线、online线上、v0.2.0、v0.3.0）在业务数据集上的指令召回率
"""
import json
import os
from pathlib import Path


def calc_recall_from_jsonl(jsonl_path):
    """从jsonl文件计算指令召回率"""
    if not os.path.exists(jsonl_path):
        return None
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if not items:
        return None

    total = len(items)
    recalled = sum(1 for i in items if i["ref_norm"] in i["hyp_norm"])
    correct = sum(1 for i in items if i["correct"])
    total_sub = sum(i["sub"] for i in items)
    total_ins = sum(i["ins"] for i in items)
    total_del = sum(i["del"] for i in items)
    total_ref = sum(i["ref_len"] for i in items)
    cer = (total_sub + total_ins + total_del) / total_ref * 100 if total_ref > 0 else 0

    return {
        "total": total,
        "recalled": recalled,
        "recall_rate": recalled / total * 100,
        "correct": correct,
        "accuracy": correct / total * 100,
        "cer": cer,
        "sub": total_sub,
        "ins": total_ins,
        "del": total_del,
    }


def find_eval_detail(base_dir, dataset_path):
    """在目录中查找eval_detail.jsonl"""
    # 尝试多种路径模式
    candidates = [
        os.path.join(base_dir, dataset_path, "test", "eval_detail.jsonl"),
        os.path.join(base_dir, dataset_path, "eval_detail.jsonl"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def main():
    eval_base = r"e:\8.音频\03 s10\eval_output"

    # 模型定义及其数据路径
    models = {}

    # seaco 基线
    models["seaco"] = {
        "s10_reflow": find_eval_detail(
            os.path.join(eval_base, "base", "reflow_eval_result", "seaco"), "s10_reflow"),
        "sp4_reflow": find_eval_detail(
            os.path.join(eval_base, "base", "reflow_eval_result", "seaco"), "sp4_reflow"),
        "s10_augment": find_eval_detail(
            os.path.join(eval_base, "base", "reflow_eval_result", "seaco"), "s10_augment"),
    }

    # online 线上模型
    models["online"] = {
        "s10_reflow": find_eval_detail(
            os.path.join(eval_base, "base", "reflow_eval_result", "online_model"), "s10_reflow"),
        "sp4_reflow": find_eval_detail(
            os.path.join(eval_base, "base", "reflow_eval_result", "online_model"), "sp4_reflow"),
        "s10_augment": find_eval_detail(
            os.path.join(eval_base, "base", "reflow_eval_result", "online_model"), "s10_augment"),
    }

    # v0.2.0
    models["v0.2.0"] = {
        "s10_reflow": find_eval_detail(
            os.path.join(eval_base, "v0.2.0", "model.pt.best", "reflow_label"), "s10_reflow"),
        "sp4_reflow": find_eval_detail(
            os.path.join(eval_base, "v0.2.0", "model.pt.best", "reflow_label"), "sp4_reflow"),
        "s10_augment": find_eval_detail(
            os.path.join(eval_base, "v0.2.0", "model.pt.best", "reflow_label"), "s10_augment"),
    }

    # v0.3.0
    models["v0.3.0"] = {
        "s10_reflow": find_eval_detail(
            os.path.join(eval_base, "v0.3.0", "model.pt.best", "reflow_label"), "s10_reflow"),
        "sp4_reflow": find_eval_detail(
            os.path.join(eval_base, "v0.3.0", "model.pt.best", "reflow_label"), "sp4_reflow"),
        "s10_augment": find_eval_detail(
            os.path.join(eval_base, "v0.3.0", "model.pt.best", "reflow_label"), "s10_augment"),
    }

    # 检查路径
    print("=" * 100)
    print("数据路径检查")
    print("=" * 100)
    for model_name, datasets in models.items():
        for ds_name, path in datasets.items():
            status = "✓" if path else "✗"
            print(f"  {status} {model_name}/{ds_name}: {path}")

    # 计算指标
    print(f"\n\n{'=' * 100}")
    print("四模型指令召回率对比（test集）")
    print("=" * 100)

    datasets = ["s10_reflow", "sp4_reflow", "s10_augment"]

    for ds in datasets:
        print(f"\n{'─' * 100}")
        print(f"【{ds}/test】")
        print(f"{'模型':<12} {'指令召回率':>10} {'CER':>10} {'句准确率':>10} {'Sub':>8} {'Ins':>8} {'Del':>8} {'未召回':>8}")
        print(f"{'─' * 100}")

        for model_name in ["seaco", "online", "v0.2.0", "v0.3.0"]:
            path = models[model_name].get(ds)
            if path:
                result = calc_recall_from_jsonl(path)
                if result:
                    not_recalled = result["total"] - result["recalled"]
                    print(f"{model_name:<12} {result['recall_rate']:>9.2f}% {result['cer']:>9.2f}% {result['accuracy']:>9.2f}% {result['sub']:>8} {result['ins']:>8} {result['del']:>8} {not_recalled:>8}")
                else:
                    print(f"{model_name:<12} (数据为空)")
            else:
                print(f"{model_name:<12} (路径未找到)")

    # 汇总对比表
    print(f"\n\n{'=' * 100}")
    print("指令召回率汇总对比")
    print("=" * 100)

    print(f"\n{'模型':<12}", end="")
    for ds in datasets:
        print(f" {ds:>15}", end="")
    print()
    print("─" * 60)

    for model_name in ["seaco", "online", "v0.2.0", "v0.3.0"]:
        print(f"{model_name:<12}", end="")
        for ds in datasets:
            path = models[model_name].get(ds)
            if path:
                result = calc_recall_from_jsonl(path)
                if result:
                    print(f" {result['recall_rate']:>14.2f}%", end="")
                else:
                    print(f" {'N/A':>15}", end="")
            else:
                print(f" {'N/A':>15}", end="")
        print()

    # vs online 改进幅度
    print(f"\n\n{'=' * 100}")
    print("vs online 线上模型的改进幅度")
    print("=" * 100)

    for ds in datasets:
        online_path = models["online"].get(ds)
        if not online_path:
            continue
        online_result = calc_recall_from_jsonl(online_path)
        if not online_result:
            continue

        print(f"\n【{ds}】online 召回率: {online_result['recall_rate']:.2f}%")
        for model_name in ["v0.2.0", "v0.3.0"]:
            path = models[model_name].get(ds)
            if path:
                result = calc_recall_from_jsonl(path)
                if result:
                    diff = result["recall_rate"] - online_result["recall_rate"]
                    print(f"  {model_name}: {result['recall_rate']:.2f}% ({diff:+.2f}%)")


if __name__ == "__main__":
    main()
