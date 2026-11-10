"""
生成指令召回率对比报告：
聚焦v0.3.0能召回但v0.4.0不能、v0.4.0能召回但v0.3.0不能的样本
去除开头唤醒词后统计
"""
import json
import os

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
    changed = True
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


def load_data(jsonl_path):
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
            recalled = ref in hyp
            results[uid] = {"uid": uid, "ref": ref, "hyp": hyp, "recalled": recalled}
    return results


def main():
    base = r"e:\8.音频\03 s10\eval_output"
    output_path = r"e:\8.音频\03 s10\11_analyse\recall_comparison_v3_v4.txt"

    datasets = {
        "s10_reflow/test": "reflow_label/s10_reflow/test/eval_detail.jsonl",
        "sp4_reflow/test": "reflow_label/sp4_reflow/test/eval_detail.jsonl",
    }

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("v0.3.0 vs v0.4.0 指令召回率对比报告\n")
        f.write("（去除开头唤醒词后，标签是否作为子串出现在识别结果中）\n")
        f.write("=" * 90 + "\n\n")

        for ds_name, ds_path in datasets.items():
            v3 = load_data(os.path.join(base, "v0.3.0/model.pt.best", ds_path))
            v4 = load_data(os.path.join(base, "v0.4.0/model.pt.best", ds_path))

            common = set(v3.keys()) & set(v4.keys())
            total = len(common)
            v3_recalled = sum(1 for uid in common if v3[uid]["recalled"])
            v4_recalled = sum(1 for uid in common if v4[uid]["recalled"])

            # 分类
            both_recalled = [uid for uid in common if v3[uid]["recalled"] and v4[uid]["recalled"]]
            both_missed = [uid for uid in common if not v3[uid]["recalled"] and not v4[uid]["recalled"]]
            v3_only = [uid for uid in common if v3[uid]["recalled"] and not v4[uid]["recalled"]]
            v4_only = [uid for uid in common if not v3[uid]["recalled"] and v4[uid]["recalled"]]

            f.write(f"\n{'═' * 90}\n")
            f.write(f"【{ds_name}】\n")
            f.write(f"{'═' * 90}\n\n")
            f.write(f"总样本: {total}\n")
            f.write(f"v0.3.0 召回: {v3_recalled} ({v3_recalled/total*100:.2f}%)\n")
            f.write(f"v0.4.0 召回: {v4_recalled} ({v4_recalled/total*100:.2f}%)\n")
            f.write(f"两者都召回: {len(both_recalled)}\n")
            f.write(f"两者都未召回: {len(both_missed)}\n")
            f.write(f"v0.3.0召回 但 v0.4.0未召回: {len(v3_only)} (v0.3.0优势)\n")
            f.write(f"v0.4.0召回 但 v0.3.0未召回: {len(v4_only)} (v0.4.0优势)\n")
            f.write(f"\n")

            # === v0.3.0能召回但v0.4.0不能 ===
            f.write(f"\n{'─' * 90}\n")
            f.write(f"v0.3.0能召回 但 v0.4.0不能 ({len(v3_only)} 条)\n")
            f.write(f"{'─' * 90}\n\n")

            # 按标签长度排序（短指令优先）
            v3_only_sorted = sorted(v3_only, key=lambda uid: len(v3[uid]["ref"]))
            for uid in v3_only_sorted:
                f.write(f"  [{uid}]\n")
                f.write(f"  标签: {v3[uid]['ref']}\n")
                f.write(f"  v0.3.0: {v3[uid]['hyp']}\n")
                f.write(f"  v0.4.0: {v4[uid]['hyp']}\n")
                f.write(f"\n")

            # === v0.4.0能召回但v0.3.0不能 ===
            f.write(f"\n{'─' * 90}\n")
            f.write(f"v0.4.0能召回 但 v0.3.0不能 ({len(v4_only)} 条)\n")
            f.write(f"{'─' * 90}\n\n")

            v4_only_sorted = sorted(v4_only, key=lambda uid: len(v4[uid]["ref"]))
            for uid in v4_only_sorted:
                f.write(f"  [{uid}]\n")
                f.write(f"  标签: {v3[uid]['ref']}\n")
                f.write(f"  v0.3.0: {v3[uid]['hyp']}\n")
                f.write(f"  v0.4.0: {v4[uid]['hyp']}\n")
                f.write(f"\n")

            # === 两者都未召回 ===
            f.write(f"\n{'─' * 90}\n")
            f.write(f"两者都未召回 ({len(both_missed)} 条)\n")
            f.write(f"{'─' * 90}\n\n")

            both_missed_sorted = sorted(both_missed, key=lambda uid: len(v3[uid]["ref"]))
            for uid in both_missed_sorted:
                f.write(f"  [{uid}]\n")
                f.write(f"  标签: {v3[uid]['ref']}\n")
                f.write(f"  v0.3.0: {v3[uid]['hyp']}\n")
                f.write(f"  v0.4.0: {v4[uid]['hyp']}\n")
                f.write(f"\n")

    print(f"报告已生成: {output_path}")

    # 打印摘要到控制台
    for ds_name, ds_path in datasets.items():
        v3 = load_data(os.path.join(base, "v0.3.0/model.pt.best", ds_path))
        v4 = load_data(os.path.join(base, "v0.4.0/model.pt.best", ds_path))
        common = set(v3.keys()) & set(v4.keys())
        total = len(common)
        v3_recalled = sum(1 for uid in common if v3[uid]["recalled"])
        v4_recalled = sum(1 for uid in common if v4[uid]["recalled"])
        v3_only = [uid for uid in common if v3[uid]["recalled"] and not v4[uid]["recalled"]]
        v4_only = [uid for uid in common if not v3[uid]["recalled"] and v4[uid]["recalled"]]
        both_missed = [uid for uid in common if not v3[uid]["recalled"] and not v4[uid]["recalled"]]

        print(f"\n【{ds_name}】")
        print(f"  v0.3.0 召回率: {v3_recalled/total*100:.2f}%")
        print(f"  v0.4.0 召回率: {v4_recalled/total*100:.2f}%")
        print(f"  v0.3.0独有召回: {len(v3_only)} 条")
        print(f"  v0.4.0独有召回: {len(v4_only)} 条")
        print(f"  两者都未召回: {len(both_missed)} 条")


if __name__ == "__main__":
    main()
