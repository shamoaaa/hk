"""
计算两个模型的指令召回率
指令召回率 = 标签文本作为子串出现在识别结果中的样本比例
"""
import json


def calc_command_recall(jsonl_path):
    """计算指令召回率"""
    total = 0
    recalled = 0
    not_recalled = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            ref = item["ref_norm"]  # 标签（去空格）
            hyp = item["hyp_norm"]  # 识别结果（去空格）
            total += 1

            # 判断标签是否作为子串出现在识别结果中
            if ref in hyp:
                recalled += 1
            else:
                not_recalled.append({
                    "uid": item["uid"],
                    "ref": ref,
                    "hyp": hyp,
                    "cer": item["cer"],
                })

    recall_rate = recalled / total * 100 if total > 0 else 0
    return total, recalled, recall_rate, not_recalled


def main():
    print("=" * 70)
    print("指令召回率对比分析")
    print("=" * 70)

    # 模型1
    path1 = r"e:\8.音频\03 s10\11_analyse\eval_detail.jsonl"
    total1, recalled1, rate1, missed1 = calc_command_recall(path1)

    print(f"\n【模型1】eval_detail.jsonl")
    print(f"  总样本数: {total1}")
    print(f"  指令召回数: {recalled1}")
    print(f"  指令未召回数: {total1 - recalled1}")
    print(f"  指令召回率: {rate1:.2f}%")

    # 模型2
    path2 = r"e:\8.音频\03 s10\11_analyse\eval_detail1.jsonl"
    total2, recalled2, rate2, missed2 = calc_command_recall(path2)

    print(f"\n【模型2】eval_detail1.jsonl")
    print(f"  总样本数: {total2}")
    print(f"  指令召回数: {recalled2}")
    print(f"  指令未召回数: {total2 - recalled2}")
    print(f"  指令召回率: {rate2:.2f}%")

    # 对比
    print(f"\n{'=' * 70}")
    print(f"对比结果:")
    print(f"  模型1 指令召回率: {rate1:.2f}%")
    print(f"  模型2 指令召回率: {rate2:.2f}%")
    print(f"  差异: {rate1 - rate2:+.2f}%")
    print(f"{'=' * 70}")

    # 分析未召回样本的错误类型分布
    print(f"\n\n{'=' * 70}")
    print("未召回样本分析")
    print(f"{'=' * 70}")

    for name, missed in [("模型1", missed1), ("模型2", missed2)]:
        print(f"\n【{name}】未召回样本数: {len(missed)}")

        # 按CER分段统计
        cer_bins = {"CER=0 (完全匹配但未召回?)": 0, "0<CER<=25%": 0,
                    "25%<CER<=50%": 0, "50%<CER<=100%": 0, "CER>100%": 0}
        for item in missed:
            cer = item["cer"] * 100  # 转为百分比
            if cer == 0:
                cer_bins["CER=0 (完全匹配但未召回?)"] += 1
            elif cer <= 25:
                cer_bins["0<CER<=25%"] += 1
            elif cer <= 50:
                cer_bins["25%<CER<=50%"] += 1
            elif cer <= 100:
                cer_bins["50%<CER<=100%"] += 1
            else:
                cer_bins["CER>100%"] += 1

        for k, v in cer_bins.items():
            print(f"    {k}: {v}")

    # 输出模型1未召回但模型2召回的样本（模型2优势）
    missed1_uids = {item["uid"] for item in missed1}
    missed2_uids = {item["uid"] for item in missed2}

    only_m1_missed = missed1_uids - missed2_uids  # 模型1没召回，模型2召回了
    only_m2_missed = missed2_uids - missed1_uids  # 模型2没召回，模型1召回了

    print(f"\n\n{'=' * 70}")
    print(f"差异分析:")
    print(f"  模型1未召回 但 模型2召回: {len(only_m1_missed)} 条（模型2优势）")
    print(f"  模型2未召回 但 模型1召回: {len(only_m2_missed)} 条（模型1优势）")
    print(f"{'=' * 70}")

    # 打印模型2未召回但模型1召回的样本（前20条）
    m2_only_missed_items = [item for item in missed2 if item["uid"] in only_m2_missed]
    m2_only_missed_items.sort(key=lambda x: x["cer"], reverse=True)

    print(f"\n模型2未召回但模型1召回的样本（前30条，按CER降序）:")
    print("-" * 70)
    for item in m2_only_missed_items[:30]:
        print(f"  [{item['uid']}] CER={item['cer']*100:.0f}%")
        print(f"    标签: {item['ref']}")
        print(f"    识别: {item['hyp']}")
        print()

    # 打印模型1未召回但模型2召回的样本（前20条）
    m1_only_missed_items = [item for item in missed1 if item["uid"] in only_m1_missed]
    m1_only_missed_items.sort(key=lambda x: x["cer"], reverse=True)

    print(f"\n模型1未召回但模型2召回的样本（前30条，按CER降序）:")
    print("-" * 70)
    for item in m1_only_missed_items[:30]:
        print(f"  [{item['uid']}] CER={item['cer']*100:.0f}%")
        print(f"    标签: {item['ref']}")
        print(f"    识别: {item['hyp']}")
        print()


if __name__ == "__main__":
    main()
