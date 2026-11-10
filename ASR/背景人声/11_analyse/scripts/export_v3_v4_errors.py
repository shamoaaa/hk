"""
导出v0.3.0和v0.4.0在业务数据集(s10_reflow/test, sp4_reflow/test)上的所有错误样本
并列展示两个版本的识别结果差异
"""
import json
import os


def load_jsonl(path):
    items = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                items[item["uid"]] = item
    return items


def main():
    base = r"e:\8.音频\03 s10\eval_output"

    datasets = {
        "s10_reflow": "reflow_label/s10_reflow/test/eval_detail.jsonl",
        "sp4_reflow": "reflow_label/sp4_reflow/test/eval_detail.jsonl",
    }

    for ds_name, ds_path in datasets.items():
        v3_path = os.path.join(base, "v0.3.0/model.pt.best", ds_path)
        v4_path = os.path.join(base, "v0.4.0/model.pt.best", ds_path)

        v3_data = load_jsonl(v3_path)
        v4_data = load_jsonl(v4_path)

        # 找出任一模型有错误的样本
        all_uids = set(v3_data.keys()) | set(v4_data.keys())
        error_uids = []
        for uid in all_uids:
            v3_item = v3_data.get(uid)
            v4_item = v4_data.get(uid)
            if (v3_item and not v3_item["correct"]) or (v4_item and not v4_item["correct"]):
                error_uids.append(uid)

        # 按CER降序排列（取两者中较大的CER）
        def sort_key(uid):
            c3 = v3_data[uid]["cer"] if uid in v3_data else 0
            c4 = v4_data[uid]["cer"] if uid in v4_data else 0
            return max(c3, c4)

        error_uids.sort(key=sort_key, reverse=True)

        # 统计
        v3_errors = sum(1 for uid in error_uids if uid in v3_data and not v3_data[uid]["correct"])
        v4_errors = sum(1 for uid in error_uids if uid in v4_data and not v4_data[uid]["correct"])
        both_error = sum(1 for uid in error_uids
                        if uid in v3_data and not v3_data[uid]["correct"]
                        and uid in v4_data and not v4_data[uid]["correct"])
        v3_only = sum(1 for uid in error_uids
                      if uid in v3_data and not v3_data[uid]["correct"]
                      and uid in v4_data and v4_data[uid]["correct"])
        v4_only = sum(1 for uid in error_uids
                      if uid in v4_data and not v4_data[uid]["correct"]
                      and uid in v3_data and v3_data[uid]["correct"])

        # 输出到文件
        output_path = os.path.join(
            r"e:\8.音频\03 s10\11_analyse",
            f"errors_v3_v4_{ds_name}.txt"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"{'=' * 90}\n")
            f.write(f"{ds_name}/test 错误样本对比 (v0.3.0 vs v0.4.0)\n")
            f.write(f"{'=' * 90}\n")
            f.write(f"总样本数: {len(all_uids)}\n")
            f.write(f"任一模型有错误: {len(error_uids)}\n")
            f.write(f"v0.3.0 错误数: {v3_errors}\n")
            f.write(f"v0.4.0 错误数: {v4_errors}\n")
            f.write(f"两者都错: {both_error}\n")
            f.write(f"仅v0.3.0错(v0.4.0对): {v3_only}\n")
            f.write(f"仅v0.4.0错(v0.3.0对): {v4_only}\n")
            f.write(f"{'=' * 90}\n\n")

            # 分三部分输出
            sections = [
                ("两者都错", lambda uid: uid in v3_data and not v3_data[uid]["correct"]
                 and uid in v4_data and not v4_data[uid]["correct"]),
                ("仅v0.3.0错 (v0.4.0正确)", lambda uid: uid in v3_data and not v3_data[uid]["correct"]
                 and uid in v4_data and v4_data[uid]["correct"]),
                ("仅v0.4.0错 (v0.3.0正确)", lambda uid: uid in v4_data and not v4_data[uid]["correct"]
                 and uid in v3_data and v3_data[uid]["correct"]),
            ]

            for section_name, filter_fn in sections:
                section_uids = [uid for uid in error_uids if filter_fn(uid)]
                section_uids.sort(key=sort_key, reverse=True)

                f.write(f"\n{'─' * 90}\n")
                f.write(f"【{section_name}】共 {len(section_uids)} 条\n")
                f.write(f"{'─' * 90}\n\n")

                for uid in section_uids:
                    v3 = v3_data.get(uid)
                    v4 = v4_data.get(uid)
                    ref = v3["ref_norm"] if v3 else v4["ref_norm"]
                    hyp3 = v3["hyp_norm"] if v3 else "N/A"
                    hyp4 = v4["hyp_norm"] if v4 else "N/A"
                    cer3 = f"{v3['cer']*100:.0f}%" if v3 else "N/A"
                    cer4 = f"{v4['cer']*100:.0f}%" if v4 else "N/A"

                    # 标记指令是否被召回
                    r3 = "✓" if ref in hyp3 else "✗"
                    r4 = "✓" if ref in hyp4 else "✗"

                    f.write(f"[{uid}]\n")
                    f.write(f"  标签:   {ref}\n")
                    f.write(f"  v0.3.0: {hyp3}  (CER={cer3}, 召回{r3})\n")
                    f.write(f"  v0.4.0: {hyp4}  (CER={cer4}, 召回{r4})\n")
                    f.write(f"\n")

        print(f"已导出: {output_path}")
        print(f"  总错误: {len(error_uids)}, 两者都错: {both_error}, "
              f"仅v0.3.0错: {v3_only}, 仅v0.4.0错: {v4_only}")


if __name__ == "__main__":
    main()
