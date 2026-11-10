"""统计 eval_detail.jsonl 的 CER，并将错误样本按格式输出到 txt"""
import json

input_file = r"e:\8.音频\03 s10\11_analyse\eval_detail1.jsonl"
output_file = r"e:\8.音频\03 s10\11_analyse\eval_errors1.txt"

total_ref_len = 0
total_sub = 0
total_ins = 0
total_del = 0
total_samples = 0
correct_samples = 0
error_entries = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        total_samples += 1
        total_ref_len += item["ref_len"]
        total_sub += item["sub"]
        total_ins += item["ins"]
        total_del += item["del"]

        if item["correct"]:
            correct_samples += 1
        else:
            cer_pct = item["cer"] * 100
            # 去除 hyp 中的空格
            hyp_clean = item["hyp"].replace(" ", "")
            error_entries.append({
                "uid": item["uid"],
                "cer_pct": cer_pct,
                "ref": item["ref"],
                "hyp": hyp_clean,
            })

overall_cer = (total_sub + total_ins + total_del) / total_ref_len if total_ref_len > 0 else 0

# 按 CER 降序排列
error_entries.sort(key=lambda x: x["cer_pct"], reverse=True)

with open(output_file, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("ASR 模型评测结果统计\n")
    f.write("=" * 60 + "\n")
    f.write(f"总样本数: {total_samples}\n")
    f.write(f"正确样本数: {correct_samples}\n")
    f.write(f"错误样本数: {total_samples - correct_samples}\n")
    f.write(f"句准确率 (Sentence Accuracy): {correct_samples / total_samples * 100:.2f}%\n")
    f.write(f"总参考字数: {total_ref_len}\n")
    f.write(f"总替换数: {total_sub}\n")
    f.write(f"总插入数: {total_ins}\n")
    f.write(f"总删除数: {total_del}\n")
    f.write(f"整体 CER: {overall_cer * 100:.2f}%\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"错误样本详情 (共 {len(error_entries)} 条，按 CER 降序排列):\n")
    f.write("-" * 60 + "\n\n")

    for entry in error_entries:
        f.write(f"[{entry['uid']}] CER={entry['cer_pct']:.0f}%\n")
        f.write(f"标签: {entry['ref']}\n")
        f.write(f"识别: {entry['hyp']}\n")
        f.write("\n")

print(f"统计完成！")
print(f"总样本: {total_samples}, 正确: {correct_samples}, 错误: {total_samples - correct_samples}")
print(f"整体 CER: {overall_cer * 100:.2f}%")
print(f"结果已输出到: {output_file}")
