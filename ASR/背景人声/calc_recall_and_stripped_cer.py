"""
计算 SP4 指令召回率 和 testset 剔除唤醒词前后的 CER。
"""
import json
import os
import re

BASE = r"E:\8.音频\03 s10\eval_output"

# 唤醒词列表（按长度降序）
WAKEWORDS = [
    "你好萤萤石", "营养萤石", "你好萤石", "你好伊思", "你好游石", "你好萤乐", "你好萤哲", "你好李石",
    "你好刘石", "控好萤石", "控楼萤石", "平好萤石", "你要萤石", "你很萤石", "怎样萤石", "你奥伊石",
    "杨萤石", "样萤石", "萤萤石", "好萤石", "你好萤", "你好石", "萤石",
]
WAKEWORDS.sort(key=len, reverse=True)


def strip_wakewords(text, max_rounds=3):
    """剥离头尾唤醒词，最多 max_rounds 轮"""
    for _ in range(max_rounds):
        stripped = False
        for ww in WAKEWORDS:
            if text.startswith(ww):
                text = text[len(ww):]
                stripped = True
                break
        for ww in WAKEWORDS:
            if text.endswith(ww):
                text = text[:-len(ww)]
                stripped = True
                break
        if not stripped:
            break
    return text


def calc_recall(detail_path):
    """计算指令召回率：剔除唤醒词后 ref_stripped in hyp_stripped"""
    total = 0
    recalled = 0
    with open(detail_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            ref = item.get("ref_norm", "")
            hyp = item.get("hyp_norm", "")
            ref_stripped = strip_wakewords(ref)
            hyp_stripped = strip_wakewords(hyp)
            if not ref_stripped:
                continue
            total += 1
            if ref_stripped in hyp_stripped:
                recalled += 1
    return recalled / total if total > 0 else None, total


def calc_cer_stripped(detail_path):
    """计算剔除唤醒词后的 CER"""
    total_ref_chars = 0
    total_errors = 0
    total_ref_chars_raw = 0
    total_errors_raw = 0
    
    with open(detail_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            ref = item.get("ref_norm", "")
            hyp = item.get("hyp_norm", "")
            
            # 原始 CER（不剔除唤醒词）
            total_ref_chars_raw += len(ref)
            # 使用已有的 cer 字段或重新计算
            if "edit_distance" in item:
                total_errors_raw += item["edit_distance"]
            
            # 剔除唤醒词后
            ref_stripped = strip_wakewords(ref)
            hyp_stripped = strip_wakewords(hyp)
            
            if not ref_stripped:
                continue
            
            total_ref_chars += len(ref_stripped)
            # 简单编辑距离近似：用字符级别
            errors = edit_distance(ref_stripped, hyp_stripped)
            total_errors += errors
    
    cer_stripped = total_errors / total_ref_chars if total_ref_chars > 0 else None
    return cer_stripped


def edit_distance(ref, hyp):
    """计算字符级编辑距离"""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    
    # 优化：使用两行 DP
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            if ref[i-1] == hyp[j-1]:
                curr[j] = prev[j-1]
            else:
                curr[j] = 1 + min(prev[j], curr[j-1], prev[j-1])
        prev, curr = curr, prev
    
    return prev[m]


def main():
    # 每版本最佳模型（用于 SP4 召回率和 testset）
    BEST_MODELS = {
        "seaco": "base_model.pt.seaco",
        "online": "base_model.pt.online",
        "v0.1.0": "v0.1.0_model.pt.ep16.4000",
        "v0.2.0": "v0.2.0_model.pt.best",
        "v0.3.0": "v0.3.0_model.pt.best",
        "v0.4.0": "v0.4.0_model.pt.best",
        "v0.5.0": "v0.5.0_model.pt.ep35",
        "v0.6.0": "v0.6.0_model.pt.ep33",
    }
    
    # ===== SP4 指令召回率 =====
    print("=" * 80)
    print("【SP4 指令召回率】")
    print("=" * 80)
    print(f"{'模型':<30} {'召回率 ↑':<12} {'有效样本数':<12}")
    print("-" * 54)
    
    for label, dir_name in BEST_MODELS.items():
        path = os.path.join(BASE, "sp4_results", dir_name, "eval_detail.jsonl")
        if os.path.exists(path):
            recall, total = calc_recall(path)
            if recall is not None:
                print(f"{label:<28} {recall*100:.2f}%      {total}")
            else:
                print(f"{label:<28} N/A")
        else:
            print(f"{label:<28} [文件缺失]")
    
    # ===== testset 剔除唤醒词前后 CER =====
    print()
    print("=" * 80)
    print("【独立测试集 CER（剔除唤醒词前后对比）】")
    print("=" * 80)
    print(f"{'模型':<30} {'原始CER ↓':<12} {'剔除唤醒词CER ↓':<16}")
    print("-" * 58)
    
    # 从 comparison_summary 获取原始 CER
    testset_summary_path = os.path.join(BASE, "testset_results", "comparison_summary.json")
    with open(testset_summary_path, "r", encoding="utf-8") as f:
        testset_data = json.load(f)
    
    for label, dir_name in BEST_MODELS.items():
        # 原始 CER
        # 需要映射 dir_name 到 comparison_summary 中的 key
        summary_key = dir_name.replace("_", "/", 1)  # base_model.pt.online -> base/model.pt.online
        raw_cer = None
        for k, v in testset_data.get("models", {}).items():
            if k == summary_key:
                raw_cer = v["overall_cer"]
                break
        
        # 剔除唤醒词后 CER
        detail_path = os.path.join(BASE, "testset_results", dir_name, "eval_detail.jsonl")
        if os.path.exists(detail_path):
            stripped_cer = calc_cer_stripped(detail_path)
            raw_str = f"{raw_cer*100:.2f}%" if raw_cer else "N/A"
            stripped_str = f"{stripped_cer*100:.2f}%" if stripped_cer else "N/A"
            print(f"{label:<28} {raw_str:<12} {stripped_str:<16}")
        else:
            print(f"{label:<28} [文件缺失]")


if __name__ == "__main__":
    main()
