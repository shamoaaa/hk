"""
找出 testset 中 v0.5.0/ep35 剔除唤醒词后仍含唤醒词残留的 11 条样本。
输出：音频名称、标签、seaco 识别结果、v0.5.0 识别结果。
"""
import json
import os

BASE = r"E:\8.音频\03 s10\eval_output\testset_results"

WAKEWORDS = [
    "你好萤萤石", "营养萤石", "你好萤石", "你好伊思", "你好游石", "你好萤乐", "你好萤哲", "你好李石",
    "你好刘石", "控好萤石", "控楼萤石", "平好萤石", "你要萤石", "你很萤石", "怎样萤石", "你奥伊石",
    "杨萤石", "样萤石", "萤萤石", "好萤石", "你好萤", "你好石", "萤石",
]
WAKEWORDS.sort(key=len, reverse=True)


def strip_wakewords(text, max_rounds=3):
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


def load_details(dir_name):
    path = os.path.join(BASE, dir_name, "eval_detail.jsonl")
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            results.append(json.loads(line))
    return results


def main():
    v05 = load_details("v0.5.0_model.pt.ep35")
    seaco = load_details("base_model.pt.seaco")
    
    print(f"找到 v0.5.0 样本数: {len(v05)}")
    print(f"找到 seaco 样本数: {len(seaco)}")
    
    # 建立 seaco 的 uid->item 映射
    seaco_map = {}
    for item in seaco:
        uid = item.get("uid", "")
        seaco_map[uid] = item
    
    # 找出 v0.5.0 剔除唤醒词后仍含唤醒词的样本
    residual_cases = []
    for item in v05:
        ref = item.get("ref_norm", "")
        hyp = item.get("hyp_norm", "")
        ref_s = strip_wakewords(ref)
        hyp_s = strip_wakewords(hyp)
        
        if not ref_s:
            continue
        
        # 检查 hyp_s 中是否包含唤醒词（且 ref_s 中不包含）
        for ww in WAKEWORDS:
            if ww in hyp_s and ww not in ref_s:
                uid = item.get("uid", "")
                seaco_item = seaco_map.get(uid, {})
                residual_cases.append({
                    "uid": uid,
                    "ref": ref,
                    "ref_stripped": ref_s,
                    "v05_hyp": hyp,
                    "v05_hyp_stripped": hyp_s,
                    "seaco_hyp": seaco_item.get("hyp_norm", "N/A"),
                    "wakeword": ww,
                })
                break
    
    print(f"\n唤醒词残留样本数: {len(residual_cases)}")
    print("=" * 100)
    
    for i, case in enumerate(residual_cases, 1):
        print(f"\n[{i}] 音频: {case['uid']}")
        print(f"    残留唤醒词: {case['wakeword']}")
        print(f"    标签(ref):     {case['ref']}")
        print(f"    seaco识别:     {case['seaco_hyp']}")
        print(f"    v0.5.0识别:    {case['v05_hyp']}")


if __name__ == "__main__":
    main()
