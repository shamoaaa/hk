"""
分析 ChildMandarin 上 v0.5.0 退化的具体错误模式/场景分类。
通过查看 v0.5.0 比 seaco 差的样本，归纳退化场景。
"""
import json
import os
import re
from collections import defaultdict

BASE = r"E:\8.音频\03 s10\eval_output\child\ChildMandarin"

WAKEWORDS = [
    "你好萤萤石", "营养萤石", "你好萤石", "你好伊思", "你好游石", "你好萤乐", "你好萤哲", "你好李石",
    "你好刘石", "控好萤石", "控楼萤石", "平好萤石", "你要萤石", "你很萤石", "怎样萤石", "你奥伊石",
    "杨萤石", "样萤石", "萤萤石", "好萤石", "你好萤", "你好石", "萤石",
]


def load_details(dir_name):
    path = os.path.join(BASE, dir_name, "eval_detail.jsonl")
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            results.append(json.loads(line))
    return results


def classify_error(ref, seaco_hyp, v05_hyp, s_errors, v_errors):
    """
    根据 ref、seaco_hyp、v05_hyp 的特征，分类退化场景。
    返回场景标签列表。
    """
    scenes = []
    ref_len = len(ref)
    v05_len = len(v05_hyp)
    
    # 1. 截断/大量删除：v0.5 输出长度远小于 ref
    if ref_len > 0 and v05_len < ref_len * 0.6 and ref_len >= 10:
        scenes.append("长句截断")
    
    # 2. 唤醒词插入
    for ww in WAKEWORDS:
        if ww in v05_hyp and ww not in ref:
            scenes.append("唤醒词插入")
            break
    
    # 3. 重复/结巴内容（ref 中有重复字词）
    # 检测连续重复字符
    has_repeat_in_ref = bool(re.search(r'(.)\1{2,}', ref)) or bool(re.search(r'(.{2,})\1{2,}', ref))
    if has_repeat_in_ref:
        scenes.append("重复/结巴语音")
    
    # 4. 拟声词/语气词为主
    onomatopoeia = set("啊哦嗯呃哈嘻呀哇噢唔嘿呜咦吧呢吗啦哎呐嗨喔")
    ref_clean = ref.replace(" ", "")
    if ref_clean and sum(1 for c in ref_clean if c in onomatopoeia) / len(ref_clean) > 0.4:
        scenes.append("语气词/拟声词")
    
    # 5. 英文/数字混合
    if re.search(r'[a-zA-Z]{2,}', ref):
        scenes.append("含英文")
    
    # 6. v0.5 输出有大量重复（模型幻觉）
    if v05_len > ref_len * 1.3 and ref_len >= 5:
        # 检查是否有重复片段
        for seg_len in range(2, 5):
            for start in range(len(v05_hyp) - seg_len * 2):
                seg = v05_hyp[start:start+seg_len]
                if v05_hyp.count(seg) >= 4 and ref.count(seg) < 2:
                    scenes.append("模型重复输出")
                    break
            else:
                continue
            break
    
    # 7. 口语化/非标准表达（含"那个"、"就是"、"然后"等口头禅）
    filler_words = ["那个", "就是", "然后", "嗯", "呃", "这个"]
    filler_count = sum(ref.count(w) for w in filler_words)
    if filler_count >= 3:
        scenes.append("口语化/多填充词")
    
    # 8. 如果没有匹配到特定场景，归为"一般替换错误"
    if not scenes:
        if v_errors - s_errors >= 3:
            # 看是替换为主还是删除为主
            scenes.append("一般识别退化")
    
    return scenes if scenes else ["一般识别退化"]


def main():
    print("加载数据...")
    seaco = load_details("base_model.pt.seaco")
    v05 = load_details("v0.5.0_model.pt.ep35")
    
    n = min(len(seaco), len(v05))
    
    # 只分析 v0.5.0 比 seaco 差的样本
    worse_cases = []
    for i in range(n):
        s = seaco[i]
        v = v05[i]
        s_errors = s["sub"] + s["ins"] + s["del"]
        v_errors = v["sub"] + v["ins"] + v["del"]
        if v_errors > s_errors:
            worse_cases.append((i, s, v, s_errors, v_errors))
    
    print(f"v0.5.0 退化样本数: {len(worse_cases)}")
    
    # 分类统计
    scene_stats = defaultdict(lambda: {"count": 0, "extra_errors": 0, "examples": []})
    
    for i, s, v, s_errors, v_errors in worse_cases:
        ref = s["ref_norm"]
        seaco_hyp = s["hyp_norm"]
        v05_hyp = v["hyp_norm"]
        
        scenes = classify_error(ref, seaco_hyp, v05_hyp, s_errors, v_errors)
        extra = v_errors - s_errors
        
        for scene in scenes:
            scene_stats[scene]["count"] += 1
            scene_stats[scene]["extra_errors"] += extra
            if len(scene_stats[scene]["examples"]) < 5:
                scene_stats[scene]["examples"].append({
                    "uid": s["uid"],
                    "ref": ref,
                    "seaco": seaco_hyp,
                    "v05": v05_hyp,
                    "s_err": s_errors,
                    "v_err": v_errors,
                })
    
    # 输出
    print(f"\n{'='*80}")
    print(f"退化场景分类统计（共 {len(worse_cases)} 条退化样本）")
    print(f"{'='*80}")
    print(f"{'场景':<20} {'样本数':<10} {'占比':<10} {'多出错误总数':<14} {'平均多出':<10}")
    print("-" * 64)
    
    sorted_scenes = sorted(scene_stats.items(), key=lambda x: x[1]["extra_errors"], reverse=True)
    for scene, stats in sorted_scenes:
        pct = stats["count"] / len(worse_cases) * 100
        avg_extra = stats["extra_errors"] / stats["count"] if stats["count"] > 0 else 0
        print(f"{scene:<18} {stats['count']:<10} {pct:.1f}%     {stats['extra_errors']:<14} {avg_extra:.1f}")
    
    # 每个场景的典型 case
    for scene, stats in sorted_scenes:
        print(f"\n{'='*80}")
        print(f"【{scene}】典型 case（{stats['count']} 条，多出 {stats['extra_errors']} 错误）")
        print(f"{'='*80}")
        for idx, ex in enumerate(stats["examples"], 1):
            print(f"\n  [{idx}] {ex['uid']}")
            print(f"    ref:   {ex['ref'][:70]}")
            print(f"    seaco: {ex['seaco'][:70]}  (err={ex['s_err']})")
            print(f"    v0.5:  {ex['v05'][:70]}  (err={ex['v_err']})")


if __name__ == "__main__":
    main()
