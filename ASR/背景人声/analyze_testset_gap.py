"""
分析 testset 上 seaco vs v0.5.0/ep35 剔除唤醒词后 CER 差异的原因。
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


def edit_distance(ref, hyp):
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
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


def analyze_model(dir_name):
    path = os.path.join(BASE, dir_name, "eval_detail.jsonl")
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            ref = item.get("ref_norm", "")
            hyp = item.get("hyp_norm", "")
            ref_s = strip_wakewords(ref)
            hyp_s = strip_wakewords(hyp)
            if not ref_s:
                continue
            ed = edit_distance(ref_s, hyp_s)
            cer = ed / len(ref_s) if len(ref_s) > 0 else 0
            # 检查是否有残留唤醒词片段
            has_extra_in_hyp = len(hyp_s) > len(ref_s) * 1.3  # hyp 比 ref 长 30% 以上
            results.append({
                "ref": ref,
                "hyp": hyp,
                "ref_s": ref_s,
                "hyp_s": hyp_s,
                "ed": ed,
                "cer": cer,
                "ref_len": len(ref_s),
                "hyp_len": len(hyp_s),
                "has_extra": has_extra_in_hyp,
            })
    return results


def main():
    print("加载 seaco 数据...")
    seaco = analyze_model("base_model.pt.seaco")
    print("加载 v0.5.0/ep35 数据...")
    v05 = analyze_model("v0.5.0_model.pt.ep35")
    
    # 总体统计
    seaco_total_ref = sum(r["ref_len"] for r in seaco)
    seaco_total_ed = sum(r["ed"] for r in seaco)
    v05_total_ref = sum(r["ref_len"] for r in v05)
    v05_total_ed = sum(r["ed"] for r in v05)
    
    print(f"\n{'='*60}")
    print(f"总体对比（剔除唤醒词后）")
    print(f"{'='*60}")
    print(f"  seaco:  CER={seaco_total_ed/seaco_total_ref*100:.2f}%  总ref字符={seaco_total_ref}  总编辑距离={seaco_total_ed}")
    print(f"  v0.5.0: CER={v05_total_ed/v05_total_ref*100:.2f}%  总ref字符={v05_total_ref}  总编辑距离={v05_total_ed}")
    print(f"  差距:   {(v05_total_ed/v05_total_ref - seaco_total_ed/seaco_total_ref)*100:.2f}%")
    
    # 分析 v0.5.0 比 seaco 差的样本
    print(f"\n{'='*60}")
    print(f"逐条对比分析")
    print(f"{'='*60}")
    
    # 按 index 对齐
    n = min(len(seaco), len(v05))
    v05_worse = 0
    v05_better = 0
    v05_same = 0
    
    worse_cases = []
    
    for i in range(n):
        s = seaco[i]
        v = v05[i]
        if v["ed"] > s["ed"]:
            v05_worse += 1
            if v["ed"] - s["ed"] >= 2:  # 差距>=2的case
                worse_cases.append((i, s, v))
        elif v["ed"] < s["ed"]:
            v05_better += 1
        else:
            v05_same += 1
    
    print(f"  v0.5.0 更差: {v05_worse} 条 ({v05_worse/n*100:.1f}%)")
    print(f"  v0.5.0 更好: {v05_better} 条 ({v05_better/n*100:.1f}%)")
    print(f"  相同:        {v05_same} 条 ({v05_same/n*100:.1f}%)")
    
    # 分析 v0.5.0 更差的原因
    print(f"\n{'='*60}")
    print(f"v0.5.0 更差的典型 case（差距>=2，前20条）")
    print(f"{'='*60}")
    
    worse_cases.sort(key=lambda x: x[2]["ed"] - x[1]["ed"], reverse=True)
    
    # 统计原因
    reason_counts = {"hyp过长(插入)": 0, "替换错误": 0, "hyp过短(删除)": 0}
    
    for idx, (i, s, v) in enumerate(worse_cases[:20]):
        diff = v["ed"] - s["ed"]
        len_diff = v["hyp_len"] - v["ref_len"]
        s_len_diff = s["hyp_len"] - s["ref_len"]
        
        if v["hyp_len"] > v["ref_len"] + 2:
            reason = "插入过多"
        elif v["hyp_len"] < v["ref_len"] - 2:
            reason = "删除过多"
        else:
            reason = "替换错误"
        
        print(f"\n  [{idx+1}] 差距={diff}, 原因={reason}")
        print(f"    ref_stripped: {v['ref_s'][:60]}")
        print(f"    seaco_hyp_s:  {s['hyp_s'][:60]}  (ed={s['ed']})")
        print(f"    v05_hyp_s:    {v['hyp_s'][:60]}  (ed={v['ed']})")
        if v["hyp_s"] != s["hyp_s"]:
            # 检查 v0.5 的 hyp 是否包含唤醒词残留
            for ww in WAKEWORDS:
                if ww in v["hyp_s"] and ww not in v["ref_s"]:
                    print(f"    ⚠️ v0.5 hyp 中包含唤醒词残留: '{ww}'")
                    break
    
    # 统计唤醒词残留
    print(f"\n{'='*60}")
    print(f"唤醒词残留分析")
    print(f"{'='*60}")
    
    seaco_ww_residual = 0
    v05_ww_residual = 0
    for r in seaco:
        for ww in WAKEWORDS:
            if ww in r["hyp_s"] and ww not in r["ref_s"]:
                seaco_ww_residual += 1
                break
    for r in v05:
        for ww in WAKEWORDS:
            if ww in r["hyp_s"] and ww not in r["ref_s"]:
                v05_ww_residual += 1
                break
    
    print(f"  seaco 剔除后仍含唤醒词: {seaco_ww_residual} 条")
    print(f"  v0.5.0 剔除后仍含唤醒词: {v05_ww_residual} 条")
    
    # 统计 hyp 长度差异
    print(f"\n{'='*60}")
    print(f"输出长度分析（hyp_len - ref_len）")
    print(f"{'='*60}")
    
    seaco_len_diffs = [r["hyp_len"] - r["ref_len"] for r in seaco]
    v05_len_diffs = [r["hyp_len"] - r["ref_len"] for r in v05]
    
    print(f"  seaco 平均多出字符: {sum(seaco_len_diffs)/len(seaco_len_diffs):.2f}")
    print(f"  v0.5.0 平均多出字符: {sum(v05_len_diffs)/len(v05_len_diffs):.2f}")
    
    seaco_insert_heavy = sum(1 for d in seaco_len_diffs if d > 3)
    v05_insert_heavy = sum(1 for d in v05_len_diffs if d > 3)
    print(f"  seaco 多出>3字符的样本: {seaco_insert_heavy} ({seaco_insert_heavy/len(seaco)*100:.1f}%)")
    print(f"  v0.5.0 多出>3字符的样本: {v05_insert_heavy} ({v05_insert_heavy/len(v05)*100:.1f}%)")


if __name__ == "__main__":
    main()
