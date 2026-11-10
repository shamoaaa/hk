"""
分析 ChildMandarin 上 seaco vs v0.5.0/ep35 的退化表现。
"""
import json
import os
from collections import defaultdict

BASE = r"E:\8.音频\03 s10\eval_output\child\ChildMandarin"


def load_details(dir_name):
    path = os.path.join(BASE, dir_name, "eval_detail.jsonl")
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            results.append(json.loads(line))
    return results


def parse_uid(uid):
    """解析 uid 获取说话人信息: ID_年龄_性别_方言_城市_设备_序号"""
    parts = uid.split("_")
    if len(parts) >= 6:
        return {
            "speaker_id": parts[0],
            "age": parts[1],
            "gender": parts[2],
            "dialect": parts[3],
            "city": parts[4],
            "device": parts[5],
        }
    return {"speaker_id": uid, "age": "?", "gender": "?", "dialect": "?", "city": "?", "device": "?"}


def main():
    print("加载数据...")
    seaco = load_details("base_model.pt.seaco")
    v05 = load_details("v0.5.0_model.pt.ep35")
    
    print(f"样本数: seaco={len(seaco)}, v0.5.0={len(v05)}")
    
    # 逐条对比
    n = min(len(seaco), len(v05))
    
    v05_worse = []
    v05_better = []
    same = 0
    
    # 按维度统计
    age_stats = defaultdict(lambda: {"seaco_errors": 0, "v05_errors": 0, "ref_chars": 0, "count": 0})
    gender_stats = defaultdict(lambda: {"seaco_errors": 0, "v05_errors": 0, "ref_chars": 0, "count": 0})
    dialect_stats = defaultdict(lambda: {"seaco_errors": 0, "v05_errors": 0, "ref_chars": 0, "count": 0})
    ref_len_stats = defaultdict(lambda: {"seaco_errors": 0, "v05_errors": 0, "ref_chars": 0, "count": 0})
    
    for i in range(n):
        s = seaco[i]
        v = v05[i]
        
        s_errors = s["sub"] + s["ins"] + s["del"]
        v_errors = v["sub"] + v["ins"] + v["del"]
        ref_len = s["ref_len"]
        
        info = parse_uid(s["uid"])
        
        # 按年龄
        age_stats[info["age"]]["seaco_errors"] += s_errors
        age_stats[info["age"]]["v05_errors"] += v_errors
        age_stats[info["age"]]["ref_chars"] += ref_len
        age_stats[info["age"]]["count"] += 1
        
        # 按性别
        gender_stats[info["gender"]]["seaco_errors"] += s_errors
        gender_stats[info["gender"]]["v05_errors"] += v_errors
        gender_stats[info["gender"]]["ref_chars"] += ref_len
        gender_stats[info["gender"]]["count"] += 1
        
        # 按方言
        dialect_stats[info["dialect"]]["seaco_errors"] += s_errors
        dialect_stats[info["dialect"]]["v05_errors"] += v_errors
        dialect_stats[info["dialect"]]["ref_chars"] += ref_len
        dialect_stats[info["dialect"]]["count"] += 1
        
        # 按句子长度分桶
        if ref_len <= 5:
            bucket = "1-5字"
        elif ref_len <= 10:
            bucket = "6-10字"
        elif ref_len <= 20:
            bucket = "11-20字"
        else:
            bucket = "20+字"
        ref_len_stats[bucket]["seaco_errors"] += s_errors
        ref_len_stats[bucket]["v05_errors"] += v_errors
        ref_len_stats[bucket]["ref_chars"] += ref_len
        ref_len_stats[bucket]["count"] += 1
        
        if v_errors > s_errors:
            v05_worse.append((i, s, v, v_errors - s_errors))
        elif v_errors < s_errors:
            v05_better.append((i, s, v, s_errors - v_errors))
        else:
            same += 1
    
    print(f"\n{'='*70}")
    print(f"逐条对比统计")
    print(f"{'='*70}")
    print(f"  v0.5.0 更差: {len(v05_worse)} 条 ({len(v05_worse)/n*100:.1f}%)")
    print(f"  v0.5.0 更好: {len(v05_better)} 条 ({len(v05_better)/n*100:.1f}%)")
    print(f"  相同:        {same} 条 ({same/n*100:.1f}%)")
    print(f"  v0.5.0 多出错误总数: {sum(x[3] for x in v05_worse)}")
    print(f"  v0.5.0 减少错误总数: {sum(x[3] for x in v05_better)}")
    
    # 按年龄分析
    print(f"\n{'='*70}")
    print(f"按年龄分析 CER")
    print(f"{'='*70}")
    print(f"{'年龄':<8} {'样本数':<8} {'seaco CER':<12} {'v0.5 CER':<12} {'差距':<10}")
    print("-" * 50)
    for age in sorted(age_stats.keys()):
        d = age_stats[age]
        s_cer = d["seaco_errors"] / d["ref_chars"] * 100 if d["ref_chars"] > 0 else 0
        v_cer = d["v05_errors"] / d["ref_chars"] * 100 if d["ref_chars"] > 0 else 0
        diff = v_cer - s_cer
        print(f"{age:<6} {d['count']:<8} {s_cer:.2f}%       {v_cer:.2f}%       {diff:+.2f}%")
    
    # 按性别分析
    print(f"\n{'='*70}")
    print(f"按性别分析 CER")
    print(f"{'='*70}")
    print(f"{'性别':<8} {'样本数':<8} {'seaco CER':<12} {'v0.5 CER':<12} {'差距':<10}")
    print("-" * 50)
    for g in sorted(gender_stats.keys()):
        d = gender_stats[g]
        s_cer = d["seaco_errors"] / d["ref_chars"] * 100 if d["ref_chars"] > 0 else 0
        v_cer = d["v05_errors"] / d["ref_chars"] * 100 if d["ref_chars"] > 0 else 0
        diff = v_cer - s_cer
        label = {"M": "男", "F": "女"}.get(g, g)
        print(f"{label}({g}){'':<2} {d['count']:<8} {s_cer:.2f}%       {v_cer:.2f}%       {diff:+.2f}%")
    
    # 按方言分析
    print(f"\n{'='*70}")
    print(f"按方言/地区分析 CER（前10）")
    print(f"{'='*70}")
    print(f"{'方言':<12} {'样本数':<8} {'seaco CER':<12} {'v0.5 CER':<12} {'差距':<10}")
    print("-" * 54)
    sorted_dialects = sorted(dialect_stats.items(), key=lambda x: (x[1]["v05_errors"]/x[1]["ref_chars"] - x[1]["seaco_errors"]/x[1]["ref_chars"]) if x[1]["ref_chars"] > 0 else 0, reverse=True)
    for dialect, d in sorted_dialects[:15]:
        s_cer = d["seaco_errors"] / d["ref_chars"] * 100 if d["ref_chars"] > 0 else 0
        v_cer = d["v05_errors"] / d["ref_chars"] * 100 if d["ref_chars"] > 0 else 0
        diff = v_cer - s_cer
        print(f"{dialect:<10} {d['count']:<8} {s_cer:.2f}%       {v_cer:.2f}%       {diff:+.2f}%")
    
    # 按句子长度分析
    print(f"\n{'='*70}")
    print(f"按句子长度分析 CER")
    print(f"{'='*70}")
    print(f"{'长度':<10} {'样本数':<8} {'seaco CER':<12} {'v0.5 CER':<12} {'差距':<10}")
    print("-" * 50)
    for bucket in ["1-5字", "6-10字", "11-20字", "20+字"]:
        d = ref_len_stats[bucket]
        s_cer = d["seaco_errors"] / d["ref_chars"] * 100 if d["ref_chars"] > 0 else 0
        v_cer = d["v05_errors"] / d["ref_chars"] * 100 if d["ref_chars"] > 0 else 0
        diff = v_cer - s_cer
        print(f"{bucket:<8} {d['count']:<8} {s_cer:.2f}%       {v_cer:.2f}%       {diff:+.2f}%")
    
    # 错误类型分析
    print(f"\n{'='*70}")
    print(f"错误类型对比（总量）")
    print(f"{'='*70}")
    seaco_sub = sum(s["sub"] for s in seaco)
    seaco_ins = sum(s["ins"] for s in seaco)
    seaco_del = sum(s["del"] for s in seaco)
    v05_sub = sum(v["sub"] for v in v05)
    v05_ins = sum(v["ins"] for v in v05)
    v05_del = sum(v["del"] for v in v05)
    print(f"  {'类型':<10} {'seaco':<10} {'v0.5.0':<10} {'差距':<10}")
    print(f"  {'替换(sub)':<8} {seaco_sub:<10} {v05_sub:<10} {v05_sub-seaco_sub:+d}")
    print(f"  {'插入(ins)':<8} {seaco_ins:<10} {v05_ins:<10} {v05_ins-seaco_ins:+d}")
    print(f"  {'删除(del)':<8} {seaco_del:<10} {v05_del:<10} {v05_del-seaco_del:+d}")
    print(f"  {'总计':<10} {seaco_sub+seaco_ins+seaco_del:<10} {v05_sub+v05_ins+v05_del:<10} {(v05_sub+v05_ins+v05_del)-(seaco_sub+seaco_ins+seaco_del):+d}")
    
    # 典型 bad case
    print(f"\n{'='*70}")
    print(f"v0.5.0 退化最严重的 case（前15条）")
    print(f"{'='*70}")
    v05_worse.sort(key=lambda x: x[3], reverse=True)
    for idx, (i, s, v, diff) in enumerate(v05_worse[:15]):
        info = parse_uid(s["uid"])
        print(f"\n  [{idx+1}] uid={s['uid']}")
        print(f"      年龄={info['age']} 性别={info['gender']} 方言={info['dialect']}")
        print(f"      ref:   {s['ref_norm'][:60]}")
        print(f"      seaco: {s['hyp_norm'][:60]}  (errors={s['sub']+s['ins']+s['del']})")
        print(f"      v0.5:  {v['hyp_norm'][:60]}  (errors={v['sub']+v['ins']+v['del']})")
        print(f"      差距: +{diff} errors")


if __name__ == "__main__":
    main()
