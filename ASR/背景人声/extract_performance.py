"""
提取各版本最佳模型在所有 test 集上的 CER，生成完整对比表。
严格基于实际数据，缺失数据标注为 N/A。
"""
import json
import os

BASE = r"E:\8.音频\03 s10\eval_output"

# 各版本最佳模型
BEST_MODELS = {
    "v0.1.0": "model.pt.ep16.4000",
    "v0.2.0": "model.pt.best",
    "v0.3.0": "model.pt.best",
    "v0.4.0": "model.pt.best",
    "v0.5.0": "model.pt.ep35",
    "v0.6.0": "model.pt.ep33",
}

# 19个test数据集
TEST_DATASETS = [
    # 业务数据
    ("s10_reflow", "reflow_label/s10_reflow/test"),
    ("sp4_reflow", "reflow_label/sp4_reflow/test"),
    ("s10_augment", "reflow_label/s10_augment/test"),
    # 公开数据集
    ("aidatatang_200zh (speech)", "speech_label/aidatatang_200zh/test"),
    ("data_aishell (speech)", "speech_label/data_aishell/test"),
    ("data_aishell_1_ios (speech)", "speech_label/data_aishell_1_ios/test"),
    ("magicdata (speech)", "speech_label/magicdata/test"),
    ("primewords (speech)", "speech_label/primewords/test"),
    ("stcmds (speech)", "speech_label/stcmds/test"),
    ("tal_asr (speech)", "speech_label/tal_asr/test"),
    ("thchs30 (speech)", "speech_label/thchs30/test"),
    # 增强数据集
    ("aidatatang_200zh (augment)", "augment_label/aidatatang_200zh/test"),
    ("data_aishell (augment)", "augment_label/data_aishell/test"),
    ("data_aishell_1_ios (augment)", "augment_label/data_aishell_1_ios/test"),
    ("magicdata (augment)", "augment_label/magicdata/test"),
    ("primewords (augment)", "augment_label/primewords/test"),
    ("stcmds (augment)", "augment_label/stcmds/test"),
    ("tal_asr (augment)", "augment_label/tal_asr/test"),
    ("thchs30 (augment)", "augment_label/thchs30/test"),
]

MODEL_COLS = ["seaco", "online", "v0.1.0", "v0.2.0", "v0.3.0", "v0.4.0", "v0.5.0", "v0.6.0"]


def load_base_data():
    """从 base 目录读取 seaco 和 online 的数据"""
    results = {"seaco": {}, "online": {}}
    
    for label_type in ["reflow", "speech", "augment"]:
        for model_type in ["seaco", "online"]:
            path = os.path.join(BASE, "base", f"{label_type}_eval_result", model_type, "combined_summary.json")
            if not os.path.exists(path):
                print(f"  [缺失] {path}")
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            key = model_type  # "seaco" or "online"
            for dir_name, dir_data in data.get("per_directory", {}).items():
                full_key = f"{label_type}_label/{dir_name}"
                results[key][full_key] = dir_data.get("overall_cer", None)
    
    return results


def load_version_data():
    """从各版本目录读取最佳模型的 test CER"""
    results = {}
    
    for version, model_name in BEST_MODELS.items():
        path = os.path.join(BASE, version, model_name, "combined_summary.json")
        if not os.path.exists(path):
            print(f"  [缺失] {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results[version] = {}
        for dir_name, dir_data in data.get("per_directory", {}).items():
            results[version][dir_name] = dir_data.get("overall_cer", None)
    
    return results


def load_fullset_data():
    """从 base/reflow 和 s10_results/sp4_results 读取全量业务数据"""
    results = {}
    
    # seaco 和 online 全量
    for model_type in ["seaco", "online"]:
        path = os.path.join(BASE, "base", "reflow_eval_result", model_type, "combined_summary.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        key_prefix = "seaco" if model_type == "seaco" else "online"
        per_dir = data.get("per_directory", {})        
        for scene in ["s10_reflow", "sp4_reflow"]:
            chars = 0
            errors = 0
            for k, v in per_dir.items():
                if k.startswith(f"{scene}/"):
                    chars += v["total_ref_chars"]
                    errors += v["substitutions"] + v["insertions"] + v["deletions"]
            if chars > 0:
                results[f"{key_prefix}_{scene}"] = errors / chars
    
    # 各版本全量
    for scene in ["s10", "sp4"]:
        path = os.path.join(BASE, f"{scene}_results", "comparison_summary.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for model_name, model_data in data.get("models", {}).items():
            results[f"{model_name}_{scene}"] = {
                "cer": model_data.get("overall_cer"),
                "ser": model_data.get("ser"),
                "acc": model_data.get("sentence_accuracy"),
            }
    
    return results


def fmt_cer(val):
    if val is None:
        return "N/A"
    return f"{val*100:.2f}%"


def main():
    print("检查数据文件...")
    base_data = load_base_data()
    version_data = load_version_data()
    fullset_data = load_fullset_data()
    
    # ===== S10 全量 =====
    print("\n" + "=" * 80)
    print("【S10 全量业务数据 CER 对比】")
    print("=" * 80)
    print(f"{'模型':<30} {'CER ↓':<12} {'SER ↓':<12} {'句准确率 ↑':<12}")
    print("-" * 66)
    
    # seaco
    val = fullset_data.get("seaco_s10_reflow")
    # 尝试从 comparison_summary 获取完整数据
    s10_path = os.path.join(BASE, "s10_results", "comparison_summary.json")
    if os.path.exists(s10_path):
        with open(s10_path, "r", encoding="utf-8") as f:
            s10_data = json.load(f)
    else:
        s10_data = {"models": {}}
    
    # seaco 和 online 从 comparison_summary 读取（含 SER）
    for key_name, label in [("base/model.pt.seaco", "seaco 原始"), ("base/model.pt.online", "online")]:
        if key_name in s10_data.get("models", {}):
            d = s10_data["models"][key_name]
            print(f"{label:<28} {fmt_cer(d['overall_cer']):<12} {fmt_cer(d['ser']):<12} {fmt_cer(d['sentence_accuracy']):<12}")
        else:
            val2 = fullset_data.get(f"{'seaco' if 'seaco' in key_name else 'online'}_s10_reflow")
            print(f"{label:<28} {fmt_cer(val2):<12} {'N/A':<12} {'N/A':<12}")
    
    # 各版本从 s10_results
    s10_path = os.path.join(BASE, "s10_results", "comparison_summary.json")
    if os.path.exists(s10_path):
        with open(s10_path, "r", encoding="utf-8") as f:
            s10_data = json.load(f)
        # 每版本选最佳
        version_best = {}
        for model_name, model_data in s10_data.get("models", {}).items():
            parts = model_name.split("/")
            version = parts[0]
            cer = model_data["overall_cer"]
            if version not in version_best or cer < version_best[version]["cer"]:
                version_best[version] = {
                    "name": model_name,
                    "cer": cer,
                    "ser": model_data["ser"],
                    "acc": model_data["sentence_accuracy"],
                }
        for version in sorted(version_best.keys()):
            if version in ("base", "v0.0.0"):
                continue
            d = version_best[version]
            print(f"{d['name']:<28} {fmt_cer(d['cer']):<12} {fmt_cer(d['ser']):<12} {fmt_cer(d['acc']):<12}")
    
    # ===== SP4 全量 =====
    print("\n" + "=" * 80)
    print("【SP4 全量业务数据 CER 对比】")
    print("=" * 80)
    print(f"{'模型':<30} {'CER ↓':<12} {'SER ↓':<12} {'句准确率 ↑':<12}")
    print("-" * 66)
    
    val = fullset_data.get("seaco_sp4_reflow")
    
    sp4_path = os.path.join(BASE, "sp4_results", "comparison_summary.json")
    if os.path.exists(sp4_path):
        with open(sp4_path, "r", encoding="utf-8") as f:
            sp4_data = json.load(f)
    else:
        sp4_data = {"models": {}}
    
    for key_name, label in [("base/model.pt.seaco", "seaco 原始"), ("base/model.pt.online", "online")]:
        if key_name in sp4_data.get("models", {}):
            d = sp4_data["models"][key_name]
            print(f"{label:<28} {fmt_cer(d['overall_cer']):<12} {fmt_cer(d['ser']):<12} {fmt_cer(d['sentence_accuracy']):<12}")
        else:
            val2 = fullset_data.get(f"{'seaco' if 'seaco' in key_name else 'online'}_sp4_reflow")
            print(f"{label:<28} {fmt_cer(val2):<12} {'N/A':<12} {'N/A':<12}")
    
    if os.path.exists(sp4_path):
        version_best = {}
        for model_name, model_data in sp4_data.get("models", {}).items():
            parts = model_name.split("/")
            version = parts[0]
            cer = model_data["overall_cer"]
            if version not in version_best or cer < version_best[version]["cer"]:
                version_best[version] = {
                    "name": model_name,
                    "cer": cer,
                    "ser": model_data["ser"],
                    "acc": model_data["sentence_accuracy"],
                }
        for version in sorted(version_best.keys()):
            if version in ("base", "v0.0.0"):
                continue
            d = version_best[version]
            print(f"{d['name']:<28} {fmt_cer(d['cer']):<12} {fmt_cer(d['ser']):<12} {fmt_cer(d['acc']):<12}")
    
    # ===== 19 数据集 test CER =====
    print("\n" + "=" * 80)
    print("【19 数据集 test 集 CER 对比（每版本最佳模型）】")
    print("=" * 80)
    
    # 表头
    header = f"{'数据集':<30}" + "".join(f"{m:<10}" for m in MODEL_COLS) + "最优"
    print(header)
    print("-" * (30 + 10 * len(MODEL_COLS) + 10))
    
    best_count = {m: 0 for m in MODEL_COLS}
    missing_data = []
    
    for ds_name, ds_key in TEST_DATASETS:
        values = {}
        
        # seaco
        values["seaco"] = base_data["seaco"].get(ds_key, None)
        # online
        values["online"] = base_data["online"].get(ds_key, None)
        
        # 各版本
        for version in ["v0.1.0", "v0.2.0", "v0.3.0", "v0.4.0", "v0.5.0", "v0.6.0"]:
            if version in version_data:
                values[version] = version_data[version].get(ds_key, None)
            else:
                values[version] = None
        
        # 记录缺失
        for m in MODEL_COLS:
            if values.get(m) is None:
                missing_data.append(f"{ds_name} -> {m}")
        
        # 格式化
        row = f"{ds_name:<28}"
        for m in MODEL_COLS:
            v = values.get(m)
            if v is not None:
                row += f"{v*100:.2f}%   "
            else:
                row += f"{'N/A':<10}"
        
        # 最优
        valid = {k: v for k, v in values.items() if v is not None}
        if valid:
            best_model = min(valid, key=valid.get)
            best_count[best_model] += 1
            row += f" {best_model}"
        else:
            row += " N/A"
        
        print(row)
    
    # 统计
    print("-" * (30 + 10 * len(MODEL_COLS) + 10))
    row = f"{'最优数量':<28}"
    for m in MODEL_COLS:
        row += f"{best_count[m]:<10}"
    print(row)
    
    # 缺失数据报告
    if missing_data:
        print(f"\n{'=' * 80}")
        print(f"【缺失数据报告】共 {len(missing_data)} 项")
        print("=" * 80)
        for item in missing_data:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
