"""
提取各版本最佳模型在所有 test 集上的 CER，生成完整对比表。
"""
import json
import os

BASE = r"E:\8.音频\03 s10\eval_output"

# 定义每个版本的最佳模型
BEST_MODELS = {
    "v0.1.0": "model.pt.ep16.4000",
    "v0.2.0": "model.pt.best",
    "v0.3.0": "model.pt.best",
    "v0.4.0": "model.pt.best",
    "v0.5.0": "model.pt.ep23",
    "v0.6.0": "model.pt.ep33",
}

# 19个test数据集的key
TEST_DATASETS = {
    # 3个业务数据集
    "s10_reflow": "reflow_label/s10_reflow/test",
    "sp4_reflow": "reflow_label/sp4_reflow/test",
    "s10_augment": "reflow_label/s10_augment/test",
    # 8个公开数据集
    "aidatatang_200zh (speech)": "speech_label/aidatatang_200zh/test",
    "data_aishell (speech)": "speech_label/data_aishell/test",
    "data_aishell_1_ios (speech)": "speech_label/data_aishell_1_ios/test",
    "magicdata (speech)": "speech_label/magicdata/test",
    "primewords (speech)": "speech_label/primewords/test",
    "stcmds (speech)": "speech_label/stcmds/test",
    "tal_asr (speech)": "speech_label/tal_asr/test",
    "thchs30 (speech)": "speech_label/thchs30/test",
    # 8个增强数据集
    "aidatatang_200zh (augment)": "augment_label/aidatatang_200zh/test",
    "data_aishell (augment)": "augment_label/data_aishell/test",
    "data_aishell_1_ios (augment)": "augment_label/data_aishell_1_ios/test",
    "magicdata (augment)": "augment_label/magicdata/test",
    "primewords (augment)": "augment_label/primewords/test",
    "stcmds (augment)": "augment_label/stcmds/test",
    "tal_asr (augment)": "augment_label/tal_asr/test",
    "thchs30 (augment)": "augment_label/thchs30/test",
}

# 读取 base seaco 和 online 的数据
def load_base_data():
    """从 base 目录读取 seaco 和 online 的 test CER"""
    results = {"seaco": {}, "online": {}}
    
    for label_type in ["reflow", "speech", "augment"]:
        for model_type in ["seaco", "online_model"]:
            path = os.path.join(BASE, "base", f"{label_type}_eval_result", model_type, "combined_summary.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                key = "seaco" if model_type == "seaco" else "online"
                for dir_name, dir_data in data.get("per_directory", {}).items():
                    results[key][f"{label_type}_label/{dir_name}"] = dir_data.get("overall_cer", None)
    
    return results

# 读取各版本最佳模型数据
def load_version_data():
    """从各版本目录读取最佳模型的 test CER"""
    results = {}
    
    for version, model_name in BEST_MODELS.items():
        # 尝试从 combined_summary.json 读取
        path = os.path.join(BASE, version, model_name, "combined_summary.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results[version] = {}
            for dir_name, dir_data in data.get("per_directory", {}).items():
                results[version][dir_name] = dir_data.get("overall_cer", None)
        else:
            # 尝试从 model_comparison_summary.json 读取
            path2 = os.path.join(BASE, version, "model_comparison_summary.json")
            if os.path.exists(path2):
                with open(path2, "r", encoding="utf-8") as f:
                    data = json.load(f)
                model_key = model_name
                models = data.get("models", {})
                if model_key in models:
                    results[version] = {}
                    for dir_name, dir_data in models[model_key].get("per_dataset", {}).items():
                        results[version][dir_name] = dir_data.get("overall_cer", None)
    
    return results

# 读取全量业务数据
def load_fullset_data():
    """从 s10_results 和 sp4_results 读取全量业务数据 CER"""
    results = {}
    
    # seaco 全量 - 从 base/reflow_eval_result 读取 (train+dev+test 合计)
    seaco_path = os.path.join(BASE, "base", "reflow_eval_result", "seaco", "combined_summary.json")
    online_path = os.path.join(BASE, "base", "reflow_eval_result", "online_model", "combined_summary.json")
    
    if os.path.exists(seaco_path):
        with open(seaco_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        per_dir = data.get("per_directory", {})
        # S10 全量 = s10_reflow 的 train+dev+test 合计
        s10_chars = sum(per_dir[k]["total_ref_chars"] for k in per_dir if k.startswith("s10_reflow/"))
        s10_errors = sum(per_dir[k]["substitutions"] + per_dir[k]["insertions"] + per_dir[k]["deletions"] for k in per_dir if k.startswith("s10_reflow/"))
        sp4_chars = sum(per_dir[k]["total_ref_chars"] for k in per_dir if k.startswith("sp4_reflow/"))
        sp4_errors = sum(per_dir[k]["substitutions"] + per_dir[k]["insertions"] + per_dir[k]["deletions"] for k in per_dir if k.startswith("sp4_reflow/"))
        results["seaco_s10"] = s10_errors / s10_chars if s10_chars > 0 else None
        results["seaco_sp4"] = sp4_errors / sp4_chars if sp4_chars > 0 else None
    
    if os.path.exists(online_path):
        with open(online_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        per_dir = data.get("per_directory", {})
        s10_chars = sum(per_dir[k]["total_ref_chars"] for k in per_dir if k.startswith("s10_reflow/"))
        s10_errors = sum(per_dir[k]["substitutions"] + per_dir[k]["insertions"] + per_dir[k]["deletions"] for k in per_dir if k.startswith("s10_reflow/"))
        sp4_chars = sum(per_dir[k]["total_ref_chars"] for k in per_dir if k.startswith("sp4_reflow/"))
        sp4_errors = sum(per_dir[k]["substitutions"] + per_dir[k]["insertions"] + per_dir[k]["deletions"] for k in per_dir if k.startswith("sp4_reflow/"))
        results["online_s10"] = s10_errors / s10_chars if s10_chars > 0 else None
        results["online_sp4"] = sp4_errors / sp4_chars if sp4_chars > 0 else None
    
    # 各版本全量数据从 s10_results / sp4_results 读取
    for scene in ["s10", "sp4"]:
        summary_path = os.path.join(BASE, f"{scene}_results", "comparison_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for model_name, model_data in data.get("models", {}).items():
                results[f"{model_name}_{scene}"] = model_data.get("overall_cer", None)
    
    return results


def main():
    base_data = load_base_data()
    version_data = load_version_data()
    fullset_data = load_fullset_data()
    
    # ===== 全量业务数据对比 =====
    print("=" * 80)
    print("【S10 全量业务数据 CER 对比】")
    print("=" * 80)
    print(f"{'模型':<30} {'CER ↓':<10} {'SER ↓':<10} {'句准确率 ↑':<12}")
    print("-" * 62)
    
    # seaco 和 online
    if fullset_data.get("seaco_s10"):
        print(f"{'seaco 原始':<28} {fullset_data['seaco_s10']*100:.2f}%")
    if fullset_data.get("online_s10"):
        print(f"{'online (v0.0.0)':<28} {fullset_data['online_s10']*100:.2f}%")
    
    # 从 s10_results 读取各版本
    s10_summary_path = os.path.join(BASE, "s10_results", "comparison_summary.json")
    with open(s10_summary_path, "r", encoding="utf-8") as f:
        s10_data = json.load(f)
    
    # 每个版本选最佳
    version_best_s10 = {}
    for model_name, model_data in s10_data.get("models", {}).items():
        # 提取版本号
        parts = model_name.split("/")
        version = parts[0]
        cer = model_data["overall_cer"]
        if version not in version_best_s10 or cer < version_best_s10[version][1]:
            version_best_s10[version] = (model_name, cer, model_data["ser"], model_data["sentence_accuracy"])
    
    for version in sorted(version_best_s10.keys()):
        name, cer, ser, acc = version_best_s10[version]
        print(f"{name:<28} {cer*100:.2f}%     {ser*100:.2f}%     {acc*100:.2f}%")
    
    # SP4
    print()
    print("=" * 80)
    print("【SP4 全量业务数据 CER 对比】")
    print("=" * 80)
    if fullset_data.get("seaco_sp4"):
        print(f"{'seaco 原始':<28} {fullset_data['seaco_sp4']*100:.2f}%")
    
    sp4_summary_path = os.path.join(BASE, "sp4_results", "comparison_summary.json")
    with open(sp4_summary_path, "r", encoding="utf-8") as f:
        sp4_data = json.load(f)
    
    version_best_sp4 = {}
    for model_name, model_data in sp4_data.get("models", {}).items():
        parts = model_name.split("/")
        version = parts[0]
        cer = model_data["overall_cer"]
        if version not in version_best_sp4 or cer < version_best_sp4[version][1]:
            version_best_sp4[version] = (model_name, cer, model_data["ser"], model_data["sentence_accuracy"])
    
    for version in sorted(version_best_sp4.keys()):
        name, cer, ser, acc = version_best_sp4[version]
        print(f"{name:<28} {cer*100:.2f}%     {ser*100:.2f}%     {acc*100:.2f}%")
    
    # ===== 19个数据集 test CER 对比 =====
    print()
    print("=" * 80)
    print("【19个数据集 test 集 CER 对比（每版本最佳模型）】")
    print("=" * 80)
    
    # 表头
    model_cols = ["seaco", "online", "v0.1.0", "v0.2.0", "v0.3.0", "v0.4.0", "v0.5.0", "v0.6.0"]
    header = f"{'数据集':<30}" + "".join(f"{m:<10}" for m in model_cols) + "最优"
    print(header)
    print("-" * len(header))
    
    best_count = {m: 0 for m in model_cols}
    
    for ds_name, ds_key in TEST_DATASETS.items():
        row = f"{ds_name:<28}"
        values = {}
        
        # seaco
        cer = base_data["seaco"].get(f"reflow_label/{ds_key.split('/')[-2]}/{ds_key.split('/')[-1]}" if "reflow" in ds_key else ds_key.replace("speech_label/", "speech_label/").replace("augment_label/", "augment_label/"))
        # 需要正确映射 key
        seaco_key = ds_key
        # base 数据的 key 格式是 "s10_reflow/test" 而不是 "reflow_label/s10_reflow/test"
        if "reflow_label/" in ds_key:
            seaco_key_base = ds_key.replace("reflow_label/", "")
            cer_seaco = base_data["seaco"].get(f"reflow_label/{seaco_key_base}", None)
            cer_online = base_data["online"].get(f"reflow_label/{seaco_key_base}", None)
        elif "speech_label/" in ds_key:
            seaco_key_base = ds_key.replace("speech_label/", "")
            cer_seaco = base_data["seaco"].get(f"speech_label/{seaco_key_base}", None)
            cer_online = base_data["online"].get(f"speech_label/{seaco_key_base}", None)
        else:
            seaco_key_base = ds_key.replace("augment_label/", "")
            cer_seaco = base_data["seaco"].get(f"augment_label/{seaco_key_base}", None)
            cer_online = base_data["online"].get(f"augment_label/{seaco_key_base}", None)
        
        values["seaco"] = cer_seaco
        values["online"] = cer_online
        
        # 各版本
        for version in ["v0.1.0", "v0.2.0", "v0.3.0", "v0.4.0", "v0.5.0", "v0.6.0"]:
            if version in version_data:
                values[version] = version_data[version].get(ds_key, None)
            else:
                values[version] = None
        
        # 格式化输出
        for m in model_cols:
            v = values.get(m)
            if v is not None:
                row += f"{v*100:.2f}%   "
            else:
                row += f"{'—':<10}"
        
        # 找最优
        valid = {k: v for k, v in values.items() if v is not None}
        if valid:
            best_model = min(valid, key=valid.get)
            best_count[best_model] += 1
            row += f" {best_model}"
        
        print(row)
    
    print("-" * len(header))
    print(f"{'最优数量':<28}", end="")
    for m in model_cols:
        print(f"{best_count[m]:<10}", end="")
    print()


if __name__ == "__main__":
    main()
