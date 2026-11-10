import json, os

BASE = r"E:\8.音频\03 s10\eval_output"

BEST_MODELS = {
    "v0.1.0": "model.pt.ep16.4000",
    "v0.2.0": "model.pt.best",
    "v0.3.0": "model.pt.best",
    "v0.4.0": "model.pt.best",
    "v0.5.0": "model.pt.ep23",
    "v0.6.0": "model.pt.ep33",
}

print("=== 各版本 combined_summary.json ===")
for v, m in BEST_MODELS.items():
    path = os.path.join(BASE, v, m, "combined_summary.json")
    exists = os.path.exists(path)
    print(f"  {v}/{m}: {'OK' if exists else 'MISSING'}")

print("\n=== base 基线数据 ===")
for label in ["reflow", "speech", "augment"]:
    for model in ["seaco", "online_model"]:
        path = os.path.join(BASE, "base", f"{label}_eval_result", model, "combined_summary.json")
        exists = os.path.exists(path)
        print(f"  base/{label}_eval_result/{model}: {'OK' if exists else 'MISSING'}")

print("\n=== 全量业务数据 ===")
for scene in ["s10_results", "sp4_results", "testset_results"]:
    path = os.path.join(BASE, scene, "comparison_summary.json")
    exists = os.path.exists(path)
    print(f"  {scene}: {'OK' if exists else 'MISSING'}")

# 检查 v0.1.0 和 v0.2.0 是否有 model_comparison_summary.json
print("\n=== model_comparison_summary.json (备选) ===")
for v in BEST_MODELS:
    path = os.path.join(BASE, v, "model_comparison_summary.json")
    exists = os.path.exists(path)
    print(f"  {v}: {'OK' if exists else 'MISSING'}")
