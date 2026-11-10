# -*- coding: utf-8 -*-
"""eval 命令 — 远程推理 + CER/召回率评估.

对齐背景人声项目 finetuned_eval.py 的推理方式：
1. 加载基座模型结构（AutoModel）
2. overlay 微调权重（torch.load + load_state_dict）
3. batch 推理
4. normalize_text + CER + stripped CER + 指令召回率
5. 输出 eval_detail.jsonl + eval_summary.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from asr_evolve.environment.remote import RemoteExecutor, RemoteConfig
from asr_evolve.pipeline.evaluation.domain_evaluator import (
    DomainEvaluator,
    normalize_text,
    strip_wakewords,
    compute_cer,
)
from asr_evolve.pipeline.models import Sample


def _load_configs():
    with open("configs/remote.yaml", encoding="utf-8") as f:
        remote_cfg = yaml.safe_load(f)
    with open("configs/model.yaml", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    return remote_cfg, model_cfg


def _get_executor(remote_cfg: dict) -> RemoteExecutor:
    r = remote_cfg["remote"]
    e = RemoteExecutor(RemoteConfig(
        host=r["host"], port=r["port"],
        username=r["username"], password=r.get("password", ""),
        key_file=r.get("key_file", ""),
        workspace=r["workspace"],
    ))
    e.connect()
    return e


def _build_eval_script(remote_cfg: dict, model_cfg: dict, checkpoint: str, test_jsonl: str, output_dir: str) -> str:
    """生成远程评估脚本（对齐 finetuned_eval.py 的推理方式）。"""
    r = remote_cfg["remote"]
    t = remote_cfg.get("training", {})
    ws = r["workspace"]
    funasr_path = t.get("funasr_path", f"{ws}/FunASR")
    base_model = t.get("model_dir", model_cfg.get("model", {}).get("model_dir", ""))
    gpu = "2"  # 用 GPU 2

    # 加载热词
    hotwords_file = f"{ws}/configs/hotwords.txt"

    script = f"""#!/bin/bash
export PYTHONPATH={funasr_path}:$PYTHONPATH
export CUDA_VISIBLE_DEVICES={gpu}

cd {ws}
mkdir -p {output_dir}

python3 << 'PYEOF'
import json, sys, os, torch
from pathlib import Path
from funasr import AutoModel

# 配置
BASE_MODEL = "{base_model}"
FINETUNE_CKPT = "{checkpoint}"
TEST_JSONL = "{test_jsonl}"
OUTPUT_DIR = "{output_dir}"
HOTWORDS_FILE = "{hotwords_file}"
BATCH_SIZE = 64

print(f"基座模型: {{BASE_MODEL}}", file=sys.stderr)
print(f"微调权重: {{FINETUNE_CKPT}}", file=sys.stderr)
print(f"测试数据: {{TEST_JSONL}}", file=sys.stderr)

# 1. 加载基座模型
model = AutoModel(model=BASE_MODEL, device="cuda:0")

# 2. Overlay 微调权重
if os.path.exists(FINETUNE_CKPT):
    print(f"加载微调权重...", file=sys.stderr)
    state_dict = torch.load(FINETUNE_CKPT, map_location="cuda:0")
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    model.model.load_state_dict(state_dict, strict=False)
    model.model.eval()
    print("微调权重加载完成", file=sys.stderr)
else:
    print(f"权重文件不存在: {{FINETUNE_CKPT}}，使用基座模型", file=sys.stderr)

# 3. 加载热词
hotword_str = ""
if os.path.exists(HOTWORDS_FILE):
    with open(HOTWORDS_FILE, "r") as f:
        hotword_str = " ".join(line.strip() for line in f if line.strip())
    print(f"热词: {{len(hotword_str.split())}} 个", file=sys.stderr)

# 4. 读测试数据
test_items = []
with open(TEST_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        test_items.append(item)
print(f"测试样本: {{len(test_items)}} 条", file=sys.stderr)

# 5. 批量推理
results = []
for batch_start in range(0, len(test_items), BATCH_SIZE):
    batch = test_items[batch_start:batch_start+BATCH_SIZE]
    batch_paths = [item["source"] for item in batch]
    
    gen_kwargs = dict(input=batch_paths, cache={{}}, batch_size=BATCH_SIZE, disable_pbar=True)
    if hotword_str:
        gen_kwargs["hotword"] = hotword_str
    
    try:
        preds = model.generate(**gen_kwargs)
    except Exception as e:
        print(f"推理失败 batch {{batch_start}}: {{e}}", file=sys.stderr)
        for item in batch:
            results.append({{"key": item["key"], "ref": item["target"], "hyp": ""}})
        continue
    
    for i, item in enumerate(batch):
        try:
            hyp = preds[i]["text"]
        except (IndexError, KeyError, TypeError):
            hyp = ""
        results.append({{"key": item["key"], "ref": item["target"], "hyp": hyp}})
    
    print(f"  进度: {{batch_start+len(batch)}}/{{len(test_items)}}", file=sys.stderr)

# 6. 写结果
output_file = os.path.join(OUTPUT_DIR, "eval_detail.jsonl")
with open(output_file, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\\n")

print(f"结果写入: {{output_file}} ({{len(results)}} 条)", file=sys.stderr)
print("EVAL_DONE")
PYEOF
"""
    return script


def cmd_eval(args):
    """评估入口：远程推理 + CER 计算。"""
    is_mock = os.environ.get("ASR_EVOLVE_MOCK", "0") == "1"
    checkpoints = args.checkpoints or "best"
    version = args.version or "latest"

    print(f"评估版本: {version}, 检查点: {checkpoints}")

    if is_mock:
        _eval_mock(version)
        return

    remote_cfg, model_cfg = _load_configs()
    r = remote_cfg["remote"]
    t = remote_cfg.get("training", {})
    ws = r["workspace"]

    # 确定检查点路径
    # 先从 train_state.json 获取最近训练的 output_dir
    state_file = Path("logs/train_state.json")
    if state_file.exists():
        train_state = json.loads(state_file.read_text(encoding="utf-8"))
        train_output = train_state.get("output_dir", "")
    else:
        train_output = f"{ws}/output/baseline_v1"

    ckpt_map = {
        "best": f"{train_output}/model.pt.best",
        "avg5": f"{train_output}/model.pt.avg5",
        "last": f"{train_output}/model.pt",
    }
    ckpt_path = ckpt_map.get(checkpoints, f"{train_output}/model.pt.{checkpoints}")

    # 测试数据（用 dev 集）
    test_jsonl = f"{ws}/data/merged/dev/dev.jsonl"

    # 远程输出目录
    eval_output = f"{ws}/output/eval_{version}"

    # 构建远程评估脚本
    script = _build_eval_script(remote_cfg, model_cfg, ckpt_path, test_jsonl, eval_output)

    # 连接远程执行
    executor = _get_executor(remote_cfg)
    try:
        # 写远程脚本
        remote_script = f"{ws}/run_eval.sh"
        rc, _, err = executor.run_command(
            f"cat > {remote_script} << 'EVALEOF'\n{script}\nEVALEOF\nchmod +x {remote_script}",
            timeout=10,
        )
        if rc != 0:
            print(f"ERROR: 写入评估脚本失败: {err.strip()}")
            return

        # 执行（推理 300 条约 2-3 分钟）
        print("正在远程推理...")
        rc, out, err = executor.run_command(f"bash {remote_script}", timeout=600)

        if "EVAL_DONE" not in out:
            print(f"ERROR: 评估未正常完成 (rc={rc})")
            if err.strip():
                print(f"  stderr: {err.strip()[-500:]}")
            return

        # 读取结果
        rc, detail_content, _ = executor.run_command(
            f"cat {eval_output}/eval_detail.jsonl", timeout=30
        )
        if rc != 0:
            print("ERROR: 无法读取评估结果")
            return

    finally:
        executor.disconnect()

    # 解析结果并计算 CER
    samples = []
    details = []
    for line in detail_content.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        ref = record.get("ref", "")
        hyp = record.get("hyp", "")
        samples.append(Sample(
            audio_path=record.get("key", ""),
            reference_text=ref,
            hypothesis_text=hyp,
        ))
        # 逐条计算详细指标
        ref_norm = normalize_text(ref)
        hyp_norm = normalize_text(hyp)
        cer, s, i, d, ref_len = compute_cer(ref, hyp)
        ref_stripped = strip_wakewords(ref_norm)
        hyp_stripped = strip_wakewords(hyp_norm)
        is_correct = (ref_norm == hyp_norm)
        details.append({
            "key": record.get("key", ""),
            "ref": ref,
            "hyp": hyp,
            "ref_norm": ref_norm,
            "hyp_norm": hyp_norm,
            "cer": round(cer, 4),
            "sub": s, "ins": i, "del": d,
            "ref_len": ref_len,
            "correct": is_correct,
        })

    # 领域评估
    evaluator = DomainEvaluator()
    report = evaluator.evaluate(samples, f"eval_{version}")

    # 保存本地结果
    results_dir = Path("results") / version
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "eval_detail.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in details),
        encoding="utf-8",
    )
    summary = {
        "version": version,
        "checkpoint": checkpoints,
        "test_set": "dev",
        "samples": len(samples),
        "cer": round(report.cer, 4),
        "stripped_cer": round(report.stripped_cer, 4),
        "recall": round(report.recall.recall, 4),
        "recall_hits": report.recall.numerator,
        "recall_total": report.recall.denominator,
    }
    (results_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 输出
    print(f"\n{'='*50}")
    print(f"评估报告: {version} (checkpoint={checkpoints})")
    print(f"{'='*50}")
    print(f"  样本数:       {len(samples)}")
    print(f"  CER:          {report.cer:.4f} ({report.cer*100:.2f}%)")
    print(f"  Stripped CER: {report.stripped_cer:.4f} ({report.stripped_cer*100:.2f}%)")
    print(f"  指令召回率:   {report.recall.recall:.4f} ({report.recall.numerator}/{report.recall.denominator})")
    print(f"{'='*50}")
    print(f"  详细结果: results/{version}/eval_detail.jsonl")
    print(f"  摘要:     results/{version}/eval_summary.json")


def _eval_mock(version: str):
    """Mock 评估模式。"""
    print(f"[MOCK] 评估版本: {version}")
    results_dir = Path("results") / version
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": version, "checkpoint": "best", "test_set": "dev",
        "samples": 50, "cer": 0.0520, "stripped_cer": 0.0380,
        "recall": 0.92, "recall_hits": 46, "recall_total": 50,
    }
    (results_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"  CER: 5.20% | Stripped CER: 3.80% | 召回率: 92%")
