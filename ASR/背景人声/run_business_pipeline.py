#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
业务数据 ASR 全流程 Pipeline（仅业务数据 s10 / sp4，不含公开数据集）

适配数据集根目录:
    /share/chenming40/云端背景人声抑制/dataset

完整流程（10 阶段）:
    [1] prepare      原始音频 + 标签 → labeled/ + unlabeled/（FunASR 格式）
    [2] quality      音频质量扫描 + 阈值判断 → quality_check/（good/bad 清单）
    [3] clean        从 labeled/unlabeled 删除问题音频（可选，破坏性）
    [4] pretag       FireRedASR + Qwen3-ASR 对 unlabeled 预标注
    [5] cross_valid  人工 vs FireRed vs Qwen 三方交叉验证 → 分歧清单
    [6] split        labeled 切分为 train/dev/test → reflow_label/
    [7] augment      仅对 train 做多说话人距离增强 → augment_label/
    [8] merge        reflow + augment 按 split 合并去重 → all_label/
    [9] jsonl        all_label/{train,dev} → train.jsonl / dev.jsonl
    [10] train       torchrun 微调 SeacoParaformer

每个阶段都会先检测产物是否已存在，已完成则跳过（除非 --force）。
你当前数据集里 prepare/quality/pretag/cross_valid 已完成，默认会自动跳过。

用法:
    # 预览全流程（强烈建议首次先 dry-run 确认路径）
    python run_business_pipeline.py --dry-run

    # 跑后半段（数据已整理好预标注好，直接组装训练数据）
    python run_business_pipeline.py --stages split augment merge jsonl

    # 跑完整流程
    python run_business_pipeline.py

    # 只处理 s10，强制重跑 split
    python run_business_pipeline.py --datasets s10_reflow --stages split --force

    # 跑训练
    python run_business_pipeline.py --stages train --gpus 0,1,2,3
"""

import os
import sys
import json
import time
import wave
import math
import random
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

SCRIPT_DIR = Path(__file__).resolve().parent

# ======================================================================
#  配置（可通过命令行参数覆盖）
# ======================================================================
DEFAULT_DATASET_ROOT = Path("/share/chenming40/云端背景人声抑制/dataset")

# 业务数据集 → 数据集根目录下的子目录
BUSINESS_DATASETS = {
    "s10_reflow": "restruct_reflow_data/s10_data",
    "sp4_reflow": "restruct_reflow_data/sp4_data",
}

OUTPUT_SUBDIR = "label"          # reflow_label / augment_label / all_label 的父目录（相对 output_root）

# 输出根目录：所有生成的中间文件/训练数据写到这里（与只读的数据集盘分离）
# /share 通常是只读，故默认放到 /data1 下
DEFAULT_OUTPUT_ROOT = Path("/data1/背景人声/work")

# 切分参数
SEED = 42
DEV_RATIO = 0.05
TEST_RATIO = 0.05

# 质量阈值（与 03_quality_report.py 对齐）
Q_MIN_DUR = 0.3
Q_MAX_DUR = 30.0
Q_RMS_MIN = 50
Q_CLIP_RATIO = 0.01
Q_SNR_MIN = 5.0
Q_EXPECT_SR = 16000

# 增强
DEFAULT_AUGMENT_CONFIG = SCRIPT_DIR / "07_augment_used" / "公开数据集增强" / "augment_config_speaker_multi.yaml"
DEFAULT_AUGMENT_FACTOR = 1

# 预训练模型根目录（默认 /data1/背景人声/pretrained_models，可用 --pretrained-root 覆盖）
# 这些模型尚未下载，需先准备好：
#   - SeacoParaformer (微调起点 + 评估基准): ModelScope iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
#   - FireRedASR-LLM-L (预标注): https://github.com/FireRedTeam/FireRedASR
#   - Qwen3-ASR-1.7B  (预标注): ModelScope/HF Qwen/Qwen3-ASR-1.7B
DEFAULT_PRETRAINED_ROOT = Path("/data1/背景人声/pretrained_models")
SEACO_MODEL_NAME = "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
FIRERED_MODEL_NAME = "FireRedASR-LLM-L"
QWEN_MODEL_NAME = "Qwen3-ASR-1.7B"

# 复用的子脚本
P06_FIRERED = SCRIPT_DIR / "06_pretag_and_compare" / "FireRedASR_pretag.py"
P06_QWEN = SCRIPT_DIR / "06_pretag_and_compare" / "QwenASR_pretag.py"
P06_CROSS = SCRIPT_DIR / "06_pretag_and_compare" / "cross_validate_labels.py"
P07_AUGMENT = SCRIPT_DIR / "07_augment_used" / "augment_manifest.py"
P09_SCP2JSONL = SCRIPT_DIR / "09_train" / "fast_scp2jsonl.py"
P09_TRAIN = SCRIPT_DIR / "09_train" / "funasr" / "bin" / "train.py"

SPLITS = ["train", "dev", "test"]
# ======================================================================


def log(msg=""):
    print(msg, flush=True)


def run_cmd(cmd, ctx, desc=""):
    """执行子进程命令，dry-run 时只打印。"""
    log(f"  $ {' '.join(str(c) for c in cmd)}")
    if ctx["dry_run"]:
        return True
    try:
        subprocess.run([str(c) for c in cmd], check=True)
        return True
    except subprocess.CalledProcessError as e:
        log(f"  [错误] {desc} 失败 (返回码 {e.returncode})")
        return False
    except FileNotFoundError as e:
        log(f"  [错误] {desc} 命令不存在: {e}")
        return False


def find_file(directory: Path, names: list):
    for n in names:
        p = directory / n
        if p.exists():
            return p
    return None


def read_kaldi_map(path: Path) -> dict:
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(None, 1)
            m[parts[0]] = parts[1] if len(parts) == 2 else ""
    return m


# ======================================================================
#  [1] prepare
# ======================================================================
def stage_prepare(ctx, datasets):
    log("=" * 70)
    log("[阶段 1] prepare — 原始音频 + 标签 → labeled/ + unlabeled/")
    log("=" * 70)
    for ds in datasets:
        ds_dir = ctx["dataset_root"] / BUSINESS_DATASETS[ds]
        labeled = ds_dir / "labeled"
        scp = labeled / "wav.scp"
        text = find_file(labeled, ["text.txt", "text"])
        if scp.exists() and text and not ctx["force"]:
            n = sum(1 for _ in open(scp, encoding="utf-8"))
            log(f"  [跳过] {ds}: labeled/ 已存在 ({n} 条) — {labeled}")
            continue
        # labeled 不存在时，01_prepare_funasr_data.py 依赖项目特定的
        # label/ready + wav/source 布局，路径写死，需手动适配后单独运行。
        log(f"  [提示] {ds}: 未检测到 labeled/，请先运行 01_prepare_funasr_data*.py")
        log(f"          （该脚本路径硬编码，需按 {ds_dir} 调整 BASE_DIR 后单独执行）")


# ======================================================================
#  [2] quality（轻量内联：读 wav 头 + RMS/静音/时长/削波）
# ======================================================================
def _analyze_wav_light(wav_path: str) -> dict:
    r = {"path": wav_path, "uid": Path(wav_path).stem, "error": None,
         "duration": 0.0, "sample_rate": 0, "channels": 0,
         "rms": 0.0, "clip_ratio": 0.0}
    try:
        if os.path.getsize(wav_path) == 0:
            r["error"] = "empty_file"
            return r
        wf = wave.open(wav_path, "rb")
    except Exception as e:
        r["error"] = f"corrupt:{str(e)[:60]}"
        return r
    try:
        ch, sw, sr = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
        n = wf.getnframes()
        r["channels"], r["sample_rate"] = ch, sr
        r["duration"] = round(n / sr, 3) if sr > 0 else 0
        if sw == 2 and n > 0:
            import struct
            raw = wf.readframes(min(n, sr * 30))
            ns = len(raw) // (sw * ch)
            if ns > 0:
                s = struct.unpack(f"<{ns * ch}h", raw[:ns * ch * 2])[::ch]
                r["rms"] = round(math.sqrt(sum(x * x for x in s) / len(s)), 1)
                r["clip_ratio"] = round(sum(1 for x in s if abs(x) >= 32000) / len(s), 6)
    except Exception as e:
        r["error"] = f"read:{str(e)[:60]}"
    finally:
        wf.close()
    return r


def _judge(r: dict) -> list:
    issues = []
    if r["error"]:
        return [r["error"]]
    if r["duration"] < Q_MIN_DUR:
        issues.append("too_short")
    if r["duration"] > Q_MAX_DUR:
        issues.append("too_long")
    if r["sample_rate"] != Q_EXPECT_SR and r["sample_rate"] > 0:
        issues.append(f"unexpected_sr:{r['sample_rate']}")
    if r["channels"] != 1 and r["channels"] > 0:
        issues.append(f"not_mono:{r['channels']}")
    if 0 < r["rms"] < Q_RMS_MIN:
        issues.append("silence_or_low_energy")
    if r["clip_ratio"] > Q_CLIP_RATIO:
        issues.append("clipping")
    return issues


def stage_quality(ctx, datasets):
    log("=" * 70)
    log("[阶段 2] quality — 音频质量扫描 + 阈值判断")
    log("=" * 70)
    for ds in datasets:
        ds_dir = ctx["dataset_root"] / BUSINESS_DATASETS[ds]
        qc_dir = ds_dir / "quality_check"
        bad_out = qc_dir / "bad_files.txt"
        if bad_out.exists() and not ctx["force"]:
            log(f"  [跳过] {ds}: quality_check/ 已存在 — {qc_dir}")
            continue

        # 扫描 labeled/wav + unlabeled/wav
        wavs = []
        for sub in [ds_dir / "labeled" / "wav", ds_dir / "unlabeled" / "wav",
                    ds_dir / "unlabeled"]:
            if sub.exists():
                wavs += [str(p) for p in sub.rglob("*.wav")]
        wavs = sorted(set(wavs))
        log(f"  [{ds}] 扫描到 {len(wavs)} 个 wav")
        if not wavs or ctx["dry_run"]:
            if ctx["dry_run"]:
                log(f"    [预览] 将质检 {len(wavs)} 个文件 → {qc_dir}")
            continue

        results = []
        with ThreadPoolExecutor(max_workers=ctx["workers"]) as ex:
            for r in ex.map(_analyze_wav_light, wavs):
                results.append(r)

        qc_dir.mkdir(parents=True, exist_ok=True)
        good, bad = 0, 0
        with open(qc_dir / "good_files.txt", "w", encoding="utf-8") as fg, \
             open(bad_out, "w", encoding="utf-8") as fb:
            for r in results:
                issues = _judge(r)
                if issues:
                    fb.write(f"{r['path']}\t{','.join(issues)}\n")
                    bad += 1
                else:
                    fg.write(f"{r['path']}\n")
                    good += 1
        log(f"    合格 {good} / 问题 {bad} → {bad_out}")


# ======================================================================
#  [3] clean（删除问题音频，破坏性，默认跳过）
# ======================================================================
def stage_clean(ctx, datasets):
    log("=" * 70)
    log("[阶段 3] clean — 删除问题音频（破坏性）")
    log("=" * 70)
    if not ctx["remove_bad"]:
        log("  [跳过] 未指定 --remove-bad，保留所有音频")
        return
    for ds in datasets:
        ds_dir = ctx["dataset_root"] / BUSINESS_DATASETS[ds]
        bad_file = ds_dir / "quality_check" / "bad_files.txt"
        if not bad_file.exists():
            log(f"  [跳过] {ds}: 无 bad_files.txt")
            continue
        bad_uids = set()
        for line in open(bad_file, encoding="utf-8"):
            if line.strip():
                bad_uids.add(Path(line.split("\t")[0]).stem)
        log(f"  [{ds}] 待删除问题音频: {len(bad_uids)} 条")
        if ctx["dry_run"]:
            log(f"    [预览] 将删除 wav 并清理 labeled/{{text.txt,wav.scp,train.jsonl}}")
            continue
        labeled = ds_dir / "labeled"
        # 删 wav
        deleted = 0
        for d in [labeled / "wav", ds_dir / "unlabeled" / "wav", ds_dir / "unlabeled"]:
            if not d.exists():
                continue
            for uid in bad_uids:
                p = d / f"{uid}.wav"
                if p.exists():
                    p.unlink()
                    deleted += 1
        # 清理索引
        for name in ["text.txt", "text", "wav.scp"]:
            fp = labeled / name
            if not fp.exists():
                continue
            kept = [l for l in open(fp, encoding="utf-8")
                    if l.strip() and l.split(None, 1)[0] not in bad_uids]
            open(fp, "w", encoding="utf-8").writelines(kept)
        log(f"    删除 {deleted} 个 wav，已清理索引")


# ======================================================================
#  [4] pretag（FireRedASR + Qwen3-ASR 对 unlabeled 预标注）
# ======================================================================
def stage_pretag(ctx, datasets):
    log("=" * 70)
    log("[阶段 4] pretag — FireRedASR + Qwen3-ASR 预标注 unlabeled")
    log("=" * 70)
    for ds in datasets:
        ds_dir = ctx["dataset_root"] / BUSINESS_DATASETS[ds]
        unlabeled_wav = ds_dir / "unlabeled" / "wav"
        if not unlabeled_wav.exists():
            unlabeled_wav = ds_dir / "unlabeled"

        fr_out = ds_dir / "firered_pretag_output"
        qw_out = ds_dir / "qwen_pretag_output"

        if (fr_out / "text.txt").exists() and not ctx["force"]:
            log(f"  [跳过] {ds}: firered_pretag_output 已存在")
        else:
            run_cmd([sys.executable, str(P06_FIRERED), "--wav-dir", unlabeled_wav,
                     "--output-dir", fr_out, "--model-dir", ctx["firered_model"],
                     "--device", "cuda:auto", "--resume"],
                    ctx, f"{ds} FireRed预标注")

        if (qw_out / "text.txt").exists() and not ctx["force"]:
            log(f"  [跳过] {ds}: qwen_pretag_output 已存在")
        else:
            run_cmd([sys.executable, str(P06_QWEN), "--wav-dir", unlabeled_wav,
                     "--output-dir", qw_out, "--model", ctx["qwen_model"],
                     "--device", "cuda:auto", "--resume"],
                    ctx, f"{ds} Qwen预标注")


# ======================================================================
#  [5] cross_valid（三方交叉验证）
# ======================================================================
def stage_cross_valid(ctx, datasets):
    log("=" * 70)
    log("[阶段 5] cross_valid — 人工 vs FireRed vs Qwen 三方交叉验证")
    log("=" * 70)
    for ds in datasets:
        ds_dir = ctx["dataset_root"] / BUSINESS_DATASETS[ds]
        human = find_file(ds_dir / "labeled", ["text.txt", "text"])
        firered = ds_dir / "firered_pretag_output" / "text.txt"
        qwen = ds_dir / "qwen_pretag_output" / "text.txt"
        out = ds_dir / "cross_validate_output"

        if (out / "validate_detail.jsonl").exists() and not ctx["force"]:
            log(f"  [跳过] {ds}: cross_validate_output 已存在")
            continue
        if not (human and firered.exists() and qwen.exists()):
            log(f"  [跳过] {ds}: 缺少 human/firered/qwen 三方标签")
            continue
        run_cmd([sys.executable, str(P06_CROSS), "--human", human,
                 "--firered", firered, "--qwen", qwen, "--output", out],
                ctx, f"{ds} 交叉验证")


# ======================================================================
#  [6] split
# ======================================================================
def stage_split(ctx, datasets):
    log("=" * 70)
    log("[阶段 6] split — labeled 切分为 train/dev/test")
    log("=" * 70)
    reflow_root = ctx["label_root"] / "reflow_label"
    for ds in datasets:
        labeled = ctx["dataset_root"] / BUSINESS_DATASETS[ds] / "labeled"
        wav_dir = labeled / "wav"
        text_file = find_file(labeled, ["text.txt", "text"])
        scp_file = find_file(labeled, ["wav.scp"])
        if not text_file or not scp_file:
            log(f"  [跳过] {ds}: 缺少 text/wav.scp")
            continue

        key2text = read_kaldi_map(text_file)
        key2wav_raw = read_kaldi_map(scp_file)

        # 路径重映射：scp 里记录的可能是旧绝对路径，统一改为「实际 labeled/wav/<原文件名>」
        # 优先用重映射后存在的路径；若不存在则回退原路径（仍存在则保留）
        key2wav = {}
        remapped, kept_orig, missing = 0, 0, 0
        use_remap = wav_dir.exists()
        for k, raw in key2wav_raw.items():
            fname = os.path.basename(raw.replace("\\", "/"))
            cand = wav_dir / fname if use_remap else None
            if cand is not None and cand.exists():
                key2wav[k] = str(cand)
                remapped += 1
            elif os.path.exists(raw):
                key2wav[k] = raw
                kept_orig += 1
            else:
                missing += 1  # 两个位置都找不到，丢弃

        common = sorted(set(key2text) & set(key2wav))
        log(f"  [{ds}] text={len(key2text)} wav(scp)={len(key2wav_raw)} "
            f"重映射={remapped} 原路径保留={kept_orig} 缺失丢弃={missing} 交集={len(common)}")
        if not common:
            log(f"  [警告] {ds}: 无有效条目（音频路径全部不存在），检查 labeled/wav 是否就位")
            continue

        keys = list(common)
        random.seed(SEED)
        random.shuffle(keys)
        n = len(keys)
        n_test, n_dev = int(n * TEST_RATIO), int(n * DEV_RATIO)
        parts = {"test": keys[:n_test], "dev": keys[n_test:n_test + n_dev],
                 "train": keys[n_test + n_dev:]}

        for split, ks in parts.items():
            out_dir = reflow_root / split
            scp_out = out_dir / f"{ds}_wav.scp"
            text_out = out_dir / f"{ds}_text"
            if ctx["dry_run"]:
                log(f"    [预览] {ds}/{split}: {len(ks)} 条 → {scp_out}")
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(scp_out, "w", encoding="utf-8") as fs, \
                 open(text_out, "w", encoding="utf-8") as ft:
                for k in sorted(ks):
                    fs.write(f"{k} {key2wav[k]}\n")
                    ft.write(f"{k} {key2text[k]}\n")
            log(f"    {ds}/{split}: {len(ks)} 条")


# ======================================================================
#  [7] augment（仅 train）
# ======================================================================
def stage_augment(ctx, datasets):
    log("=" * 70)
    log("[阶段 7] augment — 仅对 train 做多说话人距离增强")
    log("=" * 70)
    log(f"  配置: {ctx['augment_config']}  倍数: {ctx['augment_factor']}")
    reflow_root = ctx["label_root"] / "reflow_label"
    augment_root = ctx["label_root"] / "augment_label"

    for ds in datasets:
        train_scp = reflow_root / "train" / f"{ds}_wav.scp"
        train_text = reflow_root / "train" / f"{ds}_text"
        if not train_scp.exists():
            log(f"  [跳过] {ds}: 未找到 train split，请先运行 split")
            continue

        work = augment_root / "_work" / ds
        manifest = work / "train_manifest.jsonl"
        if ctx["dry_run"]:
            log(f"  [预览] {ds}: 构建 manifest + augment_manifest.py → {work}")
            continue

        # 构建 {key,wav,txt} manifest
        k2w = read_kaldi_map(train_scp)
        k2t = read_kaldi_map(train_text)
        common = sorted(set(k2w) & set(k2t))
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest, "w", encoding="utf-8") as f:
            for k in common:
                f.write(json.dumps({"key": k, "wav": k2w[k], "txt": k2t[k]},
                                   ensure_ascii=False) + "\n")
        log(f"  [{ds}] manifest: {len(common)} 条")

        ok = run_cmd([sys.executable, str(P07_AUGMENT),
                      "--config", ctx["augment_config"],
                      "--input_manifest", manifest,
                      "--output_dir", work,
                      "--splits", "unknown"], ctx, f"{ds} 增强")
        if not ok:
            continue

        src_scp = work / "wav_unknown.scp"
        src_text = work / "text_unknown.txt"
        if src_scp.exists() and src_text.exists():
            dst = augment_root / "train"
            dst.mkdir(parents=True, exist_ok=True)
            (dst / f"{ds}_wav.scp").write_text(src_scp.read_text(encoding="utf-8"), encoding="utf-8")
            (dst / f"{ds}_text").write_text(src_text.read_text(encoding="utf-8"), encoding="utf-8")
            n = sum(1 for _ in open(dst / f"{ds}_wav.scp", encoding="utf-8"))
            log(f"  [{ds}] 增强完成: {n} 条 → {dst}")


# ======================================================================
#  [8] merge
# ======================================================================
def stage_merge(ctx, datasets):
    log("=" * 70)
    log("[阶段 8] merge — reflow_label + augment_label → all_label")
    log("=" * 70)
    reflow_root = ctx["label_root"] / "reflow_label"
    augment_root = ctx["label_root"] / "augment_label"
    all_root = ctx["label_root"] / "all_label"

    for split in SPLITS:
        scp_files, text_files = [], []
        for ds in datasets:
            scp_files.append(reflow_root / split / f"{ds}_wav.scp")
            text_files.append(reflow_root / split / f"{ds}_text")
        if split == "train":
            for ds in datasets:
                scp_files.append(augment_root / split / f"{ds}_wav.scp")
                text_files.append(augment_root / split / f"{ds}_text")

        out_scp = all_root / split / "wav.scp"
        out_text = all_root / split / "text"
        if ctx["dry_run"]:
            present = [str(p) for p in scp_files if p.exists()]
            log(f"  [预览] {split}: 合并 {len(present)} 个文件 → {out_scp}")
            continue

        text_map = {}
        for tf in text_files:
            if tf.exists():
                for line in open(tf, encoding="utf-8"):
                    if line.strip():
                        p = line.rstrip("\n").split(None, 1)
                        text_map.setdefault(p[0], p[1] if len(p) == 2 else "")
        seen, sl, tl = set(), [], []
        for sf in scp_files:
            if not sf.exists():
                continue
            for line in open(sf, encoding="utf-8"):
                if not line.strip():
                    continue
                p = line.rstrip("\n").split(None, 1)
                if len(p) != 2 or p[0] in seen or p[0] not in text_map:
                    continue
                seen.add(p[0])
                sl.append(f"{p[0]} {p[1]}")
                tl.append(f"{p[0]} {text_map[p[0]]}")
        out_scp.parent.mkdir(parents=True, exist_ok=True)
        out_scp.write_text("\n".join(sl) + ("\n" if sl else ""), encoding="utf-8")
        out_text.write_text("\n".join(tl) + ("\n" if tl else ""), encoding="utf-8")
        log(f"  {split}: 合并后 {len(sl)} 条 → {out_scp}")


# ======================================================================
#  [9] jsonl
# ======================================================================
def stage_jsonl(ctx, datasets):
    log("=" * 70)
    log("[阶段 9] jsonl — all_label/{train,dev} → train.jsonl / dev.jsonl")
    log("=" * 70)
    all_root = ctx["label_root"] / "all_label"
    for split, out_name in [("train", "train.jsonl"), ("dev", "dev.jsonl")]:
        scp = all_root / split / "wav.scp"
        text = all_root / split / "text"
        out = all_root / split / out_name
        if not scp.exists():
            log(f"  [跳过] {split}: 缺少 wav.scp，请先 merge")
            continue
        run_cmd([sys.executable, str(P09_SCP2JSONL),
                 "--wav_scp", scp, "--text", text, "--output", out,
                 "--workers", "8"], ctx, f"{split} jsonl")


# ======================================================================
#  [10] train
# ======================================================================
def stage_train(ctx, datasets):
    log("=" * 70)
    log("[阶段 10] train — torchrun 微调 SeacoParaformer")
    log("=" * 70)
    all_root = ctx["label_root"] / "all_label"
    train_jsonl = all_root / "train" / "train.jsonl"
    dev_jsonl = all_root / "dev" / "dev.jsonl"
    output_dir = ctx["output_root"] / "train_outputs"

    if not train_jsonl.exists() and not ctx["dry_run"]:
        log(f"  [跳过] 缺少 {train_jsonl}，请先运行 jsonl 阶段")
        return

    gpus = ctx["gpus"]
    gpu_num = len(gpus.split(","))
    env_prefix = f"CUDA_VISIBLE_DEVICES={gpus} PYTHONPATH={SCRIPT_DIR / '09_train'}"

    cmd = [
        "torchrun", "--nnodes", "1", "--nproc_per_node", str(gpu_num),
        str(P09_TRAIN),
        f"++model='{ctx['base_model']}'",
        f"++train_data_set_list='{train_jsonl}'",
        f"++valid_data_set_list='{dev_jsonl}'",
        "++dataset_conf.batch_size=60000",
        "++dataset_conf.batch_type=token",
        f"++dataset_conf.num_workers={gpu_num}",
        "++train_conf.max_epoch=50",
        f"++train_conf.resume={'true' if ctx['resume_train'] else 'false'}",
        "++train_conf.validate_interval=4000",
        "++train_conf.save_checkpoint_interval=4000",
        "++train_conf.keep_nbest_models=5",
        "++train_conf.avg_nbest_model=5",
        "++train_conf.find_unused_parameters=true",
        "++freeze_param=['hotword_output_layer','seaco_decoder']",
        "++optim_conf.lr=0.00002",
        "++scheduler_conf.warmup_steps=30000",
        f"++output_dir='{output_dir}'",
    ]
    log(f"  环境: {env_prefix}")
    log(f"  $ {' '.join(cmd)}")
    if ctx["dry_run"]:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["PYTHONPATH"] = str(SCRIPT_DIR / "09_train") + os.pathsep + env.get("PYTHONPATH", "")
    try:
        subprocess.run(cmd, check=True, env=env)
        log(f"  训练完成 → {output_dir}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        log(f"  [错误] 训练失败: {e}")


# ======================================================================
#  主流程
# ======================================================================
STAGE_FUNCS = {
    "prepare": stage_prepare,
    "quality": stage_quality,
    "clean": stage_clean,
    "pretag": stage_pretag,
    "cross_valid": stage_cross_valid,
    "split": stage_split,
    "augment": stage_augment,
    "merge": stage_merge,
    "jsonl": stage_jsonl,
    "train": stage_train,
}
ALL_STAGES = ["prepare", "quality", "clean", "pretag", "cross_valid",
              "split", "augment", "merge", "jsonl", "train"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="业务数据 ASR 全流程 Pipeline（仅 s10/sp4）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT),
                        help=f"输出根目录（中间文件/训练数据，需可写）(default: {DEFAULT_OUTPUT_ROOT})")
    parser.add_argument("--stages", nargs="+", default=ALL_STAGES, choices=ALL_STAGES)
    parser.add_argument("--datasets", nargs="+",
                        default=list(BUSINESS_DATASETS.keys()),
                        choices=list(BUSINESS_DATASETS.keys()))
    parser.add_argument("--augment-config", default=str(DEFAULT_AUGMENT_CONFIG))
    parser.add_argument("--augment-factor", type=int, default=DEFAULT_AUGMENT_FACTOR)
    parser.add_argument("--pretrained-root", default=None,
                        help="预训练模型根目录 (default: /data1/背景人声/pretrained_models)")
    parser.add_argument("--base-model", default=None,
                        help="SeacoParaformer 模型目录（微调起点），覆盖默认")
    parser.add_argument("--firered-model", default=None,
                        help="FireRedASR-LLM-L 模型目录，覆盖默认")
    parser.add_argument("--qwen-model", default=None,
                        help="Qwen3-ASR 模型目录，覆盖默认")
    parser.add_argument("--gpus", default="0", help="训练用 GPU，逗号分隔，如 0,1,2,3")
    parser.add_argument("--no-resume-train", action="store_true",
                        help="训练不续跑，每次从预训练权重重新开始（默认续跑：断了重跑会接着练）")
    parser.add_argument("--workers", type=int, default=8, help="质检线程数")
    parser.add_argument("--remove-bad", action="store_true", help="启用 clean 删除问题音频")
    parser.add_argument("--force", action="store_true", help="强制重跑（忽略已存在产物）")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)

    # 预训练模型路径解析：命令行 > 默认 /data1/背景人声/pretrained_models
    pre_root = Path(args.pretrained_root) if args.pretrained_root \
        else DEFAULT_PRETRAINED_ROOT
    base_model = args.base_model or str(pre_root / SEACO_MODEL_NAME)
    firered_model = args.firered_model or str(pre_root / FIRERED_MODEL_NAME)
    qwen_model = args.qwen_model or str(pre_root / QWEN_MODEL_NAME)

    ctx = {
        "dataset_root": dataset_root,
        "output_root": output_root,
        "label_root": output_root / OUTPUT_SUBDIR,
        "augment_config": Path(args.augment_config),
        "augment_factor": args.augment_factor,
        "base_model": base_model,
        "firered_model": firered_model,
        "qwen_model": qwen_model,
        "gpus": args.gpus,
        "resume_train": not args.no_resume_train,
        "workers": args.workers,
        "remove_bad": args.remove_bad,
        "force": args.force,
        "dry_run": args.dry_run,
    }

    log("=" * 70)
    log("业务数据 ASR 全流程 Pipeline")
    log("=" * 70)
    log(f"数据集根目录: {dataset_root}  (只读，仅读取原始 labeled)")
    log(f"输出根目录:   {output_root}  (中间文件/训练数据)")
    log(f"标签输出:     {ctx['label_root']}")
    log(f"业务数据集:   {args.datasets}")
    log(f"执行阶段:     {args.stages}")
    log(f"预训练模型根: {pre_root}")
    log(f"  - base(seaco): {ctx['base_model']}")
    log(f"  - firered:     {ctx['firered_model']}")
    log(f"  - qwen:        {ctx['qwen_model']}")
    log(f"预览模式:     {args.dry_run}   强制重跑: {args.force}")
    log("")

    if not dataset_root.exists() and not args.dry_run:
        log(f"[错误] 数据集根目录不存在: {dataset_root}")
        sys.exit(1)

    for stage in args.stages:
        STAGE_FUNCS[stage](ctx, args.datasets)
        log("")

    log("=" * 70)
    log("Pipeline 完成")
    log(f"训练数据: {ctx['label_root']}/all_label/train/train.jsonl")
    log(f"模型输出: {output_root}/train_outputs/")
    log("=" * 70)


if __name__ == "__main__":
    main()
