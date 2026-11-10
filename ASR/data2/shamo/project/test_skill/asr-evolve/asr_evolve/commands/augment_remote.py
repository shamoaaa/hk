# -*- coding: utf-8 -*-
"""augment-remote 命令 — 远程数据增强（完整增强链）.

将本地的 augment_engine.py + augment.yaml 上传到远程，
在远程对训练数据执行完整增强链（reverb→speaker_mix→speed→EQ→volume→noise→clipping）。

功能:
  asr-evolve augment run       上传引擎+配置，执行远程增强
  asr-evolve augment status    查看远程增强产物状态
  asr-evolve augment restore   恢复原始训练数据
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml

from asr_evolve.environment.remote import RemoteExecutor, RemoteConfig


# 本地文件路径
_ENGINE_PATH = Path(__file__).parent.parent / "pipeline" / "augment_engine.py"
_AUGMENT_YAML = Path("configs/augment.yaml")


def _load_remote_cfg() -> dict:
    with open("configs/remote.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def _compute_namespace() -> str:
    """根据 augment.yaml 内容计算命名空间哈希。"""
    content = _AUGMENT_YAML.read_bytes()
    return "aug_" + hashlib.sha256(content).hexdigest()[:8]


def _build_runner_script(ws: str, namespace: str) -> str:
    """生成远程执行的 runner 脚本（调用 augment_engine）。"""
    return f'''#!/usr/bin/env python3
"""Remote augment runner — calls augment_engine.augment_single() on each sample."""
import json, os, sys
import soundfile as sf
import numpy as np

sys.path.insert(0, "{ws}/asr_evolve_remote")

from augment_engine import AudioAugmentor

WS = "{ws}"
CONFIG_PATH = f"{{WS}}/asr_evolve_remote/augment.yaml"
NAMESPACE = "{namespace}"
AUGMENTED_DIR = f"{{WS}}/data/augmented/{{NAMESPACE}}"
TRAIN_JSONL = f"{{WS}}/data/merged/train/train_original.jsonl"
if not os.path.exists(TRAIN_JSONL):
    TRAIN_JSONL = f"{{WS}}/data/merged/train/train.jsonl"
OUTPUT_JSONL = f"{{WS}}/data/merged/train/train_augmented.jsonl"

os.makedirs(AUGMENTED_DIR, exist_ok=True)

# Load engine
augmentor = AudioAugmentor(CONFIG_PATH)

# Read samples
with open(TRAIN_JSONL, "r") as f:
    samples = [json.loads(line) for line in f if line.strip()]

print(f"Namespace: {{NAMESPACE}}")
print(f"Chain: {{augmentor.chain}}")
print(f"Samples: {{len(samples)}}")

# Check for reuse
existing = [f for f in os.listdir(AUGMENTED_DIR) if f.endswith('.wav')] if os.path.isdir(AUGMENTED_DIR) else []
print(f"Existing augmented: {{len(existing)}}")

if len(existing) >= len(samples):
    print("Reusing existing augmented data")
else:
    # Build audio cache for speaker_mix (load all candidate audios)
    print("Building audio cache for speaker_mix...")
    audio_cache = {{}}
    candidate_paths = []
    for s in samples:
        wav_path = s["source"]
        candidate_paths.append(wav_path)
        if wav_path not in audio_cache:
            try:
                audio, sr = sf.read(wav_path, dtype="float64")
                if audio.ndim > 1:
                    audio = audio[:, 0]
                audio_cache[wav_path] = (audio, sr)
            except Exception:
                pass
    print(f"Audio cache: {{len(audio_cache)}} files loaded")

    # Augment
    print("Augmenting...")
    for i, sample in enumerate(samples):
        wav_path = sample["source"]
        key = sample["key"]
        out_path = os.path.join(AUGMENTED_DIR, f"aug_{{key}}.wav")

        if os.path.exists(out_path):
            continue

        if wav_path not in audio_cache:
            continue

        audio, sr = audio_cache[wav_path]
        try:
            aug_audio = augmentor.augment_single(
                audio=audio, sr=sr, wav_path=wav_path,
                aug_idx=0, file_idx=i,
                audio_cache=audio_cache,
                candidate_paths=candidate_paths,
            )
            if aug_audio is not None and not np.any(np.isnan(aug_audio)):
                # Peak normalize
                peak = np.max(np.abs(aug_audio))
                if peak > 0:
                    aug_audio = aug_audio / peak * 0.95
                sf.write(out_path, aug_audio, sr)
        except Exception as e:
            print(f"  WARN [{{i}}] {{key}}: {{e}}")

        if (i + 1) % 200 == 0:
            print(f"  {{i+1}}/{{len(samples)}}")

# Build output JSONL
out_samples = []
for s in samples:
    out_samples.append(s)
    aug_wav = os.path.join(AUGMENTED_DIR, f"aug_{{s['key']}}.wav")
    if os.path.exists(aug_wav):
        out_samples.append({{
            "key": f"aug_{{s['key']}}",
            "source": aug_wav,
            "target": s["target"],
        }})

with open(OUTPUT_JSONL, "w") as f:
    for s in out_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\\n")

print(f"Done. Original: {{len(samples)}}, Total: {{len(out_samples)}}")
'''


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_augment_run(args) -> None:
    """上传引擎+配置到远程，执行完整增强链。"""
    remote_cfg = _load_remote_cfg()
    ws = remote_cfg["remote"]["workspace"]
    namespace = _compute_namespace()

    executor = _get_executor(remote_cfg)
    try:
        remote_dir = f"{ws}/asr_evolve_remote"

        # 1. Create remote directory
        executor.run_command(f"mkdir -p {remote_dir}", timeout=5)

        # 2. Upload augment_engine.py
        engine_content = _ENGINE_PATH.read_text(encoding="utf-8")
        rc, _, err = executor.run_command(
            f"cat > {remote_dir}/augment_engine.py << 'ASR_EVOLVE_ENGINE_EOF'\n{engine_content}\nASR_EVOLVE_ENGINE_EOF",
            timeout=30,
        )
        if rc != 0:
            print(f"ERROR: upload augment_engine.py failed: {err.strip()[:200]}")
            return
        print("[OK] augment_engine.py uploaded")

        # 3. Upload augment.yaml
        yaml_content = _AUGMENT_YAML.read_text(encoding="utf-8")
        rc, _, err = executor.run_command(
            f"cat > {remote_dir}/augment.yaml << 'ASR_EVOLVE_YAML_EOF'\n{yaml_content}\nASR_EVOLVE_YAML_EOF",
            timeout=10,
        )
        if rc != 0:
            print(f"ERROR: upload augment.yaml failed: {err.strip()[:200]}")
            return
        print("[OK] augment.yaml uploaded")

        # 4. Upload runner script
        runner = _build_runner_script(ws, namespace)
        rc, _, err = executor.run_command(
            f"cat > {remote_dir}/run_augment.py << 'ASR_EVOLVE_RUNNER_EOF'\n{runner}\nASR_EVOLVE_RUNNER_EOF",
            timeout=10,
        )
        if rc != 0:
            print(f"ERROR: 上传 runner 失败: {err.strip()[:200]}")
            return

        # 5. Execute (redirect output to log file on remote for debugging)
        print(f"Running remote augmentation (namespace: {namespace})...")
        log_file = f"{remote_dir}/augment.log"
        try:
            # Run python script, save output to log, then echo exit code to stdout
            # EXIT_CODE goes to stdout (not redirected), allowing us to parse it
            rc, out, err = executor.run_command(
                f"cd {ws} && python3 {remote_dir}/run_augment.py > {log_file} 2>&1; AUGRC=$?; echo EXIT_CODE=$AUGRC; exit $AUGRC",
                timeout=1800,  # 30 min timeout for augmentation
            )
        except Exception as ex:
            print(f"ERROR: SSH exception: {type(ex).__name__}: {ex}")
            # Try to read log anyway for diagnostics
            try:
                _, log_content, _ = executor.run_command(f"tail -30 {log_file} 2>/dev/null", timeout=10)
                if log_content.strip():
                    print(f"  Partial log:")
                    for line in log_content.strip().split("\n")[-20:]:
                        print(f"    {line}")
            except Exception:
                pass
            return

        # Parse exit code from output
        actual_rc = rc
        if "EXIT_CODE=0" in out:
            actual_rc = 0
        elif "EXIT_CODE=" in out:
            try:
                actual_rc = int(out.split("EXIT_CODE=")[-1].strip())
            except ValueError:
                pass

        # Read log file for full output
        _, log_content, _ = executor.run_command(f"cat {log_file} 2>/dev/null", timeout=10)

        if actual_rc != 0:
            # Check if output JSONL was actually produced despite non-zero rc
            _, check_out, _ = executor.run_command(
                f'test -f "{ws}/data/merged/train/train_augmented.jsonl" && wc -l < "{ws}/data/merged/train/train_augmented.jsonl"',
                timeout=10,
            )
            augmented_lines = check_out.strip()
            if augmented_lines.isdigit() and int(augmented_lines) > 0:
                print(f"WARNING: script exited with rc={actual_rc} but output exists ({augmented_lines} lines). Proceeding.")
                print(log_content.strip())
            else:
                print(f"ERROR: augment failed (rc={actual_rc})")
                print(f"  Log ({log_file}):")
                lines = log_content.strip().split("\n") if log_content.strip() else ["(empty log)"]
                for line in lines[-20:]:
                    print(f"    {line}")
                return
        else:
            print(log_content.strip())

        # 6. Merge: replace train.jsonl
        merge_cmd = (
            f'AUGMENTED="{ws}/data/merged/train/train_augmented.jsonl"; '
            f'TRAIN="{ws}/data/merged/train/train.jsonl"; '
            f'BACKUP="{ws}/data/merged/train/train_original.jsonl"; '
            f'if [ -f "$AUGMENTED" ]; then '
            f'  if [ ! -f "$BACKUP" ]; then cp "$TRAIN" "$BACKUP"; fi; '
            f'  cp "$AUGMENTED" "$TRAIN"; '
            f'  echo "MERGED: $(wc -l < $TRAIN) lines"; '
            f'else echo "ERROR: no augmented file"; exit 1; fi'
        )
        rc, out, _ = executor.run_command(merge_cmd, timeout=30)
        if rc == 0:
            print(out.strip())
            print("\n[OK] Augmentation complete. Next: asr-evolve train submit")
        else:
            print(f"ERROR: merge failed")

    finally:
        executor.disconnect()


def cmd_augment_status(args) -> None:
    """查看远程增强产物状态。"""
    remote_cfg = _load_remote_cfg()
    ws = remote_cfg["remote"]["workspace"]

    executor = _get_executor(remote_cfg)
    try:
        cmds = [
            f"wc -l {ws}/data/merged/train/train.jsonl 2>/dev/null",
            f"wc -l {ws}/data/merged/train/train_augmented.jsonl 2>/dev/null",
            f"wc -l {ws}/data/merged/train/train_original.jsonl 2>/dev/null",
            f"ls -d {ws}/data/augmented/aug_*/ 2>/dev/null",
        ]
        print("远程增强状态:")
        print("━" * 40)
        for cmd in cmds:
            rc, out, _ = executor.run_command(cmd, timeout=10)
            if out.strip():
                print(f"  {out.strip()}")
        print("━" * 40)
    finally:
        executor.disconnect()


def cmd_augment_restore(args) -> None:
    """恢复原始训练数据。"""
    remote_cfg = _load_remote_cfg()
    ws = remote_cfg["remote"]["workspace"]

    executor = _get_executor(remote_cfg)
    try:
        rc, out, _ = executor.run_command(
            f"if [ -f {ws}/data/merged/train/train_original.jsonl ]; then "
            f"cp {ws}/data/merged/train/train_original.jsonl {ws}/data/merged/train/train.jsonl; "
            f"echo RESTORED; else echo NO_BACKUP; fi",
            timeout=10,
        )
        print(out.strip())
    finally:
        executor.disconnect()


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------

def cmd_augment(args) -> None:
    """augment 命令路由。"""
    action = getattr(args, "augment_action", None)
    if action == "run":
        cmd_augment_run(args)
    elif action == "status":
        cmd_augment_status(args)
    elif action == "restore":
        cmd_augment_restore(args)
    else:
        print("用法: asr-evolve augment {run|status|restore}")
