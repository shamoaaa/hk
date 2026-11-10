#!/bin/bash
"""
批量评估所有测试集
依次在 s10、sp4、周报、热词、TTS-SCFChilSC、ChildMandarin 数据集上运行 eval_on_testset.py
"""

set -e  # 遇到错误停止

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_on_testset.py"

# 公共参数（可按需修改）
MODEL_DIR="/data1/eval_output"
CKPT_PATTERN="best"
COMMON_ARGS="--model-dir ${MODEL_DIR} --ckpt-pattern ${CKPT_PATTERN}"

echo "============================================================"
echo "批量评估所有测试集"
echo "============================================================"
echo "评估脚本: ${EVAL_SCRIPT}"
echo "模型目录: ${MODEL_DIR}"
echo "权重过滤: ${CKPT_PATTERN}"
echo "============================================================"

# --- 1. S10 全量业务数据 ---
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[1/6] S10 全量业务数据"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python "${EVAL_SCRIPT}" \
    --test-scp /data1/label/reflow_origin/s10/wav.scp \
    --test-text /data1/label/reflow_origin/s10/text \
    --output-dir /data1/eval_output/s10_results \
    ${COMMON_ARGS}

# --- 2. SP4 全量业务数据 ---
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[2/6] SP4 全量业务数据"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python "${EVAL_SCRIPT}" \
    --test-scp /data1/label/reflow_origin/sp4/wav.scp \
    --test-text /data1/label/reflow_origin/sp4/text \
    --output-dir /data1/eval_output/sp4_results \
    ${COMMON_ARGS}

# --- 3. 周报独立测试集 ---
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[3/6] 周报独立测试集"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python "${EVAL_SCRIPT}" \
    --test-scp /data1/label/reflow_origin/zhoubao/wav.scp \
    --test-text /data1/label/reflow_origin/zhoubao/text \
    --output-dir /data1/eval_output/testset_results \
    ${COMMON_ARGS}

# --- 4. AISHELL 热词测试集 ---
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[4/6] AISHELL 热词测试集"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python "${EVAL_SCRIPT}" \
    --test-scp /data/share/dataset/speech_asr_aishell_hotwords_testsets/wav.scp \
    --test-text /data/share/dataset/speech_asr_aishell_hotwords_testsets/text \
    --output-dir /data1/eval_output/hotword_results/use \
    ${COMMON_ARGS}

# --- 5. TTS-SCFChilSC 儿童语音 ---
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[5/6] TTS-SCFChilSC 儿童语音"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python "${EVAL_SCRIPT}" \
    --test-scp /data/share/dataset/TTS-SCFChilSC/wav.scp \
    --test-text /data/share/dataset/TTS-SCFChilSC/text \
    --output-dir /data1/eval_output/child/TTS-SCFChilSC \
    ${COMMON_ARGS}

# --- 6. ChildMandarin 儿童语音 ---
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[6/6] ChildMandarin 儿童语音"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python "${EVAL_SCRIPT}" \
    --test-scp /data1/child_dataset/ChildMandarin/wav.scp \
    --test-text /data1/child_dataset/ChildMandarin/text \
    --output-dir /data1/eval_output/child/ChildMandarin \
    ${COMMON_ARGS}

# --- 完成 ---
echo ""
echo "============================================================"
echo "全部评估完成！"
echo "============================================================"
echo "结果目录:"
echo "  S10:            /data1/eval_output/s10_results"
echo "  SP4:            /data1/eval_output/sp4_results"
echo "  周报测试集:     /data1/eval_output/testset_results"
echo "  热词测试集:     /data1/eval_output/hotword_results/use"
echo "  TTS-SCFChilSC:  /data1/eval_output/child/TTS-SCFChilSC"
echo "  ChildMandarin:  /data1/eval_output/child/ChildMandarin"
echo "============================================================"
