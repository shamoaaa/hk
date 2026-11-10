#!/usr/bin/env python3
"""
S10 ASR 已标注数据评估脚本
使用 FunASR 对 labeled/ 目录中的音频进行识别，与人工标签对比，
计算 CER/WER/SER 等指标，输出逐条结果和汇总报告。

依赖安装:
    pip install funasr torch torchaudio jiwer

用法:
    python eval_labeled.py
    python eval_labeled.py --model sensevoice
    python eval_labeled.py --device cpu
    python eval_labeled.py --sample 1000    # 只评估随机1000条
    python eval_labeled.py --resume
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import re
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# ============ 配置 ============
BASE_DIR = Path(r'E:\8.音频\03 s10\s10_data_reflow')
LABELED_DIR = BASE_DIR / "labeled"
OUTPUT_DIR = BASE_DIR / "eval_output"

MODEL_CHOICES = {
    "paraformer": "paraformer-zh",
    "sensevoice": "iic/SenseVoiceSmall",
}
# ==============================


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    logger = logging.getLogger('eval')
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger, log_file


def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


# ============ CER 计算（不依赖 jiwer） ============

def edit_distance(ref: list, hyp: list):
    """计算编辑距离，返回 (substitutions, insertions, deletions)"""
    n, m = len(ref), len(hyp)
    # dp[i][j] = (cost, sub, ins, del)
    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, 0, i)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, j, 0)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub = dp[i - 1][j - 1]
                ins = dp[i][j - 1]
                dele = dp[i - 1][j]
                candidates = [
                    (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),      # substitution
                    (ins[0] + 1, ins[1], ins[2] + 1, ins[3]),      # insertion
                    (dele[0] + 1, dele[1], dele[2], dele[3] + 1),  # deletion
                ]
                dp[i][j] = min(candidates, key=lambda x: x[0])

    cost, s, i, d = dp[n][m]
    return s, i, d


def compute_cer(reference: str, hypothesis: str):
    """计算字符错误率 (CER)，返回 (cer, sub, ins, del, ref_len)"""
    # 去除空格，按字符切分
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))

    if len(ref_chars) == 0:
        return (1.0 if len(hyp_chars) > 0 else 0.0), 0, len(hyp_chars), 0, 0

    s, i, d = edit_distance(ref_chars, hyp_chars)
    cer = (s + i + d) / len(ref_chars)
    return cer, s, i, d, len(ref_chars)


def normalize_text(text: str) -> str:
    """文本规范化：去除标点、转小写、合并空格"""
    # 去除中英文标点
    text = re.sub(r'[，。！？、；：""''（）《》【】\[\]{},.!?;:\'"()\-—…·]', '', text)
    text = text.lower().strip()
    text = re.sub(r'\s+', '', text)
    return text


def load_progress(progress_file: Path) -> set:
    done = set()
    if progress_file.exists():
        with open(progress_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(line)
    return done


def load_labels(text_file: Path, logger) -> dict:
    """从 labeled/text.txt 加载标签，返回 {uid: text}"""
    logger.info(f"加载标签文件: {text_file}")
    labels = {}
    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            labels[parts[0]] = parts[1]
    logger.info(f"加载标签: {len(labels)} 条")
    return labels


def run_eval(args):
    output_dir = OUTPUT_DIR
    logger, log_file = setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("S10 ASR 已标注数据评估")
    logger.info("=" * 60)
    logger.info(f"标注目录: {LABELED_DIR}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"模型: {args.model} -> {MODEL_CHOICES[args.model]}")
    logger.info(f"设备: {args.device}")
    logger.info(f"Batch大小: {args.batch_size}s")
    logger.info(f"采样数: {args.sample if args.sample else '全部'}")
    logger.info(f"断点续跑: {args.resume}")
    logger.info(f"日志文件: {log_file}")

    # 1. 加载标签
    text_file = LABELED_DIR / "text.txt"
    if not text_file.exists():
        logger.error(f"标签文件不存在: {text_file}")
        logger.error("请先运行 prepare_funasr_data.py 生成 labeled/ 目录")
        return
    labels = load_labels(text_file, logger)

    # 2. 收集有标签且有wav的条目
    wav_dir = LABELED_DIR / "wav"
    eval_items = []
    missing_wav = 0
    for uid, ref_text in labels.items():
        wav_path = wav_dir / f"{uid}.wav"
        if wav_path.exists():
            eval_items.append((uid, ref_text, wav_path))
        else:
            missing_wav += 1

    logger.info(f"有效评估条目: {len(eval_items)} (缺少wav: {missing_wav})")

    # 3. 采样
    if args.sample and args.sample < len(eval_items):
        random.seed(42)
        eval_items = random.sample(eval_items, args.sample)
        logger.info(f"随机采样 {args.sample} 条进行评估")

    # 4. 断点续跑
    progress_file = output_dir / "eval_progress.txt"
    done_ids = set()
    if args.resume:
        done_ids = load_progress(progress_file)
        logger.info(f"已完成: {len(done_ids)} 条")

    pending = [item for item in eval_items if item[0] not in done_ids]
    logger.info(f"待评估: {len(pending)} 条")

    if not pending:
        logger.info("所有条目已评估完毕")
        # 直接跳到汇总
        generate_report(output_dir, logger)
        return

    # 5. 加载模型
    logger.info("正在加载 FunASR 模型...")
    load_start = time.time()

    from funasr import AutoModel

    model_name = MODEL_CHOICES[args.model]
    model_kwargs = dict(
        model=model_name,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=args.device,
    )
    if args.model == "paraformer":
        model_kwargs["punc_model"] = "ct-punc"

    model = AutoModel(**model_kwargs)
    logger.info(f"模型加载完成，耗时: {format_duration(time.time() - load_start)}")

    # 6. 批量推理 + 评估
    detail_path = output_dir / "eval_detail.jsonl"
    open_mode = 'a' if args.resume and done_ids else 'w'

    total = len(pending)
    processed = 0
    failed = 0
    infer_start = time.time()
    file_batch = args.file_batch

    # 累计统计
    total_sub, total_ins, total_del, total_ref_len = 0, 0, 0, 0
    total_correct_sentences = 0

    logger.info(f"开始评估，每批 {file_batch} 个文件...")

    with open(detail_path, open_mode, encoding='utf-8') as f_detail, \
         open(progress_file, open_mode, encoding='utf-8') as f_prog:

        for batch_start in range(0, total, file_batch):
            batch_end = min(batch_start + file_batch, total)
            batch = pending[batch_start:batch_end]
            batch_paths = [str(item[2]) for item in batch]

            batch_t0 = time.time()
            try:
                if args.model == "sensevoice":
                    from funasr.utils.postprocess_utils import rich_transcription_postprocess
                    results = model.generate(
                        input=batch_paths,
                        cache={},
                        language="auto",
                        use_itn=True,
                        batch_size_s=args.batch_size,
                        merge_vad=True,
                        merge_length_s=15,
                    )
                else:
                    results = model.generate(
                        input=batch_paths,
                        cache={},
                        batch_size_s=args.batch_size,
                    )
            except Exception as e:
                logger.error(f"批次推理失败 [{batch_start}-{batch_end}]: {e}")
                failed += len(batch)
                continue

            batch_time = time.time() - batch_t0

            for i, (uid, ref_text, wav_path) in enumerate(batch):
                try:
                    hyp_text = results[i]["text"]
                    if args.model == "sensevoice":
                        hyp_text = rich_transcription_postprocess(hyp_text)
                except (IndexError, KeyError, TypeError) as e:
                    logger.warning(f"  结果解析失败 {uid}: {e}")
                    failed += 1
                    continue

                # 规范化后计算CER
                ref_norm = normalize_text(ref_text)
                hyp_norm = normalize_text(hyp_text)
                cer, s, ins, d, ref_len = compute_cer(ref_norm, hyp_norm)

                is_correct = (ref_norm == hyp_norm)
                if is_correct:
                    total_correct_sentences += 1

                total_sub += s
                total_ins += ins
                total_del += d
                total_ref_len += ref_len

                # 写入逐条结果
                detail = {
                    "uid": uid,
                    "ref": ref_text,
                    "hyp": hyp_text,
                    "ref_norm": ref_norm,
                    "hyp_norm": hyp_norm,
                    "cer": round(cer, 4),
                    "sub": s,
                    "ins": ins,
                    "del": d,
                    "ref_len": ref_len,
                    "correct": is_correct,
                }
                f_detail.write(json.dumps(detail, ensure_ascii=False) + "\n")
                f_prog.write(f"{uid}\n")
                processed += 1

            f_detail.flush()
            f_prog.flush()

            # 进度日志
            elapsed = time.time() - infer_start
            speed = processed / elapsed if elapsed > 0 else 0
            eta = (total - batch_end) / speed if speed > 0 else 0
            pct = batch_end / total * 100
            running_cer = (total_sub + total_ins + total_del) / total_ref_len if total_ref_len > 0 else 0
            running_ser = 1 - total_correct_sentences / processed if processed > 0 else 0

            logger.info(
                f"进度: {batch_end}/{total} ({pct:.1f}%) | "
                f"本批: {batch_time:.1f}s | "
                f"成功: {processed} 失败: {failed} | "
                f"速度: {speed:.1f}条/s | "
                f"CER: {running_cer:.2%} SER: {running_ser:.2%} | "
                f"已用: {format_duration(elapsed)} 剩余: {format_duration(eta)}"
            )

    # 7. 生成汇总报告
    generate_report(output_dir, logger)


def generate_report(output_dir: Path, logger):
    """从 eval_detail.jsonl 生成汇总报告"""
    detail_path = output_dir / "eval_detail.jsonl"
    if not detail_path.exists():
        logger.warning("未找到评估详情文件")
        return

    logger.info("正在生成评估报告...")

    total_sub, total_ins, total_del, total_ref_len = 0, 0, 0, 0
    total_correct = 0
    total_count = 0
    cer_list = []

    # 按CER区间统计
    cer_buckets = defaultdict(int)  # "0%", "0-5%", "5-10%", "10-20%", "20-50%", ">50%"

    # 收集 bad cases
    bad_cases = []

    with open(detail_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            total_count += 1
            total_sub += d["sub"]
            total_ins += d["ins"]
            total_del += d["del"]
            total_ref_len += d["ref_len"]
            cer = d["cer"]
            cer_list.append(cer)

            if d["correct"]:
                total_correct += 1

            # 分桶
            if cer == 0:
                cer_buckets["0% (完全正确)"] += 1
            elif cer <= 0.05:
                cer_buckets["0-5%"] += 1
            elif cer <= 0.10:
                cer_buckets["5-10%"] += 1
            elif cer <= 0.20:
                cer_buckets["10-20%"] += 1
            elif cer <= 0.50:
                cer_buckets["20-50%"] += 1
            else:
                cer_buckets[">50%"] += 1

            if cer > 0.3:
                bad_cases.append(d)

    if total_count == 0:
        logger.warning("没有评估数据")
        return

    overall_cer = (total_sub + total_ins + total_del) / total_ref_len if total_ref_len > 0 else 0
    ser = 1 - total_correct / total_count
    avg_cer = sum(cer_list) / len(cer_list)

    # 排序 bad cases
    bad_cases.sort(key=lambda x: x["cer"], reverse=True)

    # 输出报告
    report_path = output_dir / "eval_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        def w(s=""):
            f.write(s + "\n")
            logger.info(s)

        w("=" * 60)
        w("S10 ASR 评估报告")
        w(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        w("=" * 60)
        w()
        w(f"评估总数: {total_count}")
        w(f"参考文本总字符数: {total_ref_len}")
        w()
        w("--- 整体指标 ---")
        w(f"CER (字符错误率):  {overall_cer:.2%}")
        w(f"  - 替换 (Sub): {total_sub}")
        w(f"  - 插入 (Ins): {total_ins}")
        w(f"  - 删除 (Del): {total_del}")
        w(f"平均句子CER:       {avg_cer:.2%}")
        w(f"SER (句子错误率):  {ser:.2%}")
        w(f"句子正确率:        {1 - ser:.2%} ({total_correct}/{total_count})")
        w()
        w("--- CER 分布 ---")
        bucket_order = ["0% (完全正确)", "0-5%", "5-10%", "10-20%", "20-50%", ">50%"]
        for bucket in bucket_order:
            cnt = cer_buckets.get(bucket, 0)
            pct = cnt / total_count * 100
            bar = "█" * int(pct / 2)
            w(f"  {bucket:>16s}: {cnt:>6d} ({pct:5.1f}%) {bar}")
        w()
        w(f"--- Bad Cases (CER > 30%, 共 {len(bad_cases)} 条, 展示前30) ---")
        for case in bad_cases[:30]:
            w(f"  [{case['uid']}] CER={case['cer']:.0%}")
            w(f"    标签: {case['ref']}")
            w(f"    识别: {case['hyp']}")
            w()

    logger.info(f"报告已保存: {report_path}")

    # 同时输出一份 JSON 格式的汇总
    summary = {
        "total_count": total_count,
        "total_ref_chars": total_ref_len,
        "overall_cer": round(overall_cer, 4),
        "avg_sentence_cer": round(avg_cer, 4),
        "ser": round(ser, 4),
        "sentence_accuracy": round(1 - ser, 4),
        "substitutions": total_sub,
        "insertions": total_ins,
        "deletions": total_del,
        "correct_sentences": total_correct,
        "cer_distribution": {k: cer_buckets.get(k, 0) for k in bucket_order},
        "bad_case_count": len(bad_cases),
    }
    summary_path = output_dir / "eval_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"汇总JSON: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="S10 ASR 已标注数据评估")
    parser.add_argument(
        '--model', choices=list(MODEL_CHOICES.keys()),
        default='paraformer',
        help='ASR模型 (default: paraformer)'
    )
    parser.add_argument(
        '--device', default='cuda:0',
        help='推理设备 (default: cuda:0)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=300,
        help='动态batch总时长秒 (default: 300)'
    )
    parser.add_argument(
        '--file-batch', type=int, default=500,
        help='每批文件数 (default: 500)'
    )
    parser.add_argument(
        '--sample', type=int, default=None,
        help='随机采样N条评估，不指定则全量'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='断点续跑'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
