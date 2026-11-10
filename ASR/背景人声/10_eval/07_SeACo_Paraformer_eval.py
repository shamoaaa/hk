#!/usr/bin/env python3
"""
使用离线 SeACo-Paraformer 模型评估已标注数据，计算 CER。

模型: speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
支持热词(hotword)增强。

依赖:
    pip install funasr torch torchaudio

用法:
    python SeACo_Paraformer_eval.py --wav-dir /path/to/wav --label /path/to/label.jsonl
    python SeACo_Paraformer_eval.py --wav-dir /path/to/wav --label /path/to/text.txt
    python SeACo_Paraformer_eval.py --wav-dir /path/to/dir1 /path/to/dir2 /path/to/dir3
    python SeACo_Paraformer_eval.py --hotword hotwords.txt
    python SeACo_Paraformer_eval.py --sample 1000
    python SeACo_Paraformer_eval.py --resume

多目录模式:
  指定多个 --wav-dir 时，每个目录独立评估并在 output-dir 下生成子目录报告，
  最后汇总所有目录的性能到 combined_report.txt / combined_summary.json。

标签格式支持:
  - jsonl: 每行 {"uid": "xxx", "text": "你好"} 或含 messages 的训练格式
  - text.txt: 每行 "uid 文本内容"
"""

import os
import sys
import json
import re
import time
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# 复用 pretag_utils 中的 GPU 选择
sys.path.insert(0, str(Path(__file__).parent / "06_eval_and_pretag"))
try:
    from pretag_utils import select_gpus
except ImportError:
    # 回退: 如果 pretag_utils 不在路径中，内联一个简单版本
    def select_gpus(device_arg, logger=None):
        if logger:
            logger.info(f"使用设备: {device_arg}")
        return device_arg

# ============ 配置 ============
BASE_DIR = Path(r'E:\8.音频\03 s10\s10_data_reflow')
DEFAULT_MODEL_DIR = r'E:\8.音频\02 ASR入门\models\speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
DEFAULT_WAV_DIR = BASE_DIR / "labeled" / "wav"
DEFAULT_LABEL = BASE_DIR / "labeled" / "text.txt"
DEFAULT_OUTPUT_DIR = BASE_DIR / "seaco_eval_output"
DEFAULT_HOTWORD = BASE_DIR / "hotwords" / "hotwords.txt"
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

    logger = logging.getLogger('seaco_eval')
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger, log_file


def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


# ============ CER 计算 ============

def edit_distance(ref: list, hyp: list):
    """编辑距离，返回 (substitutions, insertions, deletions)"""
    n, m = len(ref), len(hyp)
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
                    (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                    (ins[0] + 1, ins[1], ins[2] + 1, ins[3]),
                    (dele[0] + 1, dele[1], dele[2], dele[3] + 1),
                ]
                dp[i][j] = min(candidates, key=lambda x: x[0])

    cost, s, i, d = dp[n][m]
    return s, i, d


def compute_cer(reference: str, hypothesis: str):
    """计算 CER，返回 (cer, sub, ins, del, ref_len)"""
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))

    if len(ref_chars) == 0:
        return (1.0 if len(hyp_chars) > 0 else 0.0), 0, len(hyp_chars), 0, 0

    s, i, d = edit_distance(ref_chars, hyp_chars)
    cer = (s + i + d) / len(ref_chars)
    return cer, s, i, d, len(ref_chars)


def normalize_text(text: str) -> str:
    """文本规范化：去除标点、转小写、合并空格"""
    text = re.sub(r'[，。！？、；：""''（）《》【】\[\]{},.!?;:\'"()\-—…·]', '', text)
    text = text.lower().strip()
    text = re.sub(r'\s+', '', text)
    return text


# ============ 数据加载 ============

def load_labels(label_path: Path, logger) -> dict:
    """
    加载标签，支持两种格式:
      - text.txt: "uid 文本内容"
      - jsonl: {"uid":"xxx","text":"你好"} 或训练格式(从messages提取)
    返回 {uid: text}
    """
    logger.info(f"加载标签: {label_path}")
    labels = {}
    suffix = label_path.suffix.lower()

    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if suffix == '.jsonl' or suffix == '.json':
                try:
                    obj = json.loads(line)
                    # 格式1: {"uid": "xxx", "text": "你好"}
                    if "uid" in obj and "text" in obj:
                        labels[obj["uid"]] = obj["text"]
                    # 格式2: 训练格式，从 messages 中提取
                    elif "messages" in obj:
                        msgs = obj["messages"]
                        # uid 从 user content 的路径提取
                        user_content = msgs[1]["content"]
                        wav_part = user_content.split("!")[-1].split("<|endofspeech|>")[0]
                        uid = Path(wav_part).stem
                        text = msgs[-1]["content"]  # assistant 的回复
                        labels[uid] = text
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
            else:
                # text.txt 格式: "uid 文本"
                parts = line.split(None, 1)
                if len(parts) >= 2:
                    labels[parts[0]] = parts[1]

    logger.info(f"加载标签: {len(labels)} 条")
    return labels


def load_hotwords(hotword_path: Path, logger) -> str:
    """加载热词文件，返回空格分隔的热词字符串"""
    if not hotword_path or not hotword_path.exists():
        return ""
    words = []
    with open(hotword_path, 'r', encoding='utf-8') as f:
        for line in f:
            w = line.strip()
            if w:
                words.append(w)
    logger.info(f"加载热词: {len(words)} 个")
    return " ".join(words)


def load_progress(progress_file: Path) -> set:
    done = set()
    if progress_file.exists():
        with open(progress_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(line)
    return done


# ============ 主流程 ============

def run_eval_single_dir(wav_dir: Path, labels: dict, model, hotword_str: str,
                        output_dir: Path, args, logger) -> dict:
    """对单个 wav 目录执行评估，返回该目录的 summary dict（供汇总使用）"""
    output_dir.mkdir(parents=True, exist_ok=True)
    dir_name = wav_dir.name
    logger.info(f"\n{'=' * 60}")
    logger.info(f"评估目录: {wav_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"{'=' * 60}")

    # 1. 匹配 wav 文件
    eval_items = []
    missing_wav = 0
    for uid, ref_text in labels.items():
        wav_path = wav_dir / f"{uid}.wav"
        if wav_path.exists():
            eval_items.append((uid, ref_text, str(wav_path)))
        else:
            missing_wav += 1

    logger.info(f"[{dir_name}] 有效评估条目: {len(eval_items)} (缺少wav: {missing_wav})")

    if not eval_items:
        logger.warning(f"[{dir_name}] 没有可评估的条目，跳过")
        return None

    # 2. 采样
    if args.sample and args.sample < len(eval_items):
        random.seed(42)
        eval_items = random.sample(eval_items, args.sample)
        logger.info(f"[{dir_name}] 随机采样 {args.sample} 条")

    # 3. 断点续跑
    progress_file = output_dir / "progress.txt"
    done_ids = set()
    if args.resume:
        done_ids = load_progress(progress_file)
        logger.info(f"[{dir_name}] 已完成: {len(done_ids)} 条")

    pending = [item for item in eval_items if item[0] not in done_ids]
    logger.info(f"[{dir_name}] 待评估: {len(pending)} 条")

    if not pending:
        logger.info(f"[{dir_name}] 所有条目已评估完毕")
        generate_report(output_dir, logger)
        return _load_summary(output_dir)

    # 4. 批量推理 + 评估
    detail_path = output_dir / "eval_detail.jsonl"
    open_mode = 'a' if args.resume and done_ids else 'w'

    total = len(pending)
    processed = 0
    failed = 0
    infer_start = time.time()
    file_batch = args.file_batch

    total_sub, total_ins, total_del, total_ref_len = 0, 0, 0, 0
    total_correct = 0

    logger.info(f"[{dir_name}] 开始评估，每批 {file_batch} 个文件...")

    with open(detail_path, open_mode, encoding='utf-8') as f_detail, \
         open(progress_file, open_mode, encoding='utf-8') as f_prog:

        for batch_start in range(0, total, file_batch):
            batch_end = min(batch_start + file_batch, total)
            batch = pending[batch_start:batch_end]
            batch_paths = [p for _, _, p in batch]

            batch_t0 = time.time()
            try:
                gen_kwargs = dict(
                    input=batch_paths,
                    cache={},
                    batch_size_s=args.batch_size,
                )
                if hotword_str:
                    gen_kwargs["hotword"] = hotword_str
                results = model.generate(**gen_kwargs)
            except Exception as e:
                logger.error(f"[{dir_name}] 批次推理失败 [{batch_start}-{batch_end}]: {e}")
                failed += len(batch)
                continue

            batch_time = time.time() - batch_t0

            for i, (uid, ref_text, wav_path) in enumerate(batch):
                try:
                    hyp_text = results[i]["text"]
                except (IndexError, KeyError, TypeError) as e:
                    logger.warning(f"  结果解析失败 {uid}: {e}")
                    failed += 1
                    continue

                ref_norm = normalize_text(ref_text)
                hyp_norm = normalize_text(hyp_text)
                cer, s, ins, d, ref_len = compute_cer(ref_norm, hyp_norm)

                is_correct = (ref_norm == hyp_norm)
                if is_correct:
                    total_correct += 1
                total_sub += s
                total_ins += ins
                total_del += d
                total_ref_len += ref_len

                detail = {
                    "uid": uid,
                    "ref": ref_text,
                    "hyp": hyp_text,
                    "ref_norm": ref_norm,
                    "hyp_norm": hyp_norm,
                    "cer": round(cer, 4),
                    "sub": s, "ins": ins, "del": d,
                    "ref_len": ref_len,
                    "correct": is_correct,
                }
                f_detail.write(json.dumps(detail, ensure_ascii=False) + "\n")
                f_prog.write(f"{uid}\n")
                processed += 1

            f_detail.flush()
            f_prog.flush()

            # 进度
            elapsed = time.time() - infer_start
            speed = processed / elapsed if elapsed > 0 else 0
            eta = (total - batch_end) / speed if speed > 0 else 0
            pct = batch_end / total * 100
            running_cer = (total_sub + total_ins + total_del) / total_ref_len if total_ref_len > 0 else 0
            running_ser = 1 - total_correct / processed if processed > 0 else 0

            logger.info(
                f"[{dir_name}] 进度: {batch_end}/{total} ({pct:.1f}%) | "
                f"本批: {batch_time:.1f}s | "
                f"成功: {processed} 失败: {failed} | "
                f"速度: {speed:.1f}条/s | "
                f"CER: {running_cer:.2%} SER: {running_ser:.2%} | "
                f"已用: {format_duration(elapsed)} 剩余: {format_duration(eta)}"
            )

    # 5. 生成单目录报告
    generate_report(output_dir, logger)
    return _load_summary(output_dir)


def _load_summary(output_dir: Path) -> dict:
    """加载已生成的 eval_summary.json"""
    summary_path = output_dir / "eval_summary.json"
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def generate_combined_report(dir_summaries: dict, output_dir: Path, logger):
    """
    汇总多个目录的评估结果。
    dir_summaries: {dir_name: summary_dict}
    """
    logger.info("\n" + "=" * 60)
    logger.info("生成多目录汇总报告")
    logger.info("=" * 60)

    # 累加所有目录的指标
    g_sub, g_ins, g_del, g_ref_len = 0, 0, 0, 0
    g_correct, g_count = 0, 0
    cer_sum = 0.0

    for name, s in dir_summaries.items():
        g_sub += s["substitutions"]
        g_ins += s["insertions"]
        g_del += s["deletions"]
        g_ref_len += s["total_ref_chars"]
        g_correct += s["correct_sentences"]
        g_count += s["total_count"]
        cer_sum += s["avg_sentence_cer"] * s["total_count"]

    g_cer = (g_sub + g_ins + g_del) / g_ref_len if g_ref_len > 0 else 0
    g_ser = 1 - g_correct / g_count if g_count > 0 else 0
    g_avg_cer = cer_sum / g_count if g_count > 0 else 0

    bucket_order = ["0% (完全正确)", "0-5%", "5-10%", "10-20%", "20-50%", ">50%"]

    report_path = output_dir / "combined_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        def w(s=""):
            f.write(s + "\n")
            logger.info(s)

        w("=" * 70)
        w("SeACo-Paraformer 多目录汇总评估报告")
        w(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        w(f"评估目录数: {len(dir_summaries)}")
        w("=" * 70)

        # 各目录对比表
        w()
        w("--- 各目录性能对比 ---")
        w(f"{'目录':<30s} {'数量':>8s} {'CER':>8s} {'SER':>8s} {'正确率':>8s} {'字符数':>10s}")
        w("-" * 74)
        for name, s in dir_summaries.items():
            w(f"{name:<30s} {s['total_count']:>8d} "
              f"{s['overall_cer']:>7.2%} {s['ser']:>7.2%} "
              f"{s['sentence_accuracy']:>7.2%} {s['total_ref_chars']:>10d}")
        w("-" * 74)
        w(f"{'【汇总】':<30s} {g_count:>8d} "
          f"{g_cer:>7.2%} {g_ser:>7.2%} "
          f"{1 - g_ser:>7.2%} {g_ref_len:>10d}")

        # 汇总指标
        w()
        w("--- 汇总整体指标 ---")
        w(f"评估总数:          {g_count}")
        w(f"参考文本总字符数:  {g_ref_len}")
        w(f"CER (字符错误率):  {g_cer:.2%}")
        w(f"  - 替换 (Sub): {g_sub}")
        w(f"  - 插入 (Ins): {g_ins}")
        w(f"  - 删除 (Del): {g_del}")
        w(f"平均句子CER:       {g_avg_cer:.2%}")
        w(f"SER (句子错误率):  {g_ser:.2%}")
        w(f"句子正确率:        {1 - g_ser:.2%} ({g_correct}/{g_count})")

        # 汇总 CER 分布
        w()
        w("--- 汇总 CER 分布 ---")
        g_buckets = defaultdict(int)
        for name, s in dir_summaries.items():
            for bucket in bucket_order:
                g_buckets[bucket] += s.get("cer_distribution", {}).get(bucket, 0)
        for bucket in bucket_order:
            cnt = g_buckets[bucket]
            pct = cnt / g_count * 100 if g_count > 0 else 0
            bar = "█" * int(pct / 2)
            w(f"  {bucket:>16s}: {cnt:>6d} ({pct:5.1f}%) {bar}")

    logger.info(f"汇总报告已保存: {report_path}")

    # 汇总 JSON
    combined_summary = {
        "total_dirs": len(dir_summaries),
        "total_count": g_count,
        "total_ref_chars": g_ref_len,
        "overall_cer": round(g_cer, 4),
        "avg_sentence_cer": round(g_avg_cer, 4),
        "ser": round(g_ser, 4),
        "sentence_accuracy": round(1 - g_ser, 4),
        "substitutions": g_sub,
        "insertions": g_ins,
        "deletions": g_del,
        "correct_sentences": g_correct,
        "cer_distribution": {k: g_buckets[k] for k in bucket_order},
        "per_directory": {name: s for name, s in dir_summaries.items()},
    }
    summary_path = output_dir / "combined_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(combined_summary, f, ensure_ascii=False, indent=2)
    logger.info(f"汇总JSON: {summary_path}")


def run_eval(args):
    output_dir = Path(args.output_dir)
    logger, log_file = setup_logging(output_dir)

    wav_dirs = [Path(d) for d in args.wav_dir]
    multi_mode = len(wav_dirs) > 1

    logger.info("=" * 60)
    logger.info("SeACo-Paraformer 离线模型评估")
    logger.info("=" * 60)
    logger.info(f"模型目录:   {args.model_dir}")
    logger.info(f"音频目录:   {[str(d) for d in wav_dirs]}")
    logger.info(f"标签文件:   {args.label}")
    logger.info(f"输出目录:   {output_dir}")
    logger.info(f"设备:       {args.device}")
    logger.info(f"Batch大小:  {args.batch_size}s")
    logger.info(f"热词文件:   {args.hotword or '无'}")
    logger.info(f"采样数:     {args.sample or '全部'}")
    logger.info(f"断点续跑:   {args.resume}")
    logger.info(f"多目录模式: {multi_mode}")
    logger.info(f"日志文件:   {log_file}")

    # 1. 加载标签
    label_path = Path(args.label)
    if not label_path.exists():
        logger.error(f"标签文件不存在: {label_path}")
        return
    labels = load_labels(label_path, logger)

    # 2. 加载热词
    hotword_str = ""
    if args.hotword:
        hotword_str = load_hotwords(Path(args.hotword), logger)

    # 3. 加载模型（只加载一次）
    device = select_gpus(args.device, logger)
    logger.info("正在加载 SeACo-Paraformer 模型...")
    load_start = time.time()

    from funasr import AutoModel

    model = AutoModel(
        model=args.model_dir,
        device=device,
    )
    logger.info(f"模型加载完成，耗时: {format_duration(time.time() - load_start)}")

    # 4. 逐目录评估
    dir_summaries = {}  # {dir_name: summary_dict}

    for wav_dir in wav_dirs:
        if not wav_dir.exists():
            logger.warning(f"目录不存在，跳过: {wav_dir}")
            continue

        dir_name = wav_dir.name
        if multi_mode:
            sub_output_dir = output_dir / dir_name
        else:
            sub_output_dir = output_dir

        summary = run_eval_single_dir(
            wav_dir=wav_dir,
            labels=labels,
            model=model,
            hotword_str=hotword_str,
            output_dir=sub_output_dir,
            args=args,
            logger=logger,
        )
        if summary:
            dir_summaries[dir_name] = summary

    # 5. 多目录时生成汇总报告
    if multi_mode and dir_summaries:
        generate_combined_report(dir_summaries, output_dir, logger)
    elif not dir_summaries:
        logger.error("所有目录均无可评估数据")


def generate_report(output_dir: Path, logger):
    """从 eval_detail.jsonl 生成汇总报告"""
    detail_path = output_dir / "eval_detail.jsonl"
    if not detail_path.exists():
        logger.warning("未找到评估详情文件")
        return

    logger.info("生成评估报告...")

    total_sub, total_ins, total_del, total_ref_len = 0, 0, 0, 0
    total_correct = 0
    total_count = 0
    cer_list = []
    cer_buckets = defaultdict(int)
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
    bad_cases.sort(key=lambda x: x["cer"], reverse=True)

    report_path = output_dir / "eval_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        def w(s=""):
            f.write(s + "\n")
            logger.info(s)

        w("=" * 60)
        w("SeACo-Paraformer 评估报告")
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
    parser = argparse.ArgumentParser(description="SeACo-Paraformer 离线模型评估")
    parser.add_argument(
        '--model-dir', default=DEFAULT_MODEL_DIR,
        help='离线模型目录'
    )
    parser.add_argument(
        '--wav-dir', nargs='+', default=[str(DEFAULT_WAV_DIR)],
        help='wav文件目录，可指定多个目录分别评估并汇总'
    )
    parser.add_argument(
        '--label', default=str(DEFAULT_LABEL),
        help='标签文件路径 (text.txt 或 jsonl)'
    )
    parser.add_argument(
        '--output-dir', default=str(DEFAULT_OUTPUT_DIR),
        help='输出目录'
    )
    parser.add_argument(
        '--hotword', default=str(DEFAULT_HOTWORD),
        help='热词文件路径，不需要热词传空字符串'
    )
    parser.add_argument(
        '--device', default='cuda:auto',
        help='设备: cuda:0, cuda:0,1, cuda:auto, cuda:auto:2, cpu (default: cuda:auto)'
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
        help='随机采样N条评估'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='断点续跑'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
