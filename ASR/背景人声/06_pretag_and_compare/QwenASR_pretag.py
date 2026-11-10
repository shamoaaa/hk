#!/usr/bin/env python3
"""
使用 Qwen3-ASR 旗舰模型 (1.7B) 对音频数据进行预标注，生成 jsonl。

Qwen3-ASR 特点:
  - 支持 52 种语言和方言，中文方言覆盖 22 种
  - 1.7B 版本在公开基准上达到开源 SOTA
  - 支持 transformers 和 vLLM 两种后端
  - 支持长音频自动切分

依赖安装:
    pip install -U qwen-asr
    # 可选: vLLM 后端加速
    pip install -U qwen-asr[vllm]
    # 可选: FlashAttention 2 加速
    pip install -U flash-attn --no-build-isolation

目录结构:
    wavs/<日期范围>/labelled/source/*.wav   -- 已标注音频
    wavs/<日期范围>/unlabelled/source/*.wav -- 未标注音频

用法:
    # 对 wavs 目录下所有音频进行预标注（递归扫描）
    python QwenASR_pretag.py

    # 指定音频目录
    python QwenASR_pretag.py --wav-dir /data1/wavs

    # 只处理未标注的音频
    python QwenASR_pretag.py --wav-dir /data1/wavs --filter unlabelled

    # 只处理已标注的音频（用于对比）
    python QwenASR_pretag.py --wav-dir /data1/wavs --filter labelled

    # 使用轻量模型
    python QwenASR_pretag.py --model Qwen/Qwen3-ASR-0.6B

    # vLLM 加速
    python QwenASR_pretag.py --backend vllm

    # 断点续跑
    python QwenASR_pretag.py --resume

    # 采样快速验证
    python QwenASR_pretag.py --sample 100
"""

import os
import time
import json
import argparse
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from pretag_utils import (
    setup_logging, format_duration, scan_wav_files, log_duration_stats,
    load_progress, PretagWriter, SpeedMonitor, release_gpu, print_summary,
    select_gpus, prepare_pretag, run_inference_loop, get_wav_duration,
    SCAN_CACHE_FILE, SCAN_WORKERS,
)

# ============ 配置 ============
BASE_DIR = Path(r'/data1/s10/')
DEFAULT_WAV_DIR = BASE_DIR / "wavs"
DEFAULT_OUTPUT_DIR = BASE_DIR / "qwen_pretag_output"
DEFAULT_MODEL = "/data1/pretrained_models/Qwen3-ASR-1.7B"  # 本地模型路径，也支持 HF ID 如 "Qwen/Qwen3-ASR-1.7B"
# ==============================


def scan_wav_files_recursive(wav_dir: Path, logger, no_cache: bool = False,
                              filter_type: str = None) -> list:
    """
    递归扫描 wav 目录下所有 .wav 文件，支持缓存和过滤。

    目录结构:
        wavs/<日期范围>/labelled/source/*.wav
        wavs/<日期范围>/unlabelled/source/*.wav

    Args:
        wav_dir: 音频根目录
        logger: 日志器
        no_cache: 是否忽略缓存
        filter_type: 过滤类型
            - None: 扫描所有 wav 文件
            - "labelled": 只扫描 labelled/source/ 下的文件
            - "unlabelled": 只扫描 unlabelled/source/ 下的文件

    Returns:
        [(uid, path, duration), ...] 列表
    """
    cache_suffix = f"_{filter_type}" if filter_type else ""
    cache_path = wav_dir / f"wav_scan_cache{cache_suffix}.json"

    if cache_path.exists() and not no_cache:
        logger.info(f"从缓存加载扫描结果: {cache_path}")
        t0 = time.time()
        with open(cache_path, 'r', encoding='utf-8') as f:
            items = [tuple(x) for x in json.load(f)]
        logger.info(f"缓存加载 {len(items)} 条，耗时: {time.time()-t0:.1f}s")
        return items
    elif no_cache and cache_path.exists():
        logger.info("已指定 --no-cache，忽略缓存，重新扫描")

    logger.info(f"递归扫描音频目录: {wav_dir}")
    if filter_type:
        logger.info(f"过滤模式: {filter_type}")
    t0 = time.time()

    all_paths = []
    for root, dirs, files in os.walk(wav_dir):
        # 过滤逻辑
        if filter_type:
            root_lower = root.replace("\\", "/").lower()
            if filter_type == "labelled" and "/unlabelled/" in root_lower:
                continue
            if filter_type == "unlabelled" and "/labelled/" in root_lower:
                continue
            # 确保匹配目标路径
            if filter_type == "labelled" and "/labelled/" not in root_lower:
                # 允许顶层目录通过（继续递归）
                if not any(d.lower() in ("labelled", "unlabelled") for d in dirs):
                    pass  # 非叶子目录，继续
                elif "labelled" not in [d.lower() for d in dirs]:
                    continue
            if filter_type == "unlabelled" and "/unlabelled/" not in root_lower:
                if not any(d.lower() in ("labelled", "unlabelled") for d in dirs):
                    pass
                elif "unlabelled" not in [d.lower() for d in dirs]:
                    continue

        for f in files:
            if f.lower().endswith('.wav'):
                all_paths.append(os.path.join(root, f))

    logger.info(f"发现 {len(all_paths)} 个 wav 文件，多线程读取时长...")

    def _process(path):
        uid = Path(path).stem
        dur = get_wav_duration(path)
        return (uid, path, dur) if dur > 0 else None

    items = []
    skipped = 0
    with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as pool:
        for result in pool.map(_process, all_paths):
            if result:
                items.append(result)
            else:
                skipped += 1

    scan_time = time.time() - t0
    logger.info(f"有效 wav 文件: {len(items)} 个 (跳过 {skipped} 个无效)，扫描耗时: {scan_time:.1f}s")

    # 写入缓存
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(items, f)
        logger.info(f"扫描结果已缓存: {cache_path}")
    except Exception as e:
        logger.warning(f"缓存写入失败: {e}")

    return items


def load_model(args, logger):
    """加载 Qwen3-ASR 模型，返回 (model, raw_model, processor, use_raw)"""
    logger.info(f"正在加载 Qwen3-ASR 模型 ({args.backend} 后端)...")
    t0 = time.time()

    use_raw = (args.backend == "raw")

    if use_raw:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        raw_model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
        logger.info("使用 raw transformers 后端（不依赖 qwen-asr 包）")
        logger.info(f"模型加载完成，耗时: {format_duration(time.time() - t0)}")
        return None, raw_model, processor, True
    else:
        import torch
        from qwen_asr import Qwen3ASRModel

        if args.backend == "vllm":
            model = Qwen3ASRModel.LLM(
                model=args.model, gpu_memory_utilization=0.8,
                max_inference_batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            model = Qwen3ASRModel.from_pretrained(
                args.model, dtype=torch.bfloat16, device_map="auto",
                max_inference_batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
            )
        logger.info(f"模型加载完成，耗时: {format_duration(time.time() - t0)}")
        return model, None, None, False


def raw_transcribe_batch(paths, processor, raw_model, max_new_tokens):
    """用 raw transformers 逐条推理"""
    import torch
    import librosa
    texts = []
    for p in paths:
        audio, sr = librosa.load(p, sr=processor.feature_extractor.sampling_rate)
        conversation = [{"role": "user", "content": [{"type": "audio", "audio": p}]}]
        text_input = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(
            text=text_input, audios=[audio], sampling_rate=sr,
            return_tensors="pt", padding=True
        ).to(raw_model.device)
        with torch.no_grad():
            output_ids = raw_model.generate(**inputs, max_new_tokens=max_new_tokens)
        gen_ids = output_ids[:, inputs.input_ids.shape[1]:]
        text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        texts.append(text)
    return texts


def transcribe_batch(batch_paths, model, raw_model, processor, use_raw, lang_arg, max_new_tokens):
    """统一推理接口，返回文本列表"""
    if use_raw:
        return raw_transcribe_batch(batch_paths, processor, raw_model, max_new_tokens)
    else:
        results = model.transcribe(audio=batch_paths, language=lang_arg)
        return [r.text for r in results]


def process_batches(items, batch_size, writer, monitor, infer_start, total,
                    model, raw_model, processor, use_raw, lang_arg, max_new_tokens, logger):
    """批量推理主循环，返回是否需要重启"""
    for batch_start in range(0, len(items), batch_size):
        batch_end = min(batch_start + batch_size, len(items))
        batch = items[batch_start:batch_end]
        batch_paths = [p for _, p, _ in batch]

        batch_t0 = time.time()
        try:
            texts = transcribe_batch(batch_paths, model, raw_model, processor,
                                     use_raw, lang_arg, max_new_tokens)
        except Exception as e:
            logger.error(f"批次推理失败 [{batch_start}-{batch_end}]: {e}")
            logger.info("回退到逐条处理...")
            for uid, p, d in batch:
                try:
                    t = transcribe_batch([p], model, raw_model, processor,
                                         use_raw, lang_arg, max_new_tokens)
                    writer.write(uid, p, t[0])
                except Exception as e2:
                    logger.warning(f"  单条失败 {uid}: {e2}")
                    writer.failed += 1
            writer.flush()
            batch_time = time.time() - batch_t0
            writer.log_progress(batch_end, total, batch_time, len(batch), infer_start, logger)
            continue

        batch_time = time.time() - batch_t0

        for i, (uid, p, d) in enumerate(batch):
            try:
                writer.write(uid, p, texts[i])
            except (IndexError, AttributeError) as e:
                logger.warning(f"  结果解析失败 {uid}: {e}")
                writer.failed += 1

        writer.flush()
        writer.log_progress(batch_end, total, batch_time, len(batch), infer_start, logger)

        if monitor.check(batch_time, batch):
            return True

    return False


def run_pretag(args):
    output_dir = Path(args.output_dir)
    logger, log_file = setup_logging(output_dir, 'qwen_pretag')

    logger.info("=" * 60)
    logger.info("Qwen3-ASR 音频预标注")
    logger.info("=" * 60)
    logger.info(f"音频目录:   {args.wav_dir}")
    logger.info(f"输出目录:   {output_dir}")
    logger.info(f"模型:       {args.model}")
    logger.info(f"后端:       {args.backend}")
    logger.info(f"批大小:     {args.batch_size}")
    logger.info(f"语言:       {args.language or '自动检测'}")
    logger.info(f"过滤:       {args.filter or '全部'}")
    logger.info(f"断点续跑:   {args.resume}")
    logger.info(f"日志文件:   {log_file}")

    # 1. 递归扫描 + 断点续跑
    wav_items = scan_wav_files_recursive(
        Path(args.wav_dir), logger,
        no_cache=args.no_cache, filter_type=args.filter
    )
    if not wav_items:
        logger.warning("未找到任何 wav 文件，退出")
        return
    log_duration_stats(wav_items, logger)

    progress_file = output_dir / "progress.txt"
    done_ids = set()
    if args.resume:
        done_ids = load_progress(progress_file)
        logger.info(f"已完成: {len(done_ids)} 条，将跳过")

    pending = [(uid, p, d) for uid, p, d in wav_items if uid not in done_ids]
    logger.info(f"待处理: {len(pending)} 条")

    if not pending:
        if wav_items:
            logger.info("所有文件已处理完毕")
        return

    # 采样模式
    if args.sample and args.sample < len(pending):
        random.seed(42)
        pending = random.sample(pending, args.sample)
        logger.info(f"采样模式: 随机抽取 {args.sample} 条")

    # 2. 加载模型
    select_gpus(args.device, logger)
    model, raw_model, processor, use_raw = load_model(args, logger)
    lang_arg = args.language if args.language else None

    total = len(pending)

    # 3. 定义回调
    def process_fn(items, writer, monitor, infer_start, total_, logger_):
        return process_batches(items, args.batch_size, writer, monitor, infer_start, total_,
                               model, raw_model, processor, use_raw, lang_arg,
                               args.max_new_tokens, logger_)

    def reload_fn():
        nonlocal model, raw_model, processor, use_raw
        logger.info("释放 GPU 显存...")
        del model, raw_model
        release_gpu()
        time.sleep(5)
        model, raw_model, processor, use_raw = load_model(args, logger)

    # 4. 执行推理
    run_inference_loop(
        items=pending, total=total,
        output_dir=output_dir, progress_file=progress_file,
        resume=args.resume, done_ids=done_ids,
        batch_size=args.batch_size, max_slow_batches=args.max_slow_batches,
        process_fn=process_fn, reload_fn=reload_fn, logger=logger,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR 音频预标注",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 对 wavs 目录下所有音频预标注
  python QwenASR_pretag.py

  # 只处理未标注的音频
  python QwenASR_pretag.py --filter unlabelled

  # 只处理已标注的音频（用于对比验证）
  python QwenASR_pretag.py --filter labelled

  # 采样快速验证
  python QwenASR_pretag.py --sample 100

  # vLLM 加速
  python QwenASR_pretag.py --backend vllm

  # 断点续跑
  python QwenASR_pretag.py --resume
        """,
    )
    parser.add_argument('--wav-dir', default=str(DEFAULT_WAV_DIR),
                        help=f'wav 文件根目录，递归扫描 (default: {DEFAULT_WAV_DIR})')
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR), help='输出目录')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help=f'模型路径或 HF ID (default: {DEFAULT_MODEL})')
    parser.add_argument('--backend', choices=['transformers', 'vllm', 'raw'], default='transformers',
                        help='推理后端 (default: transformers)')
    parser.add_argument('--batch-size', type=int, default=128, help='批大小 (default: 128)')
    parser.add_argument('--max-new-tokens', type=int, default=256, help='最大生成token数 (default: 256)')
    parser.add_argument('--language', default=None, help='强制语言，不指定则自动检测')
    parser.add_argument('--filter', choices=['labelled', 'unlabelled'], default=None,
                        help='过滤类型: labelled=只扫描已标注, unlabelled=只扫描未标注, 不指定=全部')
    parser.add_argument('--sample', type=int, default=None,
                        help='随机采样 N 条（快速验证用）')
    parser.add_argument('--device', default='cuda:auto',
                        help='设备: cuda:0, cuda:0,1, cuda:auto, cuda:auto:2, cpu (default: cuda:auto)')
    parser.add_argument('--resume', action='store_true', help='断点续跑')
    parser.add_argument('--no-cache', action='store_true', help='忽略扫描缓存')
    parser.add_argument('--max-slow-batches', type=int, default=9,
                        help='连续慢批次数达到此值触发自动重启 (default: 9)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pretag(args)
