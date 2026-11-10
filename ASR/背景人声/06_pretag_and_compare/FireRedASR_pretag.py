#!/usr/bin/env python3
"""
使用 FireRedASR-LLM (旗舰模型) 对音频数据进行预标注，生成 jsonl。

FireRedASR-LLM 注意事项:
  - 最大支持 30s 音频输入
  - 批量推理时要求同批次音频时长相近，否则短音频可能出现重复
  - 模型参数 8.3B，需要 ~16GB 显存

依赖:
    pip install -r requirements.txt  (FireRedASR 仓库的依赖)

前置准备:
    1. git clone https://github.com/FireRedTeam/FireRedASR.git
    2. 下载模型到 pretrained_models/FireRedASR-LLM-L/
    3. 下载 Qwen2-7B-Instruct 到 pretrained_models/
    4. cd pretrained_models/FireRedASR-LLM-L && ln -s ../Qwen2-7B-Instruct

用法:
    python FireRedASR_pretag.py --wav-dir /data/share/dataset/s10_data_reflow/unlabeled
    python FireRedASR_pretag.py --wav-dir /path/to/wav --model-dir /path/to/FireRedASR-LLM-L
    python FireRedASR_pretag.py --asr-type aed --model-dir /path/to/FireRedASR-AED-L
    python FireRedASR_pretag.py --batch-size 4 --resume
"""

import time
import argparse
from pathlib import Path
from collections import defaultdict

from pretag_utils import (
    setup_logging, format_duration, scan_wav_files, log_duration_stats,
    load_progress, PretagWriter, SpeedMonitor, release_gpu, print_summary,
    select_gpus, prepare_pretag, run_inference_loop,
)

# ============ 配置 ============
BASE_DIR = Path(r'/data/share/dataset/s10_data_reflow/')
DEFAULT_WAV_DIR = BASE_DIR / "unlabeled"
DEFAULT_OUTPUT_DIR = BASE_DIR / "firered_pretag_output"
DEFAULT_MODEL_DIR = r"/data/share/pretrained_models/FireRedASR-LLM-L"
# ==============================


def create_duration_buckets(items: list, max_dur: float, bucket_width: float = 5.0):
    """按时长分桶（FireRedASR-LLM 要求同批次时长相近）"""
    buckets = defaultdict(list)
    over_limit = []
    for uid, path, dur in items:
        if dur > max_dur:
            over_limit.append((uid, path, dur))
        else:
            key = int(dur / bucket_width)
            buckets[key].append((uid, path, dur))

    sorted_items = []
    for key in sorted(buckets.keys()):
        bucket = sorted(buckets[key], key=lambda x: x[2])
        sorted_items.extend(bucket)

    return sorted_items, over_limit


def load_model(asr_type, model_dir, logger):
    """加载 FireRedASR 模型"""
    logger.info(f"正在加载 FireRedASR 模型 ({asr_type})...")
    t0 = time.time()
    from fireredasr.models.fireredasr import FireRedAsr
    model = FireRedAsr.from_pretrained(asr_type, model_dir)
    logger.info(f"模型加载完成，耗时: {format_duration(time.time() - t0)}")
    return model


def get_decode_params(asr_type, device):
    """获取解码参数"""
    use_gpu = 1 if device.startswith("cuda") else 0
    if asr_type == "llm":
        return {
            "use_gpu": use_gpu, "beam_size": 3,
            "decode_max_len": 200, "decode_min_len": 0,
            "repetition_penalty": 3.0, "llm_length_penalty": 1.0,
            "temperature": 1.0,
        }
    else:
        return {
            "use_gpu": use_gpu, "beam_size": 3, "nbest": 1,
            "decode_max_len": 300, "softmax_smoothing": 1.25,
            "aed_length_penalty": 0.6, "eos_penalty": 1.0,
        }


def process_batches(model, params, items, batch_size, writer, monitor,
                    infer_start, total, logger):
    """批量推理主循环，返回是否需要重启"""
    for batch_start in range(0, len(items), batch_size):
        batch_end = min(batch_start + batch_size, len(items))
        batch = items[batch_start:batch_end]
        batch_uids = [uid for uid, _, _ in batch]
        batch_paths = [p for _, p, _ in batch]

        batch_t0 = time.time()
        try:
            results = model.transcribe(batch_uids, batch_paths, params)
        except Exception as e:
            logger.error(f"批次推理失败 [{batch_start}-{batch_end}]: {e}")
            logger.info("回退到逐条处理...")
            for uid, p, d in batch:
                try:
                    r = model.transcribe([uid], [p], params)
                    writer.write(uid, p, r[0]["text"])
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
                writer.write(uid, p, results[i]["text"])
            except (IndexError, KeyError, TypeError) as e:
                logger.warning(f"  结果解析失败 {uid}: {e}")
                writer.failed += 1

        writer.flush()
        writer.log_progress(batch_end, total, batch_time, len(batch), infer_start, logger)

        # 速度监控
        if monitor.check(batch_time, batch):
            return True  # 需要重启

    return False


def run_pretag(args):
    output_dir = Path(args.output_dir)
    logger, log_file = setup_logging(output_dir, 'firered_pretag')

    asr_type = args.asr_type
    max_audio_dur = 30.0 if asr_type == "llm" else 60.0

    logger.info("=" * 60)
    logger.info("FireRedASR 音频预标注")
    logger.info("=" * 60)
    logger.info(f"音频目录:   {args.wav_dir}")
    logger.info(f"输出目录:   {output_dir}")
    logger.info(f"模型目录:   {args.model_dir}")
    logger.info(f"模型类型:   {asr_type} ({'旗舰LLM' if asr_type == 'llm' else 'AED'})")
    logger.info(f"最大音频时长: {max_audio_dur}s")
    logger.info(f"批大小:     {args.batch_size}")
    logger.info(f"断点续跑:   {args.resume}")
    logger.info(f"日志文件:   {log_file}")

    # 1. 扫描 + 断点续跑
    wav_items, pending, done_ids, progress_file = prepare_pretag(
        args.wav_dir, output_dir, args.resume, args.no_cache, logger
    )
    if not pending:
        if wav_items:
            logger.info("所有文件已处理完毕")
        return

    # 2. 分桶排序
    logger.info("按时长分桶排序...")
    sorted_items, over_limit = create_duration_buckets(pending, max_audio_dur)
    if over_limit:
        logger.warning(f"超过 {max_audio_dur}s 的音频: {len(over_limit)} 条，将逐条处理")

    # 3. 加载模型
    device = select_gpus(args.device, logger)
    model = load_model(asr_type, args.model_dir, logger)
    params = get_decode_params(asr_type, device)

    total = len(sorted_items) + len(over_limit)

    # 4. 定义回调
    def process_fn(items, writer, monitor, infer_start, total_, logger_):
        return process_batches(model, params, items, args.batch_size,
                               writer, monitor, infer_start, total_, logger_)

    def reload_fn():
        nonlocal model
        logger.info("释放 GPU 显存...")
        del model
        release_gpu()
        time.sleep(5)
        model = load_model(asr_type, args.model_dir, logger)

    def process_extra_fn(extra, writer, logger_):
        logger_.info(f"处理超长音频 ({len(extra)} 条)，逐条推理...")
        for idx, (uid, p, d) in enumerate(extra):
            try:
                results = model.transcribe([uid], [p], params)
                writer.write(uid, p, results[0]["text"])
            except Exception as e:
                logger_.warning(f"  超长音频失败 {uid} ({d:.1f}s): {e}")
                writer.failed += 1
            if (idx + 1) % 10 == 0:
                writer.flush()
                logger_.info(f"  超长音频进度: {idx+1}/{len(extra)}")
        writer.flush()

    # 5. 执行推理
    run_inference_loop(
        items=sorted_items, total=total,
        output_dir=output_dir, progress_file=progress_file,
        resume=args.resume, done_ids=done_ids,
        batch_size=args.batch_size, max_slow_batches=args.max_slow_batches,
        process_fn=process_fn, reload_fn=reload_fn, logger=logger,
        extra_items=over_limit, process_extra_fn=process_extra_fn,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="FireRedASR 音频预标注")
    parser.add_argument('--wav-dir', default=str(DEFAULT_WAV_DIR), help='wav文件目录')
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR), help='输出目录')
    parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR, help='FireRedASR 模型目录')
    parser.add_argument('--asr-type', choices=['llm', 'aed'], default='llm',
                        help='模型类型: llm=旗舰(8.3B), aed=高效(1.1B) (default: llm)')
    parser.add_argument('--device', default='cuda:auto',
                        help='设备: cuda:0, cuda:0,1, cuda:auto, cuda:auto:2, cpu (default: cuda:auto)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='批大小，LLM建议1-4，AED建议8-16 (default: 1)')
    parser.add_argument('--resume', action='store_true', help='断点续跑')
    parser.add_argument('--no-cache', action='store_true', help='忽略扫描缓存')
    parser.add_argument('--max-slow-batches', type=int, default=-1,
                        help='连续慢批次数达到此值触发自动重启，<=0 禁用 (default: -1)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pretag(args)
