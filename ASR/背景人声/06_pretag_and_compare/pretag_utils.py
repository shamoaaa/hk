#!/usr/bin/env python3
"""
预标注脚本公共工具模块
提供 wav 扫描、进度管理、结果写入、速度监控等共享功能。
"""

import os
import sys
import json
import struct
import time
import logging
import gc
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

SCAN_CACHE_FILE = "wav_scan_cache.json"
SCAN_WORKERS = 8


# ============ 日志 ============

def setup_logging(output_dir: Path, logger_name: str = "pretag"):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"pretag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger, log_file


def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


# ============ WAV 扫描 ============

def get_wav_duration(wav_path: str) -> float:
    """使用 soundfile 获取音频时长(秒)，兼容所有 wav 格式，失败回退到手动解析头部"""
    try:
        import soundfile as sf
        info = sf.info(wav_path)
        return info.duration
    except Exception:
        pass
    # 回退: 手动读 wav 头部（仅适用于标准 44 字节头部）
    try:
        with open(wav_path, 'rb') as f:
            header = f.read(44)
            if len(header) < 44 or header[:4] != b'RIFF':
                return -1.0
            byte_rate = struct.unpack('<I', header[28:32])[0]
            data_size = struct.unpack('<I', header[40:44])[0]
            if byte_rate == 0:
                return -1.0
            return data_size / byte_rate
    except Exception:
        return -1.0


def _process_wav_entry(path: str):
    """单个wav文件处理（供线程池调用）"""
    uid = Path(path).stem
    dur = get_wav_duration(path)
    return (uid, path, dur) if dur > 0 else None


def scan_wav_files(wav_dir: Path, logger, no_cache: bool = False) -> list:
    """
    多线程扫描wav文件，支持缓存。返回 [(uid, path, duration)] 列表。
    """
    cache_path = wav_dir / SCAN_CACHE_FILE
    if cache_path.exists() and not no_cache:
        logger.info(f"从缓存加载扫描结果: {cache_path}")
        t0 = time.time()
        with open(cache_path, 'r', encoding='utf-8') as f:
            items = [tuple(x) for x in json.load(f)]
        logger.info(f"缓存加载 {len(items)} 条，耗时: {time.time()-t0:.1f}s")
        return items
    elif no_cache and cache_path.exists():
        logger.info("已指定 --no-cache，忽略缓存，重新扫描")

    logger.info(f"扫描音频目录: {wav_dir}")
    t0 = time.time()

    all_paths = []
    try:
        for entry in os.scandir(wav_dir):
            if entry.is_file() and entry.name.lower().endswith('.wav'):
                all_paths.append(entry.path)
    except Exception:
        for root, dirs, files in os.walk(wav_dir):
            for f in files:
                if f.lower().endswith('.wav'):
                    all_paths.append(str(Path(root) / f))

    logger.info(f"发现 {len(all_paths)} 个wav文件，多线程读取时长 (workers={SCAN_WORKERS})...")

    items = []
    skipped = 0
    with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as pool:
        for result in pool.map(_process_wav_entry, all_paths):
            if result:
                items.append(result)
            else:
                skipped += 1

    scan_time = time.time() - t0
    logger.info(f"找到 {len(items)} 个有效wav文件 (跳过 {skipped} 个无效文件)，扫描耗时: {scan_time:.1f}s")

    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(items, f)
        logger.info(f"扫描结果已缓存到: {cache_path}")
    except Exception as e:
        logger.warning(f"缓存写入失败: {e}")

    return items


def log_duration_stats(wav_items: list, logger):
    """输出音频时长统计"""
    all_durs = [d for _, _, d in wav_items]
    total_hours = sum(all_durs) / 3600
    logger.info(f"音频总时长: {total_hours:.1f} 小时, "
                f"平均: {sum(all_durs)/len(all_durs):.1f}s, "
                f"最短: {min(all_durs):.2f}s, 最长: {max(all_durs):.1f}s")


# ============ 进度管理 ============

def load_progress(progress_file: Path) -> set:
    """加载已处理的uid集合"""
    done = set()
    if progress_file.exists():
        with open(progress_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(line)
    return done


# ============ 结果写入器 ============

class PretagWriter:
    """封装 jsonl/scp/text/progress 四文件写入"""

    def __init__(self, jsonl_path, scp_path, text_path, progress_path, mode='w'):
        self.f_jsonl = open(jsonl_path, mode, encoding='utf-8')
        self.f_scp = open(scp_path, mode, encoding='utf-8')
        self.f_text = open(text_path, mode, encoding='utf-8')
        self.f_prog = open(progress_path, mode, encoding='utf-8')
        self.processed = 0
        self.failed = 0
        self._base_processed = 0  # 继承自上一阶段的基数，不参与速度计算
        self._pause_total = 0.0   # 累计挂起时间（秒），从 elapsed 中扣除
        self._last_batch_end = None  # 上一批结束时间戳

    def write(self, uid: str, wav_path: str, text: str):
        wav_abs = str(Path(wav_path).resolve())
        self.f_scp.write(f"{uid} {wav_abs}\n")
        self.f_text.write(f"{uid} {text}\n")
        record = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"语音转写：<|startofspeech|>!{wav_abs}<|endofspeech|>"},
                {"role": "assistant", "content": text}
            ],
            "text_length": len(text)
        }
        self.f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.f_prog.write(f"{uid}\n")
        self.processed += 1

    def flush(self):
        self.f_jsonl.flush()
        self.f_scp.flush()
        self.f_text.flush()
        self.f_prog.flush()

    def close(self):
        for f in (self.f_jsonl, self.f_scp, self.f_text, self.f_prog):
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # 批次间隔超过此值（秒）视为系统挂起，从 elapsed 中扣除
    PAUSE_THRESHOLD = 120.0

    def log_progress(self, batch_end: int, total: int, batch_time: float,
                     batch_count: int, infer_start: float, logger):
        now = time.time()

        # 检测批次间异常间隔（系统挂起/休眠）
        if self._last_batch_end is not None:
            gap = now - batch_time - self._last_batch_end
            if gap > self.PAUSE_THRESHOLD:
                self._pause_total += gap
                logger.warning(f"检测到系统挂起 {format_duration(gap)}，已从耗时统计中扣除")
        self._last_batch_end = now

        elapsed = now - infer_start - self._pause_total
        done = self.processed + self.failed
        stage_processed = self.processed - self._base_processed
        speed = stage_processed / elapsed if elapsed > 0 else 0
        remaining = total - done
        eta = remaining / speed if speed > 0 else 0
        pct = done / total * 100 if total > 0 else 0
        logger.info(
            f"进度: {done}/{total} ({pct:.1f}%) | "
            f"本批: {batch_time:.1f}s ({batch_count}条) | "
            f"成功: {self.processed} 失败: {self.failed} | "
            f"速度: {speed:.1f}条/s | "
            f"已用: {format_duration(elapsed)} | "
            f"预计剩余: {format_duration(eta)}"
        )


# ============ 自适应速度监控 ============

class SpeedMonitor:
    """
    自适应速度监控器。
    基于 "每秒音频的处理耗时" 归一化指标，自动建立基线并检测性能退化。
    """

    WARMUP_BATCHES = 5   # 前N批建立基线
    SLOW_RATIO = 0.6     # 性能降到基线的60%即判定慢
    MAX_HISTORY = 50      # 滑动窗口大小

    def __init__(self, max_slow_batches: int, logger):
        self.max_slow_batches = max_slow_batches
        self.logger = logger
        self.slow_streak = 0
        self.history_speed = []  # time_per_sec_audio 历史

    def reset(self):
        self.slow_streak = 0
        self.history_speed = []

    def check(self, batch_time: float, batch_items: list) -> bool:
        """
        检查本批次速度是否正常。
        batch_items: [(uid, path, duration), ...]
        返回 True 表示需要重启。
        max_slow_batches <= 0 时禁用监控，始终返回 False。
        """
        if self.max_slow_batches <= 0:
            return False
        batch_total_dur = sum(d for _, _, d in batch_items)
        time_per_sec = batch_time / batch_total_dur if batch_total_dur > 0 else batch_time

        batch_count = len(self.history_speed) + 1

        if batch_count <= self.WARMUP_BATCHES:
            self.history_speed.append(time_per_sec)
            self.slow_streak = 0
            if batch_count == self.WARMUP_BATCHES:
                baseline = sum(self.history_speed) / len(self.history_speed)
                self.logger.info(
                    f"基线建立完成: {baseline:.3f}s/s音频 "
                    f"(即每秒音频需 {baseline:.3f}s 处理)"
                )
            return False

        baseline = sum(self.history_speed) / len(self.history_speed)
        threshold = baseline / self.SLOW_RATIO

        if time_per_sec > threshold:
            self.slow_streak += 1
            degradation = baseline / time_per_sec if time_per_sec > 0 else 0
            self.logger.warning(
                f"⚠ 批次异常慢 ({self.slow_streak}/{self.max_slow_batches}): "
                f"{time_per_sec:.3f}s/s音频 (基线 {baseline:.3f}, "
                f"阈值 {threshold:.3f}, 性能 {degradation:.0%}), "
                f"音频总时长 {batch_total_dur:.1f}s"
            )
        else:
            self.slow_streak = 0
            self.history_speed.append(time_per_sec)
            if len(self.history_speed) > self.MAX_HISTORY:
                self.history_speed = self.history_speed[-self.MAX_HISTORY:]

        if self.slow_streak >= self.max_slow_batches:
            self.logger.error(
                f"🔄 连续 {self.slow_streak} 个慢批次，触发自动重启！"
                f"基线: {baseline:.3f}s/s音频"
            )
            return True

        return False


def release_gpu():
    """释放 GPU 显存"""
    import torch
    torch.cuda.empty_cache()
    gc.collect()


# ============ GPU 自动选择 ============

def get_gpu_free_memory() -> list:
    """
    查询所有 GPU 的空闲显存，返回 [(gpu_id, free_mb), ...] 按空闲降序排列。
    使用 nvidia-smi 查询，失败则回退到 torch.cuda。
    """
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split("\n"):
                parts = line.strip().split(",")
                if len(parts) == 2:
                    gpu_id = int(parts[0].strip())
                    free_mb = int(parts[1].strip())
                    gpus.append((gpu_id, free_mb))
            gpus.sort(key=lambda x: -x[1])
            return gpus
    except Exception:
        pass

    # 回退: 用 torch.cuda
    import torch
    if not torch.cuda.is_available():
        return []
    gpus = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        gpus.append((i, free // (1024 * 1024)))
    gpus.sort(key=lambda x: -x[1])
    return gpus


def select_gpus(device_arg: str, logger=None) -> str:
    """
    智能 GPU 选择。根据 --device 参数返回实际使用的设备字符串。

    支持的格式:
      "cuda:0"          → 指定单卡
      "cuda:0,1,2"      → 指定多卡
      "cuda:auto"       → 自动选最空闲的 1 张卡
      "cuda:auto:N"     → 自动选最空闲的 N 张卡（不足则用全部）
      "cpu"             → CPU

    返回:
      单卡: "cuda:X"
      多卡: "cuda:X" 并设置 CUDA_VISIBLE_DEVICES
    """
    if device_arg == "cpu":
        if logger:
            logger.info("使用 CPU")
        return "cpu"

    if not device_arg.startswith("cuda"):
        if logger:
            logger.info(f"使用设备: {device_arg}")
        return device_arg

    # 解析参数
    parts = device_arg.split(":")
    # "cuda:auto" or "cuda:auto:N"
    if len(parts) >= 2 and parts[1] == "auto":
        requested = int(parts[2]) if len(parts) >= 3 else 1
        gpus = get_gpu_free_memory()
        if not gpus:
            if logger:
                logger.warning("未检测到 GPU，回退到 CPU")
            return "cpu"

        available = len(gpus)
        use_count = min(requested, available)
        selected = gpus[:use_count]

        if logger:
            logger.info(f"GPU 自动选择: 请求 {requested} 张, 可用 {available} 张, 选择 {use_count} 张")
            for gpu_id, free_mb in selected:
                logger.info(f"  GPU {gpu_id}: 空闲 {free_mb} MB")

        if use_count == 1:
            device = f"cuda:{selected[0][0]}"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(selected[0][0])
        else:
            ids = [str(g[0]) for g in selected]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
            device = "cuda:0"  # VISIBLE_DEVICES 重映射后从 0 开始

        if logger:
            logger.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
            logger.info(f"设备: {device}")
        return device

    # "cuda:0,1,2" — 指定多卡
    if len(parts) == 2 and "," in parts[1]:
        ids = parts[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = ids
        if logger:
            logger.info(f"指定多卡: CUDA_VISIBLE_DEVICES={ids}")
        return "cuda:0"

    # "cuda:0" — 指定单卡（默认行为）
    if logger:
        logger.info(f"使用设备: {device_arg}")
    return device_arg


def print_summary(logger, total, processed, failed, infer_start,
                   jsonl_path, scp_path, text_path, progress_file):
    """输出预标注汇总"""
    total_time = time.time() - infer_start
    logger.info("=" * 60)
    logger.info("预标注完成")
    logger.info(f"  总文件数: {total}")
    logger.info(f"  成功: {processed}")
    logger.info(f"  失败: {failed}")
    logger.info(f"  总耗时: {format_duration(total_time)}")
    if total_time > 0:
        logger.info(f"  平均速度: {processed / total_time:.1f} 条/s")
    logger.info(f"输出文件:")
    logger.info(f"  {jsonl_path}")
    logger.info(f"  {scp_path}")
    logger.info(f"  {text_path}")
    logger.info(f"  {progress_file}")


# ============ 公共预标注流程 ============

def prepare_pretag(wav_dir, output_dir, resume, no_cache, logger):
    """
    公共的扫描 + 断点续跑逻辑。

    返回:
        (wav_items, pending, done_ids, progress_file)
        如果无待处理文件，pending 为空列表。
    """
    wav_items = scan_wav_files(Path(wav_dir), logger, no_cache=no_cache)
    if not wav_items:
        logger.warning("未找到任何wav文件，退出")
        return wav_items, [], set(), output_dir / "progress.txt"
    log_duration_stats(wav_items, logger)

    progress_file = Path(output_dir) / "progress.txt"
    done_ids = set()
    if resume:
        done_ids = load_progress(progress_file)
        logger.info(f"已完成: {len(done_ids)} 条，将跳过")

    pending = [(uid, p, d) for uid, p, d in wav_items if uid not in done_ids]
    logger.info(f"待处理: {len(pending)} 条")
    return wav_items, pending, done_ids, progress_file


def run_inference_loop(
    items, total, output_dir, progress_file, resume, done_ids,
    batch_size, max_slow_batches,
    process_fn, reload_fn, logger,
    extra_items=None, process_extra_fn=None,
):
    """
    公共的推理主循环，封装 writer 管理、重启逻辑、print_summary。

    Args:
        items: 主要待处理列表 [(uid, path, dur), ...]
        total: 总条目数（含 extra_items）
        output_dir: 输出目录 Path
        progress_file: 进度文件 Path
        resume: 是否断点续跑
        done_ids: 已完成的 uid 集合
        batch_size: 批大小
        max_slow_batches: 连续慢批次阈值
        process_fn: 批量推理函数，签名 process_fn(items, writer, monitor, infer_start, total, logger) -> bool(need_restart)
        reload_fn: 模型重新加载函数，签名 reload_fn() -> None（内部自行释放和加载）
        logger: 日志器
        extra_items: 额外待处理列表（如 FireRed 的超长音频），可选
        process_extra_fn: 额外处理函数，签名 process_extra_fn(extra_items, writer, logger) -> None，可选
    """
    output_dir = Path(output_dir)
    jsonl_path = output_dir / "pretag.jsonl"
    scp_path = output_dir / "wav.scp"
    text_path = output_dir / "text.txt"
    open_mode = 'a' if resume and done_ids else 'w'

    monitor = SpeedMonitor(max_slow_batches, logger)
    infer_start = time.time()

    writer = PretagWriter(jsonl_path, scp_path, text_path, progress_file, open_mode)
    try:
        need_restart = process_fn(items, writer, monitor, infer_start, total, logger)

        if need_restart:
            processed_so_far = writer.processed
            failed_so_far = writer.failed
            writer.close()

            reload_fn()
            monitor.reset()
            infer_start = time.time()

            done_ids = load_progress(progress_file)
            remaining = [(uid, p, d) for uid, p, d in items if uid not in done_ids]
            if extra_items is not None:
                extra_items = [(uid, p, d) for uid, p, d in extra_items if uid not in done_ids]
            logger.info(f"重启后待处理: {len(remaining)} 条"
                        + (f" (常规) + {len(extra_items)} 条 (额外)" if extra_items else ""))

            writer = PretagWriter(jsonl_path, scp_path, text_path, progress_file, 'a')
            writer.processed = processed_so_far
            writer.failed = failed_so_far
            writer._base_processed = processed_so_far

            if remaining:
                process_fn(remaining, writer, monitor, infer_start, total, logger)

        if extra_items and process_extra_fn:
            process_extra_fn(extra_items, writer, logger)

        print_summary(logger, total, writer.processed, writer.failed,
                      infer_start, jsonl_path, scp_path, text_path, progress_file)
    finally:
        writer.close()
