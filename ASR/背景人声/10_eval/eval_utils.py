#!/usr/bin/env python3
"""
ASR 评估公用工具模块

提供 CER 计算、文本规范化、标签加载、wav 文件收集、报告生成等共享功能。
供 seaco_eval.py / finetuned_eval.py 等评估脚本复用。
"""

import os
import sys
import json
import re
import time
import random
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# 复用 pretag_utils 中的 GPU 选择
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "06_pretag_and_compare"))
try:
    from pretag_utils import select_gpus
except ImportError:
    def select_gpus(device_arg, logger=None):
        if logger:
            logger.info(f"使用设备: {device_arg}")
        return device_arg


# ============ 音频工具 ============

_wav_duration_warning_shown = False


def get_wav_duration(wav_path: str) -> float:
    """
    获取音频时长(秒)，依次尝试 soundfile 和 wave 模块。
    全部失败返回 -1.0，并在首次失败时打印具体错误原因。
    """
    global _wav_duration_warning_shown

    # 方法1: soundfile（支持多种格式）
    sf_err = None
    try:
        import soundfile as sf
        info = sf.info(wav_path)
        return info.duration
    except ImportError as e:
        sf_err = f"soundfile 未安装: {e}"
    except Exception as e:
        sf_err = f"soundfile 读取失败: {e}"

    # 方法2: 标准库 wave（仅支持 PCM WAV）
    wave_err = None
    try:
        import wave
        with wave.open(wav_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return frames / rate
            wave_err = "采样率为0"
    except Exception as e:
        wave_err = f"wave 读取失败: {e}"

    # 首次失败时打印详细错误，避免日志刷屏
    if not _wav_duration_warning_shown:
        import logging
        logger = logging.getLogger('eval')
        msg = f"[get_wav_duration] 无法获取音频时长! {sf_err}; {wave_err} (文件: {wav_path})"
        if logger.handlers:
            logger.error(msg)
        else:
            print(f"[ERROR] {msg}", file=sys.stderr)
        _wav_duration_warning_shown = True

    return -1.0


# ============ 日志 ============

def setup_logging(output_dir: Path, logger_name: str = 'eval'):
    """配置日志: 同时输出到控制台和文件"""
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

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    return logger, log_file


def format_duration(seconds: float) -> str:
    """格式化时长"""
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
    加载标签文件，支持多种格式:
      - text.txt / trans.txt: 每行 "uid 文本内容"
      - jsonl: {"uid":"xxx","text":"你好"} / {"key":"xxx","txt":"你好"} / messages 训练格式
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

            if suffix in ('.jsonl', '.json'):
                try:
                    obj = json.loads(line)
                    if "uid" in obj and "text" in obj:
                        labels[obj["uid"]] = obj["text"]
                    elif "key" in obj and "txt" in obj:
                        labels[obj["key"]] = obj["txt"]
                    elif "messages" in obj:
                        msgs = obj["messages"]
                        user_content = msgs[1]["content"]
                        wav_part = user_content.split("!")[-1].split("<|endofspeech|>")[0]
                        uid = Path(wav_part).stem
                        text = msgs[-1]["content"]
                        labels[uid] = text
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
            else:
                # text/trans 格式，支持:
                #   两列: "uid 文本内容"
                #   三列: "uid spk_id 文本内容" (tab 或空格分隔)
                parts = line.split('\t')
                if len(parts) >= 3:
                    # tab 分隔三列: filename \t spk_id \t text
                    uid = Path(parts[0].strip()).stem
                    labels[uid] = parts[-1].strip()
                elif len(parts) == 2:
                    # tab 分隔两列: uid \t text
                    uid = Path(parts[0].strip()).stem
                    labels[uid] = parts[1].strip()
                else:
                    # 空格分隔: "uid text" 或 "uid spk_id text"
                    space_parts = line.split(None, 2)
                    if len(space_parts) >= 2:
                        uid = Path(space_parts[0]).stem
                        # 如果第二列像 spk_id（不含中文），取第三列作为文本
                        if len(space_parts) == 3 and not re.search(r'[\u4e00-\u9fff]', space_parts[1]):
                            labels[uid] = space_parts[2]
                        else:
                            labels[uid] = space_parts[1] if len(space_parts) == 2 else ' '.join(space_parts[1:])

    logger.info(f"加载标签: {len(labels)} 条")
    return labels


def load_hotwords(hotword_path: Path, logger) -> str:
    """加载热词文件，返回空格分隔的热词字符串"""
    if not hotword_path or not hotword_path.exists():
        logger.info("未指定热词文件或文件不存在，跳过热词")
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
    """加载已完成的 uid 集合"""
    done = set()
    if progress_file.exists():
        with open(progress_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(line)
    return done


def _build_wav_index(wav_dir: Path, logger) -> dict:
    """
    递归扫描 wav_dir 下所有 wav 文件，构建 {uid: wav_path_str} 索引。
    如果存在同名 uid，优先保留目录层级较浅的文件。
    """
    uid_map = {}
    for f in wav_dir.rglob("*.wav"):
        uid = f.stem
        if uid not in uid_map:
            uid_map[uid] = str(f)
        else:
            if len(str(f)) < len(uid_map[uid]):
                uid_map[uid] = str(f)
    logger.info(f"递归扫描 {wav_dir}，找到 {len(uid_map)} 个 wav 文件")
    return uid_map


def collect_wav_files(wav_dir: Path, labels: dict, logger,
                     min_duration: float = 0.1) -> list:
    """
    收集 wav 文件（递归遍历子文件夹）。
    过滤掉时长低于 min_duration 秒的音频。
    返回 [(uid, ref_text_or_None, wav_path_str), ...]
    """
    uid_map = _build_wav_index(wav_dir, logger)
    eval_items = []
    missing_wav = 0
    skipped_short = 0

    if labels:
        duration_failed = 0
        for uid, ref_text in labels.items():
            if uid in uid_map:
                dur = get_wav_duration(uid_map[uid])
                if dur < 0:
                    duration_failed += 1
                    if duration_failed <= 5:
                        logger.warning(f"无法获取音频时长(仍参与评估): {uid} -> {uid_map[uid]}")
                elif dur < min_duration:
                    skipped_short += 1
                    logger.warning(f"跳过过短音频: {uid} ({dur:.3f}s) -> {uid_map[uid]}")
                    continue
                eval_items.append((uid, ref_text, uid_map[uid]))
            else:
                missing_wav += 1
        if duration_failed > 5:
            logger.warning(f"共 {duration_failed} 条音频无法获取时长（均已保留参与评估）")
        logger.info(f"匹配到 {len(eval_items)} 条 (缺少wav: {missing_wav}, 过短跳过: {skipped_short}, 时长获取失败: {duration_failed})")
    else:
        duration_failed = 0
        for uid in sorted(uid_map.keys()):
            dur = get_wav_duration(uid_map[uid])
            if dur < 0:
                duration_failed += 1
                if duration_failed <= 5:
                    logger.warning(f"无法获取音频时长(仍参与评估): {uid} -> {uid_map[uid]}")
            elif dur < min_duration:
                skipped_short += 1
                logger.warning(f"跳过过短音频: {uid} ({dur:.3f}s) -> {uid_map[uid]}")
                continue
            eval_items.append((uid, None, uid_map[uid]))
        if duration_failed > 5:
            logger.warning(f"共 {duration_failed} 条音频无法获取时长（均已保留参与评估）")
        logger.info(f"扫描到 {len(eval_items)} 个 wav 文件 (无标签模式，过短跳过: {skipped_short}, 时长获取失败: {duration_failed})")

    return eval_items


def auto_discover_labels(wav_dirs: list, logger) -> dict:
    """
    自动在每个 wav 目录的同级/父级查找 trans.txt / text.txt。
    返回 {wav_dir_str: labels_dict}
    """
    per_dir_labels = {}
    logger.info("未指定标签文件，尝试自动发现...")
    for wav_dir in wav_dirs:
        found_label = None
        search_dirs = [wav_dir, wav_dir.parent] if wav_dir.name == 'wav' else [wav_dir]
        for search_dir in search_dirs:
            for name in ('trans.txt', 'text.txt'):
                candidate = search_dir / name
                if candidate.exists():
                    found_label = candidate
                    break
            if found_label:
                break
        if found_label:
            loaded = load_labels(found_label, logger)
            per_dir_labels[str(wav_dir)] = loaded
            logger.info(f"  {wav_dir.name} -> {found_label} ({len(loaded)} 条)")
        else:
            logger.info(f"  {wav_dir.name} -> 未找到标签文件 (仅识别模式)")
    return per_dir_labels


def get_dir_display_name(wav_dir: Path) -> str:
    """获取目录的显示名称，wav 目录用父目录名"""
    name = wav_dir.name
    if name == 'wav':
        name = wav_dir.parent.name
    return name


def build_eval_items_from_scp(scp_path: Path, labels: dict, logger,
                              min_duration: float = 0.1) -> list:
    """
    从 wav.scp + labels dict 构建 eval_items，无需把 wav 文件拷贝/链接到同一目录。

    - scp 格式:  "<uid> <wav_absolute_path>" 每行一条
    - labels:    {uid: ref_text}
    - 返回:      [(uid, ref_text, wav_path_str), ...]，跳过缺失和过短音频
    """
    wav_map = {}
    with open(scp_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                wav_map[parts[0]] = parts[1]

    eval_items = []
    missing_wav = 0
    skipped_short = 0

    duration_failed = 0
    for uid, ref_text in labels.items():
        wav_path = wav_map.get(uid)
        if wav_path is None or not Path(wav_path).exists():
            missing_wav += 1
            continue
        dur = get_wav_duration(wav_path)
        if dur < 0:
            # 获取时长失败，不跳过，仅记录警告
            duration_failed += 1
            if duration_failed <= 5:
                logger.warning(f"无法获取音频时长(仍参与评估): {uid} -> {wav_path}")
        elif dur < min_duration:
            skipped_short += 1
            logger.warning(f"跳过过短音频: {uid} ({dur:.3f}s) -> {wav_path}")
            continue
        eval_items.append((uid, ref_text, wav_path))

    if duration_failed > 5:
        logger.warning(f"共 {duration_failed} 条音频无法获取时长（均已保留参与评估）")
    logger.info(
        f"从 scp 构建 eval_items: {len(eval_items)} 条 "
        f"(scp条目: {len(wav_map)}, 标签: {len(labels)}, "
        f"缺少wav: {missing_wav}, 过短跳过: {skipped_short}, "
        f"时长获取失败: {duration_failed})"
    )
    return eval_items


# ============ 单目录评估（通用） ============

def run_eval_items(eval_items: list, output_dir: Path, model, hotword_str: str,
                   args, logger, dir_name: str = "eval") -> dict:
    """
    对一组 (uid, ref_text_or_None, wav_path_str) 执行 ASR 评估。

    这是核心评估循环，与数据来源（目录扫描 / scp 文件）解耦：
      - collect_wav_files() 可提供基于目录扫描的 eval_items
      - build_eval_items_from_scp() 可提供基于 scp 的 eval_items
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"\n{'=' * 60}")
    logger.info(f"评估任务: {dir_name}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"{'=' * 60}")

    if not eval_items:
        logger.warning(f"[{dir_name}] 没有可评估的条目，跳过")
        return None

    has_labels = eval_items[0][1] is not None

    if args.sample and args.sample < len(eval_items):
        random.seed(42)
        eval_items = random.sample(eval_items, args.sample)
        logger.info(f"[{dir_name}] 随机采样 {args.sample} 条")

    progress_file = output_dir / "progress.txt"
    done_ids = set()
    if args.resume:
        done_ids = load_progress(progress_file)
        logger.info(f"[{dir_name}] 已完成: {len(done_ids)} 条")

    pending = [item for item in eval_items if item[0] not in done_ids]
    logger.info(f"[{dir_name}] 待评估: {len(pending)} 条")

    if not pending:
        logger.info(f"[{dir_name}] 所有条目已评估完毕")
        if has_labels:
            generate_report(output_dir, logger)
            return _load_summary(output_dir)
        return None

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
                    batch_size=args.batch_size,
                    disable_pbar=True,
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

                if has_labels and ref_text:
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
                else:
                    detail = {"uid": uid, "hyp": hyp_text}

                f_detail.write(json.dumps(detail, ensure_ascii=False) + "\n")
                f_prog.write(f"{uid}\n")
                processed += 1

            f_detail.flush()
            f_prog.flush()

            elapsed = time.time() - infer_start
            speed = processed / elapsed if elapsed > 0 else 0
            eta = (total - batch_end) / speed if speed > 0 else 0
            pct = batch_end / total * 100

            if has_labels:
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
            else:
                logger.info(
                    f"[{dir_name}] 进度: {batch_end}/{total} ({pct:.1f}%) | "
                    f"本批: {batch_time:.1f}s | "
                    f"成功: {processed} 失败: {failed} | "
                    f"速度: {speed:.1f}条/s | "
                    f"已用: {format_duration(elapsed)} 剩余: {format_duration(eta)}"
                )

    if has_labels:
        generate_report(output_dir, logger)
        return _load_summary(output_dir)
    else:
        logger.info(f"[{dir_name}] 无标签模式，识别结果已保存到: {detail_path}")
        return None


def _load_summary(output_dir: Path) -> dict:
    """加载已生成的 eval_summary.json"""
    summary_path = output_dir / "eval_summary.json"
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def run_eval_single_dir(wav_dir: Path, labels: dict, model, hotword_str: str,
                        output_dir: Path, args, logger) -> dict:
    """
    对单个 wav 目录执行评估（向后兼容接口）。

    内部会递归扫描 wav_dir 构建 eval_items，再调用 run_eval_items。
    如果你的数据已经是 wav.scp 格式，建议直接使用 build_eval_items_from_scp +
    run_eval_items，避免不必要的 staging 目录。
    """
    dir_name = get_dir_display_name(wav_dir)
    eval_items = collect_wav_files(wav_dir, labels, logger)
    return run_eval_items(
        eval_items=eval_items,
        output_dir=output_dir,
        model=model,
        hotword_str=hotword_str,
        args=args,
        logger=logger,
        dir_name=dir_name,
    )


# ============ 报告生成 ============

def generate_report(output_dir: Path, logger, report_title: str = "SeACo-Paraformer 评估报告"):
    """从 eval_detail.jsonl 生成单目录评估报告"""
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
            if "cer" not in d:
                continue
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
        w(report_title)
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


def generate_combined_report(dir_summaries: dict, output_dir: Path, logger,
                             report_title: str = "SeACo-Paraformer 多目录汇总评估报告"):
    """汇总多个目录的评估结果"""
    logger.info("\n" + "=" * 60)
    logger.info("生成多目录汇总报告")
    logger.info("=" * 60)

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
        w(report_title)
        w(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        w(f"评估目录数: {len(dir_summaries)}")
        w("=" * 70)

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
