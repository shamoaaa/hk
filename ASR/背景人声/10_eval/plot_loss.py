#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 FunASR 训练日志中提取 train / val 的 epoch 级别 loss，绘制在一张图上。

用法:
    # 单个日志文件
    python tools/plot_loss.py --log /data1/outputs/2026-04-28/02-52-47/train.log

    # 多个日志文件（按时间顺序自动合并，适合中断恢复的场景）
    python tools/plot_loss.py --log /data1/outputs/2026-04-27/08-53-46/train.log \
                              --log /data1/outputs/2026-04-27/12-32-39/train.log

    # 自动扫描目录下所有 train.log（按路径排序合并）
    python tools/plot_loss.py --log_dir /data1/outputs

    # 只看 attention loss
    python tools/plot_loss.py --log_dir /data1/outputs --loss_key loss_att

    # 额外绘制逐 step 细粒度曲线
    python tools/plot_loss.py --log_dir /data1/outputs --show_step

    # 指定输出路径
    python tools/plot_loss.py --log_dir /data1/outputs --save /data1/loss_curve.png

支持的 loss_key:
    loss          — 总 loss（默认）
    loss_att      — attention decoder loss
    loss_seaco    — seaco decoder loss
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境也能保存图片
import matplotlib.pyplot as plt


def find_logs(log_dir: str):
    """递归扫描目录下所有 train.log，按路径排序（日期/时间目录名天然有序）"""
    pattern = os.path.join(log_dir, "**", "train.log")
    logs = sorted(glob.glob(pattern, recursive=True))
    return logs


def parse_log(log_path: str, loss_key: str = "loss"):
    """
    解析单个日志文件，返回:
        train_epoch_loss: {epoch: loss_avg}
        val_epoch_loss:   {epoch: loss_avg}
        train_steps:      [(total_step, loss)]
        val_steps:        [(total_step, loss)]

    兼容两种日志格式:
        - 旧版: loss_avg_epoch, rank: 0
        - 新版: loss_avg_slice, rank: N（多卡各自输出）
    """
    # 匹配 train/val 行，取任意 rank（后续只保留最小 rank 避免重复）
    re_mode   = re.compile(r"- (train|val), rank: (\d+),")
    re_epoch  = re.compile(r"epoch:\s*(\d+)/(\d+)")
    re_step   = re.compile(r"total step:\s*(\d+)")
    # 兼容 loss_avg_epoch 和 loss_avg_slice
    re_avg    = re.compile(r"\(loss_avg_(?:epoch|slice):\s*([\d.eE+-]+)\)")
    re_detail = re.compile(rf"\('{re.escape(loss_key)}',\s*([\d.eE+-]+)\)")

    train_epoch_loss = {}
    val_epoch_loss   = {}
    train_steps = []   # [(total_step, loss)]
    val_steps   = []

    # 记录每个 epoch 见过的最小 rank，只取最小 rank 的数据避免多卡重复
    seen_train_rank = {}  # epoch -> min_rank
    seen_val_rank   = {}

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_mode = re_mode.search(line)
            if not m_mode:
                continue
            mode = m_mode.group(1)
            rank = int(m_mode.group(2))

            m_epoch = re_epoch.search(line)
            m_step  = re_step.search(line)
            m_avg   = re_avg.search(line)
            if not (m_epoch and m_avg):
                continue

            epoch      = int(m_epoch.group(1))
            avg_loss   = float(m_avg.group(1))
            total_step = int(m_step.group(1)) if m_step else None

            m_detail    = re_detail.search(line)
            detail_loss = float(m_detail.group(1)) if m_detail else None

            if mode == "train":
                # 只取最小 rank 的数据
                prev_rank = seen_train_rank.get(epoch)
                if prev_rank is not None and rank > prev_rank:
                    continue
                if prev_rank is None or rank < prev_rank:
                    seen_train_rank[epoch] = rank
                # 每次覆盖，最终保留该 epoch 最后一条
                train_epoch_loss[epoch] = avg_loss
                if total_step is not None and detail_loss is not None:
                    train_steps.append((total_step, detail_loss))
            else:
                prev_rank = seen_val_rank.get(epoch)
                if prev_rank is not None and rank > prev_rank:
                    continue
                if prev_rank is None or rank < prev_rank:
                    seen_val_rank[epoch] = rank
                if loss_key != "loss" and detail_loss is not None:
                    val_epoch_loss[epoch] = detail_loss
                else:
                    val_epoch_loss[epoch] = avg_loss
                if total_step is not None:
                    val_steps.append((total_step, detail_loss if detail_loss else avg_loss))

    return train_epoch_loss, val_epoch_loss, train_steps, val_steps


def merge_logs(log_paths: list, loss_key: str = "loss"):
    """合并多个日志文件的解析结果（后面的文件覆盖前面的同 epoch 数据）"""
    merged_train = {}
    merged_val   = {}
    merged_train_steps = []
    merged_val_steps   = []

    for path in log_paths:
        print(f"  读取: {path}")
        t_ep, v_ep, t_st, v_st = parse_log(path, loss_key)
        merged_train.update(t_ep)
        merged_val.update(v_ep)
        merged_train_steps.extend(t_st)
        merged_val_steps.extend(v_st)

    # 按 step 排序去重
    merged_train_steps.sort(key=lambda x: x[0])
    merged_val_steps.sort(key=lambda x: x[0])

    return merged_train, merged_val, merged_train_steps, merged_val_steps


def plot(train_epoch_loss, val_epoch_loss, train_steps, val_steps,
         loss_key="loss", save_path=None, show_step=False):
    """绘制 loss 曲线"""
    fig, axes = plt.subplots(1, 2 if show_step else 1,
                             figsize=(14 if show_step else 8, 5))
    if not show_step:
        axes = [axes]

    # ========== 子图 1: epoch 级别 ==========
    ax = axes[0]
    if train_epoch_loss:
        epochs_t = sorted(train_epoch_loss.keys())
        ax.plot(epochs_t, [train_epoch_loss[e] for e in epochs_t],
                "o-", label="Train loss (epoch avg)", color="#1f77b4", linewidth=2)
    if val_epoch_loss:
        epochs_v = sorted(val_epoch_loss.keys())
        ax.plot(epochs_v, [val_epoch_loss[e] for e in epochs_v],
                "s-", label="Val loss (epoch avg)", color="#d62728", linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(f"{loss_key}", fontsize=12)
    ax.set_title(f"Train vs Val — {loss_key} (per epoch)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # ========== 子图 2: step 级别（可选） ==========
    if show_step and len(axes) > 1:
        ax2 = axes[1]
        if train_steps:
            steps, losses = zip(*train_steps)
            ax2.plot(steps, losses, alpha=0.4, linewidth=0.5,
                     label="Train loss (per step)", color="#1f77b4")
        if val_steps:
            steps_v, losses_v = zip(*val_steps)
            ax2.scatter(steps_v, losses_v, s=60, zorder=5,
                        label="Val loss", color="#d62728", marker="s")
        ax2.set_xlabel("Total Step", fontsize=12)
        ax2.set_ylabel(f"{loss_key}", fontsize=12)
        ax2.set_title(f"Train vs Val — {loss_key} (per step)", fontsize=13)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图片已保存: {save_path}")
    else:
        default_path = "loss_curve.png"
        fig.savefig(default_path, dpi=150, bbox_inches="tight")
        print(f"图片已保存: {default_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="绘制 FunASR 训练 loss 曲线")
    parser.add_argument("--log", type=str, action="append", default=None,
                        help="训练日志文件路径，可多次指定（按顺序合并）")
    parser.add_argument("--log_dir", type=str, default=r'/data1/outputs',
                        help="日志根目录，自动扫描所有 train.log")
    parser.add_argument("--save", type=str, default=None,
                        help="输出图片路径（默认 loss_curve.png）")
    parser.add_argument("--loss_key", type=str, default="loss",
                        choices=["loss", "loss_att", "loss_seaco"],
                        help="要绘制的 loss 类型（默认 loss）")
    parser.add_argument("--show_step", action="store_true",
                        help="额外绘制逐 step 的细粒度曲线")
    args = parser.parse_args()

    # 确定日志文件列表
    # 优先用显式指定的 --log；没有再扫描 --log_dir（默认 /data1/outputs）
    log_files = []
    if args.log:
        log_files = args.log
        missing = [p for p in log_files if not os.path.exists(p)]
        if missing:
            print(f"错误: 以下日志文件不存在: {missing}")
            return
    elif args.log_dir:
        log_files = find_logs(args.log_dir)
        if not log_files:
            print(f"错误: 在 {args.log_dir} 下未找到 train.log")
            return
    else:
        parser.error("请指定 --log 或 --log_dir")

    print(f"找到 {len(log_files)} 个日志文件:")
    train_epoch, val_epoch, train_steps, val_steps = merge_logs(log_files, args.loss_key)

    print(f"\n解析完成: train {len(train_epoch)} epochs, val {len(val_epoch)} epochs")
    if train_epoch:
        for e in sorted(train_epoch.keys()):
            v = val_epoch.get(e, val_epoch.get(e + 1, None))
            v_str = f"{v:.4f}" if v is not None else "N/A"
            print(f"  Epoch {e:>2d}  train_loss={train_epoch[e]:.4f}  val_loss={v_str}")

    plot(train_epoch, val_epoch, train_steps, val_steps,
         loss_key=args.loss_key, save_path=args.save, show_step=args.show_step)


if __name__ == "__main__":
    main()
