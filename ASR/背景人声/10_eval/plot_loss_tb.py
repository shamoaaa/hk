#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 TensorBoard 事件文件中提取 train / val 的 loss，绘制在一张图上。
功能与 plot_loss.py 对齐：支持 loss_key 选择、epoch/step 双视图、多目录扫描合并。

用法:
    # 指定 tensorboard 目录
    python 10_eval/plot_loss_tb.py --tb_dir /data1/eval_output/v0.6.0/tensorboard

    # 多个 tensorboard 目录（中断恢复场景，自动合并）
    python 10_eval/plot_loss_tb.py --tb_dir /path/tb1 --tb_dir /path/tb2

    # 自动扫描父目录下所有包含 tfevents 的子目录
    python 10_eval/plot_loss_tb.py --scan_dir /data1/eval_output/v0.6.0

    # 只看 attention loss
    python 10_eval/plot_loss_tb.py --tb_dir /path/tb --loss_key loss_att

    # 额外绘制逐 step 细粒度曲线
    python 10_eval/plot_loss_tb.py --tb_dir /path/tb --show_step

    # 手动指定 steps_per_epoch（如果自动推测不准）
    python 10_eval/plot_loss_tb.py --tb_dir /path/tb --steps_per_epoch 5500

    # 指定输出路径
    python 10_eval/plot_loss_tb.py --tb_dir /path/tb --save /data1/loss_tb.png

    # 列出所有可用的 scalar tags（不绘图）
    python 10_eval/plot_loss_tb.py --tb_dir /path/tb --list_tags

支持的 loss_key:
    loss          — 总 loss（默认）
    loss_att      — attention decoder loss
    loss_seaco    — seaco decoder loss
    （也可输入 tensorboard 中的任意 tag 名称）

依赖:
    pip install tensorboard matplotlib
"""

import argparse
import glob
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境也能保存图片
import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("需要安装 tensorboard: pip install tensorboard")
    exit(1)


# ============================================================
# 数据加载
# ============================================================

def find_tb_dirs(scan_dir: str):
    """递归扫描目录，找到所有包含 tfevents 文件的目录"""
    pattern = os.path.join(scan_dir, "**", "events.out.tfevents.*")
    event_files = sorted(glob.glob(pattern, recursive=True))
    dirs = sorted(set(os.path.dirname(f) for f in event_files))
    return dirs


def load_tb_scalars(tb_dir: str):
    """从单个 tensorboard 目录加载所有 scalar 数据"""
    ea = EventAccumulator(tb_dir)
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        data[tag] = [(e.step, e.value) for e in events]

    return data, tags


def merge_tb_data(tb_dirs: list):
    """合并多个 tensorboard 目录的数据（后面的覆盖前面的同 step 数据）"""
    merged = defaultdict(dict)  # tag -> {step: value}
    all_tags = set()

    for d in tb_dirs:
        print(f"  读取: {d}")
        data, tags = load_tb_scalars(d)
        all_tags.update(tags)
        for tag, points in data.items():
            for step, value in points:
                merged[tag][step] = value

    # 转回 list 格式并排序
    result = {}
    for tag, step_dict in merged.items():
        result[tag] = sorted(step_dict.items(), key=lambda x: x[0])

    return result, sorted(all_tags)


# ============================================================
# 数据提取
# ============================================================

def find_tags(data: dict, all_tags: list, loss_key: str = "loss"):
    """
    根据 loss_key 从 tensorboard 数据中找到 train / val 对应的 tag。

    TensorBoard 的 tag 命名因框架而异，常见模式:
        - train/loss, val/loss, train/loss_att, val/loss_att
        - train_loss, val_loss
        - loss (不区分 train/val)
    """
    train_tag = None
    val_tag = None

    # 策略 1: 精确匹配 train/{loss_key} 和 val/{loss_key}
    for sep in ["/", "_"]:
        t_candidate = f"train{sep}{loss_key}"
        v_candidate = f"val{sep}{loss_key}"
        if t_candidate in data:
            train_tag = t_candidate
        if v_candidate in data:
            val_tag = v_candidate
        v2 = f"valid{sep}{loss_key}"
        if v2 in data and val_tag is None:
            val_tag = v2

    # 策略 2: 模糊匹配（tag 包含 loss_key 且包含 train/val）
    if train_tag is None or val_tag is None:
        for tag in all_tags:
            tag_lower = tag.lower()
            if loss_key.lower() in tag_lower:
                if "train" in tag_lower and train_tag is None:
                    train_tag = tag
                elif ("val" in tag_lower or "valid" in tag_lower) and val_tag is None:
                    val_tag = tag

    # 策略 3: 如果只有一个 loss_key tag（不区分 train/val），直接用它
    if train_tag is None and val_tag is None:
        if loss_key in data:
            train_tag = loss_key
            print(f"  注意: 只找到单一 tag '{loss_key}'，无法区分 train/val")

    return train_tag, val_tag


def guess_steps_per_epoch(val_steps: list, train_steps: list):
    """
    从 val 数据推测 steps_per_epoch。

    FunASR 的 validate_interval 通常是固定的（如 4000 步验证一次），
    而 epoch 边界处 val loss 会有规律性的间隔。

    更可靠的方法：如果 val 数据点间隔一致（validate_interval），
    则通过 train 数据的总 step 数 / 推测的 epoch 数来计算。
    """
    if not val_steps or len(val_steps) < 3:
        return None

    # 方法 1: 从 val 数据点的间隔推测 validate_interval
    val_step_values = [s for s, _ in val_steps]
    intervals = [val_step_values[i + 1] - val_step_values[i]
                 for i in range(len(val_step_values) - 1)]

    if not intervals:
        return None

    # 取众数作为 validate_interval
    from collections import Counter
    interval_counts = Counter(intervals)
    validate_interval = interval_counts.most_common(1)[0][0]

    # 方法 2: 从 val loss 的"跳变"推测 epoch 边界
    # epoch 结束时通常 val loss 会有明显变化模式
    # 但更简单的方法是：如果 train 数据有明显的周期性重置，用那个
    if train_steps and len(train_steps) > 100:
        train_step_values = [s for s, _ in train_steps]
        max_step = train_step_values[-1]

        # 尝试找 train loss 的周期性模式（epoch 开始时 loss 通常会跳高）
        # 简化：用 val 点数量和总 step 推算
        # 如果 val 每 validate_interval 步记录一次，那么每个 epoch 有 N 个 val 点
        # 总 val 点数 / epoch 数 = 每 epoch 的 val 点数
        # 但我们不知道 epoch 数...

        # 最终回退：返回 validate_interval 让用户手动指定
        pass

    return validate_interval


def compute_epoch_loss_from_val_pattern(val_steps: list, validate_interval: int):
    """
    从 val 数据推算 epoch 级别 loss。

    FunASR 在每个 validate_interval 步做一次验证，epoch 结束时也做一次。
    epoch 边界处的 val loss 就是该 epoch 的 val loss。

    策略：val 数据中，如果某个 step 不是 validate_interval 的整数倍，
    说明它是 epoch 结束时的验证（epoch 边界），用这些点作为 epoch loss。
    如果所有点都是整数倍，则取每个 epoch 最后一个 val 点。
    """
    if not val_steps or validate_interval <= 0:
        return {}

    # 检查是否有非整数倍的点（epoch 边界）
    epoch_boundary_points = []
    regular_points = []
    for step, value in val_steps:
        if step % validate_interval != 0:
            epoch_boundary_points.append((step, value))
        else:
            regular_points.append((step, value))

    if epoch_boundary_points:
        # 有 epoch 边界点，按顺序编号为 epoch 1, 2, 3...
        epoch_loss = {}
        for i, (step, value) in enumerate(sorted(epoch_boundary_points)):
            epoch_loss[i + 1] = value
        return epoch_loss

    # 没有明显的 epoch 边界，回退：按 validate_interval 分组
    # 假设用户会通过 --steps_per_epoch 手动指定
    return {}


def compute_epoch_loss_by_spe(step_data: list, steps_per_epoch: int):
    """根据 steps_per_epoch 将 step 数据聚合为 epoch 平均"""
    if not step_data or steps_per_epoch <= 0:
        return {}

    epoch_sums = defaultdict(float)
    epoch_counts = defaultdict(int)

    for step, value in step_data:
        epoch = (step - 1) // steps_per_epoch + 1
        epoch_sums[epoch] += value
        epoch_counts[epoch] += 1

    return {e: epoch_sums[e] / epoch_counts[e] for e in sorted(epoch_sums)}


def compute_epoch_loss_last(step_data: list, steps_per_epoch: int):
    """根据 steps_per_epoch，取每个 epoch 最后一个点的值"""
    if not step_data or steps_per_epoch <= 0:
        return {}

    epoch_last = {}
    for step, value in step_data:
        epoch = (step - 1) // steps_per_epoch + 1
        epoch_last[epoch] = value  # 后面的覆盖前面的

    return epoch_last


# ============================================================
# 绘图
# ============================================================

def plot(train_epoch_loss, val_epoch_loss, train_steps, val_steps,
         loss_key="loss", save_path=None, show_step=False):
    """绘制 loss 曲线（与 plot_loss.py 一致的绘图风格）"""
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

    # 标注最低 val loss
    if val_epoch_loss:
        best_epoch = min(val_epoch_loss, key=val_epoch_loss.get)
        best_val = val_epoch_loss[best_epoch]
        ax.annotate(f"best: ep{best_epoch}\n{best_val:.4f}",
                    xy=(best_epoch, best_val),
                    xytext=(best_epoch + 1, best_val + 0.002),
                    fontsize=9, color="#d62728",
                    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2))

    # ========== 子图 2: step 级别（可选） ==========
    if show_step and len(axes) > 1:
        ax2 = axes[1]
        if train_steps:
            steps, losses = zip(*train_steps)
            ax2.plot(steps, losses, alpha=0.4, linewidth=0.5,
                     label="Train loss (per step)", color="#1f77b4")
        if val_steps:
            steps_v, losses_v = zip(*val_steps)
            ax2.scatter(steps_v, losses_v, s=30, zorder=5,
                        label="Val loss", color="#d62728", marker="s", alpha=0.7)
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
        default_path = "loss_tb.png"
        fig.savefig(default_path, dpi=150, bbox_inches="tight")
        print(f"图片已保存: {default_path}")

    plt.close(fig)


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="从 TensorBoard 事件文件绘制 loss 曲线")
    parser.add_argument("--tb_dir", type=str, action="append", default=None,
                        help="tensorboard 事件文件目录，可多次指定（按顺序合并）")
    parser.add_argument("--scan_dir", type=str, default=None,
                        help="自动扫描该目录下所有包含 tfevents 的子目录")
    parser.add_argument("--save", type=str, default=None,
                        help="输出图片路径（默认 loss_tb.png）")
    parser.add_argument("--loss_key", type=str, default="loss",
                        help="要绘制的 loss 类型（默认 loss），也可输入 tensorboard 中的任意 tag 名")
    parser.add_argument("--show_step", action="store_true",
                        help="额外绘制逐 step 的细粒度曲线")
    parser.add_argument("--steps_per_epoch", type=int, default=None,
                        help="每个 epoch 的 step 数（手动指定，优先级最高）")
    parser.add_argument("--list_tags", action="store_true",
                        help="仅列出所有可用的 scalar tags，不绘图")
    args = parser.parse_args()

    # 确定 tensorboard 目录列表
    tb_dirs = []
    if args.scan_dir:
        tb_dirs = find_tb_dirs(args.scan_dir)
        if not tb_dirs:
            print(f"错误: 在 {args.scan_dir} 下未找到 tfevents 文件")
            return
    elif args.tb_dir:
        tb_dirs = args.tb_dir
    else:
        parser.error("请指定 --tb_dir 或 --scan_dir")

    print(f"找到 {len(tb_dirs)} 个 tensorboard 目录:")
    data, all_tags = merge_tb_data(tb_dirs)

    # 列出 tags 模式
    if args.list_tags:
        print(f"\n可用的 scalar tags ({len(all_tags)} 个):")
        for tag in all_tags:
            count = len(data.get(tag, []))
            print(f"  {tag}  ({count} points)")
        return

    if not data:
        print("错误: 未读取到任何 scalar 数据")
        return

    # 提取 train / val 数据
    loss_key = args.loss_key
    train_tag, val_tag = find_tags(data, all_tags, loss_key)

    print(f"\nloss_key: {loss_key}")
    print(f"  train tag: {train_tag or '未找到'}")
    print(f"  val   tag: {val_tag or '未找到'}")

    train_steps = data.get(train_tag, []) if train_tag else []
    val_steps = data.get(val_tag, []) if val_tag else []

    if not train_steps and not val_steps:
        print("错误: 未找到匹配的 loss 数据")
        print(f"可用 tags: {all_tags}")
        return

    print(f"\n原始数据:")
    if train_steps:
        print(f"  train: {len(train_steps)} points (step {train_steps[0][0]} ~ {train_steps[-1][0]})")
    if val_steps:
        print(f"  val:   {len(val_steps)} points (step {val_steps[0][0]} ~ {val_steps[-1][0]})")

    # 确定 steps_per_epoch
    steps_per_epoch = args.steps_per_epoch

    if steps_per_epoch is None:
        # 自动推测：从 val 数据的间隔推测 validate_interval
        validate_interval = guess_steps_per_epoch(val_steps, train_steps)
        if validate_interval:
            print(f"  推测 validate_interval = {validate_interval}")

            # 尝试从 val 数据中找 epoch 边界
            val_epoch_loss = compute_epoch_loss_from_val_pattern(val_steps, validate_interval)
            if val_epoch_loss:
                print(f"  从 val 边界点推测出 {len(val_epoch_loss)} 个 epoch")
                # 用 epoch 数反推 steps_per_epoch
                max_step = train_steps[-1][0] if train_steps else val_steps[-1][0]
                n_epochs = len(val_epoch_loss)
                steps_per_epoch = max_step // n_epochs
                print(f"  推算 steps_per_epoch ≈ {steps_per_epoch}")
            else:
                # 回退：假设 val 点均匀分布，用总 step / 合理 epoch 数估算
                if train_steps:
                    max_step = train_steps[-1][0]
                    # 常见 epoch 数范围 20~60，取 val 点数 / (max_step/validate_interval/epoch) 估算
                    n_val_per_epoch_guess = max_step // validate_interval // 40  # 假设 40 epoch
                    if n_val_per_epoch_guess > 0:
                        steps_per_epoch = validate_interval * n_val_per_epoch_guess
                        print(f"  粗略估算 steps_per_epoch ≈ {steps_per_epoch} (可能不准，建议用 --steps_per_epoch 手动指定)")
    else:
        print(f"  使用手动指定 steps_per_epoch = {steps_per_epoch}")

    # 计算 epoch 级别 loss
    train_epoch_loss = {}
    val_epoch_loss = {}

    if steps_per_epoch and steps_per_epoch > 0:
        train_epoch_loss = compute_epoch_loss_by_spe(train_steps, steps_per_epoch)
        val_epoch_loss = compute_epoch_loss_last(val_steps, steps_per_epoch)
    else:
        # 无法确定 steps_per_epoch，尝试从 val 边界推测
        if val_steps:
            validate_interval = guess_steps_per_epoch(val_steps, train_steps)
            if validate_interval:
                val_epoch_loss = compute_epoch_loss_from_val_pattern(val_steps, validate_interval)
        if not val_epoch_loss:
            print("\n警告: 无法确定 steps_per_epoch，无法绘制 epoch 级别曲线")
            print("  请使用 --steps_per_epoch 手动指定")
            print("  提示: steps_per_epoch = 训练集条数 / (batch_size_token / 平均音频token数) / gpu数")
            # 回退：直接用 step 编号作为 x 轴
            if val_steps:
                val_epoch_loss = {i + 1: v for i, (_, v) in enumerate(val_steps)}
            if train_steps and len(train_steps) <= 200:
                train_epoch_loss = {i + 1: v for i, (_, v) in enumerate(train_steps)}

    # 打印 epoch 汇总
    all_epochs = sorted(set(train_epoch_loss.keys()) | set(val_epoch_loss.keys()))
    if all_epochs:
        print(f"\nEpoch 级别汇总 ({len(all_epochs)} epochs):")
        for e in all_epochs:
            t_str = f"{train_epoch_loss[e]:.4f}" if e in train_epoch_loss else "  N/A "
            v_str = f"{val_epoch_loss[e]:.4f}" if e in val_epoch_loss else "  N/A "
            print(f"  Epoch {e:>2d}  train_loss={t_str}  val_loss={v_str}")

        # 标注最佳 epoch
        if val_epoch_loss:
            best_ep = min(val_epoch_loss, key=val_epoch_loss.get)
            print(f"\n  最佳 val loss: Epoch {best_ep} = {val_epoch_loss[best_ep]:.4f}")
    else:
        print("\n无 epoch 级别数据可显示")

    # 绘图
    plot(train_epoch_loss, val_epoch_loss, train_steps, val_steps,
         loss_key=loss_key, save_path=args.save, show_step=args.show_step)


if __name__ == "__main__":
    main()
