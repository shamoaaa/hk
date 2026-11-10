"""CLI entry point for asr-evolve.

Commands:
    status              系统概览（当前最优 CER + 最近实验）
    gpu                 远程 GPU 状态
    pipeline run        运行全流水线
    pipeline stage      运行指定阶段
    train submit        提交训练 (支持 --mode probe/full)
    train check         检查训练状态
    train stop          停止训练
    train monitor       实时监控训练 telemetry (#1 #2)
    train health        快速健康检查 (#1 #2)
    eval                评估模型
    analyse             错误归因分析
    tree view/summary/add/record/dead-end  实验树操作
    tree suggest-next   UCB1 推荐下一实验 (#4)
    strategy            搜索策略管理 (#5)
    suggest             HPO 调参建议 (#8)
    rules               决策规则管理 (#7 #11)
    reflect             失败根因提取 (#11)
    state               状态一致性检查 (#9)
    agent-report        Agent 效能报告 (#10)
    memory              查看/更新记忆
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="asr-evolve",
        description="Kiro-driven ASR model automated optimization platform",
    )
    sub = parser.add_subparsers(dest="command")

    # status
    sub.add_parser("status", help="系统概览")

    # gpu
    sub.add_parser("gpu", help="远程 GPU 状态")

    # pipeline
    pipe_parser = sub.add_parser("pipeline", help="流水线操作")
    pipe_sub = pipe_parser.add_subparsers(dest="pipe_action")

    run_p = pipe_sub.add_parser("run", help="运行全流水线")
    run_p.add_argument("--mode", choices=["aishell", "business"], default="aishell")
    run_p.add_argument("--dry-run", action="store_true")
    run_p.add_argument("--force", nargs="*", help="强制重跑的阶段")

    stage_p = pipe_sub.add_parser("stage", help="运行指定阶段")
    stage_p.add_argument("--only", required=True, help="逗号分隔的阶段名")
    stage_p.add_argument("--force", nargs="*")
    stage_p.add_argument("--mode", choices=["aishell", "business"], default="aishell")

    # train
    train_parser = sub.add_parser("train", help="训练操作")
    train_sub = train_parser.add_subparsers(dest="train_action")
    submit_p = train_sub.add_parser("submit", help="提交训练")
    submit_p.add_argument("--tag", default="", help="训练标签（默认自动生成）")
    submit_p.add_argument("--mode", choices=["probe", "full"], default="full",
                          help="训练模式: probe(快速2epoch验证) / full(完整训练)")
    train_sub.add_parser("check", help="检查训练状态")
    train_sub.add_parser("stop", help="停止训练")
    train_sub.add_parser("wait", help="阻塞等待训练完成")
    monitor_p = train_sub.add_parser("monitor", help="实时监控训练 telemetry")
    monitor_p.add_argument("--lines", type=int, default=50, help="读取日志行数")
    train_sub.add_parser("health", help="快速健康检查")

    # eval
    eval_p = sub.add_parser("eval", help="评估模型")
    eval_p.add_argument("--version", help="版本号 (v0.X.0)")
    eval_p.add_argument("--checkpoints", help="检查点列表 (best,ep50,avg5)")

    # analyse
    analyse_p = sub.add_parser("analyse", help="错误归因分析")
    analyse_p.add_argument("--results", default="results", help="结果文件/目录路径")
    analyse_p.add_argument("--test-set", default="", help="测试集名称")
    analyse_p.add_argument("--threshold", type=float, default=0.1, help="Bad case CER 阈值")
    analyse_p.add_argument("--top-n", type=int, default=20, help="混淆对 top-N")

    # tree
    tree_parser = sub.add_parser("tree", help="实验树操作")
    tree_sub = tree_parser.add_subparsers(dest="tree_action")
    tree_sub.add_parser("view", help="查看实验树")
    tree_sub.add_parser("summary", help="实验树摘要")
    add_p = tree_sub.add_parser("add", help="添加节点")
    add_p.add_argument("--parent-id", help="父节点 ID")
    add_p.add_argument("--action-desc", required=True, help="操作描述")
    add_p.add_argument("--hypothesis", default="", help="优化假设")
    record_p = tree_sub.add_parser("record", help="记录结果")
    record_p.add_argument("--node-id", required=True)
    record_p.add_argument("--cer", type=float)
    record_p.add_argument("--status", default="completed")
    dead_p = tree_sub.add_parser("dead-end", help="标记死胡同")
    dead_p.add_argument("--node-id", required=True)
    dead_p.add_argument("--reason", default="")
    suggest_p = tree_sub.add_parser("suggest-next", help="UCB1 推荐下一实验方向")
    suggest_p.add_argument("--top-n", type=int, default=5, help="推荐数量")

    # strategy (#5)
    strat_parser = sub.add_parser("strategy", help="搜索策略管理")
    strat_sub = strat_parser.add_subparsers(dest="strategy_action")
    strat_sub.add_parser("status", help="查看当前搜索阶段")
    trans_p = strat_sub.add_parser("check-transition", help="检查阶段切换")
    trans_p.add_argument("--cer", type=float, help="指定 CER (否则从实验树获取)")
    strat_sub.add_parser("params", help="当前阶段参数范围")

    # suggest (#8)
    suggest_parser = sub.add_parser("suggest", help="HPO 历史统计")

    # augment (远程数据增强)
    aug_parser = sub.add_parser("augment", help="远程数据增强")
    aug_sub = aug_parser.add_subparsers(dest="augment_action")
    aug_sub.add_parser("run", help="执行远程增强")
    aug_sub.add_parser("status", help="查看增强产物状态")
    aug_sub.add_parser("restore", help="恢复原始数据")

    # memory
    mem_p = sub.add_parser("memory", help="记忆操作")
    mem_p.add_argument("action", choices=["full", "update", "dead-ends", "insights"])

    # state
    state_parser = sub.add_parser("state", help="状态一致性检查")
    state_sub = state_parser.add_subparsers(dest="state_action")
    state_sub.add_parser("validate", help="检查所有状态文件一致性")
    summary_p = state_sub.add_parser("summary", help="结构化状态摘要")
    summary_p.add_argument("--json", dest="json_output", action="store_true", help="JSON 输出")

    # agent-report
    report_parser = sub.add_parser("agent-report", help="Agent 效能指标")
    report_parser.add_argument("--last", type=int, help="基于最近 N 轮")
    report_parser.add_argument("--update", action="store_true", help="从 HPO 历史更新数据")
    report_parser.add_argument("--json", dest="json_output", action="store_true", help="JSON 格式输出")

    # reset
    reset_parser = sub.add_parser("reset", help="系统重置")
    reset_sub = reset_parser.add_subparsers(dest="reset_action")
    reset_sub.add_parser("remote", help="清理远程产物")
    reset_sub.add_parser("local", help="重置本地记忆文件")
    reset_sub.add_parser("all", help="全部重置")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Dispatch
    if args.command == "status":
        from asr_evolve.commands.status import cmd_status
        cmd_status()
    elif args.command == "gpu":
        from asr_evolve.commands.gpu import cmd_gpu
        cmd_gpu()
    elif args.command == "pipeline":
        from asr_evolve.commands.pipeline import cmd_pipeline
        cmd_pipeline(args)
    elif args.command == "train":
        _dispatch_train(args)
    elif args.command == "eval":
        from asr_evolve.commands.evaluate import cmd_eval
        cmd_eval(args)
    elif args.command == "analyse":
        from asr_evolve.commands.analyse import cmd_analyse
        cmd_analyse(args)
    elif args.command == "tree":
        _dispatch_tree(args)
    elif args.command == "strategy":
        from asr_evolve.commands.strategy import cmd_strategy
        cmd_strategy(args)
    elif args.command == "suggest":
        from asr_evolve.commands.suggest import cmd_suggest
        cmd_suggest(args)
    elif args.command == "augment":
        from asr_evolve.commands.augment_remote import cmd_augment
        cmd_augment(args)
    elif args.command == "state":
        from asr_evolve.commands.state_validate import cmd_state
        cmd_state(args)
    elif args.command == "agent-report":
        from asr_evolve.commands.agent_report import cmd_agent_report
        cmd_agent_report(args)
    elif args.command == "memory":
        from asr_evolve.commands.memory import cmd_memory
        cmd_memory(args)
    elif args.command == "reset":
        from asr_evolve.commands.reset import cmd_reset
        cmd_reset(args)


def _dispatch_train(args):
    """Train 命令分发（含新增 monitor/health/wait）。"""
    action = getattr(args, "train_action", None)
    if action in ("submit", "check", "stop", "wait"):
        from asr_evolve.commands.train import cmd_train
        cmd_train(args)
    elif action == "monitor":
        from asr_evolve.commands.monitor import cmd_train_monitor
        cmd_train_monitor(args)
    elif action == "health":
        from asr_evolve.commands.monitor import cmd_train_health
        cmd_train_health(args)
    else:
        print("用法: asr-evolve train {submit|check|stop|wait|monitor|health}")


def _dispatch_tree(args):
    """Tree 命令分发（含新增 suggest-next）。"""
    action = getattr(args, "tree_action", None)
    if action == "suggest-next":
        from asr_evolve.commands.suggest import cmd_tree_suggest
        cmd_tree_suggest(args)
    elif action:
        from asr_evolve.commands.tree import cmd_tree
        cmd_tree(args)
    else:
        print("用法: asr-evolve tree {view|summary|add|record|dead-end|suggest-next}")


if __name__ == "__main__":
    main()
