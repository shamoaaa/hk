"""tree 命令 — 实验树操作."""
from asr_evolve.memory.experiment_tree import ExperimentTree


def cmd_tree(args):
    tree = ExperimentTree("experiment_tree.json")

    if args.tree_action == "view":
        print(tree.view())

    elif args.tree_action == "summary":
        s = tree.summary()
        print("=" * 40)
        print("实验树摘要")
        print("=" * 40)
        print(f"总节点: {s['total_nodes']}")
        print(f"已完成: {s['completed']}")
        print(f"死胡同: {s['dead_ends']}")
        print(f"待执行: {s['pending']}")
        print(f"最优节点: {s['best_node']} (CER={s['best_cer']})")
        print(f"活跃分支: {s['active_branches']}")
        print("=" * 40)

    elif args.tree_action == "add":
        node = tree.add_node(
            action_desc=args.action_desc,
            parent_id=args.parent_id,
            hypothesis=args.hypothesis,
        )
        print(f"✅ 已添加: {node.node_id} → {node.action_desc}")

    elif args.tree_action == "record":
        node = tree.record_result(
            node_id=args.node_id,
            status=args.status,
            cer=args.cer,
        )
        if node:
            print(f"✅ 已记录: {node.node_id} status={node.status} cer={node.cer}")
        else:
            print(f"❌ 节点 {args.node_id} 不存在")

    elif args.tree_action == "dead-end":
        node = tree.mark_dead_end(args.node_id, args.reason)
        if node:
            print(f"💀 已标记: {node.node_id} → dead_end")
        else:
            print(f"❌ 节点 {args.node_id} 不存在")

    else:
        print("用法: asr-evolve tree {view|summary|add|record|dead-end}")
