"""memory 命令 — 记忆管理."""
from asr_evolve.memory.state import StateManager
from asr_evolve.memory.dead_ends import DeadEndGuard
from pathlib import Path


def cmd_memory(args):
    if args.action == "full":
        # 输出完整紧凑上下文（STATE + DEAD_ENDS + INSIGHTS）
        for fname in ["STATE.md", "DEAD_ENDS.md", "INSIGHTS.md"]:
            p = Path(fname)
            if p.exists():
                print(f"\n{'='*20} {fname} {'='*20}")
                print(p.read_text(encoding="utf-8"))
    elif args.action == "update":
        sm = StateManager("STATE.md")
        sm.update_last_run()
        print("✅ STATE.md Last run 已更新")
    elif args.action == "dead-ends":
        guard = DeadEndGuard("DEAD_ENDS.md")
        entries = guard.get_entries()
        if entries:
            for e in entries:
                print(f"  ❌ {e['title']}")
        else:
            print("  （暂无死胡同）")
    elif args.action == "insights":
        p = Path("INSIGHTS.md")
        if p.exists():
            print(p.read_text(encoding="utf-8"))
        else:
            print("  （暂无洞察）")
