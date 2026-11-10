"""gpu 命令 — 远程 GPU 状态."""
import yaml
from pathlib import Path
from asr_evolve.environment.remote import RemoteExecutor, RemoteConfig


def cmd_gpu():
    """从 configs/remote.yaml 读取配置，连接服务器执行 nvidia-smi。"""
    cfg_path = Path("configs/remote.yaml")
    if not cfg_path.exists():
        print("❌ configs/remote.yaml not found")
        return

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    remote = cfg.get("remote", {})
    rc = RemoteConfig(
        host=remote.get("host", ""),
        port=remote.get("port", 22),
        username=remote.get("username", ""),
        password=remote.get("password", ""),
        key_file=remote.get("key_file", ""),
        workspace=remote.get("workspace", "/data/shamo/project/asr"),
        timeout=remote.get("timeout", 30),
    )

    executor = RemoteExecutor(rc)
    try:
        executor.connect()
        output = executor.gpu_status()
        print("=" * 50)
        print("远程 GPU 状态")
        print("=" * 50)
        print(output)
        print("=" * 50)
    except Exception as e:
        print(f"❌ 连接失败: {e}")
    finally:
        executor.disconnect()
