"""Remote SSH runner with SFTP sync for GPU-bound tasks.

本地 Kiro 通过此脚本桥接到 GPU 服务器。

核心功能：
- push:      上传配置/数据到服务器（本地修改后推送）
- pull:      下载单个文件
- sync-down: 批量下载训练产物（logs/reports/增强jsonl）
- run:       SSH 执行远程命令（流式输出）
- gpu:       查看 GPU 状态
- status:    查看服务器状态

设计原则：
- 代码同步（scripts/）由用户通过 VS Code SFTP 插件手动完成
- metadata.db 和 .kiro/ 只在本地使用，不上传
- 模型文件很大，不自动下载，需要时用 pull 单独拉取
- 增强 wav 文件只在服务器使用，不下载；只拉取增强后的 jsonl 索引

Usage:
    # 推送配置变更到服务器
    python scripts/remote_runner.py push-configs

    # 推送原始数据到服务器（首次或数据更新时）
    python scripts/remote_runner.py push-data

    # 在服务器执行命令
    python scripts/remote_runner.py run "python -m scripts.run_augmentation --all"
    python scripts/remote_runner.py run "python -m scripts.run_training --tag iter_001"

    # 下载训练/评估产物到本地
    python scripts/remote_runner.py sync-down

    # 下载单个文件
    python scripts/remote_runner.py pull logs/training/iter_001/log.txt

    # 推送单个文件
    python scripts/remote_runner.py push configs/funasr_config.yaml

    # 查看 GPU 状态
    python scripts/remote_runner.py gpu

配置: configs/remote.yaml
"""

import argparse
import os
import stat
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Optional


DEFAULT_CONFIG_PATH = "configs/remote.yaml"

# =====================================================================
# 上传规则：什么需要推送到服务器
# =====================================================================

# push-configs: 推送配置文件（本地修改后同步到服务器）
PUSH_CONFIGS = [
    "configs/funasr_config.yaml",
    "configs/post_process/",       # 热词表、正则规则等
]

# push-scripts: 推送脚本代码到服务器
PUSH_SCRIPTS = [
    "scripts/",
]

# push-data: 推送原始数据（首次部署或数据更新时）
PUSH_DATA = [
    "data/train.jsonl",
    "data/val.jsonl",
    "data/test.jsonl",
]

# =====================================================================
# 下载规则：什么需要从服务器拉取
# =====================================================================

# sync-down: 训练/评估完成后拉取的产物
SYNC_DOWN_DIRS = [
    "logs/training/",              # 训练日志和指标 jsonl
    "logs/optimization/",          # 优化日志
    "reports/evaluation/",         # 评估报告 json
    "reports/dataset/",            # 数据集分析报告
    "reports/augmentation/",       # 增强报告
]

# sync-down: 增强后的 jsonl 索引（不含 wav，wav 只在服务器用）
SYNC_DOWN_JSONL = [
    "data/augmented/train_aug.jsonl",
    "data/augmented/val_aug.jsonl",
    "data/augmented/test_aug.jsonl",
]

# sync-down: 配置版本快照（服务器训练时可能生成新版本）
SYNC_DOWN_CONFIG_VERSIONS = [
    "configs/augmentation/",
    "configs/model/",
    "configs/training/",
    "configs/post_process/",
]

# =====================================================================
# 不传输的内容（说明）
# =====================================================================
# - scripts/          → 用户通过 VS Code SFTP 手动同步
# - .kiro/            → 本地专用（skills/steering/hooks）
# - metadata.db       → 本地专用（迭代管理、知识库）
# - .venv*            → 各端独立维护
# - models/iterations/→ 太大，不自动下载，需要时用 pull 单独拉
# - data/augmented/*.wav → 只在服务器使用，不下载
# - data/musan/       → 只在服务器使用

# SFTP 排除模式
SFTP_EXCLUDE = {
    "__pycache__",
    ".pyc",
    ".venv",
    ".venv_win",
    ".venv310",
    ".pytest_cache",
    "node_modules",
    ".hypothesis",
}


def _load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load remote config from YAML."""
    try:
        import yaml
    except ImportError:
        sys.stderr.write("[remote_runner] pyyaml not installed. Run: pip install pyyaml paramiko\n")
        sys.exit(2)

    p = Path(config_path)
    if not p.exists():
        sys.stderr.write(f"[remote_runner] config not found: {config_path}\n")
        sys.stderr.write("  Please create configs/remote.yaml (see template in docs)\n")
        sys.exit(2)
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _connect(cfg: dict):
    """Open paramiko SSH client. Supports ~/.ssh/config aliases."""
    try:
        import paramiko
    except ImportError:
        sys.stderr.write("[remote_runner] paramiko not installed. Run: pip install paramiko\n")
        sys.exit(2)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Parse ~/.ssh/config for host aliases (ProxyJump, Port, IdentityFile, etc.)
    ssh_config_path = Path.home() / ".ssh" / "config"
    ssh_conf = paramiko.SSHConfig()
    if ssh_config_path.exists():
        with open(ssh_config_path, "r", encoding="utf-8") as f:
            ssh_conf.parse(f)

    host_alias = cfg["host"]
    host_info = ssh_conf.lookup(host_alias)

    # Resolve actual hostname, port, user, key from SSH config
    hostname = host_info.get("hostname", host_alias)
    port = int(cfg.get("port") or host_info.get("port", 22))
    username = cfg.get("username") or host_info.get("user", "root")

    kwargs = {
        "hostname": hostname,
        "port": port,
        "username": username,
        "timeout": cfg.get("timeout", 30),
    }

    # Key file: remote.yaml > SSH config > default
    key_filename = cfg.get("key_filename")
    if not key_filename and "identityfile" in host_info:
        key_filename = host_info["identityfile"][0]
    if key_filename:
        kwargs["key_filename"] = str(Path(key_filename).expanduser())

    password = cfg.get("password") or os.environ.get("REMOTE_PASSWORD")
    if password:
        kwargs["password"] = password

    # Handle ProxyJump / ProxyCommand from SSH config
    proxy_jump = host_info.get("proxyjump")
    proxy_command = host_info.get("proxycommand")

    if proxy_jump:
        # Resolve the jump host through SSH config too
        jump_info = ssh_conf.lookup(proxy_jump)
        jump_hostname = jump_info.get("hostname", proxy_jump)
        jump_port = int(jump_info.get("port", 22))
        jump_user = jump_info.get("user", cfg.get("username", "root"))
        jump_key = None
        if "identityfile" in jump_info:
            jump_key = str(Path(jump_info["identityfile"][0]).expanduser())

        # Create jump host connection
        jump_client = paramiko.SSHClient()
        jump_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        jump_kwargs = {
            "hostname": jump_hostname,
            "port": jump_port,
            "username": jump_user,
            "timeout": cfg.get("timeout", 30),
        }
        if jump_key:
            jump_kwargs["key_filename"] = jump_key
        if password:
            jump_kwargs["password"] = password
        jump_client.connect(**jump_kwargs)

        # Open channel through jump host to target
        jump_transport = jump_client.get_transport()
        dest_addr = (hostname, port)
        local_addr = ("127.0.0.1", 0)
        jump_channel = jump_transport.open_channel("direct-tcpip", dest_addr, local_addr)

        kwargs["sock"] = jump_channel
        # Store jump_client on main client so it doesn't get GC'd
        client._jump_client = jump_client

    elif proxy_command:
        # Use ProxyCommand (e.g., ssh -W %h:%p jumphost)
        proxy_cmd = proxy_command.replace("%h", hostname).replace("%p", str(port))
        kwargs["sock"] = paramiko.ProxyCommand(proxy_cmd)

    client.connect(**kwargs)
    return client


def _stream_exec(client, command: str, cwd: str = None) -> tuple[int, str]:
    """Execute remote command with streaming output. Returns (exit_code, captured_stdout)."""
    full_cmd = f"cd {cwd} && {command}" if cwd else command

    transport = client.get_transport()
    channel = transport.open_session()
    channel.get_pty()
    channel.exec_command(full_cmd)

    captured = []
    while True:
        if channel.recv_ready():
            chunk = channel.recv(4096).decode("utf-8", errors="replace")
            sys.stdout.write(chunk)
            sys.stdout.flush()
            captured.append(chunk)
        if channel.recv_stderr_ready():
            sys.stderr.write(channel.recv_stderr(4096).decode("utf-8", errors="replace"))
            sys.stderr.flush()
        if channel.exit_status_ready():
            while channel.recv_ready():
                chunk = channel.recv(4096).decode("utf-8", errors="replace")
                sys.stdout.write(chunk)
                captured.append(chunk)
            break
        time.sleep(0.05)

    return channel.recv_exit_status(), "".join(captured)


def _should_exclude(path: str) -> bool:
    """Check if a path should be excluded from sync."""
    parts = Path(path).parts
    for part in parts:
        if part in SFTP_EXCLUDE or part.endswith(".pyc"):
            return True
    return False


def _sftp_mkdir_p(sftp, remote_dir: str):
    """Recursively create remote directories (like mkdir -p)."""
    parts = PurePosixPath(remote_dir).parts
    current = ""
    for part in parts:
        current = current + "/" + part if current else part
        if not current.startswith("/"):
            current = "/" + current
        try:
            sftp.stat(current)
        except FileNotFoundError:
            try:
                sftp.mkdir(current)
            except IOError:
                pass  # might already exist due to race


def _upload_dir(sftp, local_dir: Path, remote_base: str, rel_prefix: str = ""):
    """Recursively upload a local directory to remote via SFTP."""
    count = 0
    for item in local_dir.rglob("*"):
        if item.is_dir():
            continue
        rel = item.relative_to(local_dir)
        rel_str = str(rel).replace("\\", "/")
        if _should_exclude(rel_str):
            continue

        if rel_prefix:
            remote_path = f"{remote_base}/{rel_prefix}/{rel_str}"
        else:
            remote_path = f"{remote_base}/{rel_str}"

        remote_parent = str(PurePosixPath(remote_path).parent)
        _sftp_mkdir_p(sftp, remote_parent)

        try:
            sftp.put(str(item), remote_path)
            count += 1
        except Exception as e:
            sys.stderr.write(f"  [WARN] upload failed: {rel_str} → {e}\n")
    return count


def _download_dir(sftp, remote_dir: str, local_base: Path, rel_prefix: str = ""):
    """Recursively download a remote directory to local via SFTP."""
    count = 0

    def _walk(remote_path: str, local_path: Path):
        nonlocal count
        try:
            items = sftp.listdir_attr(remote_path)
        except FileNotFoundError:
            return

        for item in items:
            remote_item = f"{remote_path}/{item.filename}"
            local_item = local_path / item.filename

            if stat.S_ISDIR(item.st_mode):
                local_item.mkdir(parents=True, exist_ok=True)
                _walk(remote_item, local_item)
            else:
                local_item.parent.mkdir(parents=True, exist_ok=True)
                try:
                    sftp.get(remote_item, str(local_item))
                    count += 1
                except Exception as e:
                    sys.stderr.write(f"  [WARN] download failed: {remote_item} → {e}\n")

    _walk(remote_dir, local_base)
    return count


# =====================================================================
# Subcommands
# =====================================================================

def cmd_push_configs(args: argparse.Namespace) -> int:
    """Push config files to server (after local edits)."""
    cfg = _load_config(args.config)
    client = _connect(cfg)
    try:
        sftp = client.open_sftp()
        remote_workspace = f"{cfg['remote_path']}/workspace"

        total = 0
        for path_str in PUSH_CONFIGS:
            local_path = Path(path_str)
            if local_path.is_dir():
                sys.stderr.write(f"  uploading {path_str} ...\n")
                n = _upload_dir(sftp, local_path, remote_workspace, path_str.rstrip("/"))
                total += n
            elif local_path.is_file():
                remote_file = f"{remote_workspace}/{path_str}"
                _sftp_mkdir_p(sftp, str(PurePosixPath(remote_file).parent))
                sftp.put(str(local_path), remote_file)
                total += 1
                sys.stderr.write(f"  ✓ {path_str}\n")
            else:
                sys.stderr.write(f"  [skip] {path_str} not found\n")

        sftp.close()
        sys.stderr.write(f"\n[push-configs] uploaded {total} files\n")
        return 0
    finally:
        client.close()


def cmd_push_scripts(args: argparse.Namespace) -> int:
    """Push scripts/ directory to server (after code changes)."""
    cfg = _load_config(args.config)
    client = _connect(cfg)
    try:
        sftp = client.open_sftp()
        remote_workspace = f"{cfg['remote_path']}/workspace"

        total = 0
        for path_str in PUSH_SCRIPTS:
            local_path = Path(path_str)
            if local_path.is_dir():
                sys.stderr.write(f"  uploading {path_str} ...\n")
                n = _upload_dir(sftp, local_path, remote_workspace, path_str.rstrip("/"))
                total += n
            else:
                sys.stderr.write(f"  [skip] {path_str} not found\n")

        sftp.close()
        sys.stderr.write(f"\n[push-scripts] uploaded {total} files\n")
        return 0
    finally:
        client.close()


def cmd_push_data(args: argparse.Namespace) -> int:
    """Push original data files to server (first time or data update)."""
    cfg = _load_config(args.config)
    client = _connect(cfg)
    try:
        sftp = client.open_sftp()
        remote_workspace = f"{cfg['remote_path']}/workspace"

        total = 0
        for path_str in PUSH_DATA:
            local_path = Path(path_str)
            if local_path.is_file():
                remote_file = f"{remote_workspace}/{path_str}"
                _sftp_mkdir_p(sftp, str(PurePosixPath(remote_file).parent))
                sftp.put(str(local_path), remote_file)
                total += 1
                size_mb = local_path.stat().st_size / (1024 * 1024)
                sys.stderr.write(f"  ✓ {path_str} ({size_mb:.1f} MB)\n")
            else:
                sys.stderr.write(f"  [skip] {path_str} not found\n")

        sftp.close()
        sys.stderr.write(f"\n[push-data] uploaded {total} files\n")
        return 0
    finally:
        client.close()


def cmd_sync_down(args: argparse.Namespace) -> int:
    """Download training/evaluation outputs from server to local.

    Downloads:
    - logs/training/       训练日志和指标
    - logs/optimization/   优化日志
    - reports/evaluation/  评估报告
    - reports/dataset/     数据集分析报告
    - reports/augmentation/增强报告
    - data/augmented/*.jsonl  增强后的索引文件（不含 wav）
    - configs/*/v*         配置版本快照

    不下载：
    - models/iterations/   模型文件太大，需要时用 pull 单独拉
    - data/augmented/*.wav 增强音频只在服务器用
    """
    cfg = _load_config(args.config)
    client = _connect(cfg)
    try:
        sftp = client.open_sftp()
        remote_workspace = f"{cfg['remote_path']}/workspace"

        total = 0

        # 1. 下载日志和报告目录
        for path_str in SYNC_DOWN_DIRS:
            remote_dir = f"{remote_workspace}/{path_str.rstrip('/')}"
            local_dir = Path(path_str.rstrip("/"))
            local_dir.mkdir(parents=True, exist_ok=True)
            sys.stderr.write(f"  downloading {path_str} ...\n")
            n = _download_dir(sftp, remote_dir, local_dir)
            if n > 0:
                sys.stderr.write(f"    ✓ {n} files\n")
            total += n

        # 2. 下载配置版本快照
        if not getattr(args, "skip_configs", False):
            for path_str in SYNC_DOWN_CONFIG_VERSIONS:
                remote_dir = f"{remote_workspace}/{path_str.rstrip('/')}"
                local_dir = Path(path_str.rstrip("/"))
                local_dir.mkdir(parents=True, exist_ok=True)
                n = _download_dir(sftp, remote_dir, local_dir)
                if n > 0:
                    sys.stderr.write(f"  ✓ {path_str} ({n} files)\n")
                total += n

        # 3. 下载增强 jsonl 索引（不含 wav）
        for jsonl_path in SYNC_DOWN_JSONL:
            remote_file = f"{remote_workspace}/{jsonl_path}"
            local_file = Path(jsonl_path)
            local_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                sftp.get(remote_file, str(local_file))
                total += 1
                sys.stderr.write(f"  ✓ {jsonl_path}\n")
            except FileNotFoundError:
                pass  # not generated yet

        # 4. 可选：下载模型
        if getattr(args, "include_models", False):
            remote_dir = f"{remote_workspace}/models/iterations"
            local_dir = Path("models/iterations")
            local_dir.mkdir(parents=True, exist_ok=True)
            sys.stderr.write("  downloading models/iterations/ (large!) ...\n")
            n = _download_dir(sftp, remote_dir, local_dir)
            sys.stderr.write(f"    ✓ {n} files\n")
            total += n

        sftp.close()
        sys.stderr.write(f"\n[sync-down] downloaded {total} files from {cfg['host']}\n")
        return 0
    finally:
        client.close()


def cmd_run(args: argparse.Namespace) -> int:
    """Run a command on the remote server."""
    cfg = _load_config(args.config)
    client = _connect(cfg)
    try:
        workspace = f"{cfg['remote_path']}/workspace"
        remote_python = cfg.get("remote_python", f"{workspace}/.venv/bin/python")

        cmd = args.command
        if cmd.startswith("python "):
            cmd = remote_python + " " + cmd[len("python "):]
        elif cmd.startswith("pip "):
            remote_pip = remote_python.replace("/python", "/pip")
            cmd = remote_pip + " " + cmd[len("pip "):]

        sys.stderr.write(f"[remote] {cfg['username']}@{cfg['host']}:{workspace}\n")
        sys.stderr.write(f"[remote] $ {cmd}\n")
        sys.stderr.write("─" * 60 + "\n")

        code, stdout_text = _stream_exec(client, cmd, cwd=workspace)
        sys.stderr.write(f"\n[remote] exit code: {code}\n")

        # Log remote execution to local execution_log.jsonl
        try:
            import json as _json
            from scripts.core.execution_logger import log_execution
            # Try to parse the JSON output from stdout
            # Scripts output pretty-printed JSON (indent=2), so we need to find
            # the last complete JSON object in the output
            output_payload = {}
            # Find the last '{' ... '}' block in stdout
            last_brace = stdout_text.rfind("}")
            if last_brace >= 0:
                # Search backwards for the matching opening brace
                depth = 0
                start_pos = -1
                for pos in range(last_brace, -1, -1):
                    if stdout_text[pos] == '}':
                        depth += 1
                    elif stdout_text[pos] == '{':
                        depth -= 1
                        if depth == 0:
                            start_pos = pos
                            break
                if start_pos >= 0:
                    json_str = stdout_text[start_pos:last_brace + 1]
                    try:
                        output_payload = _json.loads(json_str)
                    except _json.JSONDecodeError:
                        pass
            log_execution(f"remote:{args.command.split()[0]}", {
                "command": args.command,
                "exit_code": code,
                "output": output_payload,
            })
        except Exception:
            pass  # Don't fail if logging fails

        return code
    finally:
        client.close()


def cmd_gpu(args: argparse.Namespace) -> int:
    """Show nvidia-smi."""
    cfg = _load_config(args.config)
    client = _connect(cfg)
    try:
        code, _ = _stream_exec(client, "nvidia-smi")
        return code
    finally:
        client.close()


def cmd_status(args: argparse.Namespace) -> int:
    """Show server status."""
    cfg = _load_config(args.config)
    client = _connect(cfg)
    try:
        _stream_exec(client, "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null || echo 'No GPU'")
        sys.stdout.write("\n")
        _stream_exec(client, f"df -h {cfg['remote_path']} 2>/dev/null")
        sys.stdout.write("\n")
        _stream_exec(client, "ps aux | grep -E '(train_ds|funasr|run_training)' | grep -v grep | head -5")
        return 0
    finally:
        client.close()


def cmd_pull(args: argparse.Namespace) -> int:
    """Pull a single file from server."""
    cfg = _load_config(args.config)
    client = _connect(cfg)
    try:
        sftp = client.open_sftp()
        remote_file = f"{cfg['remote_path']}/workspace/{args.path}"
        local_file = Path(args.path)
        local_file.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote_file, str(local_file))
        sys.stderr.write(f"[pull] {remote_file} → {local_file}\n")
        sftp.close()
        return 0
    finally:
        client.close()


def cmd_push(args: argparse.Namespace) -> int:
    """Push a single file to server."""
    cfg = _load_config(args.config)
    client = _connect(cfg)
    try:
        sftp = client.open_sftp()
        local_file = Path(args.path)
        remote_file = f"{cfg['remote_path']}/workspace/{args.path}"
        remote_parent = str(PurePosixPath(remote_file).parent)
        _sftp_mkdir_p(sftp, remote_parent)
        sftp.put(str(local_file), remote_file)
        sys.stderr.write(f"[push] {local_file} → {remote_file}\n")
        sftp.close()
        return 0
    finally:
        client.close()


# =====================================================================
# Main
# =====================================================================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Bridge local Kiro to GPU server via SSH + SFTP sync."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Upload commands
    sub.add_parser("push-configs", help="Push configs to server (after local edits).")
    sub.add_parser("push-scripts", help="Push scripts/ to server (after code changes).")
    sub.add_parser("push-data", help="Push original data to server (first time or update).")

    # Download commands
    p_down = sub.add_parser("sync-down", help="Download logs/reports/jsonl from server.")
    p_down.add_argument("--include-models", action="store_true",
                        help="Also download models/iterations/ (large files).")
    p_down.add_argument("--skip-configs", action="store_true",
                        help="Skip downloading config version snapshots.")

    # Remote execution
    p_run = sub.add_parser("run", help="Execute a command on the server.")
    p_run.add_argument("command", help="Command to run (cwd=workspace).")

    # Status commands
    sub.add_parser("gpu", help="Show nvidia-smi.")
    sub.add_parser("status", help="Show server GPU/disk/process status.")

    # Single file transfer
    p_pull = sub.add_parser("pull", help="Pull one file from server.")
    p_pull.add_argument("path", help="Path relative to workspace/.")

    p_push = sub.add_parser("push", help="Push one file to server.")
    p_push.add_argument("path", help="Path relative to workspace/.")

    args = parser.parse_args(argv)

    handlers = {
        "push-configs": cmd_push_configs,
        "push-scripts": cmd_push_scripts,
        "push-data": cmd_push_data,
        "sync-down": cmd_sync_down,
        "run": cmd_run,
        "gpu": cmd_gpu,
        "status": cmd_status,
        "pull": cmd_pull,
        "push": cmd_push,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
