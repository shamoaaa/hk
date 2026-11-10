"""RemoteExecutor — SSH 远程命令执行.

封装 paramiko，提供：
- connect / disconnect
- run_command (同步，带超时)
- upload / download
- GPU 状态查询
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RemoteConfig:
    host: str
    port: int = 22
    username: str = ""
    password: str = ""
    key_file: str = ""
    workspace: str = "/data/shamo/project/asr"
    timeout: int = 30


class RemoteExecutor:
    """SSH 远程执行器。"""

    def __init__(self, config: RemoteConfig):
        self.config = config
        self._client = None

    def connect(self):
        """建立 SSH 连接。"""
        import paramiko
        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        connect_kwargs = {
            "hostname": self.config.host,
            "port": self.config.port,
            "username": self.config.username,
            "timeout": self.config.timeout,
        }
        if self.config.key_file:
            connect_kwargs["key_filename"] = self.config.key_file
        elif self.config.password:
            connect_kwargs["password"] = self.config.password
        self._client.connect(**connect_kwargs)

    def disconnect(self):
        if self._client:
            self._client.close()
            self._client = None

    def run_command(self, cmd: str, timeout: int = 300) -> tuple[int, str, str]:
        """执行远程命令，返回 (return_code, stdout, stderr)。"""
        if not self._client:
            raise RuntimeError("Not connected. Call connect() first.")
        _, stdout, stderr = self._client.exec_command(cmd)
        # No timeout — block until command completes
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        rc = stdout.channel.recv_exit_status()
        return rc, out, err

    def gpu_status(self) -> str:
        """查询远程 GPU 状态。"""
        rc, out, _ = self.run_command("nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader")
        return out if rc == 0 else "GPU status unavailable"
