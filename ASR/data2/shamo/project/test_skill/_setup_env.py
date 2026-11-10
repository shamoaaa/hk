# -*- coding: utf-8 -*-
"""检查服务器环境并安装 FunASR。"""
import paramiko, io

HOST, PORT, USER, PASS = "10.6.217.118", 20028, "root", "123456"
LOG = r"d:\project\ASR\data2\shamo\project\test_skill\_setup_env_log.txt"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
logf = io.open(LOG, "w", encoding="utf-8")

def run(cmd, timeout=30):
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    rc = stdout.channel.recv_exit_status()
    out = stdout.read().decode()
    err = stderr.read().decode()
    logf.write(f"$ {cmd}\n  rc={rc}\n  stdout: {out[:500]}\n")
    if err.strip():
        logf.write(f"  stderr: {err[:300]}\n")
    logf.flush()
    return rc, out, err

try:
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
    logf.write("connected\n\n")
    
    # 1. 基础环境
    logf.write("=== 基础环境 ===\n")
    run("cat /etc/os-release | head -5")
    run("python3 --version 2>&1 || python --version 2>&1")
    run("pip3 --version 2>&1 || pip --version 2>&1")
    run("which conda 2>&1")
    run("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1")
    run("nvcc --version 2>&1 | grep release")
    run("python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())' 2>&1")
    
    # 2. 磁盘空间
    logf.write("\n=== 磁盘空间 ===\n")
    run("df -h /data")
    
    # 3. 已有的包
    logf.write("\n=== 已有 Python 包 ===\n")
    run("pip3 list 2>/dev/null | grep -i -E 'torch|funasr|torchaudio|modelscope' || pip list 2>/dev/null | grep -i -E 'torch|funasr|torchaudio|modelscope'")
    
    client.close()
    logf.write("\ndone\n")
except Exception as e:
    logf.write(f"ERROR: {e}\n")
logf.close()
print("ENV_CHECK_DONE")
