# ASR 系统初始化重置

## 能力描述

将整个 ASR 训练优化系统恢复到"刚搭建好、还没跑过任何迭代"的初始状态。清除所有历史数据，保留代码和原始数据。

## 何时使用

- 需要从零开始重新跑优化闭环
- 历史数据混乱需要清理
- 切换到新数据集重新训练
- 系统状态异常需要彻底重置

## 执行命令

```powershell
# 预览将要删除的内容（不实际删除）
.venv_win\Scripts\python scripts/reset_system.py --dry-run --include-remote

# 执行重置（默认本地 + 远程都重置）
.venv_win\Scripts\python scripts/reset_system.py --confirm --include-remote

# 只重置本地（不动服务器）
.venv_win\Scripts\python scripts/reset_system.py --confirm

# 只重置远程（不动本地）— 需要手动执行远程命令
.venv_win\Scripts\python -m scripts.remote_runner run "rm -rf logs/training logs/optimization reports/evaluation reports/dataset reports/augmentation reports/iterations models/iterations data/augmented configs/model configs/training configs/augmentation configs/post_process && mkdir -p logs/training logs/optimization reports/evaluation reports/dataset reports/augmentation reports/iterations models/iterations data/augmented configs/model configs/training configs/augmentation configs/post_process"
```

**默认行为**：Kiro 执行重置时，始终使用 `--confirm --include-remote`（本地 + 远程都重置），除非用户明确指定只重置本地或只重置远程。

## 清除的内容

| 类别 | 路径 | 说明 |
|------|------|------|
| 数据库 | `metadata.db` | SQLite 数据库（迭代记录、反馈、知识库、策略权重等） |
| 训练日志 | `logs/training/` | 训练指标 JSONL 和 FunASR 原始日志 |
| 优化日志 | `logs/optimization/` | 优化过程日志 |
| 评估报告 | `reports/evaluation/` | 模型评估 JSON 报告 |
| 数据集报告 | `reports/dataset/` | 数据集分析报告 |
| 增强报告 | `reports/augmentation/` | 数据增强报告 |
| 迭代报告 | `reports/iterations/` | 迭代执行报告（含 latest_report.md） |
| 训练模型 | `models/iterations/` | 各迭代训练产出的模型文件 |
| 增强数据 | `data/augmented/` | 增强后的 jsonl 和 wav 文件 |
| 模型配置快照 | `configs/model/v*` | 历史模型配置版本 |
| 训练配置快照 | `configs/training/v*` | 历史训练配置版本 |
| 增强配置快照 | `configs/augmentation/v*` | 历史增强配置版本 |
| 后处理配置快照 | `configs/post_process/v*` | 历史后处理配置版本 |
| Skill 版本快照 | `knowledge/skill_versions/` | Skill 文件历史版本 |
| Skill 文件 | `.kiro/skills/*.md` | 回退到初始基线版本（从 `knowledge/skill_originals/` 恢复） |

## 保留的内容（不删除）

| 类别 | 路径 | 说明 |
|------|------|------|
| 原始数据 | `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl` | FunASR jsonl 格式原始数据 |
| MUSAN 噪声 | `data/musan/` | 增强用噪声数据 |
| 主配置 | `configs/funasr_config.yaml`, `configs/remote.yaml` | 当前生效的配置文件 |
| 脚本代码 | `scripts/` | 所有 Python 脚本 |
| Skill 基线 | `knowledge/skill_originals/` | Skill 文件的初始版本备份 |
| Steering 文件 | `.kiro/steering/` | Kiro Steering 工作流文件 |
| 虚拟环境 | `.venv_win/`, `.venv310/` | 本地和远程 Python 环境 |
| 基线模型 | `models/baseline/` | 基线模型（如有） |

## 重置后的系统状态

重置完成后，系统处于以下状态：

1. **数据库为空** — 无迭代记录、无反馈、无知识库条目、无策略权重
2. **无训练产物** — 无模型文件、无日志、无报告
3. **无增强数据** — 需要重新运行 `run_augmentation --all`
4. **配置版本目录为空** — 无历史快照（`configs/model/`、`configs/training/` 等为空）
5. **主配置保持不变** — `funasr_config.yaml` 不会被重置
6. **Skill 文件恢复到初始版本** — 从 `knowledge/skill_originals/` 恢复（需先 `--save-baseline`）
7. **远程服务器同步清空** — 使用 `--include-remote` 时，服务器上对应目录也被清空

## 重置后的标准启动流程

```powershell
cd workspace

# 1. 数据增强（必须先做，在服务器执行）
.venv_win\Scripts\python -m scripts.remote_runner run "python -m scripts.run_augmentation --all"

# 2. 创建新迭代（本地）
.venv_win\Scripts\python -m scripts.manage_iterations start --description "首轮基线训练"

# 3. 启动训练（服务器）
.venv_win\Scripts\python -m scripts.remote_runner run "python -m scripts.run_training --tag <iteration_id>"

# 4. 评估（服务器）
.venv_win\Scripts\python -m scripts.remote_runner run "python -m scripts.run_inference --model-dir models/iterations/<iteration_id> --evaluate --tag <iteration_id>"

# 5. 下载产物到本地
.venv_win\Scripts\python -m scripts.remote_runner sync-down

# 6. 记录反馈（本地）
.venv_win\Scripts\python -m scripts.record_feedback label --iteration-id <iteration_id> --cer-before X --cer-after Y
```

## 注意事项

- `--save-baseline` 在首次使用前执行一次，保存 Skill 文件的原始版本
- `--confirm` 参数是必须的，防止误操作
- `--dry-run` 可以先预览，确认无误再执行
- 远程服务器重置需要 `--include-remote`，会通过 SSH 清除服务器上对应目录（`rm -rf` + 重建空目录）
- 数据库删除后所有历史记录不可恢复，请确认不需要保留
- 主配置文件 `funasr_config.yaml` 不会被重置，如需恢复默认参数请手动修改
- 如果没有执行过 `--save-baseline`，Skill 文件不会被回退（会提示警告）
