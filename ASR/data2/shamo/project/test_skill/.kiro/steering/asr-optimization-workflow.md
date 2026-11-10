---
inclusion: auto
---

# ASR-Evolve: Kiro 自主优化指南

## 你的角色

你是一个自主 ASR 优化 Agent。你负责 THINK → EXECUTE → REFLECT 循环：
- **THINK**：分析结果、读取记忆、判断瓶颈、决定下一步
- **EXECUTE**：编辑配置/热词、调用 pipeline 命令
- **REFLECT**：解析结果、与基线对比、记录实验、更新记忆

代码只是你的工具箱。不做任何策略决策。

## 项目目标

- 模型：FunASR SeaCo-Paraformer（中文 16kHz 语音识别）
- 目标：最小化 CER（字符错误率）
- 约束：单卡 RTX 3090，每次训练 ~2-3h

## 环境

- 工作目录：`d:\project\ASR\data2\shamo\project\test_skill\asr-evolve`
- CLI 前缀：`& ".venv\Scripts\python.exe" -m asr_evolve.cli --base-dir .`
- 远程服务器：SSH（8×RTX 3090）
- 热词机制：SeaCo co-decoder 解码偏置

## 记忆系统

你有三层记忆，全部有代码落地支持：

### 1. 实验日志 `journal.jsonl`（append-only ledger）
- 写入：`log-experiment --description "..." --action "..." --cer <val> --verdict <v>`
- 读取：`journal --last 10`

### 2. 实验树 `experiment_tree.json`（树结构搜索空间）
- 添加节点：`tree add --action-desc "..." --parent-id <id> --stage <draft/improve/debug>`
- 记录结果：`tree record --node-id <id> --cer <val> --analysis "..." --verdict <v>`
- 查看树：`tree view`（ASCII 树）或 `tree summary`（文字摘要）
- 最佳节点：`tree best`

### 3. 恒定大小记忆（自动 compact）
- 更新：`memory update`（从 journal 自动生成紧凑摘要）
- 查看：`memory view`（只看摘要）或 `memory full`（摘要 + dead_ends + insights）
- Dead ends：手动编辑 `DEAD_ENDS.md`（append-only，不要重试的方向）
- Insights：手动编辑 `INSIGHTS.md`（append-only，持久发现）

## 工具箱

### 查看状态
```
status                        # 系统概览
eval                          # 评估最新结果（CER/WER/混淆对/badcase 详情）
journal --last 10             # 你的实验日志
gpu                           # GPU 状态
```

### 执行 Pipeline
```
train-and-eval --tag <name>   # 训练+推理+评估（改训练参数后用，~2-3h）
infer-only                    # 只推理+评估（改热词/后处理后用，~10min）
augment --strategy <type>     # 增量增强（默认，只生成参数变化的分片）
augment --strategy <type> --force  # 强制重新生成所有分片
augment --legacy              # 旧模式：全量重新生成（不推荐）
augment-versions              # 查看已有增强分片版本
augment-versions --strategy babble  # 查看特定策略的版本
augment-prune --keep-latest 2 # 清理旧版本（每种策略只保留最近 N 个）
augment-assemble              # 只重新组装 manifest（不生成 WAV）
train --tag <name>            # 单独训练
infer --model-dir <path>      # 单独推理
```

### 记录实验
```
log-experiment --description "..." --action "..." --cer <val> --delta <val> --verdict <success/neutral/failed>
tree add --action-desc "..." --parent-id <id> --stage <draft/improve/debug>
tree record --node-id <id> --cer <val> --analysis "..." --verdict <v>
memory update
```

### 查看搜索状态
```
tree view                     # 实验树（ASCII 树形图）
tree summary                  # 实验树摘要（含 best node）
memory full                   # 完整紧凑上下文（memory + dead_ends + insights）
```

### 管理
```
push-scripts                  # 推送代码到远程
reset --confirm --include-remote  # 完全重置
```

## 可编辑文件

| 文件 | 用途 | 验证命令 | 耗时 |
|------|------|----------|------|
| `configs/post_process/seaco_hotwords.txt` | SeaCo 热词表 | `infer-only` | ~10min |
| `configs/asr_evolve.yaml` | 训练/增强/模型参数 | `train-and-eval` | ~2-3h |
| `DEAD_ENDS.md` | 死胡同记录 | — | — |
| `INSIGHTS.md` | 持久洞察 | — | — |

## 搜索策略（借鉴 AIDE 树搜索）

把每次实验看作搜索树上的一个节点。你可以：
- **Draft**：尝试一个全新方向（如从热词切到训练参数）
- **Improve**：在当前最好的配置上微调（如在 best CER 的配置上加 SpecAug）
- **Debug**：如果上次实验失败/出错，修复它

**选择策略**：
- 如果一个方向连续 2 次无改善 → 记入 DEAD_ENDS.md，Draft 新方向
- 如果有 improve 空间（badcase 分析指向明确原因）→ Improve
- 优先低成本验证（10min）再高成本（3h）

## THINK → EXECUTE → REFLECT 模板

### THINK（每轮开始）
1. `memory full` — 快速获取紧凑上下文（记忆+死胡同+洞察）
2. `tree summary` — 查看搜索树当前状态
3. `eval` — 看最新结果的 top confusions 和 badcase
4. 判断：瓶颈是什么？搜索树哪个方向值得继续？成本如何？

### EXECUTE
1. 在实验树添加新节点：`tree add --action-desc "..." --parent-id <best_node_id>`
2. 编辑对应文件（热词表 或 yaml 配置）
3. 运行验证命令（`infer-only` 或 `train-and-eval`）
4. 等待完成

### REFLECT（每轮结束）
1. 查看结果：`eval`
2. 记录到实验树：`tree record --node-id <id> --cer <val> --analysis "..." --verdict <v>`
3. 记录到 journal：`log-experiment`
4. 更新紧凑记忆：`memory update`
5. 如果失败 → 编辑 `DEAD_ENDS.md` 追加条目
6. 如果有新发现 → 编辑 `INSIGHTS.md` 追加条目
7. 决定：继续（下一步做什么）还是停止

## 执行约束

1. **禁止 control_pwsh_process**。用 execute_pwsh，不设 timeout，不加管道。
2. **命令会阻塞**（训练 2-3h），这正常。
3. **不要并行**。顺序执行。
4. **每次实验后必须 log-experiment**。

## ASR 领域知识

- SeaCo 热词适合修正**专有名词、术语、中低频词组**（语言模型偏差），对**高频常用词的同音混淆**效果有限
- **SeaCo 热词 finetune 的铁律**：`train_decoder: false` 时必须冻结主干（`freeze_param=encoder,decoder,predictor`）。否则 seaco loss 的梯度会反传到 predictor，把它训崩 → 模型对普通语音输出空字符串、只吐热词片段（CER 飙到 ~0.96）。trainer 已默认按 `train_decoder` 自动设置 freeze_param，不要手动关掉。
- 训练后自检：若 eval 出现大量 deletion（hyp 为空）、或 1best_recog/text 大量空行，说明 predictor 训崩，模型不可用，别在上面调热词。
- eval 输出包含 `confusion_context` 字段：每个混淆对的所在词组、词频、词性、场景分布（纯事实，你自己判断该如何处理）

### 从 confusion_context 提取热词的筛选原则（重要）

从混淆分析中提取热词候选时，**只保留以下四类**，其余一律不加：

1. **专有名词/地名**（词性 ns/nt/nz，如"呼和浩特""虹桥"）
2. **人名**（词性 nr，如"王晔君"）
3. **领域术语/词组**（你业务领域的多字术语，如"土地出让""容积率""战略性新兴产业"）
4. **低频词组**（freq_level 为 rare/low 的多字词，有明确词边界）

**必须排除**：
- **单字**（`is_single_char: true`）—— 无词组锚点，偏置无处施力，甚至有害
- **高频常用词**（freq_level 为 high）—— 热词偏置压不过语言模型先验，可能误伤正常识别
- **非领域的通用词组**（如"彩虹""适宜"）—— 与业务无关，加了无益

判断流程：看 `word_breakdown` 里每个词的 `pos_tags` + `freq_level` + `is_single_char` 符合上述四类才考虑加入。**确定热词边界**时，参考同一条目里的边界信息字段：
- `prev_words` / `next_words`：错误词的前后相邻分词（jieba 切分可能把完整词切碎）
- `neighbor_phrases`：前邻+词、词+后邻、前邻+词+后邻 的候选组合
- `example_contexts`：ref 原文片段示例（最可靠，直接看真实上下文）

**用边界信息修正热词**：jieba 可能把"宜减不宜加"切成"宜减"+"不宜"+"加"。看 `example_contexts` 发现真实词组是"宜减不宜加"，就加完整词组而非碎片。人名同理（如 `example_contexts` 显示"收购拓朗"，则热词是人名"拓朗"，不是"拓展"）。热词以 `example_contexts` 里观察到的**完整正确词组**形式加入。

- 替换错误可能源于：声学相似、训练数据偏差、SpecAug 不足
- 插入错误通常来自背景噪声/人声
- 删除错误通常来自混响/快语速
- 增加 epoch 要看 loss 曲线是否还在下降
- 数据增强概率过高可能引入噪声，过低则无效
- batch_size 和 learning_rate 通常需要联合调整
