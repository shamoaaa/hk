# ASR Badcase 模式识别

## 能力描述

对 ASR 模型的 badcase（识别错误样本）进行模式识别，归纳常见错误模式，并生成针对性的优化建议。本 Skill 是闭环的核心节点：它分析评估结果，并直接对应到下一轮迭代的 **前置数据优化手段**。

## 闭环：badcase → 前置手段

**必读**：badcase 分析的结果**必须对应到 `scripts/badcase_strategy_map.py` 定义的前置手段**，才构成闭环。完整流程：

```
evaluation 输出 → badcase 模式识别 → match_strategies() → 下一轮迭代 data 增强参数
     ↑                                                              │
     └──────────────────── skill_evolution 记录反馈 ←───────────────┘
```

每次分析完 badcase 后，调用：

```python
from scripts.badcase_strategy_map import match_strategies
matched = match_strategies(evaluation_report, badcases)
# matched 是按优先级排序的推荐手段列表，含 commands + config_changes
```

或者 CLI 等价操作：查询知识库里对应类型的历史最优策略。

## 数据来源

通过 CLI 脚本获取 badcase 分析数据：

```bash
python scripts/analyze_results.py --model-path <model_path> --test-set <test_set_path>
```

## 分析规则

### 核心映射表（必须对照执行）

| badcase 触发条件 | → 前置手段名称 | CLI | 维度 |
|-----------------|---------------|-----|------|
| `insertion > substitution` | babble_augment_for_insertion | `run_augmentation.py --all --strategy babble` + `noise` | data |
| `substitution >= max(ins, del)` | hotwords_from_substitution | `extract_hotwords.py` + `apply_postprocess.py` | post_process |
| `substitution > 0 且 CER > 0.10` | specaug_for_substitution | 开启 `training.spec_aug.enabled` | train_config |
| `deletion > insertion 且 deletion > substitution * 0.5` | reverb_speed_for_deletion | `run_augmentation.py --all --strategy reverb/speed` | data |
| `overall_cer > 0.20` | full_augment_for_high_cer | `run_augmentation.py --all` | data |
| 某场景样本数 < 20 | balance_scene_augment | `preprocess_metadata.py` + 针对性增强 | data |

这张表**同时写在代码里**（`badcase_strategy_map.DATA_STRATEGIES`），修改映射时改代码表，这里的文档自动同步。

### 错误模式识别

#### 同音字/近音字替换模式

当替换错误中存在高频的同音字对时，**直接调用 extract_hotwords.py**：

```bash
python scripts/extract_hotwords.py \
    --eval-report reports/evaluation/eval_report_iter_xxx.json \
    --merge --min-count 3
```

脚本会：
1. 对每个 badcase 做编辑距离对齐
2. 提取所有替换操作的 (hyp → ref) 对
3. 统计频率，保留 ≥ 3 次的对
4. 合并到 `configs/post_process/hotwords.txt`

常见中文同音字替换（先验知识，无需等 badcase 出现）：
- "好" ↔ "号"、"已经" ↔ "以经"、"那里" ↔ "哪里"
- "在" ↔ "再"、"的" ↔ "地" ↔ "得"

#### 背景人声干扰模式

**检测方式**：插入错误占比 > 30% 或 insertion > substitution → 对应 `babble_augment_for_insertion`

- 识别文本频繁出现无关词汇（被背景人声污染的特征）
- 插入错误占比显著（> 30%）

建议：
- 添加语音增强前端模块
- 使用 MUSAN speech 数据集进行 babble 增强训练
- 调整 SeACo 模块的注意力权重

优先级：5，置信度：0.75（无场景标签时置信度略降）

#### 尾部截断模式

当删除错误集中在句末位置时：
- 检查是否与端点检测（VAD）相关
- 检查是否与音频时长分布相关
- 建议：
  - 调整端点检测参数
  - 增加包含完整句尾的训练数据
- 优先级：3，置信度：0.7

#### 数字/专有名词错误模式

当数字或专有名词的错误率显著高于普通文本时：
- 生成包含数字和专有名词的热词表
- 建议增加领域特定数据
- 优先级：3，置信度：0.75

### 场景关联分析

**注意**：本项目使用 FunASR jsonl 格式，数据中不包含 scene_type 字段。因此 badcase 分析不基于场景分组，而是基于**错误模式**进行分类。

如果未来的 test_set.jsonl 中包含了 scene_type 字段（通过人工标注或推理时附加），则可以按场景分别统计 CER。当前默认所有样本场景为 `unknown`。

### 错误严重程度评估

| CER 范围 | 严重程度 | 处理优先级 |
|---------|---------|----------|
| > 50% | 严重 | 立即关注 |
| 20%-50% | 中等 | 优先处理 |
| 10%-20% | 轻微 | 常规优化 |
| < 10% | 可接受 | 持续监控 |

## 建议生成模板

### 热词纠正建议
```json
{
  "dimension": "post_process",
  "title": "添加热词纠正规则",
  "config_changes": {
    "hotword_entries": [
      {"wrong_text": "错误词", "correct_text": "正确词", "priority": 1}
    ]
  }
}
```

### 数据增强建议
```json
{
  "dimension": "data",
  "title": "增加场景数据增强",
  "config_changes": {
    "augmentation_types": ["background_voice"],
    "target_scene": "background_voice",
    "sample_count": 1000
  }
}
```

### 模型调整建议
```json
{
  "dimension": "model",
  "title": "调整 SeACo 参数",
  "config_changes": {
    "seaco_params": {
      "co_decoder_layers": 3,
      "attention_weight": 0.7
    }
  }
}
```

## 知识库引用

分析时参考 `knowledge/knowledge_base.json` 中的：
- 历史 badcase 模式和成功的解决方案
- 不同场景下的最优后处理规则
- 错误类型与优化维度的关联经验

查询知识库中的 badcase 解决经验：

```bash
# 查询场景相关的解决方案
python scripts/manage_knowledge.py query --type scene_solution --content-match '{"scene":"background_voice"}'

# 查看知识库全貌
python scripts/manage_knowledge.py view
```

分析完成后，将发现的错误模式和解决方案写入知识库：

```bash
python scripts/manage_knowledge.py add --type scene_solution --content '{"scene":"background_voice","pattern":"homophone_substitution","solution":"hotword_correction","hotwords":["温件->文件","报高->报告"]}' --confidence 0.8 --source-iteration <iter_id>
```
