# ASR 后处理规则生成

## 能力描述

基于 badcase 分析结果，自动生成后处理规则，包括热词纠正规则、正则替换规则和语言模型重排序配置。后处理流水线按固定顺序执行：热词纠正 → 正则替换 → 语言模型重排序。

## 数据来源

通过 CLI 脚本获取 badcase 分析数据和当前后处理配置：

```bash
python scripts/analyze_results.py --model-path <model_path> --test-set <test_set_path>
```

## 分析规则

### 热词纠正规则生成

#### 触发条件
- badcase 中存在高频替换错误（同一替换对出现 ≥ 3 次）
- 替换错误涉及同音字或近音字

#### 规则生成逻辑

1. **统计替换频率**：从 badcase 的 error_details 中提取所有替换对
2. **过滤高频替换**：保留出现次数 ≥ 3 的替换对
3. **验证正确性**：确保 correct_text 在参考文本中确实出现
4. **设置优先级**：按出现频率排序，频率越高优先级越高

#### 常见热词规则

| 错误词 | 正确词 | 场景 | 优先级 |
|-------|-------|------|-------|
| 号 | 好 | 通用 | 1 |
| 以经 | 已经 | 通用 | 2 |
| 在见 | 再见 | 通用 | 2 |
| 那里 | 哪里 | 疑问句 | 3 |
| 做 | 坐 | 交通场景 | 3 |

#### 输出格式

```json
{
  "hotword_entries": [
    {
      "wrong_text": "号",
      "correct_text": "好",
      "priority": 1
    }
  ]
}
```

### 正则替换规则生成

#### 触发条件
- badcase 中存在规律性格式错误
- 存在可用正则表达式描述的错误模式

#### 常见正则规则

1. **多余空格清理**
   ```json
   {
     "pattern": "\\s+",
     "replacement": "",
     "description": "移除多余空格",
     "priority": 1
   }
   ```

2. **数字格式统一**
   ```json
   {
     "pattern": "([零一二三四五六七八九十百千万亿]+)",
     "replacement": "\\1",
     "description": "保留中文数字格式",
     "priority": 2
   }
   ```

3. **标点符号清理**
   ```json
   {
     "pattern": "[，。！？、；：""''（）【】]",
     "replacement": "",
     "description": "移除中文标点（如果目标是纯文本）",
     "priority": 3
   }
   ```

#### 规则验证
- 正则表达式必须是合法的 Python re 模式
- 替换结果不应引入新的错误
- 建议在验证集上测试规则效果

### 语言模型重排序配置

#### 触发条件
- N-best 候选中存在正确结果但未被选为 top-1
- 替换错误率较高，语言模型可能改善选择

#### 配置建议

```json
{
  "enable_lm_rerank": true,
  "lm_model_path": "/models/lm/zh_lm.bin",
  "n_best_size": 5,
  "lm_weight": 0.3
}
```

参数说明：
- `n_best_size`：候选数量，建议 3-10
- `lm_weight`：语言模型权重，建议 0.1-0.5

### 后处理流水线配置

#### 执行顺序（固定）

1. **热词纠正**（hotword）：基于精确匹配的词汇替换
2. **正则替换**（regex）：基于模式匹配的文本替换
3. **语言模型重排序**（lm_rerank）：基于语言模型得分的候选选择

#### 配置模板

```json
{
  "hotword_entries": [],
  "regex_rules": [],
  "enable_lm_rerank": false,
  "lm_model_path": null,
  "pipeline_order": ["hotword", "regex", "lm_rerank"]
}
```

### 效果评估

后处理效果通过以下指标评估：
- **CER 变化**：后处理后 CER - 后处理前 CER（负值表示改善）
- **WER 变化**：后处理后 WER - 后处理前 WER
- **规则命中率**：每条规则被应用的次数

评估命令：
```bash
python scripts/apply_postprocess.py --config configs/post_process/config.yaml --evaluate
```

## 建议生成模板

```json
{
  "dimension": "post_process",
  "title": "后处理规则优化",
  "description": "基于 badcase 分析生成的后处理规则",
  "config_changes": {
    "hotword_entries": [
      {"wrong_text": "号", "correct_text": "好", "priority": 1}
    ],
    "regex_rules": [
      {"pattern": "\\s+", "replacement": "", "description": "移除空格", "priority": 1}
    ],
    "enable_lm_rerank": true,
    "lm_model_path": "/models/lm/zh_lm.bin"
  },
  "rationale": "基于 badcase 中高频替换错误模式生成热词规则"
}
```

## 知识库引用

分析时参考 `knowledge/knowledge_base.json` 中的：
- 历史后处理规则及其效果
- 不同场景下的最优后处理配置
- 热词表更新与 CER 改善的关联

查询知识库中的后处理经验：

```bash
# 查询场景相关的后处理方案
python scripts/manage_knowledge.py query --type scene_solution --content-match '{"dimension":"post_process"}'

# 查询数据策略中与后处理相关的经验
python scripts/manage_knowledge.py query --type data_strategy
```

后处理优化完成后，将有效规则写入知识库：

```bash
python scripts/manage_knowledge.py add --type scene_solution --content '{"dimension":"post_process","rule_type":"hotword","entries_count":5,"effect":"CER improved 2%"}' --confidence 0.8 --source-iteration <iter_id>
```
