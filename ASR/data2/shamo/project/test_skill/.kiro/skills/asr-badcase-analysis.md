# ASR Badcase 分析

## 能力描述

对 badcase（识别错误样本）进行模式识别，归纳错误模式，生成针对性优化建议。

## 何时使用

- 评估报告生成后，分析错误分布
- CER 改善停滞时，深入分析 badcase 寻找突破口
- 需要决定热词提取 vs 数据增强 vs 训练参数调整时

## 执行步骤

### 1. 生成评估报告（含 badcases）
```powershell
.venv_win\Scripts\python -m scripts.analyze_results --test-set reports/evaluation/inference_iter_xxx.jsonl --tag iter_xxx
```

### 2. 查看错误分布
报告中 `error_type_distribution` 字段：
```json
{"substitution": 45, "deletion": 12, "insertion": 8}
```

### 3. 自动诊断（含错误模式分析）
```powershell
.venv_win\Scripts\python -m scripts.auto_diagnose --verbose
```
输出 `error_ratios`（各类型占比）和 `repetition_ratio`。

### 4. 策略匹配
```powershell
.venv_win\Scripts\python -m scripts.manage_strategies match --eval-report reports/evaluation/eval_report_iter_xxx.json
```

## 错误模式 → 优化动作

| 模式 | 特征 | 动作 |
|------|------|------|
| 同音字替换 | substitution 为主，高频 wrong→correct 对 | 分析候选热词 → 人工审核 → 加入 SeaCo 热词表 |
| 背景人声干扰 | insertion 多，无关词汇插入 | babble 增强 + SeACo attention_weight ↑ |
| 尾部截断 | deletion 集中在句末 | 检查 VAD / 增加 reverb 增强 |
| 重复输出 | hypothesis 中重复片段 | 增加 CTC 权重（模型层面解决） |
| 整体高 CER | 所有类型都多 | 全量增强 + 增加 epoch |
| 专有名词错误 | 特定词反复错 | 加入 SeaCo 热词表（人工确认） |

## 错误严重程度

| CER 范围 | 严重程度 | 处理 |
|---------|---------|------|
| > 50% | 严重 | 可能是标注错误，验证数据质量 |
| 20%-50% | 中等 | 优先处理 |
| 10%-20% | 轻微 | 常规优化 |
| < 10% | 可接受 | 持续监控 |

## 常见问题

- **badcase 中 CER > 50% 的样本**：优先检查是否标注错误（reference 和 audio 不匹配）
- **substitution 集中在少数字**：直接提取热词，效果立竿见影
- **场景间 CER 差异大**：针对最差场景做专项增强
- **评测统一化影响**：CER 计算前已做 NFKC + 去标点 + 去空格，空格/标点差异不会计入错误

## 知识库引用

分析完 badcase 后，将发现的模式和解决方案写入知识库供后续迭代参考：

```powershell
# 查询历史 badcase 解决经验
.venv_win\Scripts\python -m scripts.manage_knowledge query --type scene_solution

# 查询特定场景的历史方案
.venv_win\Scripts\python -m scripts.manage_knowledge query --type scene_solution --content-match "{\"scene\":\"background_voice\"}"

# 写入新发现的模式
.venv_win\Scripts\python -m scripts.manage_knowledge add --type scene_solution --content "{\"scene\":\"background_voice\",\"pattern\":\"homophone_substitution\",\"solution\":\"hotword_correction\"}" --confidence 0.8 --source-iteration iter_xxx
```
