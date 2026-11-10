# 11_analyse 目录脚本报告

本目录是流水线的**深度错误分析阶段**，消费 10 阶段评估产出的 `eval_detail.jsonl`，从中计算指令召回率、做唤醒词剥离后的 CER 重算、跨模型/跨版本错误对比，并导出可人工审阅的错误样本清单。所有脚本都是**纯后处理**——只读 jsonl，不加载模型、不做推理。

## 目录结构

```
11_analyse/
  scripts/        12 个分析脚本（本报告主体）
  data/           临时数据样本（eval_detail.jsonl / eval_errors.txt 等）
  reports/        生成的报告输出（testset / v3_vs_v4 / v3_vs_v5）
```

## 核心概念

整个目录围绕几个反复出现的概念：

| 概念 | 含义 |
|---|---|
| **指令召回率** | `ref_norm in hyp_norm`——标签作为**子串**出现在识别结果中的样本比例。衡量"指令有没有被完整识别出来"，比 CER 更贴近业务（萤石音箱要的是指令命中） |
| **唤醒词剥离** | 去除开头/结尾的"你好萤石"及其 20+ 变体后重算 CER/召回率。唤醒词残留会虚高 CER、干扰排名 |
| **错误三分类** | 跨两个模型对比时分为：两者都错 / 仅 A 错（B 对）/ 仅 B 错（A 对），定位各版本独有的优势与短板 |
| **eval_detail.jsonl 字段** | `uid / ref / hyp / ref_norm / hyp_norm / cer / sub / ins / del / ref_len / correct` |

唤醒词列表与 `model-eval-analysis.md` 规范中的完全对应（按长度降序匹配，最多剥离 3 轮）。

---

## 一、汇总统计类

### 1. `analyse_eval_detail.py` — 单文件 CER 统计 + 错误导出

| 项 | 说明 |
|---|---|
| **输入** | 硬编码单个 `eval_detail1.jsonl` |
| **输出** | `eval_errors1.txt`（统计头 + 按 CER 降序的错误样本） |

最基础的脚本：累计 sub/ins/del 算整体 CER，统计句准确率，把所有 `correct=false` 的样本按 CER 降序写出（标签/识别并列）。路径写死，无参数。

### 2. `calc_command_recall.py` — 两文件指令召回率对比

| 项 | 说明 |
|---|---|
| **输入** | 两个硬编码 jsonl（`eval_detail.jsonl` / `eval_detail1.jsonl`） |
| **输出** | 控制台打印 |

计算两个模型的指令召回率（`ref in hyp`），按 CER 分段统计未召回样本，做**差异分析**：模型1未召回但模型2召回（模型2优势）vs 反向，并打印各自独有的未召回样本（前 30 条）。

### 3. `analyse_all_models.py` — 6 模型 × 多数据集汇总

| 项 | 说明 |
|---|---|
| **输入** | `eval_output/` 下 v0.1.0~v0.3.0 各 2 权重的 `eval_detail.jsonl`（`reflow_label/{s10_reflow,sp4_reflow,s10_augment}/test`）+ `testset_results/` |
| **输出** | 控制台对比表 |

`rglob` 递归加载每个模型目录下所有 `eval_detail.jsonl`，对 3 个业务数据集 + 12_test 计算指令召回率/CER/句准确率/错误类型，逐数据集横向对比 6 个模型。重点突出 sp4_reflow（核心业务）和 s10_augment（CER 极高）。

## 二、testset 专项分析类

### 4. `analyse_testset.py` — testset 6 模型错误特征深挖

| 项 | 说明 |
|---|---|
| **输入** | `testset_results/` 下 6 个模型的 `eval_detail.jsonl` |
| **输出** | 控制台 |

最详尽的 testset 分析：
- `classify_error` 把错误分为唤醒词残留 / 多识别(环境音) / 漏识别(纯删除) / 替换错误 / 指令淹没(大量插入) / 混合错误等
- **未召回样本集合运算**：所有模型都未召回的"硬骨头"、v0.3.0 独有未召回（其他能召回它不能）、v0.2.0 独有未召回——精准定位版本弱点
- **高频混淆对**：统计 v0.3.0 的 `ref→hyp` 替换混淆对（出现≥2次）

### 5. `analyse_testset_strip_wakeword.py` — 唤醒词剥离（仅识别结果）

| 项 | 说明 |
|---|---|
| **输入** | `testset_results/` 6 模型 jsonl |
| **输出** | 控制台 |

**只剥离 hyp** 的开头/结尾唤醒词后重算 CER（自带回溯版编辑距离 `compute_cer`），对比剥离前后的召回率提升/CER 降低/插入减少，最后给出去唤醒词后排名。揭示"唤醒词残留对 CER 的虚高贡献"。

### 6. `analyse_testset_strip_wakeword_v2.py` — 唤醒词剥离（标签+识别都剥）

| 项 | 说明 |
|---|---|
| **输入** | 同上 |
| **输出** | 控制台 |

v1 的升级版，**对 ref 和 hyp 都剥离**唤醒词（更公平的对比，因为标签里也可能含唤醒词）。扩展了唤醒词变体列表（区分开头 `WAKEWORDS` / 结尾 `WAKEWORDS_END`），标签剥完为空的条目跳过，同时按召回率和 CER 双排名。这是规范里 "v2 对标签和识别结果做同样处理" 描述对应的脚本。

### 7. `export_testset_errors.py` — testset 错误样本导出（v0.2.0/v0.3.0）

| 项 | 说明 |
|---|---|
| **输入** | `testset_results/` v0.2.0、v0.3.0 jsonl |
| **输出** | `testset_errors_{model}_stripped.txt` ×2 + 控制台 |

剥离标签+识别唤醒词后，导出每个模型的全部错误样本（带 `classify_error` 分类标签、是否召回、原始 vs 剥离文本），并做两模型差异分析（v0.2.0 错但 v0.3.0 对、反向、都错）。

## 三、跨业务模型对比类

### 8. `compare_all_models_recall.py` — 4 模型业务召回率对比

| 项 | 说明 |
|---|---|
| **输入** | seaco/online 基线 + v0.2.0/v0.3.0 在 `s10_reflow,sp4_reflow,s10_augment` 的 jsonl |
| **输出** | 控制台（含路径检查、逐数据集对比、汇总表、vs online 改进幅度） |

`find_eval_detail` 容错查找路径（test 子目录或直接目录）。对比 4 个模型在 3 个业务数据集的指令召回率，并专门算 v0.2.0/v0.3.0 相对 **online 线上模型** 的改进幅度——对应规范里"必须包含 seaco 和 online"的对比规则。

### 9. `export_v3_v4_errors.py` — v0.3.0 vs v0.4.0 错误导出（原始）

| 项 | 说明 |
|---|---|
| **输入** | v0.3.0、v0.4.0 在 `s10_reflow/test`、`sp4_reflow/test` 的 jsonl |
| **输出** | `errors_v3_v4_{ds}.txt` ×2 |

**不剥离唤醒词**，直接用原始 `ref_norm`/`hyp_norm` 找出任一模型出错的样本，按"两者都错/仅 v3 错/仅 v4 错"三分类导出，每条并列展示两版本识别结果 + CER + 召回标记（✓/✗）。

### 10. `export_v3_v4_errors_stripped.py` — v0.3.0 vs v0.4.0 错误导出（剥离）

| 项 | 说明 |
|---|---|
| **输入** | 同上 |
| **输出** | `errors_v3_v4_{ds}_stripped.txt` ×2 + 控制台汇总 |

第 9 个脚本的剥离版：**只剥离开头**唤醒词后重算 CER 再对比。比原始版多一个剥离后的指标汇总表（准确率/召回率/CER/Sub/Ins/Del 的 v3 vs v4 差异）。

### 11. `recall_comparison_report.py` — v0.3.0 vs v0.4.0 召回率专项

| 项 | 说明 |
|---|---|
| **输入** | v0.3.0、v0.4.0 业务数据集 jsonl |
| **输出** | `recall_comparison_v3_v4.txt` |

聚焦**召回率维度**（剥离开头唤醒词后）：分类为两者都召回/都未召回/v3 独有召回/v4 独有召回，重点导出"v3 能召回但 v4 不能"和反向的样本（按标签长度排序，短指令优先），用于判断版本迭代是否引入召回退化。

### 12. `compare_v3_v5.py` — v0.3.0 vs v0.5.0/ep23 完整对比

| 项 | 说明 |
|---|---|
| **输入** | v0.3.0、v0.5.0/ep23 业务数据集 jsonl |
| **输出** | `reports/v3_vs_v5/errors_v3_v5_{ds}_stripped.txt` + `recall_comparison_v3_v5.txt` |

集大成者：把"错误对比报告"（第 10 个脚本逻辑）和"召回率专项报告"（第 11 个脚本逻辑）合并成一次运行，针对 v0.3.0 vs v0.5.0/ep23（规范里 S10 业务最优基准），输出到独立的 `reports/v3_vs_v5/` 目录。

---

## 几个值得注意的点

1. **编辑距离实现散落各处**：`analyse_testset_strip_wakeword.py`、`v2`、`compare_v3_v5.py`、`export_testset_errors.py`、`export_v3_v4_errors_stripped.py`、`recall_comparison_report.py` 各自带一份**带回溯**的 `compute_cer`（DP + 回溯算 sub/ins/del）。和 10_eval 的 `eval_utils.edit_distance` 算法一致但独立实现，可考虑统一复用。

2. **路径全部硬编码到 `e:\8.音频\03 s10\...`**：所有脚本无命令行参数，输入输出路径写死，换数据集/版本需直接改源码。这是典型的"一次性分析脚本"风格——为特定版本迭代临时编写。

3. **指令召回率是业务核心指标**：相比 CER，`ref in hyp`（标签是否为识别结果子串）更贴合智能音箱"指令命中"的真实需求。这也解释了为什么 steering 规范把召回率单列。

4. **唤醒词剥离的演进**：v1 只剥 hyp → v2 剥 ref+hyp → 后续脚本统一只剥开头（结尾常被截断不稳定）。唤醒词列表逐步扩充到 23 个变体，与规范完全一致。

5. **脚本是按版本迭代逐步累加的**：从命名能看出演进时间线——先有 v3/v4 对比，再有 v3/v5 对比；早期输出散落在 `11_analyse/` 根目录，后期规范到 `reports/{v3_vs_v4,v3_vs_v5}/` 子目录。`compare_v3_v5.py` 是整合最完善的最新版。

6. **与 steering 规范的对应**：本目录是规范里"指令召回率计算"和"testset 剔除唤醒词 CER"的实现载体；规范中提到的 `11_analyse/scripts/` 分析脚本和 `11_analyse/reports/` 报告输出即指此处。
