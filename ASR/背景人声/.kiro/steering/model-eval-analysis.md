---
inclusion: fileMatch
fileMatchPattern: ['**/eval_output/**', '**/11_analyse/**', '**/10_eval/**']
---

# ASR 模型性能分析规范

## 触发条件

当用户提到新版本模型评估、性能对比、CER/SER 分析、指令召回率时，按此规范执行。

## 目录约定

评估数据根目录：`E:\8.音频\03 s10\eval_output\`

- `base/{reflow,speech,augment}_eval_result/{seaco,online}/` — 基线结果（含全量 train+dev+test）
- `v0.X.0/model.pt.{best,epN,avgN}/` — 版本详细评估
  - `reflow_label/{s10_reflow,sp4_reflow,s10_augment}/{dev,test}/`
  - `speech_label/{aidatatang_200zh,data_aishell,data_aishell_1_ios,magicdata,primewords,stcmds,tal_asr,thchs30}/{dev,test}/`
  - `augment_label/{同上8个数据集}/{dev,test}/`
- `s10_results/v0.X.0_model.pt.*/` — S10 全量业务数据（~20万条）
- `sp4_results/v0.X.0_model.pt.*/` — SP4 全量业务数据（~16万条）
- `testset_results/v0.X.0_model.pt.*/` — 独立测试集（~3335条）
- `hotword_results/{use,not_use}/` — 热词评估（235条，对比使用/不使用热词的效果）
- `child/ChildMandarin/` — 儿童语音评估-ChildMandarin（40682条）
- `child/TTS-SCFChilSC/` — 儿童语音评估-TTS合成儿童语音（215条）

每个评估目录包含：`eval_summary.json`（汇总指标）、`eval_detail.jsonl`（逐条结果）。

## 数据来源说明

- **seaco/online 全量业务 CER**：从 `base/reflow_eval_result/{seaco,online}/combined_summary.json` 的 `per_directory` 中，合计 `s10_reflow/{train,dev,test}` 或 `sp4_reflow/{train,dev,test}` 的字符数和错误数计算
- **各版本全量业务 CER**：从 `s10_results/` 和 `sp4_results/` 下对应版本的 `eval_summary.json` 或 `comparison_summary.json` 读取
- **19 数据集 test CER**：从 `v0.X.0/model.pt.*/combined_summary.json` 的 `per_directory` 中读取对应 test 子目录的 `overall_cer`
- **seaco/online 的 19 数据集 CER**：从 `base/{speech,augment,reflow}_eval_result/{seaco,online}/combined_summary.json` 的 `per_directory` 中读取
- **指令召回率**：从 `{s10,sp4}_results/*/eval_detail.jsonl` 逐条计算
- **testset 剔除唤醒词 CER**：从 `testset_results/*/eval_detail.jsonl` 逐条剔除唤醒词后计算字符级编辑距离

## 分析流程（按顺序执行）

1. **确认数据完整性** — 检查新版本在 s10_results、sp4_results、testset_results、v0.X.0/ 下是否都有结果
2. **全量业务数据** — 读取 eval_summary.json 的 CER/SER，从 eval_detail.jsonl 计算指令召回率
3. **独立测试集** — 同上，额外计算剔除唤醒词后的 CER（唤醒词插入会影响排名）
4. **全数据集 CER 对比表** — 19 个数据集（3 业务 + 8 公开 + 8 增强）的 test 集 CER
5. **分析总结** — 最佳权重、vs 历史最优、部署建议、改善/退步点

## 必须对比的基准模型

- `seaco 原始` — 未微调的 SeacoParaformer 预训练模型（base/seaco）
- `online` — 当前线上部署模型（base/online）
- `v0.3.0/best` — 公开/增强数据集通用性基准
- `v0.5.0/ep35` — S10 业务数据最优基准
- `v0.6.0/ep33` — 增强数据集最优基准

## 对比规则

每次进行性能对比时，**必须同时包含 seaco 原始模型和 online 模型**，且**每个版本都必须出现**（选该版本最佳权重）。

各版本最佳权重：
- v0.1.0 → model.pt.ep16.4000
- v0.2.0 → model.pt.best
- v0.3.0 → model.pt.best
- v0.4.0 → model.pt.best
- v0.5.0 → model.pt.ep35
- v0.6.0 → model.pt.ep33

## 输出表格格式

```markdown
| 模型 | CER ↓ | SER ↓ | 句准确率 ↑ | 召回率 ↑ |
```

全数据集对比表加 `最优` 列标注每行最佳模型，末尾统计各模型最优数量。

## 指令召回率计算

从 `eval_detail.jsonl` 逐条读取 `ref_norm` 和 `hyp_norm`，剔除唤醒词后判断 `ref_stripped in hyp_stripped`。

唤醒词列表（按长度降序匹配，最多剥离3轮，头尾都剥离）：
```
你好萤萤石, 营养萤石, 你好萤石, 你好伊思, 你好游石, 你好萤乐, 你好萤哲, 你好李石,
你好刘石, 控好萤石, 控楼萤石, 平好萤石, 你要萤石, 你很萤石, 怎样萤石, 你奥伊石,
杨萤石, 样萤石, 萤萤石, 好萤石, 你好萤, 你好石, 萤石
```

## 历史基准（用于快速判断新版本优劣）

| 场景 | 最优模型 | 指标值 |
|------|---------|--------|
| S10 全量 CER | v0.5.0/ep35 | 6.82% |
| SP4 全量 CER | online | 6.91% |
| S10 reflow test CER | v0.5.0/ep35 | 7.06% |
| SP4 reflow test CER | online | 4.88% |
| 独立 testset 原始 CER | seaco | 6.09% |
| 独立 testset CER（剔除唤醒词）| v0.6.0/ep33 | 6.85% |
| 公开数据集 | v0.5.0/ep35 为主（8项中占5项最优） | 1.92%~8.20% |
| 增强数据集 | v0.6.0/ep33 为主（8项中占5项最优） | 5.51%~27.88% |
| S10 召回率 | v0.5.0/ep35 | 87.78% |
| SP4 召回率 | online | 94.54% |
| 热词(使用) CER | v0.5.0/ep35 | 7.19% |
| 热词(不使用) CER | v0.5.0/ep35 | 8.90% |
| 儿童 ChildMandarin CER | seaco | 10.72% |
| 儿童 TTS-SCFChilSC CER | v0.1.0/v0.2.0/v0.5.0/v0.6.0 并列 | 1.45% |

## SP4 指令召回率

| 模型 | 召回率 ↑ |
|------|---------|
| seaco 原始 | 79.34% |
| **online** | **94.54%** |
| v0.1.0/ep16.4000 | 92.72% |
| v0.2.0/best | 92.68% |
| v0.3.0/best | 93.24% |
| v0.4.0/best | 92.48% |
| v0.5.0/ep35 | 93.48% |
| v0.6.0/ep33 | 93.37% |

## S10 指令召回率

| 模型 | 召回率 ↑ |
|------|---------|
| seaco 原始 | 85.95% |
| online | 84.74% |
| v0.1.0/ep16.4000 | 87.03% |
| v0.2.0/best | 86.91% |
| v0.3.0/best | 87.47% |
| v0.4.0/best | 87.16% |
| **v0.5.0/ep35** | **87.78%** |
| v0.6.0/ep33 | 87.58% |

## 独立测试集 CER（剔除唤醒词前后）

| 模型 | 原始 CER ↓ | 剔除唤醒词 CER ↓ |
|------|-----------|-----------------|
| **seaco 原始** | **6.09%** | 6.09% |
| online | 7.89% | 7.74% |
| v0.1.0/ep16.4000 | 7.85% | 6.95% |
| v0.2.0/best | 7.56% | 6.76% |
| v0.3.0/best | 7.77% | 7.04% |
| v0.4.0/best | 7.31% | 6.91% |
| v0.5.0/ep35 | 8.07% | 6.90% |
| **v0.6.0/ep33** | 8.08% | **6.85%** |

## 热词评估（235条）

| 模型 | 使用热词 CER ↓ | 不使用热词 CER ↓ | 热词增益 |
|------|---------------|-----------------|---------|
| seaco 原始 | 8.24% | 9.94% | -1.70% |
| online | 8.11% | 9.94% | -1.83% |
| v0.1.0/ep16.4000 | 7.40% | 9.24% | -1.84% |
| v0.2.0/best | 7.53% | 9.16% | -1.63% |
| v0.3.0/best | 7.43% | 9.13% | -1.70% |
| v0.4.0/best | 7.64% | 9.45% | -1.81% |
| **v0.5.0/ep35** | **7.19%** | **8.90%** | -1.71% |
| v0.6.0/ep33 | 7.48% | 9.03% | -1.55% |

## 儿童语音评估

### ChildMandarin（40682条）

| 模型 | CER ↓ | SER ↓ | 句准确率 ↑ |
|------|--------|--------|------------|
| **seaco 原始** | **10.72%** | **44.02%** | **55.98%** |
| online | 11.31% | 45.39% | 54.61% |
| v0.1.0/ep16.4000 | 10.90% | 44.72% | 55.28% |
| v0.2.0/best | 10.88% | 44.75% | 55.25% |
| v0.3.0/best | 10.93% | 44.57% | 55.43% |
| v0.4.0/best | 10.92% | 44.46% | 55.54% |
| v0.5.0/ep35 | 11.09% | 44.85% | 55.15% |
| v0.6.0/ep33 | 10.92% | 44.81% | 55.19% |

### TTS-SCFChilSC（215条）

| 模型 | CER ↓ | SER ↓ | 句准确率 ↑ |
|------|--------|--------|------------|
| seaco 原始 | 1.58% | 9.30% | 90.70% |
| online | 2.11% | 12.09% | 87.91% |
| v0.1.0/ep16.4000 | 1.45% | 8.84% | 91.16% |
| v0.2.0/best | 1.45% | 8.84% | 91.16% |
| v0.3.0/best | 1.52% | 9.30% | 90.70% |
| v0.4.0/best | 1.58% | 9.30% | 90.70% |
| v0.5.0/ep35 | 1.45% | 8.84% | 91.16% |
| **v0.6.0/ep33** | **1.45%** | **8.37%** | **91.63%** |

## 关键脚本

- `E:\8.音频\03 s10\11_analyse\scripts\` — 分析脚本
- `E:\8.音频\03 s10\11_analyse\reports\` — 报告输出
- `E:\8.音频\03 s10\10_eval\batch_eval_models.py` — 批量评估入口
