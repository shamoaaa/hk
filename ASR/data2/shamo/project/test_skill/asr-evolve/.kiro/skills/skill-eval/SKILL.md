---
name: skill-eval
description: ASR 模型评估与错误分析。远程推理（基座+overlay微调权重）、CER/Stripped CER/指令召回率计算、错误归因。当闭环流程进入 OBSERVE/JUDGE 阶段需要评估指标时激活。
---

# ASR 评估 Skill

## 命令

```bash
# 评估模型（远程推理 + CER/召回）
asr-evolve eval --version <v0.X.0> --checkpoints <best|avg5|last>

# 错误归因分析
asr-evolve analyse --test-set <test_set_name> [--threshold 0.1] [--top-n 20]
```

## 推理方式（对齐背景人声 finetuned_eval.py）

1. 加载基座模型（AutoModel）获取模型结构 + tokenizer + 前端
2. torch.load 微调权重 → load_state_dict overlay
3. batch 推理（batch_size=64），支持热词
4. 输出 eval_detail.jsonl（逐条 ref/hyp/cer）

## 评估指标

| 指标 | 计算方式 | 有效阈值 |
|------|---------|---------|
| CER | normalize_text（去标点去空格）→ 编辑距离 / ref长度 | 改善 > 0.1% |
| Stripped CER | 先 normalize → 再 strip_wakewords → 编辑距离 | 同上 |
| 指令召回率 | strip(ref) in strip(hyp)，空 ref 排除 | ≥ 0.90 |

## normalize_text 规则

```python
# 去标点 → 小写 → 去所有空格
text = re.sub(r'[，。！？、；：""''（）《》【】,.!?;:...]', '', text)
text = text.lower().strip()
text = re.sub(r'\s+', '', text)
```

## 唤醒词列表（strip_wakewords）

多轮首尾剥离，按长度降序匹配：
```
你好萤萤石, 营养萤石, 你好萤石, 你好伊思, 你好游石, 你好萤乐,
你好萤哲, 你好李石, 你好刘石, 控好萤石, ...
杨萤石, 样萤石, 萤萤石, 好萤石, 你好萤, 你好石, 萤石
```

## JUDGE 分析指引

eval 输出 CER/Stripped CER/召回率数字。你自行判断是否有效：

思考框架（不是硬规则）：
- CER 改善是否显著？（参考：改善 > 0.1% 通常算有效）
- 召回率是否满足目标？（参考：业务场景期望 ≥ 90%）
- 如果 CER 降了但召回也降了 → 需要权衡
- 如果 CER 没降但召回提升了 → 可能有价值

analyse 输出的混淆对/设备 CER/bad cases 帮你定位问题根源。

## 输出文件

```
results/{version}/
├── eval_detail.jsonl    ← 逐条: key/ref/hyp/ref_norm/hyp_norm/cer/sub/ins/del/correct
└── eval_summary.json    ← 汇总: cer/stripped_cer/recall/samples
```

## 使用场景

1. 训练完成后 → `asr-evolve eval --version v0.1.0 --checkpoints best`
2. 热词修改后 → `asr-evolve eval --checkpoints best`（不重训练）
3. CER 无改善 → `asr-evolve analyse` 找方向
