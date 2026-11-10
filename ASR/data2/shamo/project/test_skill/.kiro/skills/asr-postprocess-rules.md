# ASR 后处理规则

## 能力描述

管理推理后的文本后处理管线。后处理只做**格式层清理**，不做语义纠错。语义纠错通过 SeaCo 热词偏置在解码阶段完成。

## 何时使用

- 推理结果有格式问题（多余空格、不可见字符、全角符号）
- 需要添加新的 regex 清理规则
- 需要管理 SeaCo 热词表（推理时的解码偏置）

## 当前管线配置

```yaml
post_process:
  pipeline_order: [regex]          # 只做格式清理
  hotword_file: null               # 后处理字符串替换已废弃
  seaco_hotwords_file: configs/post_process/seaco_hotwords.txt  # 解码偏置热词
  regex_rules:
    - 去不可见 Unicode 字符（BOM、零宽、NBSP）
    - 全角→半角
    - 去空格
    - 去中文标点
```

## 设计原则

| 适合后处理做 | 不适合后处理做 |
|------------|--------------|
| NFKC 标准化 | 同音字纠错（用 SeaCo） |
| 全角→半角 | 语义替换 |
| 去空格/标点 | 缺字补全 |
| 去不可见字符 | 上下文相关的修正 |

## 执行步骤

### 添加 regex 规则

```powershell
.venv_win\Scripts\python -m scripts.apply_postprocess update-config --add-regex "pattern->replacement"
```

### 管理 SeaCo 热词（解码偏置）

热词表位于 `configs/post_process/seaco_hotwords.txt`，每行一个正确词：
```
呼和浩特
限购
十宗
```

热词来源：
1. **业务提供**：公司名、人名、地名、术语（最可靠）
2. **badcase 分析推荐**：`extract_hotwords` 输出候选列表供人工审核
3. **人工审核后加入**：确认是专有名词且模型反复错才加

```powershell
# 从 badcases 提取候选热词（仅推荐，不自动生效）
.venv_win\Scripts\python -m scripts.extract_hotwords --eval-report reports/evaluation/eval_report_iter_xxx.json --output candidates.txt

# 人工审核后，手动编辑 seaco_hotwords.txt 加入确认的热词
```

### 推理时使用热词

热词在推理阶段传入 SeaCo co-decoder，不影响模型权重，随时可增删：
```powershell
# 推理时自动读取 seaco_hotwords_file 配置
.venv_win\Scripts\python -m scripts.remote_runner run "python -m scripts.run_inference --model-dir models/iterations/iter_xxx --tag iter_xxx"
```

## 常见问题

- **后处理只做格式清理**：不做语义替换，避免误伤
- **热词不是后处理**：热词通过 SeaCo 在解码阶段偏置，不是事后替换
- **评测统一化已覆盖格式问题**：CER 计算前自动做 NFKC + 去标点 + 去空格，后处理的 regex 主要影响展示文本
- **dedup 默认关闭**：说话人可能真的重复说话，不能自动去重

## 知识库引用

```powershell
# 查询后处理相关的历史方案
.venv_win\Scripts\python -m scripts.manage_knowledge query --type scene_solution --content-match "{\"dimension\":\"post_process\"}"

# 记录优化效果
.venv_win\Scripts\python -m scripts.manage_knowledge add --type scene_solution --content "{\"dimension\":\"post_process\",\"action\":\"add_seaco_hotword\",\"word\":\"十宗\",\"cer_delta\":-0.001}" --confidence 0.8 --source-iteration iter_xxx
```
