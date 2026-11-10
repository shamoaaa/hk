---
name: multiparous-sow-breeding
description: "使用 Skill 内置的独立 Python CLI 完成 radar_pig_pkg 经产猪（非后备猪）配种预警模型的数据校验、63 维特征生成、XGBoost 训练、逐实验测试、结果保存与故障排查。Use when 用户提到经产猪、非后备猪、配种模型、breeding.py、train_breeding、test_breeding、训练或测试配种预警模型，或要求不依赖项目工作目录运行该流程。"
compatibility: "Requires Python 3.8+, NumPy, pandas, scikit-learn and XGBoost. Input pickle files must be trusted."
metadata:
  author: radar-pig-team
  version: "2.0.0"
---

# 经产猪配种模型训练与测试

使用本 Skill 自带的 `scripts/breeding_cli.py` 完成输入校验、特征生成、训练和测试。该脚本封装了所需 Python 逻辑，不导入 `radar_pig_pkg`，也不从进程当前工作目录查找代码或数据。

需要核对原始 Python 实现、数据契约、特征定义、参数或差异时，读取 [`references/project-guide.md`](references/project-guide.md)。

## 硬性规则

0. **区分致命错误与逐猪跳过。** 缺少标签文件、实验不存在、某实验无可用猪属于致命错误，命令退出码为 2。单头猪的问题（标签 `-1` 淘汰、标签表没有该雷达 ID、日期断裂、行数不符、测试模式发情未配种）按原 Python 行为跳过，并在 `skipped_by_category` 和 `skipped_details` 中逐条记账，不会中断整批数据。
1. **“经产猪”不是模型标签。** 原 Python 项目没有胎次字段。必须由数据负责人确认传入的 PKL 和实验列表只属于经产猪；CLI 因此强制要求 `--confirm-multiparous-data`。
2. **`breeding` 表示配种事件，不表示胎次。** 标签来自实验 CSV 的第 3 个位置字段。
3. **只分析 Python 源码。** 不读取、解释或修改其他语言的生成文件。
4. **不依赖当前工作目录。**
   - 从本 Skill 目录定位 `scripts/breeding_cli.py`；不要从 workspace 当前目录猜测脚本位置。
   - `--feature-file`、`--label-dir`、`--model-output`、`--model-path`、`--output-dir` 都必须传绝对路径。
   - CLI 会拒绝相对路径；内部资源只通过脚本自身的 `Path(__file__).resolve()` 定位。
5. **只加载可信 pickle。** pickle 反序列化可以执行任意代码。
6. 默认保持原 Python 代码的 63 维特征名称、顺序、三日窗口和猪级首个预警评估语义。

## 首选执行入口

脚本路径为本 Skill 根目录下：

```text
scripts/breeding_cli.py
```

在 PowerShell 中应先得到它的**绝对路径**，再用装有依赖的解释器运行。以下 `$Python` 和 `$Script` 必须替换为当前环境的实际绝对路径：

```powershell
$Python = "<装有 numpy/pandas/scikit-learn/xgboost 的 python.exe 绝对路径>"
$Script = "<本 Skill 目录>\scripts\breeding_cli.py"
& $Python "$Script" --help
```

若默认 `python` 缺少依赖，先确认可用解释器，不要假设系统默认环境已装齐。

脚本有四个子命令：

- `validate`：检查 PKL、实验、猪 ID、连续日期、DataFrame、标签 CSV 和行数契约；
- `train`：生成默认 63 维特征并训练 XGBoost；
- `test`：逐实验测试，保存样本预测、猪级结果和汇总 JSON；
- `self-check`：在系统临时目录创建合成数据，完整验证训练和测试链，结束后自动清理。

## 标准工作流

### 1. 先验证脚本环境

首次使用或修改脚本后执行：

```powershell
& $Python "$Script" self-check
```

预期输出包含：

```json
{
  "status": "PASS",
  "working_directory_independent": true,
  "feature_dim": 63
}
```

若要保留合成产物，使用绝对目录：

```powershell
& $Python "$Script" self-check --output-dir "<自检产物目录绝对路径>"
```

### 2. 校验真实输入

```powershell
& $Python "$Script" validate `
  --feature-file "<特征 pickle 绝对路径>" `
  --label-dir "<标签目录绝对路径>" `
  --experiments <exp_1> <exp_2> `
  --confirm-multiparous-data
```

测试数据可加 `--mode test`，以便提前看到“发情未配种”会被剔除多少头。

只有输出中的 `valid` 为 `true` 才进入训练或测试。同时检查 `usable_pigs` 与 `skipped_by_category`：跳过量偏大时先确认标签表是否完整，再决定是否继续。常见跳过类别：

- `culled`：配种标签为 `-1`；
- `label_missing`：标签首列没有该雷达 ID；
- `oestrus_not_bred`：测试模式下发情未配种；
- `label_too_early`：配种日不晚于第二个数据日；
- `dates_gap` / `dates_too_few` / `raw_rows` / `raw_columns`：PKL 数据结构问题。

必须核对：

- PKL 顶层为 `(experiment, pig_id, dates, DataFrame)` 列表；
- `pig_id.split('_')[1]` 能匹配标签 CSV 首列；
- 日期按 `YYYYMMDD` 升序且至少连续 3 天；
- 每天 2880 行，每小时 120 行；
- DataFrame 含 `normalize_act`、`new_br`、`is_up`；
- 每个实验存在 `<experiment>.csv`，且至少有雷达 ID、发情日期、配种日期三列；
- 训练与测试实验列表互斥，并且确属经产猪。

### 3. 训练

```powershell
& $Python "$Script" train `
  --feature-file "<特征 pickle 绝对路径>" `
  --label-dir "<标签目录绝对路径>" `
  --experiments <训练实验列表> `
  --model-output "<模型输出绝对路径>\breeding.model" `
  --confirm-multiparous-data `
  --stratify
```

默认参数：

- `interval_type=-2_0`；
- `test_size=0.2`；
- `seed=42`；
- `num_boost_round=1000`；
- `early_stopping_rounds=10`；
- `nthread=4`；
- `eta=0.015`、`max_depth=6`、`subsample=0.8`、`colsample_bytree=0.8`；
- 显式 `objective=binary:logistic`。

`--stratify` 是推荐选项，可降低随机拆分后某一集合只有单类别的风险；原 Python 入口没有分层。如果显式传递以连字符开头的标签窗口，使用等号形式，例如 `--interval-type=-1_0`。

训练产物：

- 指定的模型文件；
- 同目录 `<模型文件名去后缀>_info.json`，包含特征列、参数、实验列表、样本数、标签分布和最佳迭代。

### 4. 测试

```powershell
& $Python "$Script" test `
  --feature-file "<特征 pickle 绝对路径>" `
  --label-dir "<标签目录绝对路径>" `
  --experiments <测试实验列表> `
  --model-path "<模型绝对路径>\breeding.model" `
  --output-dir "<结果目录绝对路径>" `
  --confirm-multiparous-data
```

默认阈值为严格 `probability > 0.52`。脚本会校验同名 `_info.json` 中的特征列和标签窗口（如果该文件存在）。每个实验保存：

```text
<output-dir>/
├── test_summary.json
└── <experiment>/
    ├── metrics.json
    ├── sample_predictions.csv
    └── pig_results.json
```

必须同时报告：

- 样本级 accuracy、precision、recall、F1、AUC 和 `[[TP, FP], [FN, TN]]`；
- 猪级 accuracy、precision、recall、混淆矩阵及对应猪 ID；
- 单类别实验的 AUC 为 `null`，不能伪造数值；
- 含非有限特征的测试样本会按兼容逻辑强制预测为负例。

## 内置脚本与原 Python 入口的关系

内置脚本复刻默认 63 维特征和标签逻辑，但为确保可独立运行，明确修复了原 Python 链路中的执行问题：

- 不使用易误解析的 `from utils...` 导入；
- 不依赖缺失的 `improved_feature_utils.py`；
- 训练显式设置 `binary:logistic`；
- 模型信息路径不会覆盖模型；
- 测试结果始终写入显式绝对输出目录；
- 混淆矩阵固定 `labels=[0, 1]`，单类别 AUC 返回 `null`；
- 逐猪跳过与原 Python 行为一致，但每条跳过都有类别和原因，不再静默丢数据。

这些变化意味着内置脚本是可执行、可审计的封装版本。若必须与历史模型做逐值复现，要记录 Python 包版本，并核对历史模型是否使用了不同 objective、缺失值策略或未修复逻辑。

## 验证与结果报告

完成任务前至少确认：

1. `self-check` 通过；
2. 真实输入 `validate` 返回 `valid=true`；
3. 模型可重新加载，`feature_dim=63`；
4. 训练与测试标签窗口一致；
5. 输出路径均为绝对路径且产物存在；
6. 明确报告经产猪数据范围如何确认；
7. 不能运行真实数据时，区分“脚本烟雾验证通过”和“真实训练/测试未执行”。

报告应包括：输入路径、实验列表、样本与标签分布、被跳过数量、参数、模型路径、样本级指标、猪级指标、结果目录、验证命令及剩余风险。
