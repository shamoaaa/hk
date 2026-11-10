
# radar_pig_pkg 经产猪配种训练与测试 Python 参考

本文档只总结当前 workspace 中的 Python 源码，以及 Skill 内置的独立 Python CLI。模型任务是配种预警；“经产猪”数据范围必须由输入数据和实验列表保证。

## 1. 业务语义

### 1.1 经产猪范围

项目没有胎次字段或独立的经产猪训练类。`features/feature_extract_demo.py::main` 仅通过路径是否包含 `houbei` 做近似分流：

- 包含 `houbei`：`pig_prop=2`；
- 其他路径：`pig_prop=1`。

`features/run.py::feature_extract` 按 `pig_prop` 选择活动通道：`pig_prop=2` 使用通道 6，否则使用通道 4。因此，训练和测试脚本无法自行证明某条记录属于经产猪，调用者必须确认 PKL 和实验列表的数据范围。

### 1.2 配种任务

`train_model/breeding.py` 的两个公开函数固定 `label_type='breeding'`：

- `train_breeding`：训练配种事件模型；
- `test_breeding`：测试配种事件模型；
- 标签来自 CSV 行的第 3 个位置字段 `pig_label[2]`；
- 默认 `interval_type='-2_0'`，配种日前 2 天、前 1 天和当天均为评价正例。

`train_model/oestrus.py` 当前使用 `breeding_offset`，即配种日减 2 天，而不是直接使用发情字段。

## 2. Python 项目代码地图

### 2.1 `dataset_proc`

| Python 文件 | 作用 |
|---|---|
| `dataset_proc/download_new_radar.py` | 从 OSS 下载新雷达原始数据 |
| `dataset_proc/new_radar_exchange.py` | 解码新雷达原始包并写 HDF5 |
| `dataset_proc/download_old_radar.py` | 下载旧雷达数据 |
| `dataset_proc/upload_old_radar.py` | 上传旧雷达相关数据 |
| `dataset_proc/make_faqing_dataset.py` | 按雷达划分并移动训练/测试数据 |

`make_faqing_dataset.py` 使用未固定种子的随机划分和 `shutil.move`，运行前必须备份，不能把结果视为天然可复现。

### 2.2 `features`

| Python 文件 | 作用 |
|---|---|
| `features/feature_extract_demo.py` | 批量特征提取示例入口；包含路径和胎次推断硬编码 |
| `features/run.py` | 合并雷达数据并逐小时生成特征 CSV |
| `features/core/pig_oestrus_features.py` | 活动、呼吸、起卧、存活等特征提取器 |
| `features/core/find_pulse.py` | 脉冲与呼吸信号辅助 |
| `features/core/find_pulse_env_ctrl.py` | 环境控制版本的脉冲查找 |
| `features/core/utils.py` | 滤波和信号特征辅助 |
| `features/make_pkl.py` | 将逐小时 CSV 汇总成训练/测试 PKL |

逐小时 CSV 正常为 120 行，对应 60 秒窗口、30 秒步长。一天应有 24 个文件和 2880 行。

### 2.3 `train_model`

| Python 文件 | 作用 | 关键符号 |
|---|---|---|
| `train_model/breeding.py` | 配种模型公开入口 | `train_breeding`, `test_breeding` |
| `train_model/oestrus.py` | 配种日偏移入口 | `breeding_offset` |
| `train_model/train_test_core.py` | 数据加载、拆分、XGBoost 训练、测试调度 | `train_impl`, `test_impl` |
| `train_model/utils/gen_feature_utils.py` | 63 维特征和标签窗口 | `generate_features_unified` |
| `train_model/utils/evaluate_utils_refactored.py` | 预测及样本/猪级评估 | `EvaluationConfig`, `ExperimentEvaluator` |
| `train_model/utils/common_utils.py` | 训练评估辅助函数 | `calculate_pig_sign`, `convert_features_to_numeric` |

### 2.4 顶层 `utils`

| Python 文件 | 作用 |
|---|---|
| `utils/common_utils.py` | 雷达 HDF5 合并等通用数据工具 |
| `utils/oss_utils.py` | OSS 客户端辅助 |

顶层 `utils/common_utils.py` 与 `train_model/utils/common_utils.py` 同名但职责不同，是原代码绝对导入不稳定的原因之一。

## 3. 原 Python 数据链

```text
雷达原始文件
  -> dataset_proc 中的下载/转换 Python 脚本
  -> features/run.py::feature_extract
  -> <feature_dir>/<experiment>/<pig_id>/<YYYYMMDD>/*-features.csv
  -> features/make_pkl.py::make_feature_pkl
  -> list[(experiment, pig_id, dates, DataFrame)] pickle
  -> train_model/utils/gen_feature_utils.py::generate_features_unified
  -> 63 维日级特征 + 标签
  -> train_model/train_test_core.py::train_impl / test_impl
  -> XGBoost 模型与评估结果
```

Skill 内置 CLI 从可信 PKL 开始，封装 PKL 校验、63 维特征、训练和测试；雷达下载、HDF5 转换和逐小时 CSV 生成仍属于上游数据工程。

## 4. 输入契约

### 4.1 特征 PKL

顶层必须为：

```python
[
    (
        experiment,  # str
        pig_id,      # str，split('_')[1] 必须是标签表雷达 ID
        dates,       # 连续、升序的 YYYYMMDD 列表
        raw_data,    # pandas.DataFrame
    ),
    ...,
]
```

内置 CLI 采用严格契约：

- 每头猪至少连续 3 天；
- 每天 2880 行，每小时 120 行；
- DataFrame 总行数必须等于 `len(dates) * 2880`；
- 必须含 `normalize_act`、`new_br`、`is_up`；
- 三列必须可转换为数值；
- 只加载可信 pickle。

原 `features/make_pkl.py` 的缺小时占位列不含 `new_br`，这种数据会被 CLI 明确拒绝，避免后续静默丢样本。

### 4.2 标签 CSV

每个实验必须存在：

```text
<label-dir>/<experiment>.csv
```

代码按位置读取，不依赖列名：

| 位置 | 含义 |
|---|---|
| 第 1 列 | 雷达 ID |
| 第 2 列 | 发情日期 |
| 第 3 列 | 配种日期 |

标签值允许：

- `YYYYMMDD...`：取前 8 位；
- `0`：无事件；
- `-1`：淘汰，生成特征时跳过。

测试模式会剔除“第 2 列存在发情日期但第 3 列为 0”的猪。

## 5. 默认 63 维特征

`generate_features_unified` 和内置 CLI 都使用三日窗口，并从原始 DataFrame 读取：

- `normalize_act`；
- `new_br`；
- `is_up`。

每列按 `(-1, 120)` 重塑为小时。默认时间段为每天 `[0, 7)` 和 `[20, 24)`。

固定列顺序：

- `act_01`–`act_18`：18 个活动特征；
- `br_01`–`br_20`：20 个呼吸特征；
- `up_01`–`up_16`：16 个起卧特征；
- `sleep_01`–`sleep_09`：9 个睡眠特征。

合计 63 维。模型依赖列顺序，训练和测试不得重排。

活动特征包含活动总量差、阈值 `>4`/`>15` 次数差及比值；呼吸特征包含中位数、变化量、比值、四分位数比和 `>40` 次数；起卧特征包含站立量、状态变化和连续段；睡眠特征比较两晚状态分布并计算 KL 散度。

原 Python 中可选的 7–8 点扩展实际增加 19 维而非注释中的 31 维。公开 `train_breeding` 未启用它；内置 CLI 为保证模型契约清晰，只封装默认 63 维。

## 6. 标签窗口

输出标签列：

```text
experiment, pig_id, date, label, eva_label, oneday_label, towday_label
```

`label` 仅在配种当天为 1，训练目标是 `eva_label`：

| `interval_type` | `eva_label=1` |
|---|---|
| `-2_0` | 配种日前 2 天、前 1 天、当天 |
| `-1_0` | 配种日前 1 天、当天 |
| `-1_1` | 配种日前 1 天、当天、后 1 天 |

CLI 默认 `-2_0`。训练信息会保存该值，测试时若同名模型信息 JSON 存在，CLI 会检查两者一致。

## 7. 训练：原入口与内置 CLI

### 7.1 原 Python 入口

`train_model/train_test_core.py::train_impl`：

- `train_test_split(test_size=0.2, random_state=42)`；
- 没有 `stratify`，也没有按猪或实验分组；
- `eta=0.015`、`max_depth=6`；
- `subsample=0.8`、`colsample_bytree=0.8`；
- `num_boost_round=1000`；
- `early_stopping_rounds=10`；
- 未显式设置二分类 objective；
- 用字符串替换生成训练信息路径，模型名没有 `.model` 时可能覆盖模型。

随机按样本行拆分会使同一头猪相邻日期进入训练和验证集，存在泄漏风险。

### 7.2 Skill 内置 CLI

路径：`scripts/breeding_cli.py`。脚本不导入项目包，也不从当前工作目录解析资源。所有外部路径必须是绝对路径。

训练调用：

```powershell
& $Python "<本 Skill 目录>\scripts\breeding_cli.py" train `
  --feature-file "<特征 pickle 绝对路径>" `
  --label-dir "<标签目录绝对路径>" `
  --experiments <训练实验列表> `
  --model-output "<模型输出绝对路径>\breeding.model" `
  --confirm-multiparous-data `
  --stratify
```

内置版本保留主要参数，并显式设置 `objective=binary:logistic`。`--stratify` 是可选但推荐的兼容增强；它仍不是按猪分组拆分。

模型信息安全保存到 `<model-stem>_info.json`，包含：

- Python 脚本绝对路径；
- 输入和标签绝对路径；
- 经产猪范围确认标记；
- 实验列表、标签窗口和参数；
- 63 个特征列；
- 全部/训练/验证样本数及标签分布；
- 最佳迭代与分数；
- 输入检查和跳过统计。

## 8. 测试：原入口与内置 CLI

### 8.1 原 Python 阻塞

原测试链存在以下 Python 问题：

1. `gen_feature_utils.py` 和 `evaluate_utils_refactored.py` 使用 `from utils...`，启动位置不同会找不到或导错同名模块；
2. `DataPreparator.prepare_features` 实际导入仓库中不存在的 `utils.improved_feature_utils`；
3. 可用的 `generate_features_unified` 调用被注释；
4. `test_impl` 把 `output_dir` 固定为 `None`，不会落盘；
5. 单类别实验可能导致混淆矩阵解包或 AUC 失败。

### 8.2 内置 CLI 测试

```powershell
& $Python "<本 Skill 目录>\scripts\breeding_cli.py" test `
  --feature-file "<特征 pickle 绝对路径>" `
  --label-dir "<标签目录绝对路径>" `
  --experiments <测试实验列表> `
  --model-path "<模型绝对路径>\breeding.model" `
  --output-dir "<结果目录绝对路径>" `
  --confirm-multiparous-data
```

预测规则：

- 默认严格 `probability > 0.52` 判正；
- 任一特征为非有限值时强制预测为负；
- 样本级混淆矩阵固定 `labels=[0, 1]`；
- 单类别真实标签时 AUC 写为 `null`；
- 猪级评估只保留第一次正预测；
- 有配种事件且首个预警落入 `eva_label` 窗口为 TP，否则 FN；
- 无配种事件但有预警为 FP，否则 TN。

每个实验输出 `sample_predictions.csv`、`pig_results.json`、`metrics.json`；根目录输出 `test_summary.json`。

## 9. 内置 CLI 命令契约

### 9.1 `validate`

只做输入结构和标签校验，不训练：

```powershell
& $Python "<本 Skill 目录>\scripts\breeding_cli.py" validate `
  --feature-file "<特征 pickle 绝对路径>" `
  --label-dir "<标签目录绝对路径>" `
  --experiments <实验列表> `
  --confirm-multiparous-data
```

返回码：合法为 0，输入不合法为 2。

### 9.1.1 致命错误与逐猪跳过

CLI 把问题分成两类，避免个别脏数据挡住整批可用数据：

致命错误（退出码 2）：

- 特征 pickle 无法加载或顶层结构不是列表；
- 某个请求实验缺少 `<experiment>.csv`，或标签表少于 3 列；
- 请求的实验在 PKL 中不存在；
- 某实验全部猪被跳过，或没有生成任何样本；
- 单个数据项不是长度为 4 的 tuple/list；
- 模型无法加载，或模型信息中的特征列、标签窗口与本次请求不一致。

逐猪跳过（记录在 `skipped_by_category` 与 `skipped_details`）：

| 类别 | 含义 |
|---|---|
| `culled` | 配种标签为 `-1` |
| `label_missing` | 标签首列没有该雷达 ID |
| `oestrus_not_bred` | 测试模式下发情未配种 |
| `label_too_early` | 配种日不晚于第二个数据日 |
| `label_malformed` | 标签值既不是日期也不是 `0`/`-1` |
| `dates_too_few` / `dates_format` / `dates_unsorted` / `dates_gap` | 日期不满足连续 3 天契约 |
| `raw_type` / `raw_columns` / `raw_rows` / `reshape` | DataFrame 结构或行数不符 |
| `pig_id_format` | `pig_id` 缺少雷达 ID 段 |

原 Python 代码对这些情况多为 `print` 后跳过或宽泛捕获；CLI 保留跳过行为，但强制记账，并在训练信息与测试结果中持久化。

### 9.2 `self-check`

使用系统临时目录生成合成 PKL 和标签，运行完整训练与测试，默认自动清理：

```powershell
& $Python "<本 Skill 目录>\scripts\breeding_cli.py" self-check
```

可用绝对 `--output-dir` 保留产物。该命令用于验证依赖、63 维特征、模型保存/重载、测试结果和工作目录无关性，不能替代真实数据验证。

## 9.3 真实数据运行经验

本节只记录可复用的行为结论。具体数据、标签、模型和解释器路径由使用者当场提供，不写入本文档。

### 定位输入的方式

- 特征 PKL 一般由 `features/make_pkl.py` 产出，配种数据集常与查情数据集分开保存，命名不统一，必须逐个确认内容而不是靠文件名推断。
- 标签目录集中存放 `<experiment>.csv`，一个目录通常覆盖多个批次实验。
- 真实标签日期可能带两位小时后缀，例如 10 位的 `YYYYMMDDHH`；代码只取前 8 位。
- 标签表首列常是无名索引列（读入后表头形如 `Unnamed: 0`），因此按位置而不是列名取值是必要的。
- 运行前确认解释器已装 NumPy、pandas、scikit-learn、XGBoost；某些机器的默认环境并不完整。

### 真实数据的典型跳过分布

在一个约 280 头猪的单实验数据集上实测：

- 约三分之一以上的猪配种标签为 `-1`，全部落入 `culled`；
- 少量猪的雷达 ID 不在标签表中，落入 `label_missing`；
- 个别雷达 ID 在标签表中重复，按第一行取值并记录警告；
- 其余猪可用，可正常生成 63 维特征并训练。

结论：真实数据必然包含淘汰猪和缺标签猪。逐猪跳过必须是非致命的，否则整批可用数据会被一条脏记录挡住。这也是把致命错误限定为“缺标签文件、实验不存在、无可用猪”的原因。

### 指标解读注意

- 同实验回测容易得到很高的样本级和猪级指标，但它不是泛化指标；
- 若某实验所有可用猪都有配种日期，猪级混淆矩阵不会出现真负例，precision 会等于 1，需要显式说明；
- 单猪或极少样本的实验只能证明链路可运行，不能作为效果结论；
- 真实数据中确实存在含非有限特征的测试样本，会按兼容逻辑被强制判负，需在报告中给出数量。

## 10. Python 源码中的其他风险

- 标签列完全依赖位置，无表头语义校验；
- `pig_id.split('_')[1]` 对命名格式敏感；
- 原代码在检查长度前访问 `dates[1]`；
- `replace_negative_br_with_nan` 处理 `br`，训练实际读取 `new_br`；
- `start_hour` 参数未使用；
- 多处宽泛异常捕获会静默跳过样本；
- `np.diff(up_raw)` 使用默认轴，可能与跨时间变化意图不同；
- `feature_up_15` 基于全零数组，恒为 0；
- 原训练无类别不平衡策略；
- 阈值 0.52 是固定值，不是自动搜索结果；
- 原随机行拆分存在同猪泄漏；内置 CLI 的 `--stratify` 只解决类别比例，不解决分组泄漏。

内置 CLI 为兼容历史 63 维模型，保留 `np.diff` 默认轴和恒为 0 的对应特征。若修正这些特征，必须作为新模型版本重新训练，不能与旧模型混用。

## 11. 最低验证和报告要求

执行顺序：

1. 从 Skill 自身目录取得脚本绝对路径；
2. 在任意工作目录运行 `self-check`；
3. 对真实输入运行 `validate`；
4. 确认训练与测试实验互斥且均为经产猪；
5. 训练后重新加载模型并检查 `feature_dim=63`；
6. 逐实验测试并检查所有 CSV/JSON 产物；
7. 同时报告样本级与猪级指标；
8. 明确区分合成烟雾验证和真实数据结果。

最低报告字段：

- 脚本绝对路径和 Python 包环境；
- 特征 PKL、标签目录、模型和输出绝对路径；
- 经产猪数据范围确认方式；
- 训练与测试实验列表；
- 时间窗口、样本数、标签分布和跳过统计；
- 训练参数、最佳迭代和模型信息路径；
- 样本级 accuracy/precision/recall/F1/AUC；
- 猪级混淆矩阵、accuracy/precision/recall 及猪 ID；
- 已执行命令、验证结果和仍未验证项。
