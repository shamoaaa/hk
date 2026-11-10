# 10_eval 目录脚本报告

本目录是流水线的**模型评估阶段**，对 09 阶段训练出的 SeACo-Paraformer 微调权重（以及基准模型 / 线上模型）在各类数据集上计算 CER / SER / 句准确率，并生成单模型报告、跨模型对比报告、与基线的对比报告。

## 整体架构

```
                  eval_utils.py  ← 公共底座（CER计算/标签加载/单目录评估循环/报告生成）
                       │ import
   ┌───────────────────┼─────────────────────────────────────────────┐
   ▼                   ▼                  ▼                  ▼          ▼
seaco_eval.py    finetuned_eval.py   batch_eval.py    batch_eval_   eval_on_
(基准模型,        (基准+微调权重,      (公开数据集      models.py     testset.py
 单/多目录)        单/多目录)          按 split 批量)  (多权重×多      (12_test
                                                      数据集+基线对比) 多权重对比)
                       │                                   │
   07_SeACo_Paraformer_eval.py（早期独立版，未用 eval_utils）   │
                                                              ▼
                                              gen_baseline_report.py
                                              (不跑推理,从已有结果生成基线对比)

   plot_loss.py / plot_loss_tb.py   ← 训练 loss 曲线可视化（从日志 / TensorBoard）
   batch_eval_all_testsets.sh       ← 一键评估 6 个测试集的 shell 驱动
```

所有评估输出统一为每个任务目录下的：`eval_detail.jsonl`（逐条）、`eval_report.txt`、`eval_summary.json`，多目录/多模型再额外生成 `combined_*` / `*_comparison_*`。

---

## 1. `eval_utils.py` — 评估公共工具模块（核心底座）

被除 `07_SeACo_Paraformer_eval.py` 外的所有评估脚本 import，不可独立运行。

**提供的能力：**

| 分类 | 函数 | 说明 |
|---|---|---|
| 音频 | `get_wav_duration` | soundfile → wave 双重回退取时长，失败返回 -1 |
| 日志 | `setup_logging` / `format_duration` | 控制台 + 文件日志 |
| CER | `edit_distance` / `compute_cer` | 自实现编辑距离，返回替换/插入/删除 |
| 文本 | `normalize_text` | 去标点、转小写、去空格 |
| 标签 | `load_labels` | 支持 text/trans（2列/3列/tab）、jsonl（uid+text / key+txt / messages 训练格式） |
| 热词 | `load_hotwords` | 行列表 → 空格分隔字符串 |
| 数据 | `collect_wav_files` | 递归扫描 wav 目录 + 按标签匹配 + 过滤过短音频 |
| 数据 | `build_eval_items_from_scp` | 从 wav.scp + 标签构建评估项（**无需 staging 目录**，推荐） |
| 数据 | `auto_discover_labels` | 在 wav 目录同级/父级自动找 trans.txt/text.txt |
| 评估 | `run_eval_items` | **核心评估循环**，与数据来源解耦 |
| 报告 | `generate_report` | 单目录报告（整体 CER/SER + 分布分桶 + Bad Cases>30%） |
| 报告 | `generate_combined_report` | 多目录加权汇总报告 |
| GPU | `select_gpus` | 复用 06 目录 `pretag_utils` 的 GPU 自动选择 |

**`run_eval_items` 评估循环**：接受 `(uid, ref_text_or_None, wav_path)` 列表 → 支持采样/断点续跑 → 按 `file_batch` 分批 `model.generate(input=..., hotword=...)` → 有标签则算 CER 累计、无标签则只存识别结果 → 实时打印进度（running CER/SER/速度/ETA）→ 末尾生成报告。

## 2. `07_SeACo_Paraformer_eval.py` — 早期独立评估脚本

| 项 | 说明 |
|---|---|
| **输入** | `--wav-dir`（可多个）+ `--label`（text.txt 或 jsonl）+ 可选 `--hotword` |
| **输出** | `seaco_eval_output/`（单目录）或各子目录 + `combined_*`（多目录） |

功能与 `seaco_eval.py` 基本相同，但是**自包含**——把 CER 计算、标签加载、单目录评估、报告生成全部内联在文件里（没用 `eval_utils`）。是 eval_utils 抽取之前的早期版本，与匹配 wav 时用 `wav_dir/{uid}.wav` 拼接（非递归）、`batch_size_s` 参数名等细节上与新版有差异。属于历史脚本，正式用 `seaco_eval.py`。

## 3. `seaco_eval.py` — 基准模型评估

| 项 | 说明 |
|---|---|
| **输入** | `--wav-dir`（可多个，必填）+ `--label`（可多个/不填则自动发现）+ 可选 `--hotword` |
| **输出** | `eval_output/`，多目录时每目录一子目录 + 顶层 `combined_*` |
| **模型** | 纯 SeACo-Paraformer 预训练模型（`AutoModel(model=model_dir)`） |

**逻辑**：加载标签（多文件合并，或 `auto_discover_labels` 自动找）→ 加载热词 → `select_gpus` + `AutoModel` 加载一次模型 → 逐 wav 目录调 `run_eval_single_dir` → 多目录汇总。是评估"未微调基线"的标准入口。

## 4. `finetuned_eval.py` — 微调模型评估

| 项 | 说明 |
|---|---|
| **输入** | `--base-model` + `--finetune-ckpt`（model.pt.best）+ `--wav-dir` + `--label` |
| **输出** | `finetuned_eval_output/` |

与 `seaco_eval.py` 唯一区别在**模型加载**：`load_finetuned_model` 先用 `AutoModel` 加载基准模型结构，再用 `torch.load` 读微调权重，兼容 `model_state_dict`/`state_dict`/`model` 多种 checkpoint 封装格式，`load_state_dict(strict=False)` 覆盖参数（打印 missing/unexpected）。其余评估流程完全复用 eval_utils。

## 5. `batch_eval.py` — 公开数据集批量评估

| 项 | 说明 |
|---|---|
| **输入** | `--label-root`（约定 `{split}/<dataset>_wav.scp` + `_text`），`--mode seaco/finetuned/both` |
| **输出** | `output_root/{mode}/{dataset}/{split}/` + 每数据集 `dataset_report.md` + 全局 `combined_*` |

**逻辑**：`discover_datasets` 自动发现 `<dataset>_wav.scp`/`_text` 配对 → 按 `--mode` 决定跑基准/微调/两者 → **按模型分组执行**（每个模型只加载一次，遍历所有数据集×split）→ `build_eval_items_from_scp` 直接从 scp 构建评估项 → `_generate_dataset_report` 为多 split 的数据集生成加权平均报告。用于生成基线评估结果（如 `base/augment_eval_result/`）。

## 6. `batch_eval_models.py` — 多权重批量评估 + 基线对比（最完整）★

| 项 | 说明 |
|---|---|
| **输入** | `--model-dir`（含多个 `model.pt.*`）+ `--label-root`/`--label-source` + `--baseline-dir` |
| **输出** | `output_root/{ckpt_name}/{label_source}[/{dataset}]/{split}/` + 单模型 `combined_*` + `baseline_comparison_report.md` + 跨模型 `model_comparison_report.md` |

这是规范里指定的"批量评估入口"。**逻辑**：
1. `discover_checkpoints` 扫描目录下所有 `model.pt.*` 权重
2. `discover_datasets_from_label` 发现数据集（兼容 `all_label` 的合并格式 `wav.scp`+`text` 和标准 `<dataset>_wav.scp` 格式）
3. `load_baseline_results` 从 `base/{augment,reflow,speech}_eval_result/{seaco,online_model,finetuned}/` 加载已有基线 summary，按模式分组（`eval_type → label_source` 映射）
4. 逐权重加载（baseline 不加权重 / 其余用 `load_finetuned_model`）→ 逐数据集 `run_eval_items`
5. `_generate_baseline_comparison_report`：当前权重 vs 各基线模式（seaco/online/finetuned）的 CER/SER 对比 + delta（`↓`变好`↑`变差）+ 改善/退化/持平分类汇总
6. `_generate_model_comparison_report`：跨所有权重的总览 + 逐数据集 CER/SER 对比 + 最佳权重推荐
7. `_match_task_to_baseline` 用三种策略（去前缀/直接/末两段）匹配当前任务到基线 key

## 7. `eval_on_testset.py` — 独立测试集多权重评估

| 项 | 说明 |
|---|---|
| **输入** | `--model-dir`（单目录或多版本父目录）+ `--test-scp` + `--test-text` |
| **输出** | `output_dir/{ckpt_name}/` + `comparison_report.md` / `comparison_summary.json` |

**逻辑**：`auto_discover_checkpoints` 自动判断单目录（`model.pt.*`）还是多版本（`v0.X.0/model/model.pt.*`，display_name 形如 `v0.1.0/model.pt.best`）→ 只构建一次 eval_items（所有权重共用同一测试集）→ 逐权重加载评估，每个权重后 `torch.cuda.empty_cache()` 释放显存 → `generate_comparison_report` 生成跨权重对比表（含 CER 分布对比 + 最佳/最差差距）。被 `batch_eval_all_testsets.sh` 调用。

## 8. `gen_baseline_report.py` — 离线基线对比报告（不跑推理）

| 项 | 说明 |
|---|---|
| **输入** | `--eval-dir`（已有评估结果目录）+ `--baseline-dir` |
| **输出** | 每个模型目录下 `baseline_comparison_report.md` + `_summary.json` |

**纯后处理脚本**，不加载模型、不推理。`load_model_results` 递归读 `eval_dir` 下所有 `eval_summary.json`（跳过 staging 目录），`load_baseline_results` 读基线，`generate_baseline_comparison` 生成与 `batch_eval_models.py` 内置对比报告完全一致的格式。用途：当评估结果已存在、只想重新生成/补生成对比报告时，免去重复推理。

## 9. `plot_loss.py` — 从训练日志绘制 loss 曲线

| 项 | 说明 |
|---|---|
| **输入** | `--log`（可多个）或 `--log_dir`（递归找 train.log） |
| **输出** | loss 曲线 PNG（默认 `loss_curve.png`） |

正则解析 FunASR 训练日志，提取 train/val 的 epoch 级和 step 级 loss。兼容旧版 `loss_avg_epoch` / 新版 `loss_avg_slice`，多卡时**只取最小 rank** 避免重复。支持 `--loss_key loss/loss_att/loss_seaco`、`--show_step` 双视图、多日志按时间合并（中断恢复场景）。

## 10. `plot_loss_tb.py` — 从 TensorBoard 绘制 loss 曲线

| 项 | 说明 |
|---|---|
| **输入** | `--tb_dir`（可多个）或 `--scan_dir`（递归找 tfevents） |
| **输出** | loss 曲线 PNG（默认 `loss_tb.png`） |

功能与 `plot_loss.py` 对齐，但数据源是 TensorBoard 事件文件（用 `EventAccumulator`）。`find_tags` 多策略匹配 train/val tag；`--list_tags` 列出所有可用 scalar；难点在于 TB 只记录 step 不记录 epoch，所以有一套 `guess_steps_per_epoch`（从 val 验证间隔众数推测）+ epoch 边界点识别的启发式逻辑，也支持 `--steps_per_epoch` 手动指定。会在图上标注最低 val loss 的 epoch。

## 11. `batch_eval_all_testsets.sh` — 一键评估全部测试集

驱动脚本，依次对 6 个测试集调用 `eval_on_testset.py`（统一 `--ckpt-pattern best`）：
S10 全量、SP4 全量、周报独立测试集、AISHELL 热词、TTS-SCFChilSC 儿童语音、ChildMandarin 儿童语音。各自输出到对应结果目录。`set -e` 遇错即停。

> 注意：脚本顶部用了三引号 `"""..."""` 当注释——在 bash 里这其实不是合法注释（bash 没有三引号语法），第一行会被当命令执行。实际运行可能报错或被忽略，建议改成 `#` 注释。

---

## 几个值得注意的点

1. **`eval_utils.py` 是整个目录的地基**：CER 计算/标签加载/评估循环/报告生成全部集中在此，5 个主力脚本只负责"加载什么模型 + 喂什么数据"，评估逻辑零重复。这与 06 目录里 CER 计算到处复制形成鲜明对比，是更成熟的工程组织。

2. **`build_eval_items_from_scp` 取代 staging 目录**：早期（`07_SeACo_Paraformer_eval.py`）靠 `wav_dir/{uid}.wav` 拼路径或软链接 staging 目录，新版直接从 wav.scp 读绝对路径构建评估项，省去拷贝/链接开销。

3. **微调模型加载方式统一**：`finetuned_eval` / `batch_eval` / `batch_eval_models` / `eval_on_testset` 四处都有一份几乎相同的 `load_finetuned_model`（AutoModel 加结构 + torch.load 覆盖权重 + strict=False），这块可以再抽到 eval_utils 复用。

4. **基线对比的三种匹配策略**：`_match_task_to_baseline` 处理 `label_source/dataset/split` 与基线 `dataset/split` 的 key 对齐，保证不同 label_source 下的同名数据集（如 reflow 和 speech 都可能有同名）不冲突——这是规范里"基线 key 格式为 `{label_source}/{dataset}/{split}`"的实现。

5. **与 steering 规范的对应**：`batch_eval_models.py` 是规范点名的评估入口；输出的 `combined_summary.json`（含 `per_directory`）、`eval_summary.json`、`eval_detail.jsonl` 正是规范里各项 CER / 召回率分析的数据来源；`base/{reflow,speech,augment}_eval_result/{seaco,online}/` 基线目录由 `batch_eval.py` 生成。

6. **路径硬编码 + Windows/Linux 混用**：`seaco_eval.py`/`finetuned_eval.py` 默认路径是 Windows（`E:\...`），批量脚本和 shell 是 Linux（`/data1/...`），反映本地小规模验证 + 服务器大规模评估的双环境。
