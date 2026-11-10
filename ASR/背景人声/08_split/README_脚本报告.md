# 08_split 目录脚本报告

本目录负责训练数据的**最终汇集与划分**，把前面阶段产出的各类标签数据（自有数据 reflow、增强数据、公开数据集）统一整理为 Kaldi 风格的 `wav.scp` + `text` 文件，按 `train/dev/test` 划分到固定目录布局，供下游训练直接使用。

约定的目标目录结构：

```
/data1/label/
  reflow_label/    自有数据切分（s10 / sp4）
    {train,dev,test}/<dataset>_wav.scp
    {train,dev,test}/<dataset>_text
  speech_label/    公开数据集（aidatatang/aishell/magicdata/...）
    {train,dev,test}/<dataset>_wav.scp
    {train,dev,test}/<dataset>_text
  augment_label/   07 阶段产出的增强数据
    {train,dev,test}/<dataset>_wav.scp
    {train,dev,test}/<dataset>_text
  all_label/       合并后的最终训练数据
    {train,dev,test}/wav.scp
    {train,dev,test}/text
```

## 流水线

```
自有数据                公开数据集                增强数据
text.txt + wav.scp     原始数据集目录            (来自 07 阶段)
       │                      │                        │
       ▼                      ▼                        │
s10_split_train_dev_test  public_dataset_              │
sp4_split_train_dev_test  gen_scp_txt.py               │
       │                      │                        │
       ▼                      ▼                        ▼
 reflow_label/          speech_label/            augment_label/
       └────────────────┬────────────────────────┘
                        ▼
                 merge_all_labels.py
                 (汇总 + 文件存在性校验)
                        │
                        ▼
                   all_label/
                  (训练用最终数据)
```

---

## 1. `s10_split_train_dev_test.py` — s10 自有数据集随机划分

| 项 | 说明 |
|---|---|
| **输入** | `BASE_DIR=E:\8.音频\03 s10\s10_data_reflow` 下的 `text.txt`（`uid text`）和 `wav.scp`（`uid path`） |
| **输出** | `/data1/label/reflow_label/{train,dev,test}/s10_reflow_wav.scp` 和 `s10_reflow_text` |
| **划分** | 随机切分，`train 90% / dev 5% / test 5%`，`seed=42` 可复现 |

**逻辑**：
1. `load_data` 分别读 `text.txt` 和 `wav.scp` 到 `{key: text}` 和 `{key: wav_path}` 两个字典，取**交集**得到 `common_keys`，自动剔除任意一方缺失的条目
2. `split_random` 把 keys `random.shuffle` 后按比例切分（先取 `n_test` 给 test，再取 `n_dev` 给 dev，剩下进 train）
3. `write_split` 对每个 split 按 key 排序后输出，每行写 `uid path` 或 `uid text`

**注意**：基础路径是 Windows 本地（`E:\...`），但输出路径是 Linux 服务器（`/data1/...`）。这套脚本明显是 **本地准备 → 拷贝/挂载到 Linux 训练机** 的工作流。

## 2. `sp4_split_train_dev_test.py` — sp4 自有数据集随机划分

与上一脚本几乎完全相同，**唯一差异**：

| 项 | s10 | sp4 |
|---|---|---|
| `BASE_DIR` | `s10_data_reflow` | `sp4_data\labeled` |
| `DATASET_NAME` | `s10_reflow` | `sp4_reflow` |

代码逻辑（`load_data` / `split_random` / `write_split` / `main`）逐字相同，可以考虑抽成单一脚本通过参数化处理（`--base-dir` `--dataset-name`），现在这是明显的复制粘贴。

## 3. `public_dataset_gen_scp_txt.py` — 公开数据集格式归一化

把 8 个常用公开中文 ASR 数据集各自的转写格式统一转为 `wav.scp` + `text`，输出到 `LABEL_ROOT=/data1/code/label/{train,dev,test}/`。

| 项 | 说明 |
|---|---|
| **输入** | 各数据集的原始目录（路径硬编码在每个 `gen_*` 函数里，使用前需要修改 `root` 变量） |
| **输出** | `/data1/code/label/{split}/<dataset_name>_wav.scp` + `<dataset_name>_text` |
| **入口** | `__main__` 里按顺序调用 8 个 `gen_*` 函数 |

**支持的 8 个数据集**（每个数据集独立一个 `gen_*` 函数处理特定格式）：

| 函数 | 数据集 | 转写格式特点 | split 来源 |
|---|---|---|---|
| `gen_aidatatang_200zh` | aidatatang_200zh | 单个 transcript txt（`utt_id text`）+ `corpus/{split}/<spk>/<utt>.wav` | 数据集自带 |
| `gen_data_aishell` | AISHELL-1 | 转写**分词格式需去空格拼接**；`wav/{split}/<spk>/<utt>.wav` | 数据集自带 |
| `gen_data_aishell_1_ios` | data_aishell_1 iOS | 每个 split 自带 `wav.scp`（相对路径，需转绝对）+ `trans.txt`；目录命名特殊（`data/` → train） | 数据集自带，名称重映射 |
| `gen_magicdata` | MagicData | `TRANS.txt` tab 分隔三列（`utt_id\tspk_id\ttext`），需跳过表头；wav 路径含 `.wav` 后缀处理 | 数据集自带 |
| `gen_primewords` | PrimeWords | 单个 JSON（含 `file/user_id/text/length`）；text 分词需去空格；wav 用文件名前两位作哈希分桶路径 | **无划分，按说话人 90/5/5 划分**，`seed=42` |
| `gen_stcmds` | ST-CMDS | 102600 对 `wav` + `txt` 平铺同一目录；从文件名正则 `(P\d+)` 提说话人 ID | **无划分，按说话人 90/5/5 划分** |
| `gen_tal_asr` | TAL_ASR | 单个 transcript txt + `wav/{split}/<spk>/<utt>.wav` | 数据集自带 |
| `gen_thchs30` | THCHS-30 | 每个 wav 配 `.wav.trn`（三行：汉字/拼音/音素，**取第 1 行去空格**）；trn 可能是相对路径引用，需追踪到实际文件 | 数据集自带 |

**通用约束**：
- utt_id 都用 `wav_path.stem`（不带扩展名）
- 路径都用 `wav_path.resolve()` 转绝对路径
- `write_scp_text` 是统一的输出函数，每个 split 一对文件
- 缺转写或文件丢失会计入 `missing` 并打印警告，但不中断

**对无预设划分的数据集**（PrimeWords / ST-CMDS）：
- **按说话人级别划分**而不是条目级别——避免同一说话人同时出现在 train/dev/test 造成数据泄漏
- 都用 `seed=42` 保证可复现
- ratio 默认 `dev=5% / test=5%`

## 4. `merge_all_labels.py` — 三类数据汇总到 all_label

| 项 | 说明 |
|---|---|
| **输入** | `LABEL_ROOT=/data1/label` 下三个子目录：`augment_label`、`reflow_label`、`speech_label`，每个里面是 `{train,dev,test}/<dataset>_wav.scp` + `<dataset>_text` |
| **输出** | `/data1/label/all_label/{train,dev,test}/wav.scp` 和 `text`（每个 split 一对总文件） |

**核心优化（性能关键点）**：

直接给 200w+ 条目的 wav.scp 逐行 `Path.exists()` 会触发海量 `stat` 系统调用，扫描 2 小时都跑不完。这里用了一个**两阶段扫描**：

1. `collect_wav_dirs`：从所有 `*_wav.scp` 解析出涉及到的**音频目录集合**（`os.path.dirname` 去重）
2. `scan_existing_files`：对每个目录一次性 `os.listdir` 把所有文件路径加进一个 `set`
3. 后续逐行处理时，用 `wav_path in existing_set` 做 O(1) 查表，**220w 条秒级完成**

**合并流程（每个 split 独立处理）**：

1. `merge_scp`：
   - 按字典序遍历三个源目录的所有 `*_wav.scp`
   - **去重**（同 key 重复出现只保留首次）
   - **存在性校验**（不在 `existing_files` 里的剔除并加入 `missing_paths`）
   - 每 1w 行打印一次实时速度和缺失数
   - 返回保留下来的 `valid_keys`
2. `merge_text`：
   - 同样遍历三个源目录的 `*_text`
   - 去重 + **过滤**（不在 `valid_keys` 里的跳过，保证 scp 和 text 严格对齐）
3. 校验 `len(scp_lines) == len(text_lines)`，不一致打印警告
4. 末尾打印缺失文件汇总（最多前 50 条）和总计统计

**为什么把 `existing_files` 做成 set 而不是逐个检查？**
- 220w 条 × 几个 ms 的 stat = 1+ 小时
- 改为目录粒度 listdir + set 查表 = 几分钟（每个目录 listdir 一次，单次开销均摊到几万条上）

---

## 几个值得注意的点

1. **三个 reflow 切分脚本的高度重复**：`s10_split_train_dev_test.py` 和 `sp4_split_train_dev_test.py` 几乎完全相同，只差 `BASE_DIR` 和 `DATASET_NAME` 两个常量。可以重构成一个带 `--base-dir --dataset-name` 参数的通用脚本。

2. **路径硬编码**：所有脚本的输入输出路径全部写死在源代码里（`E:\8.音频\...` 和 `/data1/label/...`），不接受命令行参数。改用环境路径需要直接编辑文件——不太友好但确实是实验代码常态。

3. **划分策略的两种粒度**：
   - **条目级随机**（s10/sp4/aidatatang 等已自带 split 的数据集）：简单直接
   - **说话人级随机**（PrimeWords / ST-CMDS）：避免说话人在 train/dev/test 间泄漏，对小数据集尤其重要
   
   两种粒度都用 `seed=42` 保证可复现。

4. **`merge_all_labels.py` 的去重 + 校验逻辑很细**：同时做了 (a) key 重复检测（同一个 utt 在多源目录出现）、(b) 文件存在性校验（音频文件实际存在磁盘上）、(c) scp/text 严格对齐校验。这套校验在数据量上百万条时极其关键，否则训练阶段才发现"路径不存在"会浪费很多时间。

5. **`public_dataset_gen_scp_txt.py` 的格式陷阱**：
   - **AISHELL-1 / THCHS-30 / PrimeWords 都是分词格式**，需要去空格拼回连续中文
   - **MagicData 的 TRANS.txt 有表头**，需要跳过
   - **THCHS-30 的 trn 文件可能是相对路径引用**而不是真实内容，需要追踪
   - **ST-CMDS 没有 transcript 集中文件**，每个 wav 配独立 txt
   
   把这些差异封装在各自的 `gen_*` 函数里是合理的——每个数据集的怪癖都隔离开来。

6. **同名 ipynb 文件**：目录里还有个 `public_dataset_gen_scp_txt.ipynb`，按命名应该是同一份脚本的开发版/调试版 notebook，没有读取的必要。
