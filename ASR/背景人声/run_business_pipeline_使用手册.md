# 业务数据 ASR Pipeline 使用手册

`run_business_pipeline.py` 的完整操作流程：从环境安装 → 数据/模型准备 → 执行 pipeline → 训练 → 训练后评估/精简。

适用：仅业务数据（s10 / sp4），不含公开数据集。

---

## 0. 环境信息

- 服务器：Linux + GPU，**CUDA 驱动 12.4**
- 内网 pip 镜像：`af.hikvision.com.cn`（无法直连公网，pip 自动走此镜像）
- conda：miniforge，环境名 `asr`
- **项目代码目录**：`/data1/背景人声/背景人声`
- **数据集目录（只读）**：`/share/chenming40/云端背景人声抑制/dataset`
- **输出目录（中间产物/训练数据/模型）**：`/data1/背景人声/work`（`/share` 通常只读，故所有产出统一写到 `/data1`，可用 `--output-root` 覆盖）

---

## 1. 安装环境

> 重点：服务器连不上公网，**所有 pip 命令都不要加 `--index-url https://...`**，否则会覆盖内网镜像导致连接失败。直接 `pip install` 即可走内网镜像。

### 1.1 创建 conda 环境

```bash
conda create -n asr python=3.10 -y
conda activate asr
```

### 1.2 安装 PyTorch（匹配 CUDA 12.4）

⚠️ 内网镜像默认会装最新版 torch（如 2.12+cu130），与 12.4 驱动**不匹配**，必须指定版本。

```bash
# 先看镜像里有哪些版本可选（此命令会报错并列出可用版本）
pip install torch== 2>&1 | head -5

# 安装 cu124 编译版（推荐）
pip install torch==2.4.1+cu124 torchaudio==2.4.1+cu124

# 若镜像无 +cu124 后缀包，退而用不带后缀的稳定版（cu121 运行时兼容 12.4 驱动）
pip install torch==2.4.1 torchaudio==2.4.1
```

### 1.3 安装核心依赖

```bash
pip install numpy scipy soundfile funasr pyyaml tqdm tensorboard modelscope
```

| 包 | 用途 |
|---|---|
| numpy / scipy / soundfile | 增强引擎 + jsonl 生成 |
| funasr | 训练引擎（train 阶段） |
| pyyaml | 读增强配置 |
| tqdm | 进度条 |
| tensorboard | 训练 loss 记录 |
| modelscope | 下载预训练模型 |

### 1.4 验证环境

```bash
python -c "import torch, numpy; print('torch:', torch.__version__, '| cuda:', torch.version.cuda, '| avail:', torch.cuda.is_available(), '| gpus:', torch.cuda.device_count())"
```

**必须看到 `avail: True`**，cuda 版本为 12.1 或 12.4，无 numpy 报错。否则回到 1.2 重装匹配版本。

```bash
python -c "import funasr, soundfile, scipy, yaml, tqdm; print('依赖 OK')"
```

> pretag 阶段（FireRed/Qwen 预标注）的依赖可选，数据已预标注完则无需安装。

---

## 2. 准备数据和模型

### 2.1 数据目录（已就位的部分）

```
/share/chenming40/云端背景人声抑制/dataset/      # ← 数据集根目录
└── restruct_reflow_data/
    ├── s10_data/labeled/{wav/, wav.scp, text.txt}
    └── sp4_data/labeled/{wav/, wav.scp, text.txt}
```

> 下文命令统一用 `--dataset-root /share/chenming40/云端背景人声抑制/dataset` 指定。

### 2.2 下载预训练模型（训练前必需）

```bash
modelscope download \
  --model iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
  --local_dir "/data1/背景人声/pretrained_models/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
```

放置后的结构：
```
/data1/背景人声/pretrained_models/
└── speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/
```

> 这是脚本默认的模型根目录（`--pretrained-root` 可覆盖）。FireRedASR-LLM-L、Qwen3-ASR 仅 pretag 阶段需要，数据已预标注完则跳过。

---

## 3. 执行 Pipeline

`run_business_pipeline.py` 共 10 个阶段，每阶段检测产物，已完成自动跳过。

| 阶段 | 说明 | 你的情况 |
|---|---|---|
| prepare | 整理 labeled/unlabeled | 已完成，跳过 |
| quality | 质量扫描+判断 | 已完成，跳过 |
| clean | 删除问题音频（需 --remove-bad） | 可选，默认跳过 |
| pretag | FireRed+Qwen 预标注 | 已完成，跳过 |
| cross_valid | 三方交叉验证 | 已完成，跳过 |
| **split** | labeled 切 train/dev/test | **要跑** |
| **augment** | 仅 train 多说话人距离增强 | **要跑** |
| **merge** | reflow+augment 合并 | **要跑** |
| **jsonl** | 生成 train/dev.jsonl | **要跑** |
| train | torchrun 微调 | 要跑 |

### 3.1 第一步：dry-run 核对路径（必做）

```bash
conda activate asr
cd /data1/背景人声/背景人声     # 项目代码目录

python run_business_pipeline.py \
    --dataset-root "/share/chenming40/云端背景人声抑制/dataset" \
    --dry-run
```

检查打印出来的：labeled 路径、模型路径、各 split 条目数。确认无误再正式跑。

### 3.2 第二步：组装训练数据（split → augment → merge → jsonl）

```bash
python run_business_pipeline.py \
    --dataset-root "/share/chenming40/云端背景人声抑制/dataset" \
    --stages split augment merge jsonl
```

产出（默认在 `--output-root`，即 `/data1/背景人声/work/label/`）：
```
/data1/背景人声/work/label/
├── reflow_label/{train,dev,test}/{s10_reflow,sp4_reflow}_wav.scp + _text
├── augment_label/train/{s10_reflow,sp4_reflow}_wav.scp + _text
└── all_label/
    ├── train/wav.scp + text + train.jsonl   ← 训练输入
    ├── dev/wav.scp + text + dev.jsonl        ← 验证输入
    └── test/wav.scp + text
```

### 3.3 第三步：训练

```bash
# 单卡
python run_business_pipeline.py \
    --dataset-root "/share/chenming40/云端背景人声抑制/dataset" \
    --stages train --gpus 0

# 多卡（如 4 卡）
python run_business_pipeline.py \
    --dataset-root "/share/chenming40/云端背景人声抑制/dataset" \
    --stages train --gpus 0,1,2,3
```

产出：`/data1/背景人声/work/train_outputs/model.pt.ep* / model.pt.best / model.pt.avg5`（= `<output-root>/train_outputs/`）

建议后台运行并记录日志：
```bash
nohup python run_business_pipeline.py \
    --dataset-root "/share/chenming40/云端背景人声抑制/dataset" \
    --stages train --gpus 0,1,2,3 \
    > train.log 2>&1 &

tail -f train.log
```

---

## 4. 训练之后：评估 → 精简 → 部署

`run_business_pipeline.py` 到 train 阶段为止；训练完成后用 `10_eval/`、`11_analyse/`、`13_slim_model_for_inference.py` 这几个独立脚本做评估、分析、精简。

> 路径说明：下面命令都假设训练产物在 `--output-root` 默认位置 `/data1/背景人声/work`。
> 评估脚本（`batch_eval_models.py` 等）自带的默认路径是 `/data1/label`、`/data1/eval_output`、`/data1/pretrained_models`，与本 pipeline 的输出根不一致，所以下面统一用参数显式覆盖。
> 先设两个变量方便复制：
>
> ```bash
> BASE_MODEL=/data1/背景人声/pretrained_models/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
> TRAIN_OUT=/data1/背景人声/work/train_outputs
> ```

### 4.1 看训练 loss 曲线（可选，先确认训练正常）

> 注意：FunASR 用 hydra，**`train.log` 写在「启动训练的目录」下的 `outputs/<日期>/<时间>/train.log`**，不是 `train_outputs/`（那里放的是权重）。先定位日志：
>
> ```bash
> ls outputs/*/*/train.log        # 或： find . -name train.log
> ```

```bash
# 递归扫描 outputs/ 下所有 train.log（中断续训产生的多段日志会自动按时间合并）
python 10_eval/plot_loss.py --log_dir outputs

# 或指定单个日志文件
python 10_eval/plot_loss.py --log outputs/2026-06-22/01-32-08/train.log
```

产出 `loss_curve.png`，看 train/val loss 是否正常下降、有没有过拟合，挑出合适的 epoch。

### 4.2 评估微调模型（reflow dev/test，主入口）

`batch_eval_models.py` 是规范指定的批量评估入口，会扫描 `--model-dir` 下所有 `model.pt.*` 权重，对 `--label-root` 的数据集逐个算 CER/SER，并可与基线对比。

```bash
python 10_eval/batch_eval_models.py \
    --model-dir "$TRAIN_OUT" \
    --base-model "$BASE_MODEL" \
    --label-root /data1/背景人声/work/label \
    --label-source reflow_label \
    --splits dev test \
    --ckpt-pattern best \
    --baseline-dir "" \
    --output-root /data1/背景人声/work/eval_output
```

- `--ckpt-pattern best`：只评 `model.pt.best`；去掉则评估全部权重并生成跨权重对比（`model_comparison_report.md`）。
- `--baseline-dir ""`：跳过基线对比（首次没有 `base/` 基线结果时）；有基线结果再传其目录，会额外生成 `baseline_comparison_report.md`。
- 产出：`/data1/背景人声/work/eval_output/<权重名>/reflow_label/{dev,test}/` 下的 `eval_summary.json`、`eval_detail.jsonl`、`eval_report.txt`。

### 4.3 独立测试集评估（可选）

对单个测试集（如 reflow test、周报测试集、儿童语音等）用 `eval_on_testset.py`，输入一对 `wav.scp` + `text`：

```bash
python 10_eval/eval_on_testset.py \
    --model-dir "$TRAIN_OUT" \
    --base-model "$BASE_MODEL" \
    --test-scp  /data1/背景人声/work/label/reflow_label/test/s10_reflow_wav.scp \
    --test-text /data1/背景人声/work/label/reflow_label/test/s10_reflow_text \
    --output-dir /data1/背景人声/work/eval_output/s10_testset \
    --ckpt-pattern best
```

sp4 同理把 `s10_reflow` 换成 `sp4_reflow`。一次性跑多个测试集可参考 `10_eval/batch_eval_all_testsets.sh`（注意里面路径是 `/data1/eval_output`、`/data1/label/reflow_origin`，需先按你的实际路径改）。

### 4.4 精简模型用于推理部署

`13_slim_model_for_inference.py` 去掉优化器/调度器等训练状态，只留模型权重，体积更小、加载更快。**该脚本没有命令行参数，入口路径写死在文件末尾**（当前是 Windows 路径），先改成你的权重再运行：

```python
# 13_slim_model_for_inference.py 末尾 __main__：
input_file  = "/data1/背景人声/work/train_outputs/model.pt.best"
output_file = "/data1/背景人声/work/train_outputs/model.pt"   # 精简后用于部署
```

```bash
python 13_slim_model_for_inference.py
```

### 4.5 深入性能分析（可选）

`11_analyse/scripts/` 下的脚本基于 4.2/4.3 产出的 `eval_detail.jsonl` / `eval_summary.json` 做指令召回率、错误模式、跨版本对比等分析，报告写到 `11_analyse/reports/`。分析口径见 `.kiro/steering/model-eval-analysis.md`。

---

## 5. 常用参数

```
--dataset-root <路径>      数据集根目录（必填，指向你的 dataset/，只读）
--output-root <路径>       输出根目录（中间文件/训练数据/模型，需可写；默认 /data1/背景人声/work）
--stages <阶段...>         指定执行哪些阶段
--datasets s10_reflow      只处理 s10（默认 s10+sp4）
--augment-factor 2         每条原始音频增强几份（默认 1）
--augment-config <yaml>    换增强配置（默认多说话人距离增强）
--pretrained-root <路径>   模型放别处时整体改根
--base-model <路径>        单独指定微调起点模型（SeacoParaformer）
--firered-model <路径>     FireRedASR-LLM-L 模型目录（仅 pretag 阶段用）
--qwen-model <路径>        Qwen3-ASR 模型目录（仅 pretag 阶段用）
--gpus 0,1,2,3             训练用 GPU
--no-resume-train          训练不续跑，每次从预训练权重重新开始（默认续跑：断了重跑会接着练）
--workers 8                质检线程数
--remove-bad               启用 clean 阶段删除问题音频
--force                    强制重跑（忽略已存在产物）
--dry-run                  预览不执行
```

---

## 6. 常见操作

**只跑 s10：**
```bash
python run_business_pipeline.py --dataset-root "/share/chenming40/云端背景人声抑制/dataset" \
    --datasets s10_reflow --stages split augment merge jsonl
```

**改了增强配置想重新增强：**（加 --force 覆盖旧产物）
```bash
python run_business_pipeline.py --dataset-root "/share/chenming40/云端背景人声抑制/dataset" \
    --stages augment merge jsonl --force
```

**增强倍数调到 2：**
```bash
python run_business_pipeline.py --dataset-root "/share/chenming40/云端背景人声抑制/dataset" \
    --stages augment merge jsonl --augment-factor 2 --force
```

**模型放在别的目录：**
```bash
python run_business_pipeline.py --dataset-root "/share/chenming40/云端背景人声抑制/dataset" \
    --stages train --gpus 0 \
    --base-model /your/path/to/speech_seaco_paraformer_large_..._vocab8404-pytorch
```

---

## 7. 注意事项

1. **首次务必 `--dry-run`** 核对路径，避免跑一半才发现路径错。
2. **pip 不要加 `--index-url`**，走内网镜像；torch 必须装 cu124/cu121 匹配 12.4 驱动的版本。
3. **增强只作用于 train**，dev/test 保持原始，无数据泄漏。
4. **重跑某阶段要加 `--force`**，否则检测到产物已存在会跳过。
5. **split/merge/jsonl 用 CPU 即可**；train 需 GPU + 已下载的预训练模型。
6. **切分用固定 seed=42**，可复现；train/dev/test = 90%/5%/5%。
7. **所有产出写到 `--output-root`（默认 `/data1/背景人声/work`），不写回只读的 `/share` 数据集盘**；训练数据在 `<output-root>/label/all_label/`，模型在 `<output-root>/train_outputs/`。

---

## 8. 一键安装脚本

存为 `setup_env.sh`，在服务器执行 `bash setup_env.sh`：

```bash
#!/bin/bash
set -e
conda create -n asr python=3.10 -y
source activate asr
# torch 匹配 CUDA 12.4 驱动（走内网镜像，不加 --index-url）
pip install torch==2.4.1+cu124 torchaudio==2.4.1+cu124 || pip install torch==2.4.1 torchaudio==2.4.1
pip install numpy scipy soundfile funasr pyyaml tqdm tensorboard modelscope
python -c "import torch; print('cuda:', torch.version.cuda, '| avail:', torch.cuda.is_available(), '| gpus:', torch.cuda.device_count())"
echo "完成。conda activate asr 后使用"
```
