# S10 ASR 数据集工具链

## 脚本概览

| 脚本 | 说明 |
|------|------|
| `01_prepare_funasr_data.py` | 整理有标签音频到 `labeled/`，未标注音频到 `unlabeled/`，生成 FunASR 训练格式 |
| `01_prepare_funasr_data_sp4.py` | SP4 数据集整理：合并已有 text/wav.scp（含增量 add 文件），Linux→本地路径重映射，复制/移动 wav 到 labeled/unlabeled 目录，生成 FunASR 训练格式 |
| `02_audio_quality_check.py` | 音频数据质量扫描，输出逐条分析结果 (`quality_detail.jsonl`) |
| `03_quality_report.py` | 基于 `quality_detail.jsonl` 按可配置阈值判断问题，生成统计报告和文件清单 |
| `04_sort_bad_files.py` | 按问题类型分类拷贝问题音频 |
| `05_remove_bad_files.py` | 从 `labeled/unlabeled` 中删除问题音频 |
| `07_SeACo_Paraformer_eval.py` | SeACo-Paraformer 离线评估已标注数据 CER（支持热词增强、多目录模式） |
| `10_eval/seaco_eval.py` | SeACo-Paraformer 本地模型评估（支持多标签格式、热词增强、多目录评估+汇总、随机采样、断点续跑、自动 GPU 选择、短音频过滤） |
| `10_eval/batch_eval.py` | 公开数据集批量评估：自动发现 `public_dataset_gen_scp_txt.py` 生成的数据集，批量运行 SeACo 基准模型和/或微调模型评估，支持多数据集×多 split×多模型组合，生成数据集级和全局汇总报告 |
| `10_eval/batch_eval_models.py` | 多模型权重批量评估：扫描模型目录下所有权重文件（`model.pt.*`），对 `/data1/label` 下的数据集进行批量评估，支持多标签源、多 split、采样评估、baseline 对比（含基线对比报告），生成跨模型对比报告；支持 `--report-only` 模式从已有评估结果直接生成基线对比报告（不跑推理） |
| `10_eval/gen_baseline_report.py` | 基线对比报告独立生成：从已有评估结果目录读取 `eval_summary.json`，与基线结果对比，生成 `baseline_comparison_report.md`（不需要重新跑模型推理），支持单模型或多模型父目录、指定基线模式过滤 |
| `10_eval/eval_on_testset.py` | 指定测试集多模型权重批量评估：`--test-scp` 和 `--test-text` 为必填参数，支持单目录模式（直接扫描 `model.pt.*`）和多版本模式（自动扫描子目录），逐个加载并评估 CER/SER，生成跨模型对比报告（Markdown + JSON），支持指定版本过滤、采样快速验证、同时评估基准模型、断点续跑 |
| `10_eval/plot_loss_tb.py` | TensorBoard Loss 曲线绘制：从 tfevents 文件提取 train/val loss，支持多目录合并、自动扫描、loss\_key 选择（loss/loss\_att/loss\_seaco/任意 tag）、epoch/step 双视图、自动标注最佳 val loss、手动指定 steps\_per\_epoch、列出可用 tags |
| `07_augment_used/batch_augment_public.py` | 公开数据集批量增强（多进程版）：基于 `public_dataset_gen_scp_txt.py` 生成的标签目录结构，按数据集采样 10% 进行多说话人增强，各 split 完全隔离（train/dev/test 的主说话人和背景说话人互不交叉），条目级多进程并行，支持断点续跑 |
| `07_augment_used/augment_short_in_meeting.py` | 短音频嵌入会议背景增强：将 ≤2秒的清晰短音频（唤醒词/短指令）随机嵌入 AliMeeting 远场多人会议背景片段中，模拟真实会议干扰场景，增强音频输出到 `output_dir/wav/` 子目录，主说话人和背景目录均带索引缓存加速二次扫描，支持可选混响/底噪后处理、可配置清晰度过滤阈值（`min_rms`/`max_clip_ratio`）、归一化模式（`main_region`/`global`）、独立增益调整（`main_gain_db`/`bg_gain_db`）、断点续跑，支持传入标签文件（`--text_file`）和 wav.scp（`--scp_file`）以生成增强标签 |
| `07_augment_used/augment_dir.py` | 基于目录的数据增强（无标签），扫描 wav 目录直接增强，不需要 manifest，支持断点续跑。与 `07_augment/augment_dir.py` 功能相同，使用本目录下的 `augment_engine.py` 和配置文件 |
| `07_augment_used/augment_manifest.py` | 基于 manifest 的数据增强（带标签），读取 JSONL manifest 按 split 隔离增强，内置 LRU 音频缓存避免 OOM，输出增强音频和 Kaldi 格式 wav.scp + text，支持断点续跑 |
| `11_analyse/analyse_eval_detail.py` | 统计 `eval_detail.jsonl` 的整体 CER/SER，并将错误样本按 CER 降序输出到可读 txt 文件 |
| `11_analyse/calc_command_recall.py` | 计算两个模型的指令召回率（标签文本作为子串出现在识别结果中的比例），对比分析未召回样本差异 |
| `11_analyse/analyse_all_models.py` | 多模型错误特征对比分析：递归加载 `eval_output` 下 6 个模型（3 版本 × 2 权重）的 `eval_detail.jsonl`，对比指令召回率、CER、句准确率、错误类型分布，覆盖业务测试集和 12\_test 测试集 |
| `11_analyse/analyse_testset.py` | testset\_results 6 模型错误深度分析：指令召回率、错误分类（唤醒词残留/多识别/替换/漏识别/指令淹没）、跨模型未召回对比、高频混淆对提取 |
| `11_analyse/analyse_testset_strip_wakeword.py` | 去除识别结果开头/结尾唤醒词（"你好萤石"及变体）后重新计算 CER 和指令召回率，对比 6 个模型去唤醒词前后的改善幅度并排名 |
| `11_analyse/analyse_testset_strip_wakeword_v2.py` | 去除标签和识别结果开头/结尾唤醒词后重新计算 CER 和指令召回率（v1 仅处理识别结果，v2 对标签和识别结果做同样处理），扩展唤醒词变体列表，对比 6 个模型去唤醒词前后的改善幅度并排名 |
| `11_analyse/compare_all_models_recall.py` | 四模型（seaco 基线、online 线上、v0.2.0、v0.3.0）在业务数据集（s10\_reflow / sp4\_reflow / s10\_augment）上的指令召回率对比，含 CER/句准确率/错误类型统计及 vs online 改进幅度分析 |
| `11_analyse/export_v3_v4_errors.py` | 导出 v0.3.0 和 v0.4.0 在业务数据集（s10\_reflow/test、sp4\_reflow/test）上的所有错误样本，按"两者都错/仅v0.3.0错/仅v0.4.0错"分类并列展示两版本识别差异 |
| `11_analyse/export_v3_v4_errors_stripped.py` | 去除标签和识别结果开头唤醒词后，重新计算 CER 并对比 v0.3.0 和 v0.4.0 在业务数据集上的错误（消除唤醒词残留对 CER 的干扰） |
| `11_analyse/recall_comparison_report.py` | 生成 v0.3.0 vs v0.4.0 指令召回率对比报告：去除开头唤醒词后，聚焦"v0.3.0能召回但v0.4.0不能"和"v0.4.0能召回但v0.3.0不能"的样本，按标签长度排序输出 |
| `11_analyse/compare_v3_v5.py` | 生成 v0.3.0 vs v0.5.0/ep23 完整对比报告：去除开头唤醒词后的错误对比（按"两者都错/仅v0.3.0错/仅v0.5.0错"分类）+ 指令召回率专项对比（聚焦两版本召回差异样本） |
| `check_eval_data.py` | 检查评估数据完整性：验证各版本 combined\_summary.json、base 基线数据、全量业务数据、model\_comparison\_summary.json 是否存在 |
| `extract_best_model_cer.py` | 提取各版本最佳模型在 19 个 test 数据集上的 CER，生成全量业务数据（S10/SP4）对比表 + 全数据集 CER 对比表（含最优模型标注） |
| `extract_performance.py` | 提取各版本最佳模型在全量业务数据（S10/SP4）和 19 个 test 数据集上的 CER/SER/句准确率，生成完整对比表（含最优模型标注和缺失数据报告） |
| `calc_recall_and_stripped_cer.py` | 计算各版本最佳模型的 SP4 指令召回率（剔除唤醒词后 ref⊆hyp）和独立测试集剔除唤醒词前后 CER 对比 |
| `analyze_testset_gap.py` | 分析 testset 上 seaco vs v0.5.0/ep35 剔除唤醒词后 CER 差异的原因：逐条对比编辑距离、统计唤醒词残留、输出长度差异分析 |
| `find_wakeword_residual.py` | 找出 testset 中 v0.5.0/ep35 剔除唤醒词后识别结果仍含唤醒词残留的样本，输出音频名称、标签、seaco 和 v0.5.0 识别结果对比 |
| `analyze_child_gap.py` | 分析 ChildMandarin 数据集上 seaco vs v0.5.0/ep35 的退化表现：按年龄/性别/方言/句子长度多维度对比 CER，统计错误类型分布，输出典型 bad case |
| `analyze_child_error_patterns.py` | 分析 ChildMandarin 上 v0.5.0 退化样本的具体错误模式/场景分类：唤醒词插入、长句截断、模型重复输出、语气词/拟声词、重复/结巴语音等，按场景统计退化贡献并输出典型 case |
| `12_test/gen_test_text.py` | 从 `wavs/` 各日期文件夹中整合 label.txt，生成测试集 `text` 和 `wav.scp` |
| `12_test/trim_wakeword.py` | 【服务器端】使用 FunASR 时间戳对齐检测音频中唤醒词"你好萤石"的位置（只做检测不做裁剪），自动过滤损坏/空文件，输出 `trim_log.jsonl` 供本地裁剪脚本使用 |
| `12_test/verify_trim_energy.py` | 基于短时能量分析校验 `trim_log.jsonl` 中唤醒词时间戳边界的合理性，在 boundary 附近寻找能量低谷验证切分点，输出校验结果和统计报告，支持可视化 |
| `12_test/ChildMandarin/prepare_for_eval.py` | ChildMandarin 数据集预处理：递归扫描 speaker 子目录，从 JSON 标注读取 uid/text，检查音频格式（16kHz 单声道），生成 `eval_on_testset.py` 所需的 `wav.scp` 和 `text` |
| `07_augment/augment_data.py` | 数据增强入口（向后兼容），内部委托给 `augment_manifest.py` |
| `07_augment/augment_manifest.py` | 基于 manifest 的数据增强（带标签），按 split 隔离增强，输出增强音频和训练格式文件 |
| `07_augment/augment_dir.py` | 基于目录的数据增强（无标签），扫描 wav 目录直接增强，不需要 manifest，支持断点续跑 |
| `07_augment/augment_engine.py` | 增强引擎模块，提供 `AudioAugmentor` 类供其他脚本导入使用（不直接运行） |
| `07_augment/manifest_to_scp.py` | 将三字段 manifest.jsonl 转换为 Kaldi 格式 `wav.scp` + `text.txt`，按 split 自动切分 |
| `run_pipeline.py` | 按顺序执行各阶段脚本（6步流水线） |
| `audio_review_server.py` | Web 版音频人工审核打分工具 |
| `gen_hotwords.py` | 从标签文件生成热词表 |
| `merge_scp.py` | 合并多个 wav.scp / text.txt 文件（按 key 去重，保留首次出现） |
| `json_jsonl_convert.py` | JSON 与 JSONL 互转工具 |
| `fix_scp_paths.py` | 批量替换 wav.scp 中的音频路径前缀（支持预览和输出到新文件） |
| `fix_unlabeled_paths.py` | 修复结果文件中 `unlabeled/xxx.wav` → `unlabeled/wav/xxx.wav` 的路径问题 |
| `gen_wav_scp.py` | 为指定目录生成 Kaldi 格式 `wav.scp` |
| `download_aishell4.py` | 下载 AISHELL-4 数据集（支持断点续传/自动解压） |
| `download_alimeeting.py` | 下载 AliMeeting (M2MeT) 数据集（支持断点续传/自动解压） |
| `download_magicdata.py` | 通过 ModelScope 下载 MagicData-Read 数据集，自动尝试多个 dataset ID |
| `utils/public_dataset_gen_scp_txt.py` | 公开 ASR 数据集批量转换为 Kaldi 格式 `wav.scp` + `text`（支持 8 种数据集） |
| `analyze_audio_distribution.py` | 音频音量/SNR/推测距离分布分析：扫描 wav 目录，统计 RMS 音量(dBFS)、SNR(dB)、推测录音距离，输出统计摘要、分布直方图(PNG)、详细数据(CSV) |
| `slim_model_for_inference.py` | 精简 FunASR SeacoParaformer 训练 checkpoint：移除优化器/调度器等训练状态，只保留模型权重，减小体积加快加载 |
| `s10_data_reflow/apply_reflow.py` | 将 Label Studio 重新标注结果回流到 text.txt 和 wav.scp（剔除/更新/备份） |
| `s10_data_reflow/download_models.py` | 下载 FunASR 所需模型（paraformer-zh、fa-zh）到本地 `F:\pretrained_models` |
| `s10_data_reflow/analyze_trigger.py` | 分析标注数据中唤醒词（"你好萤石"等）的分布：统计纯唤醒词 vs 唤醒词+指令的数量和位置模式 |
| `s10_data_reflow/count_trigger_cmd.py` | 统计 text.txt 中"触发词+命令"条目数量，按触发词前缀分类计数并展示示例 |
| `s10_data_reflow/count_hao_yingshi.py` | 分析 text.txt 中"好萤石"+命令的记录（排除"你好萤石"），按前缀变体统计分布并展示命令模式 |
| `s10_data_reflow/move_only_trigger.py` | 将 text.txt 中仅含触发词（"你好萤石"等）的音频移动到 `only_trigger/` 目录，从训练数据中分离纯唤醒词样本 |
| `s10_data_reflow/remove_only_trigger.py` | 从 labeled copy 中彻底删除仅含触发词的记录：删除 wav 音频、清理 text.txt 和 wav.scp 对应行 |
| `s10_data_reflow/remove_partial_trigger.py` | 从 text.txt 和 wav.scp 中删除仅含部分触发词（萤石、好萤石等不含"你好"前缀）的记录 |
| `s10_data_reflow/copy_yingshi_prefix.py` | 将"萤石"前缀（不含"你好萤石"和"好萤石"）的音频拷贝到 `萤石_prefix/` 目录，生成文本列表供查看 |
| `s10_data_reflow/move_trigger_cmd.py` | 将"你好萤石+命令"和"好萤石+命令"的音频移动到 `trigger_cmd/` 目录，生成列表文件供服务器端强制对齐裁剪 |
| `s10_data_reflow/labeled copy/code/cut_trigger_funasr/trim_trigger_funasr.py` | 基于 FunASR MonotonicAligner 强制对齐模型对"触发词+命令"音频做字级时间戳对齐，裁剪掉触发词前缀，只保留命令部分，支持 batch 推理、整批失败时自动降级逐条处理 |
| `s10_data_reflow/labeled copy/code/cut_trigger_funasr/apply_trim_result.py` | 将 `trim_trigger_funasr.py` 的裁剪结果回写到 text.txt / wav.scp：成功的更新文本为去掉触发词的命令并替换 wav/ 中的音频，失败的从标签文件中删除 |
| `s10_data_reflow/split_train_dev_test.py` | 按设备ID将 text.txt / wav.scp 切分为 train / dev / test（同设备数据不跨 split，避免数据泄漏） |
| `08_split/reflow_split_train_dev_test.py` | 按设备ID（`02020adeXXXX`）将 s10\_data\_reflow 的 text.txt / wav.scp 切分为 train / dev / test |
| `08_split/sp4_split_train_dev_test.py` | 按设备ID（key 第二段，如 `K17175903`）将 SP4 labeled 的 text.txt / wav.scp 切分为 train / dev / test |
| `08_split/s10_split_train_dev_test.py` | 按条目完全随机将 s10\_data\_reflow 的 text.txt / wav.scp 切分为 train 90% / dev 5% / test 5%（seed=42 可复现） |
| `08_split/merge_all_labels.py` | 将 augment\_label / reflow\_label / speech\_label 中的数据按 train/dev/test 拼接汇总到 all\_label/（按 key 去重，预扫描文件集合加速存在性检查） |
| `08_split/public_dataset_gen_scp_txt.py` | 公开 ASR 数据集批量转换为 Kaldi 格式 `wav.scp` + `text`（支持 8 种数据集：aidatatang\_200zh / AISHELL-1 / AISHELL-1 iOS / MagicData / PrimeWords / ST-CMDS / TAL\_ASR / THCHS-30） |
| `s10-dataset-struct.md` | 数据集结构总览文档 |

### `utils/`

| 脚本 | 说明 |
|------|------|
| `public_dataset_gen_scp_txt.py` | 公开 ASR 数据集批量转换为 Kaldi 格式 `wav.scp` + `text`（支持 8 种数据集） |
| `genetate_scp.py` | 为指定目录生成 Kaldi 格式 `wav.scp`（扫描 wav 文件，输出 `<uid> <绝对路径>`） |

### `10_eval/`

| 脚本 | 说明 |
|------|------|
| `seaco_eval.py` | SeACo-Paraformer 本地模型评估（支持多标签格式、热词增强、多目录评估+汇总、随机采样、断点续跑、自动 GPU 选择、短音频过滤） |
| `batch_eval.py` | 公开数据集批量评估：自动发现 `public_dataset_gen_scp_txt.py` 生成的数据集，批量运行 SeACo 基准模型和/或微调模型评估，支持多数据集×多 split×多模型组合，生成数据集级和全局汇总报告 |
| `batch_eval_models.py` | 多模型权重批量评估：扫描模型目录下所有权重文件（`model.pt.*`），对 `/data1/label` 下的数据集进行批量评估，支持多标签源、多 split、采样评估、baseline 对比（含基线对比报告），生成跨模型对比报告；支持 `--report-only` 模式从已有评估结果直接生成基线对比报告（不跑推理） |
| `gen_baseline_report.py` | 基线对比报告独立生成：从已有评估结果目录读取 `eval_summary.json`，与基线结果对比，生成 `baseline_comparison_report.md`（不需要重新跑模型推理），支持单模型或多模型父目录、指定基线模式过滤 |
| `eval_on_testset.py` | 指定测试集多模型权重批量评估：`--test-scp` 和 `--test-text` 为必填参数，支持单目录模式和多版本模式（自动扫描子目录），逐个加载并评估，生成跨模型对比报告（Markdown + JSON），支持指定版本过滤 |
| `eval_utils.py` | ASR 评估公用工具模块（CER 计算、文本规范化、标签加载、wav 收集、日志配置、单目录评估、报告生成、多目录汇总报告等），供 `seaco_eval.py` / `finetuned_eval.py` 等评估脚本复用 |
| `plot_loss_tb.py` | 从 TensorBoard 事件文件提取 train/val loss 绘制曲线图：支持多目录合并（中断恢复场景）、自动扫描 tfevents、按 loss\_key 选择指标、epoch/step 双视图、自动标注最佳 val loss epoch、手动指定 steps\_per\_epoch、列出所有可用 tags |

### `11_analyse/`

| 脚本 | 说明 |
|------|------|
| `analyse_eval_detail.py` | 统计 `eval_detail.jsonl` 的整体 CER/SER，并将错误样本按 CER 降序输出到可读 txt 文件 |
| `calc_command_recall.py` | 计算两个模型的指令召回率（标签文本作为子串出现在识别结果中的比例），对比分析未召回样本差异 |
| `analyse_all_models.py` | 多模型错误特征对比分析：递归加载 6 个模型的评估结果，对比指令召回率、CER、句准确率、错误类型分布 |
| `analyse_testset.py` | testset\_results 6 模型错误深度分析：指令召回率、错误分类、跨模型未召回对比、高频混淆对提取 |
| `analyse_testset_strip_wakeword.py` | 去除识别结果开头/结尾唤醒词后重新计算 CER 和指令召回率，对比 6 个模型去唤醒词前后的改善幅度并排名 |
| `analyse_testset_strip_wakeword_v2.py` | 去除标签和识别结果开头/结尾唤醒词后重新计算 CER 和指令召回率（双向去除），扩展唤醒词变体列表，对比改善幅度并排名 |
| `compare_all_models_recall.py` | 四模型（seaco/online/v0.2.0/v0.3.0）在业务数据集上的指令召回率对比，含 CER/句准确率/错误类型统计及 vs online 改进幅度 |
| `export_v3_v4_errors.py` | 导出 v0.3.0 和 v0.4.0 在业务数据集上的所有错误样本，按"两者都错/仅v0.3.0错/仅v0.4.0错"分类并列展示识别差异 |
| `export_v3_v4_errors_stripped.py` | 去除标签和识别结果开头唤醒词后重新计算 CER，对比 v0.3.0 和 v0.4.0 错误（消除唤醒词残留干扰） |
| `recall_comparison_report.py` | v0.3.0 vs v0.4.0 指令召回率对比：去除开头唤醒词后，聚焦两版本召回差异样本，按标签长度排序输出 |
| `compare_v3_v5.py` | v0.3.0 vs v0.5.0/ep23 完整对比报告：去唤醒词后错误对比 + 指令召回率专项对比，输出两份报告 |

### `12_test/`

| 脚本 | 说明 |
|------|------|
| `gen_test_text.py` | 从 `wavs/` 各日期文件夹中整合 `label.txt`，生成测试集 `text` 和 `wav.scp` |
| `trim_wakeword.py` | 【服务器端】使用 FunASR 时间戳对齐检测音频中唤醒词"你好萤石"位置，自动过滤损坏/空文件，输出 `trim_log.jsonl` 供本地裁剪 |
| `verify_trim_energy.py` | 基于短时能量分析校验唤醒词边界合理性，在 boundary 附近寻找能量低谷验证切分点，输出校验结果和统计报告 |
| `ChildMandarin/prepare_for_eval.py` | ChildMandarin 数据集预处理：从 JSON 标注读取 uid/text，检查音频格式，生成 `wav.scp` 和 `text` |

### `06_pretag_and_compare/`

| 脚本 | 说明 |
|------|------|
| `pretag_utils.py` | 预标注脚本公共工具模块（日志、wav 扫描、进度管理、结果写入、GPU 自动选择） |
| `FunASR_eval_labeled.py` | FunASR 评估已标注数据 CER/SER |
| `FunASR_pretag_unlabeled.py` | FunASR 预标注未标注数据 |
| `FireRedASR_pretag.py` | FireRedASR 预标注（LLM/AED，自动分桶 + 断点续跑） |
| `QwenASR_pretag.py` | Qwen3-ASR 预标注（transformers/vllm/raw 三种后端，断点续跑） |
| `sort_error_samples.py` | 按类别拷贝错误样本音频（支持交叉验证/比对两种模式），方便人工复核 |
| `review_labels.py` | 人工复核标签工具：生成 review.txt 预填候选标签 + 合并回写 text.txt |
| `compare_pretag.py` | 比对两个预标注模型的标签结果，计算 CER，输出不一致清单 |
| `cross_validate_labels.py` | 三方交叉验证（人工 vs FireRedASR vs Qwen3），找出人工标注错误 |
| `cross_validate_review_server.py` | Web 版交叉验证人工复核工具，播放音频并展示三方标签，支持选择/编辑/剔除 |
| `export_labeled_check.py` | 从交叉验证结果导出不一致样本到 `labeled_check/`，生成三方 text.txt / train.jsonl / wav.scp |

## 输出目录结构

均在 `s10_data_reflow/` 下：

```
s10_data_reflow/
├── labeled/                  # 已标注数据 (wav/ + train.jsonl + wav.scp + text.txt)
├── unlabeled/                # 未标注音频
├── quality_check/            # 质量检查结果 (quality_detail.jsonl + 报告 + bad_files.txt + bad_sorted/)
├── hotwords/                 # 热词表 (hotwords.txt + hotwords_detail.txt + ...)
├── eval_output/              # FunASR 评估结果 (eval_detail.jsonl + eval_report.txt + eval_summary.json)
├── seaco_eval_output/        # SeACo-Paraformer 评估结果
├── pretag_output/            # FunASR 预标注结果 (pretag.jsonl + wav.scp + text.txt)
├── firered_pretag_output/    # FireRedASR 预标注结果 (+ progress.txt)
├── qwen_pretag_output/       # Qwen3-ASR 预标注结果 (+ progress.txt)
├── augmented/                # 数据增强输出 (按 split 子目录存放增强音频 + manifest.jsonl)
├── compare_output/           # 两模型比对结果 (compare_detail.jsonl + diff_list.txt + compare_report.txt + compare_summary.json)
├── cross_validate_output/    # 三方交叉验证结果 (validate_detail.jsonl + models_agree.txt + all_differ.txt + ...)
├── labeled_check/            # 不一致样本导出 (wav/ + 三方 text.txt/train.jsonl + wav.scp + validate_detail.json)
├── review/                   # 人工复核 (review.txt + merge 备份)
└── error_sorted/             # 错误样本分类拷贝 (按 category 或 CER 区间子目录 + labels.txt)
```

---

## 各脚本详细用法

### 01\_prepare\_funasr\_data

整理有标签音频到 `labeled/`，未标注音频到 `unlabeled/`。通过脚本内 `COPY_MODE` 变量控制：`True` = 复制，`False` = 移动（默认移动）。

```bash
python 01_prepare_funasr_data.py
```

输出到 `s10_data_reflow/`：

- `labeled/wav/` — 已标注音频
- `labeled/train.jsonl` — FunASR ChatML 格式训练数据
- `labeled/wav.scp` — Kaldi 格式音频路径列表
- `labeled/text.txt` — Kaldi 格式文本标注
- `unlabeled/` — 未标注音频
- `unlabeled/wav.scp` — 未标注音频的 Kaldi 格式路径列表（自动生成）

### 01\_prepare\_funasr\_data\_sp4

SP4 数据集整理脚本。与 `01_prepare_funasr_data.py` 的区别：SP4 数据集已有 `text` / `wav.scp` 文件（含增量 `_add` 文件），无需从 label 目录扫描；`wav.scp` 中路径为 Linux 服务器路径，需重映射到本地 `trimed_wav/` 下。

功能流程：

1. 读取多个 `text` / `wav.scp` 文件（含增量 add 文件），合并为完整标签集
2. 将 Linux 服务器路径重映射为本地 Windows 路径，校验文件存在性
3. 匹配标签与音频
4. 将匹配到的 wav 复制/移动到 `labeled/wav/` 目录，生成 FunASR 训练所需的 jsonl、wav.scp、text.txt
5. 将未标注音频复制/移动到 `unlabeled/` 目录，并生成 wav.scp

```bash
python 01_prepare_funasr_data_sp4.py
```

使用前需修改脚本内的配置变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `BASE_DIR` | SP4 数据根目录 | `/data1/code/output/sp4` |
| `OUTPUT_DIR` | 标注数据输出目录 | `BASE_DIR / "labeled"` |
| `UNLABELED_DIR` | 未标注数据输出目录 | `BASE_DIR / "unlabeled"` |
| `TEXT_FILES` | text 标签文件列表（含增量 add 文件） | `[text, text_0924_add, text_20241204_add]` |
| `WAV_SCP_FILES` | wav.scp 文件列表（含增量 add 文件） | `[wav.scp, wav_0924_add.scp, wav_20241204_add.scp]` |
| `WAV_ALL_SCP` | 全量音频列表（含未标注），用于识别 unlabeled | `wav_all.scp` |
| `LINUX_PREFIX` | Linux 路径截取前缀 | `sp4_data/` |
| `COPY_MODE` | 文件操作模式（`True` = 复制，`False` = 移动） | `False` |

路径重映射规则：`wav.scp` 中的 Linux 路径（如 `/home/.../sp4_data/trimed_wav/...`）会截取 `sp4_data/` 之后的相对路径，拼接到 `BASE_DIR` 下。

输出到 `{OUTPUT_DIR}/`（即 `labeled/`）：

- `wav/` — 已标注音频（从 `trimed_wav/` 复制/移动过来）
- `train.jsonl` — FunASR ChatML 格式训练数据
- `wav.scp` — Kaldi 格式音频路径列表
- `text.txt` — Kaldi 格式文本标注

输出到 `{UNLABELED_DIR}/`（即 `unlabeled/`）：

- `*.wav` — 未标注音频文件（从 `trimed_wav/` 和 `wav_all.scp` 中识别的无标签音频）
- `wav.scp` — 未标注音频的 Kaldi 格式路径列表

> 依赖：仅 Python 标准库（`json`, `shutil`, `pathlib`, `collections`）

### 02\_audio\_quality\_check

> 依赖：仅 Python 标准库（`wave`, `struct`, `math`）
>
> 仅采集原始数值，不做阈值判断。

```bash
# 默认 4 进程扫描所有目录
python 02_audio_quality_check.py

# 8 进程加速
python 02_audio_quality_check.py --workers 8

# 只扫描原始 source 目录
python 02_audio_quality_check.py --source-only
```

输出到 `s10_data_reflow/quality_check/`：

- `quality_detail.jsonl` — 逐条分析结果（duration / rms / snr\_db / dc\_offset / clipping\_ratio 等原始数值）

后续运行 `03_quality_report.py` 基于 jsonl 按阈值判断，生成报告和文件清单。

### 03\_quality\_report

> 依赖：仅 Python 标准库
>
> 前置：先运行 `02_audio_quality_check.py` 生成 `quality_detail.jsonl`
>
> 所有过滤阈值在此脚本配置，调整后只需重跑本脚本，无需重新扫描。

```bash
python 03_quality_report.py

# 自定义过滤阈值
python 03_quality_report.py --min-dur 0.5 --max-dur 20
python 03_quality_report.py --rms-threshold 100 --snr-threshold 10
python 03_quality_report.py --clipping-ratio 0.005 --dc-offset 300
```

阈值参数（括号内为默认值）：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--min-dur` | 最短时长秒 | 0.3 |
| `--max-dur` | 最长时长秒 | 30.0 |
| `--rms-threshold` | 静音 RMS 阈值 | 50 |
| `--clipping-ratio` | 削波样本占比阈值 | 0.01 |
| `--dc-offset` | 直流偏移绝对值阈值 | 500 |
| `--snr-threshold` | 最低 SNR 阈值 dB | 5.0 |

输出到 `s10_data_reflow/quality_check/`：

- `quality_report.txt` — 文本统计报告（时长/问题类型/采样率/RMS/SNR/DC偏移 分布）
- `quality_summary.json` — JSON 汇总
- `bad_files.txt` — 问题文件清单（路径 `\t` 问题类型）
- `good_files.txt` — 合格文件清单

### 04\_sort\_bad\_files

按问题类型分类拷贝到 `quality_check/bad_sorted/`。

```bash
python 04_sort_bad_files.py
```

输出到 `s10_data_reflow/quality_check/bad_sorted/`：

- `too_short/` — 时长过短
- `silence_or_low_energy/` — 静音/低能量
- `clipping/` — 削波
- `...` — 其他问题类型子目录

### 05\_remove\_bad\_files

```bash
# 先预览，看看会删什么
python 05_remove_bad_files.py --dry-run

# 确认无误后执行
python 05_remove_bad_files.py
```

操作：

- 删除 `labeled/wav/` 和 `unlabeled/` 中的问题音频
- 清理 `labeled/text.txt`、`wav.scp`、`train.jsonl` 中对应条目

### run\_pipeline

按顺序执行数据处理各阶段脚本（6 步流水线）：

| 步骤 | 脚本 | 说明 |
|------|------|------|
| 1 | `01_prepare_funasr_data.py` | 整理数据 |
| 2 | `02_audio_quality_check.py` | 音频质量扫描 |
| 3 | `03_quality_report.py` | 生成质量报告 |
| 4 | `04_sort_bad_files.py` | 分类问题音频 |
| 5 | `05_remove_bad_files.py` | 删除问题音频（可选，需 `--remove-bad`） |
| 6 | `FireRedASR_pretag.py` | FireRedASR 预标注 |

```bash
python run_pipeline.py                    # 执行 1-4, 6（跳过删除）
python run_pipeline.py --remove-bad       # 包含删除步骤
python run_pipeline.py --start 3 --end 5  # 只执行第 3~5 步
python run_pipeline.py --skip 4           # 跳过第 4 步
python run_pipeline.py -y                 # 跳过确认提示
python run_pipeline.py --continue-on-error
python run_pipeline.py --dry-run          # 预览
python run_pipeline.py --list             # 列出所有步骤
```

### FunASR\_eval\_labeled

> 依赖：`pip install funasr torch torchaudio`

```bash
python 06_pretag_and_compare/FunASR_eval_labeled.py
python 06_pretag_and_compare/FunASR_eval_labeled.py --sample 1000
python 06_pretag_and_compare/FunASR_eval_labeled.py --model sensevoice
python 06_pretag_and_compare/FunASR_eval_labeled.py --device cpu
python 06_pretag_and_compare/FunASR_eval_labeled.py --resume
```

输出到 `s10_data_reflow/eval_output/`：

- `eval_detail.jsonl` — 逐条结果（uid / ref / hyp / CER / sub / ins / del）
- `eval_report.txt` — 可读报告（CER / SER / 分布 / bad cases）
- `eval_summary.json` — JSON 汇总

### FunASR\_pretag\_unlabeled

> 依赖：`pip install funasr torch torchaudio`

```bash
python 06_pretag_and_compare/FunASR_pretag_unlabeled.py
python 06_pretag_and_compare/FunASR_pretag_unlabeled.py --model sensevoice
python 06_pretag_and_compare/FunASR_pretag_unlabeled.py --device cpu
python 06_pretag_and_compare/FunASR_pretag_unlabeled.py --resume
```

输出到 `s10_data_reflow/pretag_output/`：

- `pretag.jsonl` — FunASR ChatML 格式
- `wav.scp` — Kaldi 格式音频路径
- `text.txt` — Kaldi 格式文本标注

### FireRedASR\_pretag

> 依赖：`pip install -r requirements.txt`（FireRedASR 仓库）
>
> 前置：`git clone` FireRedASR + 下载模型 + 链接 Qwen2-7B-Instruct
>
> 共享模块：`pretag_utils.py`（日志/扫描/写入/GPU 自动选择）

```bash
python 06_pretag_and_compare/FireRedASR_pretag.py --wav-dir /path/to/wav
python 06_pretag_and_compare/FireRedASR_pretag.py --asr-type aed --batch-size 8
python 06_pretag_and_compare/FireRedASR_pretag.py --resume
python 06_pretag_and_compare/FireRedASR_pretag.py --no-cache
python 06_pretag_and_compare/FireRedASR_pretag.py --max-slow-batches 3
python 06_pretag_and_compare/FireRedASR_pretag.py --max-slow-batches 0   # 禁用速度监控
```

输出到 `s10_data_reflow/firered_pretag_output/`：

- `pretag.jsonl` — FunASR ChatML 格式
- `wav.scp` — Kaldi 格式音频路径
- `text.txt` — Kaldi 格式文本标注
- `progress.txt` — 断点续跑进度

显存参考：

| 模式 | 显存 | batch size | 最大音频时长 |
|------|------|------------|-------------|
| LLM | ~16 GB | 1–4 | 30s |
| AED | ~8 GB | 8–16 | 60s |

### QwenASR\_pretag

> 依赖：`pip install -U qwen-asr`（或 `qwen-asr[vllm]`）；可选 `pip install -U flash-attn --no-build-isolation`（FlashAttention 2 加速）
>
> 共享模块：`pretag_utils.py`（日志/扫描/写入/GPU 自动选择）
>
> 模型：Qwen3-ASR 旗舰模型（1.7B），支持 52 种语言和 22 种中文方言

使用 Qwen3-ASR 对音频数据进行预标注，生成 jsonl / wav.scp / text.txt。支持 transformers、vLLM、raw 三种推理后端，支持断点续跑、自动速度监控与模型重载。

默认使用本地模型路径 `/data1/pretrained_models/Qwen3-ASR-1.7B`，也支持传入 HuggingFace ID（如 `Qwen/Qwen3-ASR-1.7B`）自动下载。

```bash
python 06_pretag_and_compare/QwenASR_pretag.py --wav-dir /path/to/wav
python 06_pretag_and_compare/QwenASR_pretag.py --model Qwen/Qwen3-ASR-0.6B
python 06_pretag_and_compare/QwenASR_pretag.py --backend vllm
python 06_pretag_and_compare/QwenASR_pretag.py --backend raw    # 仅需 transformers
python 06_pretag_and_compare/QwenASR_pretag.py --batch-size 64
python 06_pretag_and_compare/QwenASR_pretag.py --max-new-tokens 512
python 06_pretag_and_compare/QwenASR_pretag.py --language zh    # 强制指定语言
python 06_pretag_and_compare/QwenASR_pretag.py --resume --no-cache
python 06_pretag_and_compare/QwenASR_pretag.py --max-slow-batches 9
python 06_pretag_and_compare/QwenASR_pretag.py --max-slow-batches 0    # 禁用速度监控
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--wav-dir` | wav 文件目录（递归扫描） | `wavs/` |
| `--output-dir` | 输出目录 | `qwen_pretag_output/` |
| `--model` | 模型路径或 HuggingFace ID（支持本地路径如 `/data1/pretrained_models/Qwen3-ASR-1.7B`，也支持 HF ID 如 `Qwen/Qwen3-ASR-1.7B`） | `/data1/pretrained_models/Qwen3-ASR-1.7B` |
| `--backend` | 推理后端：`transformers` / `vllm` / `raw` | `transformers` |
| `--batch-size` | 批大小 | 128 |
| `--max-new-tokens` | 最大生成 token 数 | 256 |
| `--language` | 强制语言（不指定则自动检测） | `None` |
| `--device` | 设备（支持 `cuda:auto` / `cuda:auto:N` / 指定卡号 / `cpu`） | `cuda:auto` |
| `--resume` | 断点续跑（跳过已完成的 uid） | — |
| `--no-cache` | 忽略 wav 扫描缓存 | — |
| `--max-slow-batches` | 连续慢批次达此值触发自动重启（0 禁用） | 9 |

输出到 `qwen_pretag_output/`（或 `--output-dir` 指定的目录）：

- `pretag.jsonl` — FunASR ChatML 格式
- `wav.scp` — Kaldi 格式音频路径
- `text.txt` — Kaldi 格式文本标注
- `progress.txt` — 断点续跑进度
- `qwen_pretag_*.log` — 运行日志

扫描缓存（存放在 `--wav-dir` 目录下）：

- `wav_scan_cache_recursive_all.json` — 全量扫描缓存（无 `--filter` 时）
- `wav_scan_cache_recursive_labelled.json` — `--filter labelled` 扫描缓存
- `wav_scan_cache_recursive_unlabelled.json` — `--filter unlabelled` 扫描缓存

> 缓存为空时自动触发重新扫描，避免因异常产生的空缓存阻塞后续运行。使用 `--no-cache` 可强制忽略缓存。

后端对比：

| 后端 | 说明 | 依赖 |
|------|------|------|
| `transformers` | 默认，使用 qwen-asr 包封装 | `qwen-asr` |
| `vllm` | 高吞吐加速，适合大批量 | `qwen-asr[vllm]` |
| `raw` | 直接用 transformers + librosa，逐条推理 | `transformers`（不依赖 `qwen-asr`） |

GPU 自动选择（`pretag_utils.select_gpus`）：

两个预标注脚本均可通过 `--device` 参数使用智能 GPU 选择功能：

| 格式 | 说明 |
|------|------|
| `cuda:0` | 指定单卡 |
| `cuda:0,1,2` | 指定多卡（自动设置 `CUDA_VISIBLE_DEVICES`） |
| `cuda:auto` | 自动选最空闲的 1 张卡 |
| `cuda:auto:N` | 自动选最空闲的 N 张卡（不足则用全部） |
| `cpu` | CPU |

优先使用 `nvidia-smi` 查询空闲显存，失败时回退到 `torch.cuda.mem_get_info`。

### pretag\_utils（公共工具模块）

> 依赖：`torch`（GPU 相关功能）；其余为 Python 标准库
>
> 路径：`06_pretag_and_compare/pretag_utils.py`

预标注脚本的公共工具模块，被 `FireRedASR_pretag.py`、`QwenASR_pretag.py`、`FunASR_pretag_unlabeled.py` 等脚本共享。不直接运行，而是作为库导入使用。

提供的核心功能：

| 组件 | 说明 |
|------|------|
| `setup_logging()` | 统一日志配置（控制台 + 文件双输出） |
| `scan_wav_files()` | 多线程 wav 扫描（直接读 wav 头获取时长，支持 JSON 缓存加速二次扫描） |
| `prepare_pretag()` | 公共的扫描 + 断点续跑逻辑，返回待处理列表 |
| `PretagWriter` | 封装 jsonl / wav.scp / text.txt / progress.txt 四文件同步写入，内置进度日志（速度、ETA） |
| `run_inference_loop()` | 公共推理主循环，封装 writer 管理、重启逻辑、额外项处理、汇总输出 |
| `SpeedMonitor` | 自适应速度监控器，自动建立基线并检测性能退化，连续慢批次达阈值触发重启（`max_slow_batches <= 0` 时禁用监控） |
| `select_gpus()` | 智能 GPU 选择（`cuda:auto` / `cuda:auto:N` / 指定单卡多卡 / CPU） |

`run_inference_loop` 重启流程：

1. 保存当前进度（processed / failed 计数）
2. 关闭 writer，调用 `reload_fn()` 释放并重新加载模型
3. 重新打开 writer（append 模式），继承已处理计数，设置 `_base_processed` 使速度/ETA 只计算重启后的增量
4. 处理剩余项 + 额外项（如 FireRedASR 的超长音频）

输出文件（由调用方指定目录）：

- `pretag.jsonl` — FunASR ChatML 格式训练数据
- `wav.scp` — Kaldi 格式音频路径
- `text.txt` — Kaldi 格式文本标注
- `progress.txt` — 已处理 uid 列表（断点续跑依据）
- `wav_scan_cache.json` — wav 扫描缓存（存放在 wav 目录下）

### 07\_SeACo\_Paraformer\_eval

> 依赖：`pip install funasr torch torchaudio`
>
> 模型：`speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`（离线）
>
> 支持热词（hotword）增强、多目录独立评估

```bash
python 07_SeACo_Paraformer_eval.py
python 07_SeACo_Paraformer_eval.py --wav-dir /path/to/wav --label /path/to/text.txt
python 07_SeACo_Paraformer_eval.py --wav-dir /path/to/dir1 /path/to/dir2 /path/to/dir3
python 07_SeACo_Paraformer_eval.py --model-dir /path/to/seaco_model
python 07_SeACo_Paraformer_eval.py --hotword hotwords.txt
python 07_SeACo_Paraformer_eval.py --hotword ""              # 不用热词
python 07_SeACo_Paraformer_eval.py --device cpu
python 07_SeACo_Paraformer_eval.py --batch-size 500 --file-batch 1000
python 07_SeACo_Paraformer_eval.py --sample 1000
python 07_SeACo_Paraformer_eval.py --resume
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-dir` | 离线模型目录 | 脚本内置路径 |
| `--wav-dir` | wav 文件目录，可指定多个 | `labeled/wav` |
| `--label` | 标签文件路径（text.txt 或 jsonl） | `labeled/text.txt` |
| `--output-dir` | 输出目录 | `seaco_eval_output` |
| `--hotword` | 热词文件路径，传空字符串禁用 | `hotwords/hotwords.txt` |
| `--device` | 推理设备 | `cuda:0` |
| `--batch-size` | 动态 batch 总时长秒 | 300 |
| `--file-batch` | 每批文件数 | 500 |
| `--sample` | 随机采样 N 条评估 | 全部 |
| `--resume` | 断点续跑 | — |

标签格式：`text.txt`（`uid 文本` 两列或 `uid spk_id 文本` 三列，tab/空格分隔）或 jsonl（训练格式）

多目录模式：指定多个 `--wav-dir` 时，每个目录独立评估并在 output-dir 下生成子目录报告，最后汇总所有目录的性能到 `combined_report.txt` / `combined_summary.json`。

输出到 `s10_data_reflow/seaco_eval_output/`：

- `eval_detail.jsonl` — 逐条结果（uid / ref / hyp / CER / sub / ins / del）
- `eval_report.txt` — 可读报告（CER / SER / 分布 / bad cases）
- `eval_summary.json` — JSON 汇总
- `progress.txt` — 断点续跑进度
- `combined_report.txt` — 多目录汇总报告（多目录模式）
- `combined_summary.json` — 多目录汇总 JSON（多目录模式）

### 10\_eval/seaco\_eval（SeACo-Paraformer 本地模型评估）

> 依赖：`pip install funasr torch torchaudio`
>
> 模型：`speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`（离线本地模型）
>
> 共享模块：`pretag_utils.py`（GPU 自动选择、wav 时长读取）

使用本地 SeACo-Paraformer 模型对指定数据集进行 ASR 识别并计算 CER/SER 等评估指标。支持多种标签格式（text.txt / jsonl / manifest jsonl / ChatML 训练格式）、热词增强、多目录独立评估+汇总报告、随机采样、断点续跑、自动 GPU 选择。自动过滤时长过短的音频（默认 < 0.1s），避免特征提取窗口溢出导致推理失败。

```bash
# 基本用法
python 10_eval/seaco_eval.py --wav-dir /path/to/wav --label /path/to/text.txt

# 使用 jsonl 标签
python 10_eval/seaco_eval.py --wav-dir /path/to/wav --label /path/to/label.jsonl

# 多目录评估
python 10_eval/seaco_eval.py --wav-dir /path/to/dir1 /path/to/dir2 /path/to/dir3 --label /path/to/text.txt

# 使用热词
python 10_eval/seaco_eval.py --wav-dir /path/to/wav --label /path/to/text.txt --hotword /path/to/hotwords.txt

# 随机采样 1000 条
python 10_eval/seaco_eval.py --wav-dir /path/to/wav --label /path/to/text.txt --sample 1000

# 断点续跑
python 10_eval/seaco_eval.py --wav-dir /path/to/wav --label /path/to/text.txt --resume

# 仅识别（不计算 CER）
python 10_eval/seaco_eval.py --wav-dir /path/to/wav

# 指定 GPU
python 10_eval/seaco_eval.py --wav-dir /path/to/wav --label /path/to/text.txt --device cuda:0
python 10_eval/seaco_eval.py --wav-dir /path/to/wav --label /path/to/text.txt --device cuda:auto
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-dir` | 本地模型目录 | 脚本内置路径 |
| `--wav-dir` | wav 文件目录（必填），可指定多个目录分别评估并汇总 | — |
| `--label` | 标签文件路径（text.txt 或 jsonl），不指定则仅识别不计算 CER | `None` |
| `--output-dir` | 输出目录 | `10_eval/eval_output` |
| `--hotword` | 热词文件路径（每行一个热词） | `None` |
| `--device` | 推理设备（支持 `cuda:auto` 自动选卡） | `cuda:auto` |
| `--batch-size` | 每次推理的样本数，值越大显存占用越高 | 64 |
| `--file-batch` | 每批文件数（控制进度日志粒度） | 5000 |
| `--sample` | 随机采样 N 条评估 | 全部 |
| `--resume` | 断点续跑（跳过已完成的 uid） | — |

标签格式支持：

| 格式 | 说明 |
|------|------|
| `text.txt`（两列） | 每行 `uid 文本内容`（空格或 tab 分隔） |
| `text.txt`（三列） | 每行 `uid spk_id 文本内容`（tab 或空格分隔，自动识别第二列为 spk_id 并取第三列作为文本；uid 支持带扩展名的文件路径，自动提取 stem） |
| `jsonl` | 每行 `{"uid":"xxx","text":"你好"}` |
| manifest jsonl | 每行 `{"key":"xxx","txt":"你好"}`（manifest 格式） |
| ChatML jsonl | 含 `messages` 字段的训练格式，自动从中提取 uid 和文本 |

多目录模式：指定多个 `--wav-dir` 时，每个目录独立评估并在 output-dir 下生成子目录报告，最后汇总所有目录的性能到 `combined_report.txt` / `combined_summary.json`。

输出到 `10_eval/eval_output/`（或 `--output-dir` 指定的目录）：

- `eval_detail.jsonl` — 逐条结果（uid / ref / hyp / CER / sub / ins / del / correct）
- `eval_report.txt` — 可读报告（CER / SER / CER 分布直方图 / bad cases）
- `eval_summary.json` — JSON 汇总
- `progress.txt` — 断点续跑进度（已完成的 uid 列表）
- `eval_*.log` — 运行日志（带时间戳）
- `combined_report.txt` — 多目录汇总报告（多目录模式）
- `combined_summary.json` — 多目录汇总 JSON（多目录模式）

### 10\_eval/batch\_eval（公开数据集批量评估）

> 依赖：`pip install funasr torch torchaudio soundfile`
>
> 共享模块：`eval_utils.py`（日志、标签加载、`build_eval_items_from_scp`、`run_eval_items`、汇总报告、GPU 自动选择）
>
> 前置：先运行 `utils/public_dataset_gen_scp_txt.py` 生成标准目录结构的 `{dataset}_wav.scp` + `{dataset}_text`

基于 `public_dataset_gen_scp_txt.py` 生成的标准目录结构，自动发现所有可用数据集，批量运行 SeACo-Paraformer 基准模型和/或微调模型评估。支持多数据集 × 多 split × 多模型的组合评估，按模型分组加载（避免重复加载），逐数据集逐 split 评估后生成汇总报告。直接从 `wav.scp` 构建评估条目，无需创建临时符号链接目录。

目录结构约定：

```
{label_root}/{train,dev,test}/{dataset}_wav.scp
{label_root}/{train,dev,test}/{dataset}_text
```

```bash
# 默认评估所有数据集的 train+dev+test 集（基准+微调模型）
python 10_eval/batch_eval.py

# 仅评估基准模型
python 10_eval/batch_eval.py --mode seaco

# 仅评估微调模型
python 10_eval/batch_eval.py --mode finetuned

# 只评估指定数据集
python 10_eval/batch_eval.py --datasets aidatatang_200zh data_aishell

# 只评估 test 集
python 10_eval/batch_eval.py --splits test

# 采样评估
python 10_eval/batch_eval.py --sample 500

# 使用热词
python 10_eval/batch_eval.py --hotword /path/to/hotwords.txt

# 断点续跑
python 10_eval/batch_eval.py --resume
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 评估模式：`seaco`=基准模型，`finetuned`=微调模型，`both`=两者 | `both` |
| `--label-root` | 标签根目录（含 train/dev/test 子目录） | `/data1/label/augment_label/` |
| `--datasets` | 指定数据集名称（空格分隔），不指定则评估全部已发现的数据集 | 全部 |
| `--splits` | 评估哪些集合（空格分隔） | `train dev test` |
| `--seaco-model` | SeACo 基准模型目录 | 脚本内置路径 |
| `--finetune-base` | 微调模型的基准模型目录 | 脚本内置路径 |
| `--finetune-ckpt` | 微调权重路径（`.pt` 文件） | 脚本内置路径 |
| `--output-root` | 输出根目录 | `/data1/eval_output/base/augment_eval_result` |
| `--hotword` | 热词文件路径（每行一个热词） | `None` |
| `--device` | 推理设备（支持 `cuda:auto` 自动选卡） | `cuda:auto` |
| `--batch-size` | 每次推理的样本数 | 16 |
| `--file-batch` | 文件批大小（控制进度日志粒度） | 5000 |
| `--sample` | 随机采样 N 条评估 | 全部 |
| `--resume` | 断点续跑（跳过已完成的 uid） | — |

工作流程：

1. 扫描 `{label_root}/{split}/` 下的 `*_wav.scp` + `*_text` 文件对，自动发现数据集
2. 按模型分组（避免重复加载模型）：先加载一次模型，评估该模型下所有数据集×split 组合
3. 对每个评估任务：通过符号链接将 scp 中的 wav 组织到临时 staging 目录，复制 text 为 `trans.txt` 供 `eval_utils` 自动发现
4. 调用 `eval_utils.run_eval_single_dir()` 执行单目录评估
5. 为每个数据集生成数据集级汇总报告（`dataset_report.md`），对比该数据集各 split 的 CER/SER，多 split 时计算加权平均
6. 所有任务完成后调用 `eval_utils.generate_combined_report()` 生成全局汇总报告

微调模型加载：加载基准模型后，从 `--finetune-ckpt` 读取权重（自动识别 `model_state_dict` / `state_dict` / `model` 等常见 key），通过 `load_state_dict(strict=False)` 加载并记录缺失/多余的权重。

输出到 `{output_root}/{mode}/`（如 `eval_output/seaco/`）：

- `{dataset}/{split}/eval_detail.jsonl` — 逐条评估结果
- `{dataset}/{split}/eval_report.txt` — 单数据集单 split 可读报告
- `{dataset}/{split}/eval_summary.json` — 单数据集单 split JSON 汇总
- `{dataset}/dataset_report.md` — 数据集级汇总报告（对比各 split 的 CER/SER，多 split 时含加权平均）
- `combined_report.txt` — 全局多数据集汇总报告（CER/SER 对比表）
- `combined_summary.json` — 全局多数据集汇总 JSON
- `staging/{dataset}/{split}/wav/` — 临时符号链接目录（评估用）
- `batch_{mode}_*.log` — 运行日志

### 10\_eval/batch\_eval\_models（多模型权重批量评估）

> 依赖：`pip install funasr torch torchaudio soundfile`
>
> 共享模块：`eval_utils.py`（日志、标签加载、GPU 自动选择、单目录评估、汇总报告生成）
>
> 前置：模型目录下已有 `model.pt.*` 权重文件；`/data1/label` 下已有标签目录（`augment_label`、`reflow_label`、`speech_label`、`all_label` 等）

扫描指定模型目录下的所有权重文件（`model.pt.*`），对 `/data1/label` 下的数据集进行批量评估，并生成跨模型对比报告。支持指定标签源、数据集、split、采样评估、baseline 对比等。

目录结构约定：

```
模型目录:  /data1/eval_output/v0.1.0/model.pt.best, model.pt.ep16.4000, ...
标签目录:  /data1/label/{augment_label,reflow_label,speech_label,all_label}/{train,dev,test}/
```

```bash
# 评估模型目录下所有权重
python 10_eval/batch_eval_models.py --model-dir /data1/eval_output/v0.1.0

# 指定标签源和集合
python 10_eval/batch_eval_models.py --model-dir /data1/eval_output/v0.1.0 \
    --label-source all_label --splits dev test

# 只评估指定权重文件
python 10_eval/batch_eval_models.py --model-dir /data1/eval_output/v0.1.0 \
    --ckpt-pattern "model.pt.best"

# 指定数据集
python 10_eval/batch_eval_models.py --model-dir /data1/eval_output/v0.1.0 \
    --datasets aidatatang_200zh data_aishell

# 同时评估基准模型作为 baseline 对比
python 10_eval/batch_eval_models.py --model-dir /data1/eval_output/v0.1.0 --with-baseline

# 不与基线对比（传空字符串禁用）
python 10_eval/batch_eval_models.py --model-dir /data1/eval_output/v0.1.0 \
    --baseline-dir ""

# 指定其他基线目录
python 10_eval/batch_eval_models.py --model-dir /data1/eval_output/v0.1.0 \
    --baseline-dir /data1/eval_output/v0.0.9

# 采样评估（快速验证）
python 10_eval/batch_eval_models.py --model-dir /data1/eval_output/v0.1.0 --sample 500

# 使用 reflow_label
python 10_eval/batch_eval_models.py --model-dir /data1/eval_output/v0.1.0 \
    --label-source reflow_label

# 使用多个标签源
python 10_eval/batch_eval_models.py --model-dir /data1/eval_output/v0.1.0 \
    --label-source augment_label reflow_label speech_label

# 仅生成基线对比报告（不跑推理，从已有评估结果目录读取）
python 10_eval/batch_eval_models.py \
    --report-only /data1/eval_output/base/augment_eval_result/finetuned

# 指定基线目录生成对比报告
python 10_eval/batch_eval_models.py \
    --report-only /data1/eval_output/v0.1.0/model.pt.best \
    --baseline-dir /data1/eval_output/base
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-dir` | 模型权重目录，扫描其中的 `model.pt.*` 文件（正常模式必填，`--report-only` 模式不需要） | — |
| `--base-model` | 基准模型目录（SeACo-Paraformer） | 脚本内置路径 |
| `--ckpt-pattern` | 权重文件名过滤，逗号分隔多个模式（如 `"best,avg"`） | 全部 `model.pt.*` |
| `--with-baseline` | 同时评估基准模型作为 baseline 对比 | — |
| `--baseline-dir` | 基线评估结果目录，用于生成对比报告（传空字符串禁用） | `/data1/eval_output/base` |
| `--label-root` | 标签根目录 | `/data1/label` |
| `--label-source` | 标签源目录名（空格分隔多个），位于 `label-root` 下 | `augment_label reflow_label speech_label` |
| `--datasets` | 指定数据集名称（空格分隔），不指定则评估全部已发现的数据集 | 全部 |
| `--splits` | 评估哪些集合（空格分隔） | `train dev test` |
| `--output-root` | 输出根目录 | `/data1/eval_output` |
| `--hotword` | 热词文件路径（每行一个热词） | `None` |
| `--device` | 推理设备（支持 `cuda:auto` 自动选卡） | `cuda:auto` |
| `--batch-size` | 推理 batch 大小 | 8 |
| `--file-batch` | 文件批大小（控制进度日志粒度） | 5000 |
| `--sample` | 随机采样 N 条评估（快速验证） | 全部 |
| `--resume` | 断点续跑（跳过已完成的 uid） | — |
| `--report-only` | 仅生成基线对比报告模式：指定已有评估结果目录（含 `eval_summary.json`），不跑推理，直接从已有结果生成对比报告。需配合 `--baseline-dir` 使用 | `None` |

工作流程：

1. 扫描 `--model-dir` 下的 `model.pt.*` 权重文件（支持 `model.pt.best`、`model.pt.ep<N>.<step>`、`model.pt.avg<N>`、`model.pt`）
2. 从 `--label-source` 指定的标签源目录中发现可用数据集：合并后的 `all_label` 格式（`wav.scp` + `text`）或标准格式（`{dataset}_wav.scp` + `{dataset}_text`）
3. 逐模型加载：基准模型结构 + 微调权重（自动识别 `model_state_dict` / `state_dict` / `model` 等常见 checkpoint key），通过 `load_state_dict(strict=False)` 加载
4. 对每个评估任务：通过符号链接将 scp 中的 wav 组织到临时 staging 目录，复制 text 为 `trans.txt` 供 `eval_utils` 自动发现
5. 调用 `eval_utils.run_eval_single_dir()` 执行单目录评估
6. 每个模型评估完成后生成单模型汇总报告
7. 若指定了 `--baseline-dir` 且基线目录下有有效结果，为每个模型生成与多基线模式的对比报告（Markdown + JSON）。基线结果按模式（mode）分组加载（seaco / online\_model / finetuned），key 格式为 `{label_source}/{dataset}/{split}`（通过 eval\_type 目录名映射到 label\_source，如 `augment_eval_result` → `augment_label`），确保不同来源的同名数据集不冲突。报告按固定顺序（SeACo基准 → 线上模型 → 旧微调）展示各模式对比，包含：加权 CER/SER 总览（当前模型 vs 各基线模式及 delta）、逐数据集 CER/SER 对比（每列一个基线模式）、按基线模式分别统计的改善/退化汇总。通过多种策略自动匹配当前评估结果与基线（去 label\_source 前缀、直接匹配、末两段匹配）
8. 所有模型评估完成后生成跨模型对比报告（Markdown + JSON），包含模型总览表格、按数据集 CER/SER 对比、最佳模型推荐

`--report-only` 模式工作流程：

1. 从指定的已有评估结果目录中递归查找 `eval_summary.json` 文件（支持 `{dataset}/{split}/eval_summary.json` 和 `{split}/eval_summary.json` 两种目录结构）
2. 加载 `--baseline-dir` 下的基线评估结果
3. 直接生成基线对比报告（`baseline_comparison_report.md` + `baseline_comparison_summary.json`），输出到评估结果目录下
4. 不加载模型、不跑推理，适合评估完成后补生成或更新对比报告

输出到 `{output_root}/`（默认 `/data1/eval_output/`）：

- `{ckpt_name}/{dataset}/{split}/eval_detail.jsonl` — 逐条评估结果
- `{ckpt_name}/{dataset}/{split}/eval_report.txt` — 单数据集单 split 可读报告
- `{ckpt_name}/{dataset}/{split}/eval_summary.json` — 单数据集单 split JSON 汇总
- `{ckpt_name}/combined_report.txt` — 单模型多数据集汇总报告
- `{ckpt_name}/combined_summary.json` — 单模型多数据集汇总 JSON
- `{ckpt_name}/baseline_comparison_report.md` — 单模型与多基线模式对比报告（指定 `--baseline-dir` 时生成，按 seaco / online\_model / finetuned 分别对比，含加权 CER/SER 总览、逐数据集 CER/SER 对比、按模式分别统计的改善/退化汇总）
- `{ckpt_name}/baseline_comparison_summary.json` — 单模型与多基线模式对比 JSON 汇总（含 baseline\_modes 列表、per\_dataset 逐数据集指标及各模式 cer\_delta/ser\_delta）
- `{ckpt_name}/staging/` — 临时符号链接目录（评估用）
- `{ckpt_name}/eval_{ckpt_name}_*.log` — 运行日志
- `model_comparison_report.md` — 跨模型对比报告（多模型时生成，含总览表格、按数据集 CER/SER 对比、最佳模型推荐）
- `model_comparison_summary.json` — 跨模型对比 JSON 汇总

### 10\_eval/gen\_baseline\_report（基线对比报告独立生成）

> 依赖：仅 Python 标准库（`json`, `pathlib`, `os`, `argparse`, `datetime`）
>
> 路径：`10_eval/gen_baseline_report.py`
>
> 前置：已有评估结果目录（含 `eval_summary.json`）和基线评估结果目录

从已有的评估结果目录中读取 `eval_summary.json`，与基线结果对比，生成 `baseline_comparison_report.md`。不需要重新跑模型推理，只需要已有的评估结果。适合评估完成后独立生成或更新对比报告。

目录结构约定：

```
新模型评估结果:  /data1/eval_output/v0.1.0/<ckpt_name>/<label_source>/<dataset>/<split>/eval_summary.json
基线评估结果:    /data1/eval_output/base/{augment,reflow,speech}_eval_result/{seaco,finetuned,online_model}/<dataset>/<split>/eval_summary.json
```

```bash
# 为指定评估结果目录生成基线对比报告
python 10_eval/gen_baseline_report.py --eval-dir /data1/eval_output/v0.1.0/model.pt.best

# 为整个版本目录下所有模型生成对比报告
python 10_eval/gen_baseline_report.py --eval-dir /data1/eval_output/v0.1.0

# 指定基线目录
python 10_eval/gen_baseline_report.py --eval-dir /data1/eval_output/v0.1.0 \
    --baseline-dir /data1/eval_output/base

# 只对比指定基线模式
python 10_eval/gen_baseline_report.py --eval-dir /data1/eval_output/v0.1.0 \
    --baseline-modes seaco online_model

# 指定输出文件名
python 10_eval/gen_baseline_report.py --eval-dir /data1/eval_output/v0.1.0/model.pt.best \
    --output baseline_comparison_report.md
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--eval-dir` | 模型评估结果目录（单个模型或包含多个模型的父目录），必填 | — |
| `--baseline-dir` | 基线评估结果目录 | `/data1/eval_output/base` |
| `--baseline-modes` | 只对比指定基线模式（空格分隔，如 `seaco online_model finetuned`） | 全部可用模式 |
| `--output` | 输出报告文件名 | `baseline_comparison_report.md` |

工作流程：

1. 加载基线评估结果：从 `--baseline-dir` 下按 `{eval_type}/{mode}/{dataset}/{split}/eval_summary.json` 结构递归加载，按模式（seaco / online\_model / finetuned）分组。`eval_type` 目录名自动映射为对应的 `label_source`（`augment_eval_result` → `augment_label`、`reflow_eval_result` → `reflow_label`、`speech_eval_result` → `speech_label`），基线 key 格式为 `{label_source}/{dataset}/{split}`，确保不同来源的同名数据集不冲突
2. 自动判断 `--eval-dir` 是单个模型目录还是包含多个模型的父目录：子目录名像模型权重名（`model.pt.*`）且内部有 `eval_summary.json` 则认为是父目录；否则检查 `eval_dir` 本身是否有评估结果；无评估结果则返回空列表
3. 逐模型从评估结果目录递归加载所有 `eval_summary.json`（跳过 `staging` 目录）
4. 通过多种策略自动匹配当前评估结果与基线（去 label\_source 前缀、直接匹配、末两段匹配）
5. 生成对比报告：加权 CER/SER 总览、逐数据集 CER/SER 对比、按基线模式分别统计的改善/退化汇总

输出到评估结果目录下（每个模型目录内）：

- `baseline_comparison_report.md` — 与多基线模式对比的 Markdown 报告（含加权 CER/SER 总览、逐数据集 CER/SER 对比表、改善/退化汇总）
- `baseline_comparison_report_summary.json` — 对比 JSON 汇总（含 overall 各基线模式指标及 delta、per\_dataset 逐数据集详细指标）

### 10\_eval/eval\_on\_testset（12\_test 测试集多模型权重批量评估）

> 依赖：`pip install funasr torch torchaudio soundfile`
>
> 路径：`10_eval/eval_on_testset.py`
>
> 共享模块：`eval_utils.py`（日志、标签加载、GPU 自动选择、`build_eval_items_from_scp`、`run_eval_items`、汇总报告生成）
>
> 前置：测试集目录下已有 `wav.scp` 和 `text`（如 `12_test/gen_test_text.py` 或 `ChildMandarin/prepare_for_eval.py` 生成）；模型目录下已有 `model.pt.*` 权重文件

在指定测试集上批量评估多个模型权重的 CER/SER，最终生成跨模型对比报告。与 `batch_eval_models.py` 的区别：本脚本专注于单一测试集的快速评估对比，无需配置标签源/split 等复杂参数，适合日常训练后快速验证模型效果。`--test-scp` 和 `--test-text` 为必填参数，可灵活指定任意测试集（如 12\_test、ChildMandarin、SP4 等）。

支持两种目录结构：

1. **单目录模式**：`--model-dir` 指向包含 `model.pt.*` 的目录，直接扫描该目录下的权重文件
2. **多版本模式**：`--model-dir` 指向包含多个版本子目录的父目录，自动扫描各子目录下的权重文件

多版本目录结构示例：

```
/data1/eval_output/
├── v0.1.0/model/model.pt.best, model.pt.ep16.4000
├── v0.2.0/model/model.pt.best, model.pt.ep14.4000
└── v0.3.0/model/model.pt.best, model.pt.ep27.4000
```

```bash
# 多版本模式（自动扫描子目录）
python 10_eval/eval_on_testset.py --test-scp /data1/12_test/wav.scp --test-text /data1/12_test/text --model-dir /data1/eval_output

# 单目录模式
python 10_eval/eval_on_testset.py --test-scp /data1/12_test/wav.scp --test-text /data1/12_test/text --model-dir /data1/eval_output/v0.1.0/model

# 只评估 best 权重
python 10_eval/eval_on_testset.py --test-scp /data1/12_test/wav.scp --test-text /data1/12_test/text --ckpt-pattern "best"

# 只评估指定版本
python 10_eval/eval_on_testset.py --test-scp /data1/12_test/wav.scp --test-text /data1/12_test/text --versions v0.2.0 v0.3.0

# 采样快速验证
python 10_eval/eval_on_testset.py --test-scp /data1/12_test/wav.scp --test-text /data1/12_test/text --sample 100

# 同时评估基准模型（不加载微调权重）
python 10_eval/eval_on_testset.py --test-scp /data1/12_test/wav.scp --test-text /data1/12_test/text --with-baseline

# 评估儿童语音数据集
python 10_eval/eval_on_testset.py --test-scp /data1/child_dataset/ChildMandarin/wav.scp --test-text /data1/child_dataset/ChildMandarin/text --output-dir /data1/eval_output/child/ChildMandarin

# 使用热词
python 10_eval/eval_on_testset.py --test-scp /data1/12_test/wav.scp --test-text /data1/12_test/text --hotword /path/to/hotwords.txt

# 断点续跑
python 10_eval/eval_on_testset.py --test-scp /data1/12_test/wav.scp --test-text /data1/12_test/text --resume
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-dir` | 模型权重目录（单目录模式指向含 `model.pt.*` 的目录；多版本模式指向含版本子目录的父目录） | `/data1/eval_output` |
| `--base-model` | 基准模型目录（SeACo-Paraformer） | 脚本内置路径 |
| `--test-scp` | 测试集 `wav.scp` 路径（**必填**） | — |
| `--test-text` | 测试集 `text` 路径（**必填**） | — |
| `--output-dir` | 输出目录 | `/data1/eval_output/testset_results` |
| `--ckpt-pattern` | 权重文件名过滤，逗号分隔多个关键词（如 `"best,avg5"`） | 全部 `model.pt.*` |
| `--versions` | 指定版本目录名（空格分隔），不指定则评估全部版本（仅多版本模式有效） | 全部 |
| `--with-baseline` | 同时评估基准模型（不加载微调权重）作为对比 | — |
| `--hotword` | 热词文件路径（每行一个热词） | `None` |
| `--device` | 推理设备（支持 `cuda:auto` 自动选卡） | `cuda:auto` |
| `--batch-size` | 推理 batch 大小 | 8 |
| `--file-batch` | 文件批大小（控制进度日志粒度） | 5000 |
| `--sample` | 随机采样 N 条（快速验证用） | 全部 |
| `--resume` | 断点续跑（跳过已完成的 uid） | — |

工作流程：

1. 检查测试集文件（`wav.scp` + `text`）是否存在
2. 判断目录模式：若 `--model-dir` 下直接有 `model.pt.*` 文件则为单目录模式，否则扫描子目录（多版本模式）
3. 扫描权重文件（支持 `model.pt.best`、`model.pt.ep<N>`、`model.pt.avg<N>`、`model.pt`），可通过 `--ckpt-pattern` 和 `--versions` 过滤
4. 加载测试集标签，构建 eval\_items（只需构建一次，所有模型共用）
5. 逐模型加载并评估：基准模型结构 + 微调权重（自动识别 `model_state_dict` / `state_dict` / `model` 等常见 checkpoint key），评估完释放显存
6. 生成跨模型对比报告（Markdown + JSON），包含 CER/SER 对比表、CER 分布对比、最佳模型推荐

输出到 `10_eval/testset_results/`（或 `--output-dir` 指定的目录）：

- `{ckpt_name}/eval_detail.jsonl` — 逐条评估结果（uid / ref / hyp / CER / sub / ins / del / correct）
- `{ckpt_name}/eval_report.txt` — 单模型可读报告
- `{ckpt_name}/eval_summary.json` — 单模型 JSON 汇总
- `comparison_report.md` — 跨模型对比 Markdown 报告（CER/SER 对比表、CER 分布对比、最佳模型推荐）
- `comparison_summary.json` — 跨模型对比 JSON 汇总
- `testset_eval_*.log` — 运行日志

### 10\_eval/eval\_utils（ASR 评估公用工具模块）

> 依赖：`soundfile`（音频时长读取）；其余为 Python 标准库
>
> 路径：`10_eval/eval_utils.py`
>
> 共享模块：`pretag_utils.py`（GPU 自动选择）

ASR 评估脚本的公共工具模块，被 `seaco_eval.py`、`finetuned_eval.py` 等评估脚本共享。不直接运行，而是作为库导入使用。

```python
from eval_utils import (
    setup_logging, format_duration, get_wav_duration, select_gpus,
    compute_cer, normalize_text, load_labels, load_hotwords, load_progress,
    collect_wav_files, auto_discover_labels, get_dir_display_name,
    build_eval_items_from_scp, run_eval_items,
    run_eval_single_dir, generate_report, generate_combined_report,
)
```

提供的核心功能：

| 组件 | 说明 |
|------|------|
| `get_wav_duration()` | 获取音频时长（秒），依次尝试 soundfile 和标准库 wave 模块，兼容多种 wav 格式；全部失败返回 -1.0，并在首次失败时打印具体错误原因（soundfile 未安装/读取失败、wave 读取失败等），后续失败不再重复打印以避免日志刷屏 |
| `setup_logging()` | 统一日志配置（控制台 + 文件双输出），支持自定义 logger 名称，日志文件带时间戳 |
| `format_duration()` | 将秒数格式化为 `H:MM:SS` 时长字符串 |
| `select_gpus()` | 智能 GPU 选择（从 `pretag_utils` 导入，支持 `cuda:auto` / `cuda:auto:N` / 指定单卡多卡 / CPU） |
| `edit_distance()` | 编辑距离计算，返回 (substitutions, insertions, deletions) |
| `compute_cer()` | 计算字符错误率 CER，返回 (cer, sub, ins, del, ref_len) |
| `normalize_text()` | 文本规范化：去除中英文标点、转小写、去空格 |
| `load_labels()` | 加载标签文件，支持 text.txt（两列 `uid text` 或三列 `uid spk_id text`，tab/空格分隔，uid 自动提取 stem）/ jsonl / manifest jsonl / ChatML 训练格式，返回 `{uid: text}` |
| `load_hotwords()` | 加载热词文件，返回空格分隔的热词字符串 |
| `load_progress()` | 加载断点续跑进度文件，返回已完成的 uid 集合 |
| `collect_wav_files()` | 递归收集 wav 文件并匹配标签，过滤过短音频，返回 `[(uid, ref_text, wav_path), ...]` |
| `build_eval_items_from_scp()` | 从 `wav.scp` + labels dict 直接构建 `eval_items`（无需把 wav 拷贝/链接到同一目录），过滤缺失和过短音频，返回 `[(uid, ref_text, wav_path), ...]` |
| `auto_discover_labels()` | 自动在 wav 目录同级/父级查找 trans.txt / text.txt 标签文件 |
| `get_dir_display_name()` | 获取目录显示名称（wav 目录自动用父目录名） |
| `run_eval_items()` | 核心评估循环（与数据来源解耦）：对一组 `(uid, ref, wav_path)` 执行采样 → 断点续跑 → 批量推理 → CER 计算 → 生成报告，返回 summary dict；配合 `collect_wav_files()`（目录扫描）或 `build_eval_items_from_scp()`（scp 文件）使用 |
| `run_eval_single_dir()` | 向后兼容包装：内部递归扫描 `wav_dir` 构建 eval_items 后调用 `run_eval_items()`；如果数据已是 wav.scp 格式，建议直接用 `build_eval_items_from_scp` + `run_eval_items` 跳过 staging 目录 |
| `generate_report()` | 从 `eval_detail.jsonl` 生成单目录评估报告（`eval_report.txt` + `eval_summary.json`），支持自定义报告标题 |
| `generate_combined_report()` | 汇总多个目录的评估结果，生成多目录汇总报告（`combined_report.txt` + `combined_summary.json`），支持自定义报告标题 |

两种构建 `eval_items` 的方式：

| 方式 | 适用场景 | 调用 |
|------|----------|------|
| 目录扫描 | wav 已组织在同一目录下，可能伴随 `trans.txt` | `collect_wav_files(wav_dir, labels, logger)` |
| scp 直读 | 已有 `wav.scp` 文件，wav 分散在不同路径，不想拷贝/软链 | `build_eval_items_from_scp(scp_path, labels, logger)` |

`generate_report()` 报告内容：整体 CER/SER 指标、替换/插入/删除统计、CER 分布直方图（6 个区间）、Bad Cases 列表（CER > 30% 的前 30 条）。

`generate_combined_report()` 报告内容：各目录性能对比表（数量/CER/SER/正确率/字符数）、汇总整体指标、汇总 CER 分布。

输出文件（由调用方指定目录）：

- `eval_{timestamp}.log` — 运行日志（`setup_logging` 自动创建，带时间戳命名）
- `eval_detail.jsonl` — 逐条评估结果（`run_eval_items` / `run_eval_single_dir` 生成）
- `eval_report.txt` — 单目录可读评估报告（`generate_report` 生成）
- `eval_summary.json` — 单目录 JSON 汇总（`generate_report` 生成）
- `combined_report.txt` — 多目录汇总报告（`generate_combined_report` 生成）
- `combined_summary.json` — 多目录汇总 JSON（`generate_combined_report` 生成）
- `progress.txt` — 断点续跑进度（`run_eval_items` / `run_eval_single_dir` 生成）

### 10\_eval/plot\_loss\_tb（TensorBoard Loss 曲线绘制）

> 依赖：`pip install tensorboard matplotlib`
>
> 路径：`10_eval/plot_loss_tb.py`

从 TensorBoard 事件文件中提取 train / val 的 loss 数据，绘制在一张图上。支持多目录合并（中断恢复训练场景）、自动扫描包含 `tfevents` 的子目录、按 `loss_key` 选择不同 loss 指标（含任意自定义 tag）、epoch 级别和 step 级别双视图、自动标注最佳 val loss epoch。

```bash
# 指定 tensorboard 目录
python 10_eval/plot_loss_tb.py --tb_dir /data1/eval_output/v0.6.0/tensorboard

# 多个 tensorboard 目录（中断恢复场景，自动按 step 合并）
python 10_eval/plot_loss_tb.py --tb_dir /path/tb1 --tb_dir /path/tb2

# 自动扫描父目录下所有包含 tfevents 的子目录
python 10_eval/plot_loss_tb.py --scan_dir /data1/eval_output/v0.6.0

# 只看 attention loss
python 10_eval/plot_loss_tb.py --tb_dir /path/tb --loss_key loss_att

# 额外绘制逐 step 细粒度曲线
python 10_eval/plot_loss_tb.py --tb_dir /path/tb --show_step

# 手动指定 steps_per_epoch（如果自动推测不准）
python 10_eval/plot_loss_tb.py --tb_dir /path/tb --steps_per_epoch 5500

# 指定输出路径
python 10_eval/plot_loss_tb.py --tb_dir /path/tb --save /data1/loss_tb.png

# 列出所有可用的 scalar tags（不绘图）
python 10_eval/plot_loss_tb.py --tb_dir /path/tb --list_tags
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--tb_dir` | TensorBoard 事件文件目录，可多次指定（按顺序合并，后面的覆盖前面的同 step 数据） | — |
| `--scan_dir` | 自动递归扫描该目录下所有包含 `tfevents` 的子目录 | — |
| `--save` | 输出图片路径 | `loss_tb.png`（当前目录） |
| `--loss_key` | 要绘制的 loss 类型，也可输入 TensorBoard 中的任意 tag 名 | `loss` |
| `--show_step` | 额外绘制逐 step 的细粒度曲线（双子图） | — |
| `--steps_per_epoch` | 每个 epoch 的 step 数（手动指定，优先级最高；不指定则自动推测） | `None`（自动） |
| `--list_tags` | 仅列出所有可用的 scalar tags 及数据点数量，不绘图 | — |

支持的 `loss_key`：

| 值 | 说明 |
|------|------|
| `loss` | 总 loss（默认） |
| `loss_att` | attention decoder loss |
| `loss_seaco` | seaco decoder loss |
| 任意 tag | TensorBoard 中的任意 scalar tag 名称 |

Tag 匹配策略（按优先级）：

1. 精确匹配 `train/{loss_key}` 和 `val/{loss_key}`（或 `train_{loss_key}` / `valid_{loss_key}`）
2. 模糊匹配：tag 同时包含 `loss_key` 和 `train` / `val`
3. 单一 tag：仅存在 `loss_key` 本身（无法区分 train/val）

Epoch 推测策略（当未指定 `--steps_per_epoch` 时）：

1. 从 val 数据点间隔推测 `validate_interval`（取间隔众数）
2. 检测 val 数据中非 `validate_interval` 整数倍的点作为 epoch 边界（FunASR 在 epoch 结束时额外做一次验证）
3. 若检测到 epoch 边界点，用其数量反推 `steps_per_epoch`
4. 若无法确定，回退为直接用 val 点序号作为 x 轴，并提示用户手动指定

输出文件：

- `loss_tb.png`（或 `--save` 指定的路径）— Loss 曲线图（PNG，150 DPI）
  - 左图：epoch 级别 train vs val loss，自动标注最佳 val loss 所在 epoch
  - 右图（`--show_step` 时）：step 级别细粒度曲线（train 为半透明折线，val 为散点）

> 提示：如果自动推测的 epoch 划分不准确，使用 `--steps_per_epoch` 手动指定。计算公式：`steps_per_epoch = 训练集条数 / (batch_size_token / 平均音频token数) / gpu数`。

### 07\_augment\_used/batch\_augment\_public（公开数据集批量增强，多进程版）

> 依赖：`pip install numpy scipy pyyaml soundfile`
>
> 共享模块：`augment_engine.py`（增强引擎）
>
> 前置：先运行 `utils/public_dataset_gen_scp_txt.py` 生成标准目录结构的 `{dataset}_wav.scp` + `{dataset}_text`

基于 `public_dataset_gen_scp_txt.py` 生成的标签目录结构，从每个数据集中随机采样指定比例（默认 10%）的数据进行多说话人数据增强。

核心约束：

1. 每个数据集采样 `sample_ratio` 比例作为主说话人
2. 背景说话人从同一 split 的所有数据集中采样同比例
3. **各 split 完全隔离**：train 的主说话人和背景说话人只来自 train，dev 只来自 dev，test 只来自 test

多进程策略：

- **条目级并行**：同一数据集内的条目分配到多个 worker 并行增强
- 每个 worker 独立创建 `AudioAugmentor` 实例和 `_SimpleAudioCache`（按需加载，无大小限制，单条任务内使用）
- 背景说话人池（只读列表）通过 `initializer` 共享给所有 worker
- 任务数少于 worker 数时自动回退到单进程执行

支持断点续跑：自动扫描输出目录已有文件，跳过已完成的增强（由 YAML 配置中的 `resume: true` 控制）。

目录结构约定：

```
{label_root}/{train,dev,test}/{dataset}_wav.scp
{label_root}/{train,dev,test}/{dataset}_text
```

```bash
# 使用默认配置增强所有数据集（自动检测 CPU 核数）
python 07_augment_used/batch_augment_public.py

# 指定 worker 数
python 07_augment_used/batch_augment_public.py --workers 8

# 只增强指定数据集
python 07_augment_used/batch_augment_public.py --datasets aidatatang_200zh data_aishell

# 自定义采样比例和增强倍数
python 07_augment_used/batch_augment_public.py --sample-ratio 0.2 --factor 3
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 增强配置 YAML 文件路径 | 同目录下 `augment_config_speaker_multi.yaml` |
| `--label-root` | 标签根目录（含 train/dev/test 子目录） | `/data1/code/label` |
| `--output-root` | 增强输出根目录 | `/data1/code/label/augmented` |
| `--datasets` | 指定数据集名称（空格分隔），不指定则增强全部已发现的数据集 | 全部 |
| `--splits` | 增强哪些集合（空格分隔） | `train dev test` |
| `--sample-ratio` | 采样比例（主说话人和背景说话人池均按此比例采样） | `0.1` |
| `--factor` | 增强倍数（覆盖 yaml 中的 `augment_factor`） | yaml 配置值 |
| `--seed` | 随机种子 | `42` |
| `--workers` | Worker 进程数（`0` = 自动检测 CPU 核数 - 1） | `0` |

工作流程：

1. 扫描 `{label_root}/{split}/` 下的 `*_wav.scp` + `*_text` 文件对，自动发现数据集
2. 按 split 构建背景说话人池：每个 split 的背景说话人只从该 split 的所有数据集中选取
3. 对各 split 的背景说话人池按 `sample_ratio` 采样
4. 逐 split 逐数据集：对主说话人按 `sample_ratio` 采样，使用 `multiprocessing.Pool` 将条目分配到多个 worker 并行增强
5. 每个数据集增强完成后输出 Kaldi 格式的 scp 和 text 文件

输出到 `{output_root}/{split}/`：

- `{dataset_name}/` — 增强音频目录（`{key}_aug{N}.wav`）
- `{dataset_name}_wav.scp` — Kaldi 格式音频路径列表（`<aug_key> <绝对路径>`）
- `{dataset_name}_text` — Kaldi 格式文本标注（`<aug_key> <text>`）

### 07\_augment\_used/augment\_short\_in\_meeting（短音频嵌入会议背景增强）

> 依赖：`pip install numpy scipy soundfile`
>
> 背景数据：AliMeeting Train\_Ali\_far 远场录音（`download_alimeeting.py` 下载）

将 ≤2秒的清晰短音频（唤醒词、短指令等）随机嵌入 AliMeeting 远场多人会议背景片段中，模拟真实会议干扰场景。背景片段从 AliMeeting 8 声道远场录音中按需截取 4-8 秒，优先选取含活跃语音的段落（能量最高候选），每次随机选声道增加多样性。主说话人目录和背景音频目录均带索引缓存（`.main_index.json` / `.bg_index.json`），首次扫描后保存文件列表和时长/元信息，后续运行通过目录指纹（文件名列表 MD5）检测变化，未变化时直接加载缓存跳过逐文件 `sf.info` 调用。支持独立增益调整（`main_gain_db`/`bg_gain_db`，在 SNR 混合前分别对主说话人和背景施加 dB 增益）、可选混响/高斯底噪后处理、峰值归一化（以主说话人区域为参考）、断点续跑。API 层（`run_augment()` 函数）支持传入 `min_rms`/`max_clip_ratio` 自定义清晰度过滤阈值、`normalize_mode`/`target_peak` 控制归一化行为、`resume` 控制断点续跑、`text_file`（原始标签文件）和 `scp_file`（原始 wav.scp 文件）参数用于生成增强后的标签文件。

```bash
# 基本用法
python 07_augment_used/augment_short_in_meeting.py \
    --main_dir  E:/8.音频/03_s10/clean_short \
    --bg_dir    F:/asr_datasets/alimeeting/Train_Ali_far/audio_dir \
    --output_dir E:/8.音频/03_s10/augmented_meeting

# 自定义增强倍数和 SNR 范围
python 07_augment_used/augment_short_in_meeting.py \
    --main_dir  E:/path/to/short_wavs \
    --bg_dir    F:/asr_datasets/alimeeting/Train_Ali_far/audio_dir \
    --output_dir E:/path/to/output \
    --factor 5 --snr_min 3 --snr_max 25

# 启用混响和底噪后处理
python 07_augment_used/augment_short_in_meeting.py \
    --main_dir  E:/path/to/short_wavs \
    --bg_dir    F:/asr_datasets/alimeeting/Train_Ali_far/audio_dir \
    --output_dir E:/path/to/output \
    --add_reverb --add_noise

# 自定义背景截取时长和主音频时长上限
python 07_augment_used/augment_short_in_meeting.py \
    --main_dir  E:/path/to/short_wavs \
    --bg_dir    F:/asr_datasets/alimeeting/Train_Ali_far/audio_dir \
    --output_dir E:/path/to/output \
    --crop_min 5 --crop_max 15 --max_main_duration 2.0
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--main_dir` | 主说话人 wav 目录（清晰短音频，≤2秒）（必填） | — |
| `--bg_dir` | 背景说话人 wav 目录（AliMeeting 远场录音） | `F:\asr_datasets\alimeeting\Train_Ali_far\audio_dir` |
| `--output_dir` | 输出增强音频目录（必填） | — |
| `--factor` | 每条主音频生成几条增强样本 | `3` |
| `--snr_min` | 主说话人相对背景 SNR 下限（dB），>0 保证主说话人比背景响 | `5.0` |
| `--snr_max` | 主说话人相对背景 SNR 上限（dB） | `20.0` |
| `mix.main_gain_db`（YAML） | 主说话人额外增益（dB），在 SNR 混合前施加（0=不变，-6=减半，+6=加倍） | `0.0` |
| `mix.bg_gain_db`（YAML） | 背景声额外增益（dB），在 SNR 混合前施加（0=不变，-6=减半，-12=很安静） | `0.0` |
| `--max_main_duration` | 主说话人最大时长（秒），超过则跳过 | `2.0` |
| `--min_rms` | 主说话人最小 RMS，低于此值视为静音跳过 | `0.02` |
| `--max_clip_ratio` | 主说话人最大削波比例（peak≥0.99 的采样点占比），超过则跳过 | `0.05` |
| `--crop_min` | 背景截取最短秒数 | `4.0` |
| `--crop_max` | 背景截取最长秒数 | `8.0` |
| `--seed` | 随机种子 | `42` |
| `--add_reverb` | 对混合结果加合成混响 | 关闭 |
| `--reverb_rt60_min` | 混响 RT60 下限（秒） | `0.1` |
| `--reverb_rt60_max` | 混响 RT60 上限（秒） | `0.4` |
| `--add_noise` | 加高斯底噪 | 关闭 |
| `--noise_snr_min` | 底噪 SNR 下限（dB） | `30.0` |
| `--noise_snr_max` | 底噪 SNR 上限（dB） | `45.0` |
| `--no_normalize` | 不做峰值归一化（默认以主说话人区域峰值为参考归一化到 0.6） | 关闭 |
| `--text_file` | 原始标签文件路径（格式: `key 文本`），提供后生成增强标签（`text` + `wav.scp`） | `None` |
| `--scp_file` | 原始 wav.scp 文件路径（格式: `key /path/to/wav`），用于将 wav 路径映射到 key；不提供则用文件名 stem 作为 key | `None` |

增强流程：

1. 扫描主说话人目录，过滤 ≤ `max_main_duration` 秒且清晰的 wav 文件（RMS ≥ `min_rms` 排除近似静音、削波比例 ≤ `max_clip_ratio` 排除严重削波；首次扫描后缓存到 `.main_index.json`，后续运行目录未变化时直接加载）
2. 从 AliMeeting 远场录音中按需截取背景片段（首次扫描后缓存到 `.bg_index.json`）：随机选文件 → 随机选声道 → 采样多个候选位置取能量最高者（活跃语音优先）
3. 独立增益调整：在 SNR 混合前，分别对背景音频施加 `bg_gain_db` 增益、对主说话人音频施加 `main_gain_db` 增益（dB 转线性增益：`10^(dB/20)`）
4. 将主说话人音频按随机 SNR（均匀分布于 `[snr_min, snr_max]`）混入背景片段的随机位置
5. 可选后处理：合成混响（直达声 + 早期反射 + 晚期尾巴）、高斯底噪
6. 峰值归一化（默认 `main_region` 模式：以主说话人所在区域峰值为参考，归一化到 `target_peak`=0.6，约 -4.4dBFS，确保主说话人始终清晰突出），检查有效性（NaN/Inf 检测）

输出到 `{output_dir}/`：

- `wav/{stem}_meetbg{N}.wav` — 增强音频（`stem` 为原始文件名，`N` 为增强序号 0\~factor-1），存放在 `wav/` 子目录下
- `text` — 增强标签文件（提供 `--text_file` 时生成，格式: `{stem}_meetbg{N} 文本`）
- `wav.scp` — 增强音频路径列表（提供 `--text_file` 时生成，格式: `{stem}_meetbg{N} 绝对路径`）

断点续跑：自动检测 `{output_dir}/wav/` 下已有文件，跳过已存在的增强结果。

### 07\_augment\_used/augment\_dir（基于目录的数据增强，无标签）

> 依赖：`pip install numpy scipy pyyaml soundfile`
>
> 共享模块：同目录下 `augment_engine.py`（增强引擎）
>
> 配置文件：`07_augment_used/公开数据集增强/augment_config_speaker_multi.yaml`（默认），可通过 `--config` 切换

基于目录的数据增强，不需要 manifest 和标签文件。递归扫描输入目录中的 wav 文件，预加载所有音频到内存缓存（供 speaker\_mix / concatenate 使用），对每条音频执行增强链，输出到指定目录。适用于无标签音频的批量增强场景。

与 `07_augment/augment_dir.py` 功能相同，区别在于使用本目录（`07_augment_used/`）下的 `augment_engine.py` 和配置文件，便于独立调试和实验不同的增强配置。

支持断点续跑：自动扫描输出目录已有文件，跳过已完成的增强（由 YAML 配置中的 `resume: true` 控制），中断后重跑无需从头开始。

```bash
# 使用默认配置（config 自动定位到脚本同目录下 公开数据集增强/augment_config_speaker_multi.yaml）
python 07_augment_used/augment_dir.py

# 指定输入输出目录
python 07_augment_used/augment_dir.py \
    --config 07_augment_used/公开数据集增强/augment_config_speaker_multi.yaml \
    --input_dir /path/to/clean_wavs \
    --output_dir /path/to/augmented

# 指定增强倍数（覆盖 yaml 中的 augment_factor）
python 07_augment_used/augment_dir.py \
    --input_dir /path/to/clean_wavs \
    --output_dir /path/to/augmented \
    --factor 3
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 增强配置 YAML 文件路径 | 脚本同目录下 `公开数据集增强/augment_config_speaker_multi.yaml`（相对路径自动解析） |
| `--input_dir` | 输入 wav 目录（递归扫描所有 `.wav` 文件） | 脚本内置路径 |
| `--output_dir` | 输出增强音频目录 | 脚本内置路径 |
| `--factor` | 增强倍数（覆盖 yaml 中的 `augment_factor`） | yaml 配置值 |

工作流程：

1. 递归扫描输入目录下所有 `.wav` 文件
2. 预加载所有音频到内存缓存（`audio_cache`），用于 speaker\_mix / concatenate 策略选取背景音频
3. 逐文件逐增强副本执行增强链（由 YAML 配置的 `chain` 决定策略顺序）
4. 断点续跑：输出目录中已存在的 `{stem}_aug{N}.wav` 文件会被自动跳过
5. 增强后音频经峰值归一化，含 NaN/Inf 的无效结果自动丢弃

输出到 `{output_dir}/`：

- `{stem}_aug{N}.wav` — 增强后的音频文件（`stem` 为原始文件名不含扩展名，`N` 为增强副本索引 0\~factor-1）

### 07\_augment\_used/augment\_manifest（基于 manifest 的数据增强）

> 依赖：`pip install numpy soundfile pyyaml`
>
> 共享模块：同目录下 `augment_engine.py`（增强引擎，提供 `AudioAugmentor` 类）
>
> 配置文件：`07_augment_used/公开数据集增强/augment_config_speaker_multi.yaml`（默认），可通过 `--config` 切换

读取 JSONL manifest 文件（每行含 `key`/`wav`/`txt` 字段），自动按 split（train/dev/test/unknown）隔离处理。多说话人混合时背景音频只从同一 split 中选取，避免数据泄露。内置 `LRUAudioCache`（有界 LRU 音频缓存，默认上限 2000 条），按需加载并自动淘汰最久未用的音频，避免大数据集一次性预加载导致 OOM。

与 `07_augment/augment_manifest.py` 功能相同，区别在于使用本目录（`07_augment_used/`）下的 `augment_engine.py` 和配置文件，便于独立调试和实验不同的增强配置。

支持断点续跑：自动扫描输出目录已有文件，跳过已完成的增强（由 YAML 配置中的 `resume: true` 控制），中断后重跑无需从头开始。

```bash
# 使用默认配置增强 dev 和 test 集
python 07_augment_used/augment_manifest.py

# 指定配置和输入 manifest
python 07_augment_used/augment_manifest.py \
    --config 07_augment_used/augment_config_speaker.yaml \
    --input_manifest /path/to/manifest.json \
    --output_dir /path/to/augmented

# 只增强 train 集
python 07_augment_used/augment_manifest.py \
    --input_manifest /path/to/manifest.json \
    --output_dir /path/to/augmented \
    --splits train

# 增强多个 split
python 07_augment_used/augment_manifest.py \
    --input_manifest /path/to/manifest.json \
    --output_dir /path/to/augmented \
    --splits train,dev,test
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 增强配置 YAML 文件路径 | 脚本同目录下 `公开数据集增强/augment_config_speaker_multi.yaml` |
| `--input_manifest` | 输入 manifest 文件路径（JSON Lines，每行含 `key`/`wav`/`txt`） | `E:\8.音频\02 ASR入门\demo\dataset\manifest.json` |
| `--output_dir` | 增强音频输出目录（会自动创建 split 子目录） | `E:\8.音频\02 ASR入门\demo\dataset\augment` |
| `--splits` | 要增强的数据集划分，逗号分隔 | `dev,test` |

工作流程：

1. 读取 manifest 全部条目，按 wav 路径中的 split 关键词自动分组
2. 对每个需要增强的 split：注册音频路径到 LRU 缓存（懒加载模式，验证文件可读性）
3. 逐条目逐增强副本执行增强链（由 YAML 配置的 `chain` 决定策略顺序）
4. 断点续跑：输出目录中已存在的 `{key}_aug{N}.wav` 文件会被自动跳过
5. 增强后音频含 NaN/Inf 的无效结果自动丢弃并记录警告
6. 按 split 分别写入 wav.scp 和 text.txt

输出到 `{output_dir}/`：

- `{split}/{key}_aug{N}.wav` — 按 split 分目录存放增强音频
- `wav_{split}.scp` — Kaldi 格式音频路径列表（`<utterance_id> <wav_path>`）
- `text_{split}.txt` — Kaldi 格式文本标注（`<utterance_id> <text>`）

### 07\_augment/augment\_data（向后兼容入口）

> 依赖：`augment_manifest.py`、`augment_engine.py`

数据增强的向后兼容入口脚本。本文件保留原有命令行接口，内部直接委托给 `augment_manifest.py` 的 `main()` 函数执行。

```bash
# 与直接运行 augment_manifest.py 等价
python 07_augment/augment_data.py --config 07_augment/augment_config.yaml \
    --input_manifest data/manifest.jsonl \
    --output_dir data/augmented
```

参数、输出与 `augment_manifest.py` 完全一致，详见下方 `augment_manifest` 章节。

### 07\_augment/augment\_manifest（基于 manifest 的数据增强）

> 依赖：`pip install numpy scipy pyyaml soundfile`
>
> 配置文件：`07_augment/augment_config.yaml`

读取 JSONL manifest 文件，自动按 split（train/dev/test）隔离处理，多说话人混合时背景音频只从同一 split 中选取，避免数据泄露。内置 `LRUAudioCache`（有界 LRU 音频缓存，默认上限 2000 条），按需加载并自动淘汰最久未用的音频，避免大数据集一次性预加载导致 OOM，同时保持 speaker\_mix / concatenate 的缓存命中率。

```bash
python 07_augment/augment_manifest.py --config 07_augment/augment_config.yaml \
    --input_manifest data/manifest.jsonl \
    --output_dir data/augmented

# 只增强 train 集
python 07_augment/augment_manifest.py --config 07_augment/augment_config.yaml \
    --input_manifest data/manifest.jsonl \
    --output_dir data/augmented --splits train

# 增强 dev 和 test 集
python 07_augment/augment_manifest.py --config 07_augment/augment_config.yaml \
    --input_manifest data/manifest.jsonl \
    --output_dir data/augmented --splits dev,test
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 增强配置 YAML 文件路径 | 脚本内置路径 |
| `--input_manifest` | 输入 manifest 文件路径（JSON Lines，每行含 `key`/`wav`/`txt`） | 脚本内置路径 |
| `--output_dir` | 增强音频输出目录（会自动创建 train/dev/test 子目录） | 脚本内置路径 |
| `--splits` | 要增强的数据集划分，逗号分隔 | `dev,test` |

输出（按 split 分别生成）：

- `{output_dir}/{split}/{key}_aug{N}.wav` — 按 split 分目录存放增强音频
- `{output_dir}/wav_{split}.scp` — Kaldi 格式音频路径列表（`<utterance_id> <wav_path>`）
- `{output_dir}/text_{split}.txt` — Kaldi 格式文本标注（`<utterance_id> <text>`）

### 07\_augment/augment\_dir（基于目录的数据增强）

> 依赖：`pip install numpy scipy pyyaml soundfile`
>
> 配置文件：`07_augment/augment_config_speaker_multi.yaml`（默认），可通过 `--config` 切换为其他配置

基于目录的数据增强，不需要 manifest 和标签文件。递归扫描输入目录中的 wav 文件，对每条音频执行增强链，输出到指定目录。适用于无标签音频的批量增强场景。支持断点续跑：自动扫描输出目录已有文件，跳过已完成的增强，中断后重跑无需从头开始。

```bash
python 07_augment/augment_dir.py \
    --input_dir /path/to/clean_wavs \
    --output_dir /path/to/augmented

# 使用其他配置文件
python 07_augment/augment_dir.py --config 07_augment/augment_config.yaml \
    --input_dir /path/to/clean_wavs \
    --output_dir /path/to/augmented

# 指定增强倍数（覆盖 yaml 中的 augment_factor）
python 07_augment/augment_dir.py \
    --input_dir /path/to/clean_wavs \
    --output_dir /path/to/augmented \
    --factor 3
```

断点续跑：输出目录中已存在的 `{stem}_aug{N}.wav` 文件会被自动跳过，进度日志中会显示跳过数量。无需额外参数，重复运行即可续跑。

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 增强配置 YAML 文件路径 | 脚本内置路径 |
| `--input_dir` | 输入 wav 目录（递归扫描所有 `.wav` 文件） | 脚本内置路径 |
| `--output_dir` | 输出增强音频目录 | 脚本内置路径 |
| `--factor` | 增强倍数（覆盖 yaml 中的 `augment_factor`） | yaml 配置值 |

输出：

- `{output_dir}/{stem}_aug{N}.wav` — 增强后的音频文件（`stem` 为原始文件名不含扩展名）

### 07\_augment/manifest\_to\_scp（manifest 转 wav.scp + text）

> 依赖：仅 Python 标准库（`json`, `argparse`, `collections`）

将三字段 manifest.jsonl（每行含 `key`/`wav`/`txt`）转换为 Kaldi 格式的 `wav.scp` 和 `text.txt`。根据 wav 路径中出现的 `/train/`、`/dev/`、`/test/` 自动判断 split，按 split 分别输出文件。

```bash
# 基本用法（输出到 manifest 同目录）
python 07_augment/manifest_to_scp.py manifest.jsonl

# 指定输出目录
python 07_augment/manifest_to_scp.py manifest.jsonl --output_dir ./output
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `manifest`（位置参数） | 输入的三字段 manifest.jsonl 路径 | （必填） |
| `--output_dir` | 输出目录 | 与 manifest 同目录 |

split 检测规则：路径中包含 `/train/`、`/dev/`、`/test/` 时归入对应 split，否则归为 `unknown`。

输出（按 split 分别生成）：

- `{output_dir}/wav_{split}.scp` — Kaldi 格式音频路径列表（`<key> <wav_path>`）
- `{output_dir}/text_{split}.txt` — Kaldi 格式文本标注（`<key> <text>`）

### 07\_augment 增强策略与配置

> 以下策略和配置说明适用于 `augment_manifest.py`、`augment_dir.py` 和 `augment_data.py`。

已实现的增强策略（16 种）：

| 策略 | 说明 | 配置参数 |
|------|------|----------|
| `gaussian_noise` | 高斯白噪声叠加 | `snr_range: [5, 20]` |
| `speaker_mix` | 多说话人混合（近场+远场，逐条独立处理：混响/低通/音调/SNR，支持多种时间对齐模式和距离因子关联采样） | `snr_range`, `max_speakers`, `random_insert`, `overlap_mode`, `min_overlap`, `bg_reverb`, `bg_rt60_range`, `bg_lowpass_freq`, `bg_lowpass_range`, `bg_pitch_shift`, `bg_semitone_range`, `bg_distance_correlated` |
| `reverb` | 改进合成 RIR 房间混响（直达声 + 早期反射 + 晚期混响尾巴） | `rt60_range: [0.2, 0.8]` |
| `real_rir` | 真实 RIR 卷积混响（需提供 RIR 文件目录，如 OpenSLR-26/28） | `rir_dir` |
| `speed_perturb` | 速度扰动（重采样） | `factor_range: [0.9, 1.1]` |
| `volume_perturb` | 音量扰动 | `gain_range: [-6, 6]` |
| `noise_from_file` | 环境噪声叠加（需提供噪声目录） | `noise_dir`, `snr_range` |
| `pitch_shift` | 音调扰动（保持时长不变，resample 实现） | `semitone_range: [-2, 2]` |
| `eq_random` | 随机均衡器（频响随机化，模拟不同麦克风/设备） | `num_bands`, `gain_range` |
| `random_crop_pad` | 随机裁剪/填充（增加时间偏移鲁棒性） | `target_sec`, `pad_mode` |
| `time_stretch` | 时间拉伸（Phase Vocoder，保持音调不变） | `rate_range: [0.9, 1.1]` |
| `random_silence` | 随机静音插入（模拟停顿和犹豫） | `max_inserts`, `duration_range` |
| `codec_simulate` | 编解码失真模拟（G.711 μ-law / A-law） | `codecs: [ulaw, alaw]` |
| `bandpass_filter` | 带通滤波（模拟电话信道等） | `low_freq`, `high_freq`, `order` |
| `clipping` | 削波失真模拟（模拟麦克风过载） | `threshold_range: [0.3, 0.8]` |
| `concatenate` | 多条短音频拼接（训练长序列处理能力） | `max_concat`, `gap_range` |

`speaker_mix` 通过 `overlap_mode` 控制背景人声与主音频的时间对齐方式（`random_insert` 作为兼容参数保留，`overlap_mode` 优先）：

| 模式 | 说明 |
|------|------|
| `full` | 背景音全程覆盖主音频（循环填充） |
| `random` | 随机位置插入，保证 `min_overlap` 比例的重叠（默认） |
| `mid_insert` | 主说话人说到一半时插入背景声，持续到结尾 |
| `bg_first` | 背景声先出现，主说话人稍后加入 |
| `bg_tail` | 主说话人说完后出现背景声 |
| `bg_longer` | 背景声比主音频长，主音频结束后背景声还在继续 |
| `auto` | 每次随机选择以上模式之一（推荐，最大多样性） |

`min_overlap`（默认 0.8）控制背景声与主音频的最小重叠比例，确保混合区域不会过短。

`bg_distance_correlated`（默认 `false`）启用距离因子关联采样：为每条背景人声采样一个统一的距离因子 `t ∈ [0, 1]`（`t=0` 最近，`t=1` 最远），SNR、RT60、低通截止频率均由同一个 `t` 驱动，保证物理一致性——近处的背景人声音量大（SNR 低）、混响弱（RT60 小）、高频保留（lowpass 高）；远处则相反。关闭时各参数独立随机采样。

配置文件 `augment_config.yaml` 中通过 `chain` 列表指定默认增强策略的执行顺序，`augment_factor` 控制每条原始音频生成的增强样本数，`seed` 控制随机种子保证可复现。

支持按目录指定不同的增强链（`folder_chains`）：在配置文件中添加 `folder_chains` 字段，按路径子串匹配音频文件，首个命中的规则生效，未命中时回退到默认 `chain`。示例：

```yaml
folder_chains:
  "/phone_channel/":
    chain: [bandpass_filter, codec_simulate, gaussian_noise]
  "/speaker_data/":
    chain: [speaker_mix, reverb, volume_perturb]
```

### 07\_augment/augment\_engine（增强引擎模块）

> 依赖：`pip install numpy scipy pyyaml soundfile`
>
> 配置文件：`07_augment/augment_config.yaml`

音频数据增强引擎，提供 `AudioAugmentor` 类，包含所有增强策略的实现。本模块不直接运行，而是作为库由 `augment_manifest.py` 和 `augment_dir.py` 导入使用。

```python
from augment_engine import AudioAugmentor

augmentor = AudioAugmentor("augment_config.yaml")

# 统一增强入口：对单条音频执行完整增强链
result = augmentor.augment_single(
    audio=audio_array,       # np.ndarray 输入音频
    sr=16000,                # 采样率
    wav_path="/path/to.wav", # 用于 _resolve_chain 匹配增强链
    aug_idx=0,               # 增强副本索引
    file_idx=0,              # 文件索引（用于确定性随机种子）
    audio_cache=cache_dict,  # 可选，{path: (audio, sr)} 缓存，供 speaker_mix/concatenate 使用
    candidate_paths=paths,   # 可选，可用于混合的候选音频路径列表
)
# 返回峰值归一化后的增强音频（np.ndarray），含 NaN/Inf 时返回 None
```

核心方法：

| 方法 | 说明 |
|------|------|
| `augment_single()` | 统一增强入口，根据 `wav_path` 自动匹配增强链，按链式顺序执行所有已启用策略，返回归一化后的音频 |
| `mix_speakers()` | 多说话人混合，通过 `overlap_mode` 支持 7 种时间对齐模式（full/random/mid\_insert/bg\_first/bg\_tail/bg\_longer/auto），`min_overlap` 控制最小重叠比例 |
| `apply_reverb()` | 改进合成 RIR 房间混响：直达声脉冲（0ms）+ 稀疏早期反射（5-50ms，3-8 个随机衰减脉冲）+ 密集晚期混响尾巴（50ms-RT60，指数衰减高斯噪声），FFT 卷积并保持原始音量水平 |
| `_sample_normal_clipped()` | 正态分布采样并 clip 到 `[low, high]`，以中点为均值、范围/4 为标准差（约 95% 落在范围内），比均匀分布更贴近真实场景（典型值集中，极端值少见）。用于 reverb RT60 等参数采样 |
| `_resolve_chain()` | 根据音频路径匹配 `folder_chains`，未命中时回退到默认 `chain` |
| `_detect_split()` | 从路径中检测 train/dev/test 划分 |
| `normalize_peak()` | 峰值归一化，仅在峰值超过 1.0 时缩放（防止削波），保留原始动态范围 |

已实现的增强策略（16 种）详见上方「07\_augment 增强策略与配置」章节。

配置通过 `augment_config.yaml` 加载，支持 `chain`（默认增强链）、`folder_chains`（按路径匹配不同增强链）和 `resume`（断点续跑开关，默认 `true`，由 `augment_dir.py` / `augment_manifest.py` 读取以决定是否跳过已有输出文件）。

输出：本模块无独立输出，由调用方决定输出文件位置与格式。

### compare\_pretag

> 依赖：仅 Python 标准库

比对两个预标注模型的标签结果。

```bash
python 06_pretag_and_compare/compare_pretag.py
python 06_pretag_and_compare/compare_pretag.py --a firered_pretag_output/text.txt --b qwen_pretag_output/text.txt
python 06_pretag_and_compare/compare_pretag.py --a firered_pretag_output/pretag.jsonl --b qwen_pretag_output/pretag.jsonl
python 06_pretag_and_compare/compare_pretag.py --threshold 0.2    # CER>=20% 视为不一致
```

输出到 `s10_data_reflow/compare_output/`：

- `compare_detail.jsonl` — 逐条比对结果（uid / text\_a / text\_b / CER / sub / ins / del / match）
- `diff_list.txt` — 超阈值不一致清单（按 CER 降序，方便人工审核）
- `compare_report.txt` — 统计报告（一致率 / CER 分布 / 差异最大 30 条）
- `compare_summary.json` — JSON 汇总

### cross\_validate\_labels

> 依赖：仅 Python 标准库

三方交叉验证：人工标注 vs FireRedASR vs Qwen3-ASR，重点找出人工标注可能出错的条目。

```bash
python 06_pretag_and_compare/cross_validate_labels.py
python 06_pretag_and_compare/cross_validate_labels.py --human labeled/text.txt --firered firered_pretag_output/text.txt --qwen qwen_pretag_output/text.txt
python 06_pretag_and_compare/cross_validate_labels.py --threshold 0.1    # 放宽一致性判定
```

判定分类：

| 分类 | 含义 | 备注 |
|------|------|------|
| `all_agree` | 三方一致 | — |
| `models_agree` | 两模型一致但与人工不同 | ⚠️ 人工可能错了（重点关注） |
| `human_firered` | 人工与 FireRedASR 一致，Qwen 不同 | — |
| `human_qwen` | 人工与 Qwen 一致，FireRedASR 不同 | — |
| `all_differ` | 三方都不同 | ⚠️ 需复核 |

输出到 `s10_data_reflow/cross_validate_output/`：

- `validate_detail.jsonl` — 逐条详情（uid / category / 三方文本 / 三组 CER）
- `models_agree.txt` — 人工可能出错清单（按 CER 降序，优先看这个）
- `all_differ.txt` — 三方都不同清单
- `human_firered.txt` — 人工 = FireRed，Qwen 不同
- `human_qwen.txt` — 人工 = Qwen，FireRed 不同
- `validate_report.txt` — 统计报告（分类统计 + 典型案例）
- `validate_summary.json` — JSON 汇总

### cross\_validate\_review\_server

> 依赖：`pip install flask`
>
> 前置：先运行 `cross_validate_labels.py` 生成 `validate_detail.jsonl`

Web 版交叉验证人工复核工具。加载 `cross_validate_labels.py` 的输出，在浏览器中播放音频，展示人工/FireRedASR/Qwen3-ASR 三方标签及 CER，提供选择按钮和可编辑输入框，支持剔除信号差样本。

默认数据根目录为 `sp4_data/`（脚本内 `BASE_DIR` 变量），所有默认路径均相对于此目录。

```bash
python 06_pretag_and_compare/cross_validate_review_server.py
python 06_pretag_and_compare/cross_validate_review_server.py --detail validate_detail.jsonl --wav-dir labeled/wav
python 06_pretag_and_compare/cross_validate_review_server.py --category models_agree
python 06_pretag_and_compare/cross_validate_review_server.py --port 5001
```

浏览器打开 `http://127.0.0.1:5001`

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--detail` | `validate_detail.jsonl` 文件路径 | `{BASE_DIR}/cross_validate_output/validate_detail.jsonl` |
| `--wav-dir` | wav 文件目录 | `{BASE_DIR}/labeled/wav` |
| `--category` | 只加载指定分类的条目（如 `models_agree`、`all_differ`） | 全部分类 |
| `--decision-file` | 复核决策记录文件路径 | `{BASE_DIR}/cross_validate_output/review_decisions.jsonl` |
| `--port` | 端口号 | `5001` |
| `--host` | 监听地址 | `127.0.0.1` |

复核选项快捷键：

| 按键 | 选项 | 含义 |
|------|------|------|
| 1 | human | 采用人工标签 |
| 2 | firered | 采用 FireRedASR 结果 |
| 3 | qwen | 采用 Qwen3-ASR 结果 |
| E | edit | 手动编辑预测结果（聚焦编辑框） |
| D | bad | 信号差，剔除该样本 |

导航：`←` / `→` 前后翻，`S` 跳到未复核

界面展示每条样本的分类标签（如「两模型一致 人工不同」）和三组 CER（H-F / H-Q / F-Q），点击标签行可将对应文本填入编辑框。

Qwen 标签在展示和选择时均自动去除末尾中英文标点（`，。！？、；：…·,.!?;:`），避免标点差异污染标签。

复核决策保存在 `review_decisions.jsonl`（每行 `{uid, choice, text}`）。

### export\_labeled\_check

> 依赖：仅 Python 标准库（`json`, `shutil`, `argparse`, `pathlib`）
>
> 前置：先运行 `cross_validate_labels.py` 生成 `validate_detail.jsonl`

从交叉验证结果中导出不一致样本（非 `all_agree`）到独立的 `labeled_check/` 目录，生成人工/FireRedASR/Qwen3-ASR 三方的 text.txt、ChatML 格式 train.jsonl 和 wav.scp，方便在训练服务器上直接使用。Qwen 标签自动去除末尾标点。

默认数据根目录为 `sp4_data/`（脚本内 `BASE_DIR` 变量），所有默认路径均相对于此目录。

```bash
python 06_pretag_and_compare/export_labeled_check.py

# 只导出特定类别
python 06_pretag_and_compare/export_labeled_check.py --category models_agree

# 自定义路径
python 06_pretag_and_compare/export_labeled_check.py --detail validate_detail.jsonl --wav-dir labeled/wav --output labeled_check

# 不拷贝 wav（仅生成标签文件）
python 06_pretag_and_compare/export_labeled_check.py --no-copy-wav

# 自定义 scp/jsonl 中的 wav 路径前缀（用于 Linux 服务器）
python 06_pretag_and_compare/export_labeled_check.py --wav-prefix /data/share/dataset/s10_data_reflow/labeled_check/wav
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--detail` | `validate_detail.jsonl` 文件路径 | `{BASE_DIR}/cross_validate_output/validate_detail.jsonl` |
| `--wav-dir` | 源 wav 文件目录 | `{BASE_DIR}/labeled/wav` |
| `--output` | 输出目录 | `{BASE_DIR}/labeled_check` |
| `--category` | 只导出指定类别（`models_agree` / `all_differ` / `human_firered` / `human_qwen`） | 全部非 `all_agree` |
| `--wav-prefix` | scp/jsonl 中使用的 wav 路径前缀（用于生成 Linux 服务器可用的路径） | `/data/share/dataset/s10_data_reflow/labeled_check/wav` |
| `--no-copy-wav` | 不拷贝 wav 文件，仅生成标签文件 | — |

输出到 `{BASE_DIR}/labeled_check/`（默认 `sp4_data/labeled_check/`）：

- `wav/` — 不一致样本的 wav 文件
- `wav.scp` — Kaldi 格式音频路径
- `human_text.txt` — 人工标注（`uid text`）
- `firered_text.txt` — FireRedASR 结果
- `qwen_text.txt` — Qwen3-ASR 结果（已去末尾标点）
- `human_train.jsonl` — 人工标注 ChatML 格式
- `firered_train.jsonl` — FireRedASR ChatML 格式
- `qwen_train.jsonl` — Qwen3-ASR ChatML 格式（已去末尾标点）
- `validate_detail.json` — 所有不一致条目详情（JSON 数组）

### sort\_error\_samples

> 依赖：仅 Python 标准库（`json`, `shutil`, `argparse`）
>
> 前置：先运行 `cross_validate_labels.py`（交叉验证模式）或 `compare_pretag.py`（比对模式）生成详情 jsonl

根据比对/交叉验证结果，将错误样本音频按类别拷贝到对应子目录，每个子目录下同时生成 `labels.txt` 记录 uid 和各方标签，方便人工复核。

```bash
# 交叉验证模式（默认），按 category 分类拷贝
python 06_pretag_and_compare/sort_error_samples.py

# 比对模式，按 CER 区间分类拷贝
python 06_pretag_and_compare/sort_error_samples.py --mode compare

# 自定义路径
python 06_pretag_and_compare/sort_error_samples.py --detail validate_detail.jsonl --wav_scp labeled/wav.scp --output sorted_errors/

# 只拷贝特定类别
python 06_pretag_and_compare/sort_error_samples.py --categories models_agree all_differ
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 运行模式：`cross_validate`（按 category）或 `compare`（按 CER 区间） | `cross_validate` |
| `--detail` | 详情 jsonl 文件路径 | 按模式自动选择 |
| `--wav_scp` | wav.scp 文件路径（uid → wav 路径映射） | `labeled/wav.scp` |
| `--output` | 输出目录 | `error_sorted` |
| `--categories` | 只拷贝指定类别（空格分隔） | 全部类别 |

交叉验证模式（`cross_validate`）输出子目录：

| 子目录 | 含义 |
|--------|------|
| `models_agree/` | 两模型一致但与人工不同（人工可能错） |
| `all_differ/` | 三方都不同（需复核） |
| `human_firered/` | 人工 = FireRed，Qwen 不同 |
| `human_qwen/` | 人工 = Qwen，FireRed 不同 |

比对模式（`compare`）输出子目录：

| 子目录 | 含义 |
|--------|------|
| `cer_05_10/` | CER 5%–10% |
| `cer_10_20/` | CER 10%–20% |
| `cer_20_50/` | CER 20%–50% |
| `cer_above_50/` | CER > 50% |

输出到 `s10_data_reflow/error_sorted/`：

- `{category}/` — 按类别分目录存放的 wav 文件
- `{category}/labels.txt` — 该类别下每条样本的 uid 和各方标签

### review\_labels

> 依赖：仅 Python 标准库（`json`, `shutil`, `argparse`）
>
> 前置：先运行 `cross_validate_labels.py`（交叉验证模式）或 `compare_pretag.py`（比对模式）生成详情 jsonl

人工复核标签工具，分两步使用：

1. `generate` — 从交叉验证/比对结果生成 `review.txt`，预填候选标签（自动选择最可能正确的标签填入 `FINAL:` 行）
2. 人工在编辑器中逐条复核，修改 `FINAL:` 行
3. `merge` — 解析 `review.txt`，将修正结果回写到 `text.txt`（自动备份原文件）

```bash
# 生成 review.txt（交叉验证模式，默认只处理需复核的类别）
python 06_pretag_and_compare/review_labels.py generate

# 生成 review.txt（比对模式）
python 06_pretag_and_compare/review_labels.py generate --mode compare

# 只生成特定类别
python 06_pretag_and_compare/review_labels.py generate --categories models_agree all_differ

# 人工复核后，合并回写
python 06_pretag_and_compare/review_labels.py merge

# 合并时指定路径
python 06_pretag_and_compare/review_labels.py merge --review review.txt --text labeled/text.txt

# 试运行（不实际写入，只输出统计）
python 06_pretag_and_compare/review_labels.py merge --dry-run
```

参数说明（`generate` 子命令）：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 运行模式：`cross`（交叉验证）或 `compare`（双模型比对） | `cross` |
| `--detail` | 详情 jsonl 文件路径 | 按模式自动选择 |
| `--wav_scp` | wav.scp 文件路径（uid → wav 路径映射） | `labeled/wav.scp` |
| `--review` | 输出 review.txt 路径 | `review/review.txt` |
| `--categories` | 只处理指定类别（空格分隔） | 全部需复核类别 |

参数说明（`merge` 子命令）：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--review` | review.txt 文件路径 | `review/review.txt` |
| `--text` | 要回写的 text.txt 路径 | `labeled/text.txt` |
| `--dry-run` | 试运行，不实际写入 | — |

`review.txt` 格式：

```
# uid: xxx
# category: models_agree
# wav: E:/.../.../xxx.wav
# human:   播放今天得天气
# firered: 播放今天的天气
# qwen:    播放今天的天气
FINAL: 播放今天的天气
```

输出到 `s10_data_reflow/review/`：

- `review.txt` — 待复核标签文件（`generate` 生成，人工编辑后供 `merge` 读取）
- `text.txt.bak.{timestamp}` — `merge` 回写前的 text.txt 备份

---

### s10\_data\_reflow/apply\_reflow（Label Studio 标注回流）

> 依赖：仅 Python 标准库（`json`, `shutil`, `collections`）
>
> 前置：Label Studio 导出的标注 JSON 文件（如 `1411_S10智能音箱数据_2_20260414.json`），以及原始 `text.txt` / `wav.scp`

将 Label Studio 重新标注结果回流到训练数据文件 `text.txt` 和 `wav.scp`。根据标注中的备注标签和人工校正文本，执行剔除或更新操作，原文件自动备份为 `.bak`。

处理逻辑：

1. 备注标签为「听不清」或「剔除」的音频 → 从 `text.txt` 和 `wav.scp` 中删除对应条目
2. 其他音频 → 用标注中的 `content_human` 字段替换 `text.txt` 中的文本
3. 无 `content_human` 或文本为空的条目 → 同样从 `text.txt` 和 `wav.scp` 中剔除

使用前需修改脚本内的路径配置变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `JSON_FILE` | Label Studio 导出的标注 JSON 文件路径 | 脚本同目录下 `1411_S10智能音箱数据_2_20260414.json` |
| `TEXT_FILE` | 原始 text.txt 路径 | 脚本同目录下 `text.txt` |
| `WAV_SCP` | 原始 wav.scp 路径 | 脚本同目录下 `wav.scp` |

```bash
python s10_data_reflow/apply_reflow.py
```

备份机制：首次运行时自动将 `text.txt` 和 `wav.scp` 备份为 `.bak` 文件；若 `.bak` 已存在则跳过备份（始终从 `.bak` 读取原始数据进行处理）。

Label Studio JSON 格式要求：每条标注需包含 `data.audio`（音频路径，用于提取 key）和 `annotations[].result[]`，其中 `from_name="备注标签"` 的 choices 字段用于判断剔除，`from_name="content_human"` 的 text 字段用于文本更新。脚本使用 `json.JSONDecoder.raw_decode` 解析，兼容末尾有多余字符的 JSON 文件。

输出（原地更新）：

- `text.txt` — 更新后的 Kaldi 格式文本标注（剔除 + 文本替换）
- `wav.scp` — 更新后的 Kaldi 格式音频路径列表（剔除对应条目）
- `text.txt.bak` — 原始 text.txt 备份
- `wav.scp.bak` — 原始 wav.scp 备份

控制台输出处理统计（剔除/更新/保留数量）和一致性验证（text.txt 与 wav.scp 行数是否一致）。

### s10\_data\_reflow/download\_models（下载 FunASR 模型）

> 依赖：`pip install funasr`
>
> 用途：在本地预下载 FunASR 推理所需的模型文件，供 `trim_trigger_funasr.py` 等脚本使用

将 FunASR 所需的预训练模型下载到本地 `F:\pretrained_models` 目录。下载的模型包括：

| 模型 | 说明 |
|------|------|
| `paraformer-zh` | 主 ASR 模型（中文语音识别，带时间戳） |
| `fa-zh` | 强制对齐时间戳模型（用于字级别时间戳对齐） |

```bash
python s10_data_reflow/download_models.py
```

无命令行参数。如需修改保存路径，编辑脚本内的 `SAVE_DIR` 变量（默认 `F:\pretrained_models`）。

输出到 `F:\pretrained_models/`：

- `paraformer-zh/` — paraformer-zh 模型文件目录
- `fa-zh/` — fa-zh 强制对齐模型文件目录

控制台输出：各模型下载进度和完成状态，最终列出目录内容。

### s10\_data\_reflow/analyze\_trigger（唤醒词分布分析）

> 依赖：仅 Python 标准库（`json`, `re`）
>
> 前置：`s10_data_reflow/` 目录下已有 Label Studio 导出的标注 JSON 文件

分析标注数据中唤醒词（"你好萤石"及其同音变体）的出现模式。统计三类情况：

1. **纯唤醒词**：标注文本仅包含唤醒词本身（去除标点和空格后匹配）
2. **唤醒词 + 指令**：标注文本中包含唤醒词，但还有其他内容（如"你好萤石，今天天气怎么样"）
3. **唤醒词不在开头**：唤醒词出现在文本中间或末尾的异常情况

识别的唤醒词变体：`你好萤石`、`你好莹石`、`你好荧石`、`你好影石`

使用前需修改脚本内的路径变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| JSON 文件路径 | Label Studio 导出的标注 JSON | `s10_data_reflow/1410_S10智能音箱数据_1_20260414.json` |

```bash
python s10_data_reflow/analyze_trigger.py
```

输出：控制台打印统计结果（无文件输出），包括：

- 含唤醒词的总条目数
- 纯唤醒词（无指令）的数量和示例
- 唤醒词 + 指令的数量和示例
- 唤醒词不在句首的异常条目数量和示例

### s10\_data\_reflow/count\_trigger\_cmd（统计触发词+命令条目）

> 依赖：仅 Python 标准库（`os`, `collections`）
>
> 前置：`s10_data_reflow/labeled copy/` 目录下已有 `text.txt`

统计 `text.txt` 中以触发词开头且后面跟有命令内容的条目数量。按触发词前缀分类计数，并展示前 20 条示例。

识别的触发词前缀（按长度优先匹配）：
- 四字：`你好萤石`、`你好莹石`、`你好荧石`、`你好影石`
- 三字：`好萤石`、`好莹石`、`好荧石`、`好影石`
- 二字：`萤石`、`莹石`、`荧石`、`影石`

匹配规则：去除文本开头的标点（`，,、`）和空格后，匹配最长的触发词前缀；去掉触发词后仍有剩余内容才计为"触发词+命令"。

使用前需修改脚本内的路径变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `TEXT` | text.txt 路径 | `E:\8.音频\03 s10\s10_data_reflow\labeled copy\text.txt` |

```bash
python s10_data_reflow/count_trigger_cmd.py
```

输出：控制台打印统计结果（无文件输出），包括：

- 触发词+命令的总条目数
- 按触发词前缀分类的条目数量（按数量降序排列）
- 前 20 条文本示例

### s10\_data\_reflow/count\_hao\_yingshi（分析"好萤石"前缀记录）

> 依赖：仅 Python 标准库（`os`, `collections`）
>
> 前置：`s10_data_reflow/labeled copy/` 目录下已有 `text.txt`

分析 `text.txt` 中以"好萤石"系列前缀开头（排除"你好萤石"系列）且后面跟有命令内容的记录。用于了解"好萤石+指令"这类短触发词样本的分布和命令模式。

识别的前缀：
- 匹配：`好萤石`、`好莹石`、`好荧石`、`好影石`
- 排除：`你好萤石`、`你好莹石`、`你好荧石`、`你好影石`

匹配规则：去除文本开头的标点（`，,、`）后，先排除以"你好萤石"系列开头的条目，再匹配"好萤石"系列前缀；去掉前缀后仍有剩余内容才计入结果。

使用前需修改脚本内的路径变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `TEXT` | text.txt 路径 | `E:\8.音频\03 s10\s10_data_reflow\labeled copy\text.txt` |

```bash
python s10_data_reflow/count_hao_yingshi.py
```

输出：控制台打印统计结果（无文件输出），包括：

- "好萤石"+命令的总条目数
- 按前缀变体分布（如 `好萤石: N 条`、`好莹石: N 条`）
- 前 50 条原始文本示例
- 前 50 条去掉前缀后的命令内容（格式：`[前缀] + '命令'`）

### s10\_data\_reflow/move\_only\_trigger（移动纯触发词音频）

> 依赖：仅 Python 标准库（`os`, `shutil`）
>
> 前置：`s10_data_reflow/labeled copy/` 目录下已有 `text.txt`、`wav.scp` 和 `wav/` 音频目录

将 `text.txt` 中文本内容仅为触发词（"你好萤石"及其同音变体）的音频从 `wav/` 移动到 `only_trigger/` 目录，用于从训练数据中分离纯唤醒词样本。

触发词匹配规则：去除标点（`，,、`）和空格后，文本完全等于以下任一变体即判定为纯触发词：
- `你好萤石`、`你好莹石`、`你好荧石`、`你好影石`

处理流程：

1. 从 `text.txt` 找出文本内容仅为触发词的 key
2. 在 `wav/` 目录中查找对应音频（`{key}.wav`）
3. 将匹配到的音频移动到 `only_trigger/` 目录
4. `text.txt` 和 `wav.scp` 暂不修改，等确认后再处理

使用前需修改脚本内的路径变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `BASE_DIR` | 数据根目录 | `E:\8.音频\03 s10\s10_data_reflow\labeled copy` |
| `TEXT_FILE` | text.txt 路径 | `{BASE_DIR}/text.txt` |
| `WAV_SCP` | wav.scp 路径 | `{BASE_DIR}/wav.scp` |
| `WAV_DIR` | 音频文件目录 | `{BASE_DIR}/wav` |
| `OUTPUT_DIR` | 输出目录（移动目标） | `{BASE_DIR}/only_trigger` |

```bash
python s10_data_reflow/move_only_trigger.py
```

输出：

- `{BASE_DIR}/only_trigger/` — 仅含触发词的音频文件（从 `wav/` 移动过来）

控制台输出：找到的纯触发词记录数、成功移动的音频数、未找到的音频数、`only_trigger/` 目录中的文件列表预览（前 5 个 + 总数）。

> 注意：当前版本仅移动音频文件，不修改 `text.txt` 和 `wav.scp`。确认移动结果无误后需手动或另行处理标签文件中对应条目的清理。

### s10\_data\_reflow/remove\_only\_trigger（删除纯触发词记录）

> 依赖：仅 Python 标准库（`os`）
>
> 前置：`s10_data_reflow/labeled copy/` 目录下已有 `text.txt`、`wav.scp` 和 `wav/` 音频目录
>
> 关联：`move_only_trigger.py` 的后续步骤——前者仅移动音频，本脚本彻底从训练数据中清除纯触发词条目

从 `labeled copy/` 中彻底删除仅含触发词的记录：删除 `wav/` 中残留的音频文件，并从 `text.txt` 和 `wav.scp` 中移除对应行。适用于已通过 `move_only_trigger.py` 确认分离结果后，需要清理标签文件的场景。

触发词匹配规则：去除标点（`，,、`）和空格后，文本完全等于以下任一变体即判定为纯触发词：
- `你好萤石`、`你好莹石`、`你好荧石`、`你好影石`

处理流程：

1. 从 `text.txt` 找出文本内容仅为触发词的 key
2. 删除 `wav/` 中对应的音频文件（已被 `move_only_trigger.py` 移走的不会重复处理）
3. 原地过滤 `text.txt`，移除匹配的行
4. 原地过滤 `wav.scp`，移除匹配的行

使用前需修改脚本内的路径变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `BASE_DIR` | 数据根目录 | `E:\8.音频\03 s10\s10_data_reflow\labeled copy` |
| `TEXT_FILE` | text.txt 路径 | `{BASE_DIR}/text.txt` |
| `WAV_SCP` | wav.scp 路径 | `{BASE_DIR}/wav.scp` |
| `WAV_DIR` | 音频文件目录 | `{BASE_DIR}/wav` |

```bash
python s10_data_reflow/remove_only_trigger.py
```

输出：

- 原地修改 `{BASE_DIR}/text.txt` — 移除纯触发词行
- 原地修改 `{BASE_DIR}/wav.scp` — 移除纯触发词行
- 删除 `{BASE_DIR}/wav/` 中残留的纯触发词音频文件

控制台输出：需删除的条目数、从 `wav/` 删除的音频数、`text.txt` 和 `wav.scp` 的行数变化（原始 → 保留，删除数）。

> 注意：本脚本会原地修改 `text.txt` 和 `wav.scp`，不会自动备份。建议在运行前确认 `move_only_trigger.py` 的移动结果无误，或手动备份文件。

### s10\_data\_reflow/copy\_yingshi\_prefix（拷贝"萤石"前缀音频供查看）

> 依赖：仅 Python 标准库（`os`, `shutil`）
>
> 前置：`s10_data_reflow/labeled copy/` 目录下已有 `text.txt` 和 `wav/` 音频目录

将 `text.txt` 中以"萤石/莹石/荧石/影石"开头（但不以"你好萤石"或"好萤石"等长前缀开头）且后面有内容的音频拷贝到 `萤石_prefix/` 目录，同时生成文本列表 `list.txt`，方便人工查看这类"萤石+指令"样本。

匹配规则：

1. 去除文本开头的标点（`，,、`）
2. 排除长前缀：`你好萤石/莹石/荧石/影石`、`好萤石/莹石/荧石/影石`
3. 匹配短前缀：`萤石/莹石/荧石/影石` 且后面有剩余内容

使用前需修改脚本内的路径变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `BASE` | 数据根目录 | `E:\8.音频\03 s10\s10_data_reflow\labeled copy` |
| `TEXT` | text.txt 路径 | `{BASE}/text.txt` |
| `WAV_DIR` | 音频文件目录 | `{BASE}/wav` |
| `OUTPUT_DIR` | 输出目录 | `{BASE}/萤石_prefix` |

```bash
python s10_data_reflow/copy_yingshi_prefix.py
```

输出到 `{BASE}/萤石_prefix/`：

- `*.wav` — 匹配到的音频文件（从 `wav/` 拷贝）
- `list.txt` — 文本列表（每行 `key 文本`）

控制台输出：找到的匹配记录数、拷贝音频数、未找到的音频数。

### s10\_data\_reflow/cut\_trigger\_funasr/trim\_trigger\_funasr（FunASR 强制对齐裁剪触发词）

> 依赖：`pip install funasr soundfile`
>
> 模型：`speech_timestamp_prediction-v1-16k-offline`（FunASR MonotonicAligner，字级时间戳预测）
>
> 前置：`move_trigger_cmd.py` 已将"触发词+命令"音频和 `list.txt` 输出到 `trigger_cmd/` 目录；已通过 `download_models.py` 或手动下载好 `speech_timestamp_prediction-v1-16k-offline` 到本地模型目录

使用 FunASR 强制对齐模型（MonotonicAligner）对"触发词+命令"音频做字级时间戳对齐，取触发词最后一个字的结束时间作为裁剪起点，裁剪掉触发词部分后只保留命令部分。支持 batch 推理（默认 batch 32）；整批推理失败时自动降级为逐条处理；裁剪后若剩余长度不足 100ms 则视为失败；触发词字符范围超出对齐结果长度也会失败。

支持的触发词前缀（按长度降序排列匹配）：

- 4 字：`你好萤石`、`你好莹石`、`你好荧石`、`你好影石`
- 3 字：`好萤石`、`好莹石`、`好荧石`、`好影石`

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_dir` | 输入目录（需包含 `list.txt` 和 `*.wav`） | 必填 |
| `--output_dir` | 裁剪结果输出目录 | 必填 |
| `--model_dir` | 本地模型根目录（需包含 `speech_timestamp_prediction-v1-16k-offline/` 子目录） | 必填 |
| `--batch_size` | 推理 batch 大小 | 32 |
| `--device` | 推理设备 | `cuda:0` |

输入要求：

- `{input_dir}/list.txt` — tab 分隔四列：`key`、`original_text`、`prefix`、`command`（`#` 开头为注释，由 `move_trigger_cmd.py` 生成）
- `{input_dir}/{key}.wav` — 与 `list.txt` 中 key 对应的原始音频

```bash
# 服务器上运行（Linux）
python s10_data_reflow/labeled\ copy/code/cut_trigger_funasr/trim_trigger_funasr.py \
    --input_dir /path/to/trigger_cmd \
    --output_dir /path/to/trimmed \
    --model_dir /path/to/pretrained_models/funasr \
    --device cuda:0

# 调整 batch size
python .../trim_trigger_funasr.py --input_dir ... --output_dir ... --model_dir ... --batch_size 64

# CPU 推理
python .../trim_trigger_funasr.py --input_dir ... --output_dir ... --model_dir ... --device cpu
```

对齐文本处理：用 `original_text` 去掉中英文逗号和空格后作为对齐文本；裁剪起点 = `timestamp[len(prefix_clean)-1][1]`（即触发词最后一个字的结束时间，毫秒）。

输出到 `{output_dir}/`：

- `{key}.wav` — 裁剪后的音频（只保留命令部分，原采样率保留）
- `result.txt` — 成功裁剪列表，tab 分隔三列：`key`、`command`、`trim_start_ms`（首行 `# key\tcommand\ttrim_start_ms`）
- `failed.txt` — 失败记录，tab 分隔四列：`key`、`original_text`、`prefix`、`reason`（原因含 `wav_not_found` / `empty_result` / `error:xxx` / `prefix_out_of_range` / `trim_too_short`）

控制台输出：模型加载信息、每 200 条或完成时输出进度（`进度: X/Y, 成功: N, 失败: M`），最终汇总成功/失败条数及输出路径。

### s10\_data\_reflow/cut\_trigger\_funasr/apply\_trim\_result（裁剪结果回写到 text.txt / wav.scp）

> 依赖：仅 Python 标准库（`os`, `shutil`, `argparse`）
>
> 前置：已运行 `trim_trigger_funasr.py` 生成 `trimmed/result.txt` 和 `trimmed/failed.txt`

将 `trim_trigger_funasr.py` 的裁剪结果应用回 `labeled copy/` 的 `text.txt` 和 `wav.scp`：

- 成功裁剪的条目：将 `text.txt` 中的原文本更新为裁剪后的 `command`（去掉触发词前缀），同时将 `trimmed/{key}.wav` 拷贝覆盖到 `{base_dir}/wav/{key}.wav`
- 失败的条目：从 `text.txt` 和 `wav.scp` 中删除对应行，并从 `wav/` 中删除对应音频

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--trimmed_dir` | 裁剪结果目录（`trim_trigger_funasr.py` 的 `--output_dir`） | 必填 |
| `--base_dir` | labeled copy 目录（包含 `text.txt`、`wav.scp`、`wav/`） | 必填 |

```bash
python s10_data_reflow/labeled\ copy/code/cut_trigger_funasr/apply_trim_result.py \
    --trimmed_dir /path/to/trimmed \
    --base_dir /path/to/labeled_copy
```

输出（原地修改 `base_dir`）：

- `{base_dir}/text.txt` — 原地更新：成功条目文本改为裁剪后的 command，失败条目删除
- `{base_dir}/wav.scp` — 原地更新：删除失败条目
- `{base_dir}/wav/{key}.wav` — 成功条目替换为裁剪后的音频，失败条目删除

> 注意：本脚本会原地修改标签文件和 wav 目录，不会自动备份。建议运行前手动备份 `text.txt` / `wav.scp` / `wav/`，或先 diff 检查 `result.txt` / `failed.txt`。

### s10\_data\_reflow/split\_train\_dev\_test（按设备ID切分 train/dev/test）

> 依赖：仅 Python 标准库（`re`, `random`, `pathlib`, `collections`）
>
> 前置：`s10_data_reflow/` 目录下已有 `text.txt` 和 `wav.scp`（如由 `apply_reflow.py` 处理后的文件）

将 `s10_data_reflow/` 下的 `text.txt` 和 `wav.scp` 按设备ID维度随机切分为 train / dev / test 三个子集。设备ID 从 key 中提取（`02020adeXXXX` 格式），同一设备的所有数据归入同一个 split，避免数据泄漏。

切分比例：train 90% / dev 5% / test 5%（按设备数量计算，dev 和 test 各至少 1 个设备）。

使用前可修改脚本内的配置变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `TEXT_FILE` | 输入 text.txt 路径 | 脚本同目录下 `text.txt` |
| `WAV_SCP` | 输入 wav.scp 路径 | 脚本同目录下 `wav.scp` |
| `LABEL_ROOT` | 输出根目录（含 train/dev/test 子目录） | `/data1/code/label` |
| `DATASET_NAME` | 数据集名称（用于输出文件名前缀） | `s10_reflow` |
| `SEED` | 随机种子 | `42` |
| `DEV_RATIO` | dev 集设备占比 | `0.05` |
| `TEST_RATIO` | test 集设备占比 | `0.05` |

```bash
python s10_data_reflow/split_train_dev_test.py
```

输出到 `{LABEL_ROOT}/{train,dev,test}/`（默认 `/data1/code/label/{train,dev,test}/`）：

- `s10_reflow_wav.scp` — Kaldi 格式音频路径列表（`<uid> <wav_path>`）
- `s10_reflow_text` — Kaldi 格式文本标注（`<uid> <text>`）

控制台输出：数据加载统计（文本/音频/设备数量）、各 split 的设备数和条目数、写出路径和条目数。

### 08\_split/reflow\_split\_train\_dev\_test（s10\_reflow 按设备ID切分 train/dev/test）

> 依赖：仅 Python 标准库（`re`, `random`, `pathlib`, `collections`）
>
> 前置：脚本同目录下已有 `text.txt` 和 `wav.scp`

与 `s10_data_reflow/split_train_dev_test.py` 功能相同，将 `text.txt` 和 `wav.scp` 按设备ID（`02020adeXXXX` 格式）维度随机切分为 train / dev / test。同一设备的所有数据归入同一个 split，避免数据泄漏。

切分策略：先打乱设备顺序，再按数据量从大到小贪心分配，确保 dev/test 各约占目标比例（默认各 5%），允许 1.3 倍容差。dev 和 test 各至少 1 个设备。

使用前可修改脚本内的配置变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `TEXT_FILE` | 输入 text.txt 路径 | 脚本同目录下 `text.txt` |
| `WAV_SCP` | 输入 wav.scp 路径 | 脚本同目录下 `wav.scp` |
| `LABEL_ROOT` | 输出根目录（含 train/dev/test 子目录） | `/data1/code/label` |
| `DATASET_NAME` | 数据集名称（用于输出文件名前缀） | `s10_reflow` |
| `SEED` | 随机种子 | `42` |
| `DEV_RATIO` | dev 集占比 | `0.05` |
| `TEST_RATIO` | test 集占比 | `0.05` |

```bash
python 08_split/reflow_split_train_dev_test.py
```

输出到 `{LABEL_ROOT}/{train,dev,test}/`（默认 `/data1/code/label/{train,dev,test}/`）：

- `s10_reflow_wav.scp` — Kaldi 格式音频路径列表（`<uid> <wav_path>`）
- `s10_reflow_text` — Kaldi 格式文本标注（`<uid> <text>`）

### 08\_split/sp4\_split\_train\_dev\_test（SP4 按设备ID切分 train/dev/test）

> 依赖：仅 Python 标准库（`random`, `pathlib`, `collections`）
>
> 前置：SP4 labeled 目录下已有 `text.txt` 和 `wav.scp`（如由 `01_prepare_funasr_data_sp4.py` 生成）

将 SP4 数据集的 `text.txt` 和 `wav.scp` 按设备ID维度随机切分为 train / dev / test 三个子集。设备ID 从 key 的第二段提取（如 `20230701140335-K17175903-0-20027-2-1` 中的 `K17175903`），同一设备的所有数据归入同一个 split，避免数据泄漏。

切分策略：先打乱设备顺序，再按数据量从大到小贪心分配，确保 dev/test 各约占目标比例（默认各 5%），允许 1.3 倍容差。dev 和 test 各至少 1 个设备。

使用前需修改脚本内的配置变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `BASE_DIR` | SP4 labeled 数据目录 | `E:\8.音频\03 s10\sp4_data\labeled` |
| `TEXT_FILE` | 输入 text.txt 路径 | `{BASE_DIR}/text.txt` |
| `WAV_SCP` | 输入 wav.scp 路径 | `{BASE_DIR}/wav.scp` |
| `LABEL_ROOT` | 输出根目录（含 train/dev/test 子目录） | `/data1/code/label` |
| `DATASET_NAME` | 数据集名称（用于输出文件名前缀） | `sp4` |
| `SEED` | 随机种子 | `42` |
| `DEV_RATIO` | dev 集占比 | `0.05` |
| `TEST_RATIO` | test 集占比 | `0.05` |

```bash
python 08_split/sp4_split_train_dev_test.py
```

输出到 `{LABEL_ROOT}/{train,dev,test}/`（默认 `/data1/code/label/{train,dev,test}/`）：

- `sp4_wav.scp` — Kaldi 格式音频路径列表（`<uid> <wav_path>`）
- `sp4_text` — Kaldi 格式文本标注（`<uid> <text>`）

控制台输出：数据加载统计（文本/音频/设备数量）、各 split 的设备数和条目数、写出路径和条目数。

### 08\_split/s10\_split\_train\_dev\_test（s10\_reflow 按条目随机切分 train/dev/test）

> 依赖：仅 Python 标准库（`random`, `pathlib`）
>
> 前置：`s10_data_reflow` 目录下已有 `text.txt` 和 `wav.scp`

将 s10\_data\_reflow 的 `text.txt` 和 `wav.scp` 按条目完全随机切分为 train / dev / test 三个子集。与 `reflow_split_train_dev_test.py`（按设备ID切分）不同，本脚本不考虑设备维度，直接对所有条目随机打乱后按比例划分。

切分策略：取 text.txt 和 wav.scp 的 key 交集，使用固定随机种子（seed=42）打乱后按比例切分：train 90% / dev 5% / test 5%。

使用前需修改脚本内的配置变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `BASE_DIR` | s10\_data\_reflow 数据目录 | `E:\8.音频\03 s10\s10_data_reflow` |
| `TEXT_FILE` | 输入 text.txt 路径 | `{BASE_DIR}/text.txt` |
| `WAV_SCP` | 输入 wav.scp 路径 | `{BASE_DIR}/wav.scp` |
| `LABEL_ROOT` | 输出根目录（含 train/dev/test 子目录） | `/data1/label/reflow_label` |
| `DATASET_NAME` | 数据集名称（用于输出文件名前缀） | `s10_reflow` |
| `SEED` | 随机种子 | `42` |
| `DEV_RATIO` | dev 集占比 | `0.05` |
| `TEST_RATIO` | test 集占比 | `0.05` |

```bash
python 08_split/s10_split_train_dev_test.py
```

输出到 `{LABEL_ROOT}/{train,dev,test}/`（默认 `/data1/label/reflow_label/{train,dev,test}/`）：

- `s10_reflow_wav.scp` — Kaldi 格式音频路径列表（`<uid> <wav_path>`）
- `s10_reflow_text` — Kaldi 格式文本标注（`<uid> <text>`）

控制台输出：数据加载统计（文本/音频/交集条目数量）、各 split 的条目数、写出路径和条目数。

### 08\_split/merge\_all\_labels（多来源标签汇总合并）

> 依赖：仅 Python 标准库（`os`, `sys`, `time`, `pathlib`, `collections`）
>
> 前置：各来源标签目录（`augment_label`、`reflow_label`、`speech_label`）下已按 train/dev/test 划分好 `*_wav.scp` 和 `*_text` 文件（如由 `reflow_split_train_dev_test.py`、`sp4_split_train_dev_test.py` 等脚本生成）

将多个来源标签目录中的数据按 train/dev/test 拼接汇总到统一的 `all_label/` 目录。每个 split 下的所有 `*_wav.scp` 和 `*_text` 文件分别合并为一个 `wav.scp` 和 `text`，按第一列 key 去重（保留首次出现的条目）。

性能优化：先从所有 wav.scp 中提取音频文件所在目录，一次性扫描建立文件路径集合（set），后续用 set 查找替代逐个 `Path.exists()`，220 万文件可在秒级完成。

使用前可修改脚本内的配置变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LABEL_ROOT` | 标签根目录 | `/data1/label` |
| `SOURCE_DIRS` | 要合并的来源目录名列表 | `["augment_label", "reflow_label", "speech_label"]` |
| `SPLITS` | 要处理的 split 列表 | `["train", "dev", "test"]` |
| `OUTPUT_DIR` | 输出目录 | `{LABEL_ROOT}/all_label` |

输入目录结构约定：

```
{LABEL_ROOT}/augment_label/{train,dev,test}/*_wav.scp, *_text
{LABEL_ROOT}/reflow_label/{train,dev,test}/*_wav.scp, *_text
{LABEL_ROOT}/speech_label/{train,dev,test}/*_wav.scp, *_text
```

```bash
python 08_split/merge_all_labels.py
```

执行流程：

1. **收集音频目录**：从所有来源的 `*_wav.scp` 中提取音频文件路径的目录部分，汇总为目录集合。
2. **预扫描文件**：一次性扫描所有音频目录，建立完整的文件路径集合（set），用于后续快速查找。扫描过程中逐目录打印进度和文件数量。
3. **合并 wav.scp**：按 split 逐个合并，用预扫描的 set 检查文件存在性。每处理 10 万行打印进度（百分比、速度、缺失数）。
4. **合并 text**：只保留 wav.scp 中有效 key 对应的文本条目。
5. **缺失文件汇总**：所有 split 合并完成后，汇总打印缺失文件路径（最多显示前 50 个）。
6. **总计统计**：打印各 split 和总体的 wav.scp / text 行数，以及总耗时。

合并规则：

1. **wav.scp 合并**：按每行第一列（空格分隔的 key/uid）去重，同一 key 只保留首次出现的行。来源目录和文件均按字典序扫描，排在前面的条目优先保留。合并时通过预扫描的文件集合检查音频文件是否存在，不存在的条目记录并剔除。
2. **text 合并**：同样按 key 去重，且只保留 wav.scp 中存在的 key 对应的文本条目（即 wav 缺失或去重被剔除的 key，text 中也会跳过）。
3. 合并完成后会检查 `wav.scp` 与 `text` 行数是否一致，不一致时输出 ⚠️ 警告。

输出到 `{LABEL_ROOT}/all_label/{train,dev,test}/`：

- `wav.scp` — 合并去重后的 Kaldi 格式音频路径列表（`<uid> <wav_path>`）
- `text` — 合并去重后的 Kaldi 格式文本标注（`<uid> <text>`）

控制台输出：音频目录扫描进度、wav.scp 合并进度（每 10 万行）、每个 split 的合并统计（有效行数、来源文件数、去重数量、缺失剔除数量、text 跳过数量）、缺失文件汇总（前 50 个）、总计统计、总耗时。

### 08\_split/public\_dataset\_gen\_scp\_txt（公开 ASR 数据集转 wav.scp + text）

> 依赖：仅 Python 标准库（`json`, `re`, `random`, `pathlib`, `collections`）
>
> 与 `utils/public_dataset_gen_scp_txt.py` 功能相同，放置在 `08_split/` 目录下便于与其他数据划分脚本统一管理

批量将多种公开中文 ASR 数据集转换为统一的 Kaldi 格式 `wav.scp`（`<utt_id> <wav_absolute_path>`）和 `text`（`<utt_id> <transcript>`），按 train/dev/test 划分输出。脚本内为每个数据集定义了独立的处理函数，运行时依次处理全部数据集。

支持的数据集（8 种）：

| 数据集 | 函数 | 划分方式 | 特殊处理 |
|--------|------|----------|----------|
| aidatatang\_200zh | `gen_aidatatang_200zh()` | 预设 train/dev/test | — |
| data\_aishell (AISHELL-1) | `gen_data_aishell()` | 预设 train/dev/test | 分词格式去空格 |
| data\_aishell\_1 iOS | `gen_data_aishell_1_ios()` | data→train, dev, test | wav.scp 相对路径转绝对路径 |
| MagicData | `gen_magicdata()` | 预设 train/dev/test | TRANS.txt tab 分隔三列；自动跳过表头行；utt\_id 自动去除 `.wav` 后缀 |
| PrimeWords | `gen_primewords()` | 按说话人随机划分 90/5/5 | JSON 转写，分词去空格 |
| ST-CMDS | `gen_stcmds()` | 按说话人随机划分 90/5/5 | 每条一个 .txt 文件 |
| TAL\_ASR | `gen_tal_asr()` | 预设 train/dev/test | — |
| THCHS-30 | `gen_thchs30()` | 预设 train/dev/test | .wav.trn 三行取第一行去空格；自动跟踪路径引用（如 `../data/xxx.wav.trn`）到实际文件 |

```bash
# 运行全部数据集转换（需先修改脚本内各数据集的 root 路径）
python 08_split/public_dataset_gen_scp_txt.py
```

使用前需修改脚本内各数据集函数中的 `root = Path(...)` 为实际数据集路径。全局输出目录由脚本顶部的 `LABEL_ROOT` 变量控制（默认 `/data1/code/label`）。

对于无预设划分的数据集（PrimeWords、ST-CMDS），按说话人随机划分，随机种子固定为 42，保证可复现。

输出到 `{LABEL_ROOT}/{split}/`（即 `/data1/code/label/{train,dev,test}/`）：

- `{dataset_name}_wav.scp` — Kaldi 格式音频路径列表（`<utt_id> <绝对路径>`）
- `{dataset_name}_text` — Kaldi 格式文本标注（`<utt_id> <transcript>`）

> 本脚本也有对应的 Jupyter Notebook 版本：`08_split/public_dataset_gen_scp_txt.ipynb`，功能相同，适合交互式逐数据集运行。

### analyze\_audio\_distribution（音频音量/SNR/距离分布分析）

> 依赖：Python 标准库（`wave`, `struct`, `math`, `csv`, `concurrent.futures`）；可选 `matplotlib`（生成分布图）
>
> 无前置依赖，可独立运行

扫描指定目录下的 wav 文件，统计每条音频的 RMS 音量（dBFS）、SNR（dB，基于帧能量排序估算）和推测录音距离（基于 RMS + SNR 的简单模型），输出控制台统计摘要、文本直方图、分布图（PNG）和详细数据（CSV）。支持多进程并行分析。

距离推测模型：综合 RMS（权重 0.6）和 SNR（权重 0.4）评分，将录音分为 4 档：`<0.3m`（贴近麦克风）、`0.3-1m`（近场）、`1-3m`（中距离）、`>3m`（远场）。

```bash
# 默认分析 sp4_data/labeled_check/wav 目录
python analyze_audio_distribution.py

# 指定 wav 目录
python analyze_audio_distribution.py --wav-dir "E:\8.音频\03 s10\s10_data_reflow\labeled\wav"

# 8 进程加速
python analyze_audio_distribution.py --workers 8

# 只分析前 1000 个文件（调试用）
python analyze_audio_distribution.py --limit 1000
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--wav-dir` | wav 文件目录（递归扫描） | `E:\8.音频\03 s10\sp4_data\labeled_check\wav` |
| `--workers` | 并行进程数 | `4` |
| `--limit` | 只分析前 N 个文件（调试用） | 全部 |

输出到 `{wav-dir}/../audio_analysis/`：

- `audio_analysis.csv` — 逐条详细数据（uid / path / duration / sample\_rate / rms / rms\_dbfs / peak / peak\_dbfs / snr\_db / est\_distance / error）
- `audio_distribution.png` — 分布图（RMS 直方图、SNR 直方图、RMS vs SNR 散点图、距离分布饼图），需安装 `matplotlib`

控制台输出：

- 统计摘要（均值、范围、P5/P25/P50/P75/P95）
- 推测距离分布（各档数量和占比）
- 文本直方图（RMS 和 SNR 的分桶分布）

> 如需生成分布图，请安装 matplotlib：`pip install matplotlib`

### audio\_review\_server

> 依赖：`pip install flask`

```bash
python audio_review_server.py
python audio_review_server.py --wav-dir "E:\8.音频\03 s10\s10_data_reflow\unlabeled" --label-file ""
python audio_review_server.py --port 8080
```

浏览器打开 `http://127.0.0.1:5000`

打分快捷键：

| 按键 | 分数 | 含义 |
|------|------|------|
| Q | -1 | 信号很差 |
| 0 | 0 | 标签错误 |
| 1 | 1 | 萤石字段 |
| 2 | 2 | 多人说话 |
| 3 | 3 | 信号一般 |
| 4 | 4 | 信号很好 |

导航：`←` / `→` 前后翻，`S` 跳到未打分

打分记录保存在 `review_scores.jsonl`。

### gen\_hotwords

> 依赖：`pip install jieba`

```bash
python gen_hotwords.py
python gen_hotwords.py --top 500 --min-freq 5
```

输出到 `s10_data_reflow/hotwords/`：

- `hotwords.txt` — 纯热词表（可直接用于 FunASR / SeACo hotword 参数）
- `hotwords_detail.txt` — 带频次的详细版
- `frequent_commands.txt` — 高频完整指令
- `contact_names.txt` — 联系人名

### slim\_model\_for\_inference

> 依赖：`pip install torch`

精简 FunASR SeacoParaformer 训练 checkpoint 用于推理部署。移除优化器状态、调度器状态等训练相关数据，只保留模型权重，大幅减小文件体积，加快加载速度。

```bash
# 直接修改脚本底部的 input_file / output_file 路径后运行
python slim_model_for_inference.py
```

也可作为模块导入使用：

```python
from slim_model_for_inference import slim_checkpoint

slim_checkpoint("path/to/model.pt.ep23", "path/to/model.pt.ep23_slim")
```

参数说明（`slim_checkpoint` 函数）：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input_path` | 原始 checkpoint 文件路径 | （必填） |
| `output_path` | 精简后输出路径 | 同目录生成 `_slim` 后缀文件 |

处理逻辑：

1. 加载 checkpoint 到 CPU
2. 分析 checkpoint 结构（打印所有顶层 key 及类型）
3. 提取模型权重：
   - 优先查找 `model_state_dict` / `state_dict` / `model` key
   - 若整个 dict 全为 tensor，视为已是 state\_dict
   - 否则移除已知训练相关 key（`optimizer`、`scheduler`、`epoch`、`step` 等）
4. 保存精简后的权重文件

输出：

- `{input_path}_slim`（或指定的 `output_path`）— 仅含模型权重的 PyTorch checkpoint

控制台输出精简前后的文件大小和减少比例。

### json\_jsonl\_convert

> 依赖：仅 Python 标准库（`json`, `sys`, `pathlib`）

JSON 与 JSONL 格式互转工具。根据输入文件后缀自动判断转换方向，支持自动推断输出路径（换后缀）。

```bash
# JSON → JSONL
python json_jsonl_convert.py input.json output.jsonl

# JSONL → JSON
python json_jsonl_convert.py input.jsonl output.json

# 自动推断输出路径（换后缀）
python json_jsonl_convert.py input.json      # → input.jsonl
python json_jsonl_convert.py input.jsonl     # → input.json
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| 第 1 个参数 | 输入文件路径（`.json` 或 `.jsonl`） | （必填） |
| 第 2 个参数 | 输出文件路径 | 自动推断（换后缀） |

转换规则：

- `.json` → `.jsonl`：读取 JSON 数组（或单个对象），逐条写为 JSONL
- `.jsonl` → `.json`：逐行读取 JSONL，合并为 JSON 数组（`indent=2` 格式化输出）

输出：指定路径或与输入同目录的换后缀文件。

### fix\_scp\_paths

> 依赖：仅 Python 标准库（`argparse`, `sys`）

批量替换 wav.scp 中的音频路径前缀。适用于数据集迁移到新目录或新服务器后，需要统一更新 scp 文件中的路径。支持预览模式和输出到新文件。

```bash
# 替换路径前缀
python fix_scp_paths.py wav.scp --old "E:\\old\\path" --new "E:\\new\\path"

# 预览模式（不写入，只显示变更）
python fix_scp_paths.py wav.scp --old "/root/data" --new "/data/asr" --dry-run

# 输出到新文件（不覆盖原文件）
python fix_scp_paths.py wav.scp --old "/old" --new "/new" --output wav_fixed.scp
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `scp`（位置参数） | wav.scp 文件路径 | （必填） |
| `--old` | 要替换的旧路径子串 | （必填） |
| `--new` | 替换为的新路径子串 | （必填） |
| `--output` | 输出文件路径 | 覆盖原文件 |
| `--dry-run` | 预览模式，不写入文件（展示前 5 条变更） | — |

输出：原地修改 scp 文件（或写入 `--output` 指定的新文件）。

### fix\_unlabeled\_paths

> 依赖：仅 Python 标准库（`re`, `argparse`, `pathlib`）

修复结果文件中的 unlabeled 音频路径：将 `.../unlabeled/xxx.wav` 替换为 `.../unlabeled/wav/xxx.wav`。扫描预标注输出目录下的 `wav.scp` 和 `pretag.jsonl`，使用负向前瞻正则确保已正确的路径不会被重复替换。

```bash
# 默认修复 /data/share/dataset/s10_data_reflow 下的文件
python fix_unlabeled_paths.py

# 指定数据根目录
python fix_unlabeled_paths.py --base-dir /path/to/s10_data_reflow

# 只预览不修改
python fix_unlabeled_paths.py --dry-run
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--base-dir` | 数据根目录 | `/data/share/dataset/s10_data_reflow` |
| `--dry-run` | 只预览不修改 | — |

扫描范围：

| 目录 | 文件 |
|------|------|
| `firered_pretag_output/` | `wav.scp`, `pretag.jsonl`（递归） |
| `qwen_pretag_output/` | `wav.scp`, `pretag.jsonl`（递归） |
| `compare_output/` | `wav.scp`, `pretag.jsonl`（递归） |
| `unlabeled/` | `wav.scp` |

输出：原地修复上述文件中的路径（`--dry-run` 时仅打印统计，不写入）。

### gen\_wav\_scp

> 依赖：仅 Python 标准库（`os`, `argparse`, `pathlib`）

为指定目录下的所有 wav 文件生成 Kaldi 格式的 `wav.scp`（每行 `<uid> <绝对路径>`），uid 取文件名（不含扩展名），按 uid 字典序排列。

```bash
# 默认扫描 unlabeled/ 目录，输出到该目录下的 wav.scp
python gen_wav_scp.py

# 指定 wav 目录
python gen_wav_scp.py --wav-dir /path/to/wav

# 指定输出路径
python gen_wav_scp.py --wav-dir /path/to/wav --output /path/to/wav.scp
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--wav-dir` | wav 文件目录 | `s10_data_reflow/unlabeled` |
| `--output` | 输出 wav.scp 路径 | `{wav-dir}/wav.scp` |

输出：

- `wav.scp` — Kaldi 格式音频路径列表（`uid 绝对路径`，按 uid 排序）

### merge\_scp

> 依赖：仅 Python 标准库（`argparse`, `os`）

合并多个 wav.scp 和 text.txt 文件，按第一列 key 去重（保留首次出现的条目）。适用于将原始数据与增强数据的 scp/text 合并为统一的训练文件。

```bash
# 合并原始 + 增强的 wav.scp 和 text.txt
python merge_scp.py --scp original/wav.scp augment/wav_train.scp --text original/text.txt augment/text_train.txt --output_dir combined/

# 只合并 wav.scp
python merge_scp.py --scp a.scp b.scp c.scp --output_dir combined/

# 只合并 text.txt
python merge_scp.py --text a.txt b.txt --output_dir combined/
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--scp` | 要合并的 wav.scp 文件列表（空格分隔多个路径） | — |
| `--text` | 要合并的 text.txt 文件列表（空格分隔多个路径） | — |
| `--output_dir` | 输出目录 | （必填） |

去重规则：按每行第一列（空格分隔的 key/uid）去重，同一 key 只保留首次出现的行。输入文件的顺序决定优先级——排在前面的文件中的条目优先保留。

输出到 `{output_dir}/`：

- `wav.scp` — 合并去重后的 Kaldi 格式音频路径列表（仅当指定 `--scp` 时生成）
- `text.txt` — 合并去重后的 Kaldi 格式文本标注（仅当指定 `--text` 时生成）

### download\_magicdata（MagicData-Read 数据集下载，ModelScope 源）

> 依赖：`pip install modelscope`
>
> 对应 shell 版本：`download_magicdata.sh`（通过 curl 从 OpenSLR 下载）

通过 ModelScope CLI 下载 MagicData-Read 数据集。脚本内预设了多个候选 dataset ID，依次尝试下载，首个成功即停止。适用于国内网络环境下无法直接访问 OpenSLR 的场景。

```bash
python download_magicdata.py
```

脚本内配置变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `TARGET_DIR` | 下载目标目录 | `D:\asr_dataset\magicdata` |
| `DATASET_IDS` | ModelScope 上的候选 dataset ID 列表 | `["speech_asr/speech_asr_magicdata_zhcn", "MagicData/MagicData-RAMC"]` |

工作流程：

1. 创建目标目录（如不存在）
2. 依次尝试 `DATASET_IDS` 中的每个 dataset ID，调用 `modelscope download --dataset <id> --local_dir <TARGET_DIR>`
3. 首个下载成功的 ID 即停止；全部失败时提示手动到 ModelScope 搜索确认

输出到 `D:\asr_dataset\magicdata\`：

- ModelScope 下载的数据集文件（具体内容取决于 dataset ID 对应的数据集结构）

> 如需从 OpenSLR 源下载（需科学上网），可使用 shell 版本 `download_magicdata.sh`，通过 curl 断点续传下载 test/dev/train 三个 tar.gz 包。

### utils/public\_dataset\_gen\_scp\_txt（公开数据集转 wav.scp + text）

> 依赖：仅 Python 标准库（`json`, `re`, `random`, `pathlib`, `collections`）

批量将多种公开中文 ASR 数据集转换为统一的 Kaldi 格式 `wav.scp`（`<utt_id> <wav_absolute_path>`）和 `text`（`<utt_id> <transcript>`），按 train/dev/test 划分输出。脚本内为每个数据集定义了独立的处理函数，运行时依次处理全部数据集。

支持的数据集（8 种）：

| 数据集 | 函数 | 划分方式 | 特殊处理 |
|--------|------|----------|----------|
| aidatatang\_200zh | `gen_aidatatang_200zh()` | 预设 train/dev/test | — |
| data\_aishell (AISHELL-1) | `gen_data_aishell()` | 预设 train/dev/test | 分词格式去空格 |
| data\_aishell\_1 iOS | `gen_data_aishell_1_ios()` | data→train, dev, test | wav.scp 相对路径转绝对路径 |
| MagicData | `gen_magicdata()` | 预设 train/dev/test | TRANS.txt tab 分隔三列；自动跳过表头行；utt\_id 自动去除 `.wav` 后缀 |
| PrimeWords | `gen_primewords()` | 按说话人随机划分 90/5/5 | JSON 转写，分词去空格 |
| ST-CMDS | `gen_stcmds()` | 按说话人随机划分 90/5/5 | 每条一个 .txt 文件 |
| TAL\_ASR | `gen_tal_asr()` | 预设 train/dev/test | — |
| THCHS-30 | `gen_thchs30()` | 预设 train/dev/test | .wav.trn 三行取第一行去空格；自动跟踪路径引用（如 `../data/xxx.wav.trn`）到实际文件 |

```bash
# 运行全部数据集转换（需先修改脚本内各数据集的 root 路径）
python utils/public_dataset_gen_scp_txt.py
```

使用前需修改脚本内各数据集函数中的 `root = Path(...)` 为实际数据集路径。全局输出目录由脚本顶部的 `LABEL_ROOT` 变量控制（默认 `/data1/code/label`）。

对于无预设划分的数据集（PrimeWords、ST-CMDS），按说话人随机划分，随机种子固定为 42，保证可复现。

输出到 `{LABEL_ROOT}/{split}/`：

- `{dataset_name}_wav.scp` — Kaldi 格式音频路径列表（`<utt_id> <绝对路径>`）
- `{dataset_name}_text` — Kaldi 格式文本标注（`<utt_id> <transcript>`）

> 本脚本也有对应的 Jupyter Notebook 版本：`utils/public_dataset_gen_scp_txt.ipynb`，功能相同，适合交互式逐数据集运行。

### utils/genetate\_scp（生成 wav.scp）

> 依赖：仅 Python 标准库（`os`, `argparse`, `pathlib`）

为指定目录下的所有 `.wav` 文件生成 Kaldi 格式的 `wav.scp`（每行 `<uid> <绝对路径>`），uid 取文件名（不含扩展名），按 uid 字典序排列。适用于快速为未标注音频目录生成 scp 文件，供后续评估或预标注脚本使用。

```bash
# 默认扫描 s10_data_reflow/unlabeled/wav/ 目录，输出到 s10_data_reflow/unlabeled/wav.scp
python utils/genetate_scp.py

# 指定 wav 目录
python utils/genetate_scp.py --wav-dir /path/to/wav

# 指定输出路径
python utils/genetate_scp.py --wav-dir /path/to/wav --output /path/to/wav.scp
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--wav-dir` | wav 文件目录（非递归，仅扫描当前目录下的 `.wav` 文件） | `/data/share/dataset/s10_data_reflow/unlabeled/wav` |
| `--output` | 输出 wav.scp 路径 | `/data/share/dataset/s10_data_reflow/unlabeled/wav.scp` |

输出：

- `wav.scp` — Kaldi 格式音频路径列表（`uid 绝对路径`，按 uid 字典序排列），自动创建输出目录

输出格式示例：

```
20230206214908-J67780092-0-20027-2-1 /data/share/dataset/s10_data_reflow/unlabeled/wav/20230206214908-J67780092-0-20027-2-1.wav
20230207082414-J67780092-0-20027-2-1 /data/share/dataset/s10_data_reflow/unlabeled/wav/20230207082414-J67780092-0-20027-2-1.wav
```

> 注：根目录下的 `gen_wav_scp.py` 功能相同，默认路径不同（指向 `s10_data_reflow/unlabeled`）。本脚本位于 `utils/` 目录，默认路径指向 `/data/share/dataset/s10_data_reflow/unlabeled/wav`，适合在服务器端直接使用。

### 11\_analyse/analyse\_eval\_detail（评估结果统计与错误样本导出）

> 依赖：仅 Python 标准库（`json`）
>
> 前置：已有 `eval_detail.jsonl` 文件（由 `seaco_eval.py` / `batch_eval.py` 等评估脚本生成）

统计 `eval_detail.jsonl` 中的整体 CER（字错误率）和句准确率，并将所有识别错误的样本按 CER 降序排列输出到可读的 txt 文件，方便人工查看 bad cases。

```bash
python 11_analyse/analyse_eval_detail.py
```

脚本内配置变量（直接修改脚本）：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `input_file` | 输入的 eval_detail.jsonl 路径 | `e:\8.音频\03 s10\11_analyse\eval_detail2.jsonl` |
| `output_file` | 输出的错误样本 txt 路径 | `e:\8.音频\03 s10\11_analyse\eval_errors2.txt` |

输入文件格式（`eval_detail.jsonl`，每行一个 JSON）：

```json
{"uid": "xxx", "ref": "参考文本", "hyp": "识别文本", "cer": 0.125, "sub": 1, "ins": 0, "del": 0, "ref_len": 8, "correct": false}
```

输出到 `11_analyse/eval_errors2.txt`：

- 统计摘要区：总样本数、正确/错误样本数、句准确率、总参考字数、总替换/插入/删除数、整体 CER
- 错误样本详情区：按 CER 降序排列，每条包含 uid、CER 百分比、标签文本、识别文本

输出格式示例：

```
============================================================
ASR 模型评测结果统计
============================================================
总样本数: 1000
正确样本数: 600
错误样本数: 400
句准确率 (Sentence Accuracy): 60.00%
总参考字数: 8500
总替换数: 320
总插入数: 45
总删除数: 30
整体 CER: 4.65%
============================================================

错误样本详情 (共 400 条，按 CER 降序排列):
------------------------------------------------------------

[AT0016W0440_aug0] CER=100%
标签: 可能至少要分五份
识别: 美 女 联 系 不 到 了

[AT0011W0020_aug0] CER=117%
标签: 是最好的朋友
识别: 我 想 找 个 童 一 会
```

### 11\_analyse/calc\_command\_recall（指令召回率对比分析）

> 依赖：仅 Python 标准库（`json`）
>
> 前置：已有两个模型的 `eval_detail.jsonl` 或 `eval_detail1.jsonl` 文件（由 `seaco_eval.py` / `batch_eval.py` 等评估脚本生成）

计算两个模型的指令召回率并进行对比分析。指令召回率定义为：标签文本（`ref_norm`）作为子串出现在识别结果（`hyp_norm`）中的样本比例。即使识别结果包含多余内容（如尾部噪声、重复），只要完整包含了标签指令文本，就视为"召回"。

```bash
python 11_analyse/calc_command_recall.py
```

脚本内配置变量（直接修改脚本）：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `path1` | 模型1 的 eval_detail.jsonl 路径 | `e:\8.音频\03 s10\11_analyse\eval_detail.jsonl` |
| `path2` | 模型2 的 eval_detail.jsonl 路径 | `e:\8.音频\03 s10\11_analyse\eval_detail1.jsonl` |

输入文件格式（`eval_detail.jsonl`，每行一个 JSON）：

```json
{"uid": "xxx", "ref_norm": "打开客厅灯", "hyp_norm": "打开客厅灯你好萤石", "cer": 0.8, ...}
```

输出（直接打印到控制台，无输出文件）：

- 两个模型各自的指令召回率统计（总样本数、召回数、未召回数、召回率）
- 对比结果（两模型召回率差异）
- 未召回样本按 CER 分段统计（CER=0 / 0~25% / 25~50% / 50~100% / >100%）
- 差异分析：模型1未召回但模型2召回的样本数（模型2优势）、模型2未召回但模型1召回的样本数（模型1优势）
- 各方向差异样本详情（前30条，按 CER 降序）

适用场景：

- 对比新旧模型在"指令是否被正确识别出来"维度的表现差异
- 分析 CER 指标之外的实用性指标（即使 CER 不为 0，只要指令被完整包含就算召回）
- 找出两个模型各自的优势/劣势样本

### 11\_analyse/analyse\_all\_models（多模型错误特征对比分析）

> 依赖：仅 Python 标准库（`json`, `os`, `pathlib`, `collections`）
>
> 前置：已通过 `eval_on_testset.py` 或 `batch_eval_models.py` 等评估脚本在 `eval_output` 下生成各模型的 `eval_detail.jsonl` 文件

对 `eval_output` 目录下 3 个版本 × 2 个权重 = 6 个模型的评估结果进行横向对比分析。重点关注：各数据集类别的错误分布、指令召回率对比、错误类型（替换/插入/删除）分布差异。

分析覆盖的数据集：

| 数据集 | 路径（相对于模型评估目录） | 说明 |
|--------|--------------------------|------|
| `s10_reflow` | `reflow_label/s10_reflow/test` | S10 回流数据测试集 |
| `sp4_reflow` | `reflow_label/sp4_reflow/test` | SP4 回流数据测试集（核心业务数据集） |
| `s10_augment` | `reflow_label/s10_augment/test` | S10 增强数据测试集 |
| `12_test` | `testset_results/` 下各模型子目录 | 12\_test 独立测试集 |

```bash
python 11_analyse/analyse_all_models.py
```

脚本内配置变量（直接修改脚本）：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `eval_base` | 评估结果根目录 | `e:\8.音频\03 s10\eval_output` |
| `models` | 模型名称→路径映射（6 个模型） | `v0.1.0/best`, `v0.1.0/ep16`, `v0.2.0/best`, `v0.2.0/ep14`, `v0.3.0/best`, `v0.3.0/ep27` |

目录结构约定：

```
eval_output/
├── v0.1.0/model.pt.best/reflow_label/s10_reflow/test/eval_detail.jsonl
├── v0.1.0/model.pt.ep16.4000/reflow_label/sp4_reflow/test/eval_detail.jsonl
├── v0.2.0/model.pt.best/...
├── v0.3.0/model.pt.ep27.4000/...
└── testset_results/
    ├── v0.1.0_model.pt.best/eval_detail.jsonl
    ├── v0.2.0_model.pt.best/eval_detail.jsonl
    └── ...
```

输出（直接打印到控制台，无输出文件）：

- 全数据集×全模型对比表格（指令召回率、CER、句准确率、替换/插入/删除数）
- `sp4_reflow/test` 详细对比（核心业务数据集，含未召回数）
- `s10_augment/test` 分析（CER 极高的数据集）
- `s10_reflow/test` 分析
- `12_test` 测试集对比（从 `testset_results/` 读取）

指标说明：

| 指标 | 定义 |
|------|------|
| 指令召回率 | `ref_norm` 作为子串出现在 `hyp_norm` 中的比例 |
| CER | 字符错误率 = (替换 + 插入 + 删除) / 参考字符总数 |
| 句准确率 | CER=0 的样本占比 |
| 未召回 | 指令未被完整识别出的样本数 |

### 11\_analyse/analyse\_testset（testset\_results 多模型错误深度分析）

> 依赖：仅 Python 标准库（`json`, `os`, `collections`）
>
> 前置：`eval_output/testset_results/` 下各模型子目录中已有 `eval_detail.jsonl`（由 `eval_on_testset.py` 生成）

针对 `testset_results` 下 6 个模型（3 版本 × 2 权重）的评估结果进行错误深度分析。重点关注指令召回率（标签文本是否作为子串出现在识别结果中）、错误分类分布、跨模型未召回对比、高频混淆对提取。

```bash
python 11_analyse/analyse_testset.py
```

脚本内配置变量（直接修改脚本）：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `base` | testset\_results 根目录 | `e:\8.音频\03 s10\eval_output\testset_results` |
| `models` | 模型显示名→子目录名映射（6 个模型） | `v0.1.0/best` → `v0.1.0_model.pt.best`, ..., `v0.3.0/ep27` → `v0.3.0_model.pt.ep27.4000` |

目录结构约定：

```
eval_output/testset_results/
├── v0.1.0_model.pt.best/eval_detail.jsonl
├── v0.1.0_model.pt.ep16.4000/eval_detail.jsonl
├── v0.2.0_model.pt.best/eval_detail.jsonl
├── v0.2.0_model.pt.ep14.4000/eval_detail.jsonl
├── v0.3.0_model.pt.best/eval_detail.jsonl
└── v0.3.0_model.pt.ep27.4000/eval_detail.jsonl
```

输出（直接打印到控制台，无输出文件）：

1. **总览表格** — 各模型的指令召回率、CER、句准确率、Sub/Ins/Del 数、未召回数
2. **错误分类分布** — 每个模型的错误按以下类别统计：
   - 唤醒词残留（识别结果中包含"你好萤石"等唤醒词）
   - 多识别(环境音)（指令被召回但有大量多余内容）
   - 多识别(轻微)（指令被召回但有少量多余内容）
   - 漏识别(纯删除)（仅有删除错误）
   - 替换错误（有替换但无插入）
   - 指令淹没(大量插入)（插入数超过参考长度）
   - 混合错误（替换+插入同时存在）
3. **跨模型未召回对比** — 所有模型都未召回的"硬骨头"样本（前 30 条）
4. **v0.3.0/best 独有未召回** — v0.3.0 未召回但 v0.2.0 能召回的样本（前 20 条）
5. **v0.2.0/best 独有未召回** — v0.2.0 未召回但 v0.3.0 能召回的样本（前 20 条）
6. **高频混淆对** — v0.3.0/best 中出现 2 次以上的 ref→hyp 替换模式（前 30 对）

指标说明：

| 指标 | 定义 |
|------|------|
| 指令召回率 | `ref_norm` 作为子串出现在 `hyp_norm` 中的比例 |
| CER | 字符错误率 = (替换 + 插入 + 删除) / 参考字符总数 |
| 句准确率 | CER=0 的样本占比 |
| 未召回 | 指令未被完整识别出的样本数 |

### 11\_analyse/analyse\_testset\_strip\_wakeword（去唤醒词后重新评测）

> 依赖：仅 Python 标准库（`json`, `os`, `re`, `collections`）
>
> 前置：`eval_output/testset_results/` 下各模型子目录中已有 `eval_detail.jsonl`（由 `eval_on_testset.py` 生成）

去除识别结果（`hyp_norm`）开头和结尾的唤醒词"你好萤石"及其常见变体后，重新计算 CER 和指令召回率。用于量化唤醒词残留对评测指标的影响，帮助判断模型真实的指令识别能力。

唤醒词变体列表：`你好萤石`、`你好萤`、`你好石`、`你好刘石`、`你好伊思`、`你好游石`、`你好萤乐`、`你好萤哲`、`萤石`、`石`、`控好萤石`、`控楼萤石`

处理逻辑：循环去除 `hyp_norm` 开头和结尾匹配的唤醒词变体（贪心匹配，直到无变化为止），然后用动态规划重新计算编辑距离（Sub/Ins/Del）和 CER。

```bash
python 11_analyse/analyse_testset_strip_wakeword.py
```

脚本内配置变量（直接修改脚本）：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `base` | testset\_results 根目录 | `e:\8.音频\03 s10\eval_output\testset_results` |
| `models` | 模型显示名→子目录名映射（6 个模型） | `v0.1.0/best` → `v0.1.0_model.pt.best`, ..., `v0.3.0/ep27` → `v0.3.0_model.pt.ep27.4000` |

输出（直接打印到控制台，无输出文件）：

1. **原始 vs 去唤醒词对比表** — 每个模型两行（原始 / 去唤醒词），展示指令召回率、CER、句准确率、Sub/Ins/Del 数
2. **改善幅度对比** — 各模型去唤醒词后的召回率提升、CER 降低、准确率提升、Ins 减少量
3. **去唤醒词后最终排名** — 按指令召回率降序、CER 升序排列

指标说明：

| 指标 | 定义 |
|------|------|
| 指令召回率 | `ref_norm` 作为子串出现在处理后 `hyp` 中的比例 |
| CER | 字符错误率 = (替换 + 插入 + 删除) / 参考字符总数 |
| 句准确率 | CER=0 的样本占比 |

### 11\_analyse/analyse\_testset\_strip\_wakeword\_v2（去唤醒词后重新评测 v2 — 双向去除）

> 依赖：仅 Python 标准库（`json`, `os`, `re`）
>
> 前置：`eval_output/testset_results/` 下各模型子目录中已有 `eval_detail.jsonl`（由 `eval_on_testset.py` 生成）

与 v1 的区别：v1 仅去除识别结果（`hyp_norm`）中的唤醒词；**v2 对标签（`ref_norm`）和识别结果（`hyp_norm`）都做同样的唤醒词去除处理**，更准确地反映模型在纯指令部分的识别能力。此外 v2 扩展了唤醒词变体列表（新增 `平好萤石`、`你要萤石`、`你很萤石`、`杨萤石`、`样萤石`、`营养萤石`、`怎样萤石`、`你奥伊石` 等），并区分了开头变体和结尾变体（结尾更宽松）。

唤醒词变体列表（开头）：`你好萤石`、`你好萤`、`你好石`、`你好刘石`、`你好伊思`、`你好游石`、`你好萤乐`、`你好萤哲`、`你好李石`、`控好萤石`、`控楼萤石`、`萤萤石`、`你好萤萤石`、`好萤石`、`平好萤石`、`你要萤石`、`你很萤石`、`杨萤石`、`样萤石`、`营养萤石`

唤醒词变体列表（结尾，额外包含）：`怎样萤石`、`你奥伊石`

处理逻辑：

1. 对每条样本的 `ref_norm` 和 `hyp_norm` 分别循环去除开头/结尾匹配的唤醒词变体（按长度降序贪心匹配，最多 5 轮迭代）
2. 若标签去除唤醒词后为空，则跳过该条（不计入有效条数）
3. 用动态规划重新计算编辑距离（Sub/Ins/Del）和 CER
4. 重新计算指令召回率（去唤醒词后的 `ref_stripped` 是否为 `hyp_stripped` 的子串）

```bash
python 11_analyse/analyse_testset_strip_wakeword_v2.py
```

脚本内配置变量（直接修改脚本）：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `base` | testset\_results 根目录 | `e:\8.音频\03 s10\eval_output\testset_results` |
| `models` | 模型显示名→子目录名映射（6 个模型） | `v0.1.0/best` → `v0.1.0_model.pt.best`, ..., `v0.3.0/ep27` → `v0.3.0_model.pt.ep27.4000` |
| `WAKEWORDS` | 开头唤醒词变体列表（按长度降序） | 见脚本 |
| `WAKEWORDS_END` | 结尾唤醒词变体列表（按长度降序，比开头更宽松） | 见脚本 |

输出（直接打印到控制台，无输出文件）：

1. **原始 vs 去唤醒词对比表** — 每个模型两行（原始 / 去唤醒词），展示召回率、CER、准确率、Sub/Ins/Del、有效条数、变更条数
2. **改善幅度对比** — 各模型去唤醒词后的召回率变化、CER 降低、准确率提升、Ins 减少、Sub 变化
3. **去唤醒词后最终排名（按指令召回率降序）**
4. **去唤醒词后最终排名（按 CER 升序）**

指标说明：

| 指标 | 定义 |
|------|------|
| 指令召回率 | 去唤醒词后的 `ref_stripped` 作为子串出现在 `hyp_stripped` 中的比例 |
| CER | 字符错误率 = (替换 + 插入 + 删除) / 参考字符总数（去唤醒词后重新计算） |
| 句准确率 | CER=0 的样本占比 |
| 变更条数 | ref 或 hyp 经唤醒词去除后发生变化的样本数 |

### 11\_analyse/compare\_all\_models\_recall（四模型指令召回率对比）

> 依赖：仅 Python 标准库（`json`, `os`, `pathlib`）
>
> 前置：`eval_output/` 下各模型评估目录中已有 `eval_detail.jsonl`（由 `batch_eval_models.py` 或 `seaco_eval.py` 生成）

对比 4 个模型（seaco 基线、online 线上模型、v0.2.0、v0.3.0）在业务数据集（s10\_reflow / sp4\_reflow / s10\_augment）test 集上的指令召回率。指令召回率定义为：标签文本（`ref_norm`）作为子串出现在识别结果（`hyp_norm`）中的比例。同时统计 CER、句准确率、Sub/Ins/Del 错误类型分布，并计算各版本相对 online 线上模型的改进幅度。

```bash
python 11_analyse/compare_all_models_recall.py
```

脚本内配置变量（直接修改脚本）：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `eval_base` | 评估结果根目录 | `e:\8.音频\03 s10\eval_output` |
| `models` | 模型名→数据集路径映射（4 个模型 × 3 个数据集） | 见脚本 `main()` 函数 |

数据路径约定：

```
eval_output/
├── base/reflow_eval_result/seaco/{s10_reflow,sp4_reflow,s10_augment}/test/eval_detail.jsonl
├── base/reflow_eval_result/online_model/{s10_reflow,sp4_reflow,s10_augment}/test/eval_detail.jsonl
├── v0.2.0/model.pt.best/reflow_label/{s10_reflow,sp4_reflow,s10_augment}/test/eval_detail.jsonl
└── v0.3.0/model.pt.best/reflow_label/{s10_reflow,sp4_reflow,s10_augment}/test/eval_detail.jsonl
```

输出（直接打印到控制台，无输出文件）：

1. **数据路径检查** — 列出各模型×数据集的 jsonl 路径及是否存在
2. **分数据集详细对比** — 每个数据集的 4 模型指令召回率、CER、句准确率、Sub/Ins/Del、未召回数
3. **指令召回率汇总对比表** — 4 模型 × 3 数据集的召回率矩阵
4. **vs online 改进幅度** — v0.2.0 和 v0.3.0 相对 online 线上模型的召回率差值

指标说明：

| 指标 | 定义 |
|------|------|
| 指令召回率 | `ref_norm` 作为子串出现在 `hyp_norm` 中的样本占比 |
| CER | 字符错误率 = (Sub + Ins + Del) / 参考字符总数 |
| 句准确率 | `correct=true` 的样本占比 |
| 未召回 | `ref_norm` 未出现在 `hyp_norm` 中的样本数 |

### 11\_analyse/export\_v3\_v4\_errors（v0.3.0 vs v0.4.0 错误样本导出）

> 依赖：仅 Python 标准库（`json`, `os`）
>
> 前置：`eval_output/` 下 v0.3.0 和 v0.4.0 的评估目录中已有 `eval_detail.jsonl`（由 `batch_eval_models.py` 生成）

导出 v0.3.0 和 v0.4.0 两个模型版本在业务数据集（s10\_reflow/test、sp4\_reflow/test）上的所有错误样本，并列展示两个版本的识别结果差异。按"两者都错"、"仅 v0.3.0 错（v0.4.0 正确）"、"仅 v0.4.0 错（v0.3.0 正确）"三部分分类输出，每条样本标注 CER 和指令召回状态（标签文本是否作为子串出现在识别结果中）。

```bash
python 11_analyse/export_v3_v4_errors.py
```

脚本内配置变量（直接修改脚本）：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `base` | 评估结果根目录 | `e:\8.音频\03 s10\eval_output` |
| `datasets` | 数据集名→eval\_detail.jsonl 相对路径映射 | `s10_reflow`, `sp4_reflow` |

数据路径约定：

```
eval_output/
├── v0.3.0/model.pt.best/reflow_label/s10_reflow/test/eval_detail.jsonl
├── v0.3.0/model.pt.best/reflow_label/sp4_reflow/test/eval_detail.jsonl
├── v0.4.0/model.pt.best/reflow_label/s10_reflow/test/eval_detail.jsonl
└── v0.4.0/model.pt.best/reflow_label/sp4_reflow/test/eval_detail.jsonl
```

输出到 `11_analyse/`：

- `errors_v3_v4_s10_reflow.txt` — s10\_reflow 数据集错误对比
- `errors_v3_v4_sp4_reflow.txt` — sp4\_reflow 数据集错误对比

输出文件格式：

```
============================ 头部统计 ============================
总样本数 / 任一模型有错误数 / v0.3.0错误数 / v0.4.0错误数
两者都错 / 仅v0.3.0错 / 仅v0.4.0错

──────────────── 分段详情 ────────────────
【两者都错】共 N 条（按 CER 降序）
[uid]
  标签:   <ref_norm>
  v0.3.0: <hyp_norm>  (CER=XX%, 召回✓/✗)
  v0.4.0: <hyp_norm>  (CER=XX%, 召回✓/✗)

【仅v0.3.0错 (v0.4.0正确)】共 N 条
...

【仅v0.4.0错 (v0.3.0正确)】共 N 条
...
```

指标说明：

| 指标 | 定义 |
|------|------|
| CER | 字符错误率（来自 eval\_detail.jsonl 中的 `cer` 字段） |
| 召回 ✓/✗ | 标签文本（`ref_norm`）是否作为子串出现在识别结果（`hyp_norm`）中 |

### 11\_analyse/export\_v3\_v4\_errors\_stripped（去唤醒词后错误对比）

> 依赖：仅 Python 标准库（`json`, `os`, `collections`）
>
> 前置：`eval_output/` 下 v0.3.0 和 v0.4.0 的评估目录中已有 `eval_detail.jsonl`（由 `batch_eval_models.py` 生成）

与 `export_v3_v4_errors.py` 功能类似，但在对比前先去除标签（`ref_norm`）和识别结果（`hyp_norm`）开头的唤醒词（"你好萤石"及 20+ 种变体），然后重新计算 CER。这样可以消除唤醒词残留对 CER 的干扰，更准确地反映模型在指令识别部分的真实性能差异。

唤醒词去除规则：
- 支持 23 种唤醒词变体（按长度降序匹配，最多迭代 3 次）
- 仅去除文本开头的唤醒词，不处理结尾
- 去除后若标签为空则跳过该样本

```bash
python 11_analyse/export_v3_v4_errors_stripped.py
```

脚本内配置变量（直接修改脚本）：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `base` | 评估结果根目录 | `e:\8.音频\03 s10\eval_output` |
| `output_dir` | 输出目录 | `e:\8.音频\03 s10\11_analyse` |
| `datasets` | 数据集名→eval\_detail.jsonl 相对路径映射 | `s10_reflow`, `sp4_reflow` |

数据路径约定（同 `export_v3_v4_errors.py`）：

```
eval_output/
├── v0.3.0/model.pt.best/reflow_label/s10_reflow/test/eval_detail.jsonl
├── v0.3.0/model.pt.best/reflow_label/sp4_reflow/test/eval_detail.jsonl
├── v0.4.0/model.pt.best/reflow_label/s10_reflow/test/eval_detail.jsonl
└── v0.4.0/model.pt.best/reflow_label/sp4_reflow/test/eval_detail.jsonl
```

输出到 `11_analyse/`：

- `errors_v3_v4_s10_reflow_stripped.txt` — s10\_reflow 去唤醒词后错误对比
- `errors_v3_v4_sp4_reflow_stripped.txt` — sp4\_reflow 去唤醒词后错误对比

控制台输出汇总表格（每个数据集）：

```
总样本 / 准确率 / 召回率 / CER / Sub / Ins / Del（v0.3.0 vs v0.4.0 及差异）
错误分布：两者都错 / 仅v0.3.0错 / 仅v0.4.0错
```

输出文件格式：

```
============================ 头部统计 ============================
总样本数
v0.3.0: 准确率=XX% 召回率=XX% CER=XX%
v0.4.0: 准确率=XX% 召回率=XX% CER=XX%
两者都错 / 仅v0.3.0错 / 仅v0.4.0错

──────────────── 分段详情 ────────────────
【两者都错】共 N 条（按 CER 降序）
[uid]
  标签:   <去唤醒词后的ref>
  v0.3.0: <去唤醒词后的hyp>  (CER=XX%, 召回✓/✗)
  v0.4.0: <去唤醒词后的hyp>  (CER=XX%, 召回✓/✗)

【仅v0.3.0错 (v0.4.0正确)】共 N 条
...

【仅v0.4.0错 (v0.3.0正确)】共 N 条
...
```

与 `export_v3_v4_errors.py` 的区别：

| 对比项 | `export_v3_v4_errors.py` | `export_v3_v4_errors_stripped.py` |
|--------|--------------------------|----------------------------------|
| 唤醒词处理 | 不处理，直接使用原始 ref\_norm / hyp\_norm | 去除开头唤醒词后再对比 |
| CER 计算 | 使用 eval\_detail.jsonl 中的原始 CER | 去唤醒词后重新计算 CER（内置编辑距离） |
| 召回定义 | ref\_norm 是否为 hyp\_norm 的子串 | 去唤醒词后的 ref 是否为去唤醒词后的 hyp 的子串 |
| 适用场景 | 查看原始错误全貌 | 排除唤醒词干扰，聚焦指令识别差异 |

### 11\_analyse/recall\_comparison\_report（v0.3.0 vs v0.4.0 指令召回率对比）

> 依赖：仅 Python 标准库（`json`, `os`）
>
> 前置：`eval_output/v0.3.0/model.pt.best/` 和 `eval_output/v0.4.0/model.pt.best/` 下已有 `reflow_label/{s10_reflow,sp4_reflow}/test/eval_detail.jsonl`

生成 v0.3.0 与 v0.4.0 的指令召回率对比报告。去除标签和识别结果开头的唤醒词（"你好萤石"及 22 种变体）后，判断标签是否作为子串出现在识别结果中（即"召回"），聚焦以下三类样本：

- **v0.3.0 能召回但 v0.4.0 不能**（v0.3.0 优势）
- **v0.4.0 能召回但 v0.3.0 不能**（v0.4.0 优势）
- **两者都未召回**

各类样本按标签长度升序排列（短指令优先），便于快速定位高频短指令的召回退化。

```bash
python 11_analyse/recall_comparison_report.py
```

无命令行参数，配置通过脚本内变量修改：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `base` | eval\_output 根目录 | `e:\8.音频\03 s10\eval_output` |
| `output_path` | 输出报告路径 | `e:\8.音频\03 s10\11_analyse\recall_comparison_v3_v4.txt` |
| `datasets` | 数据集名称→eval\_detail.jsonl 相对路径映射 | s10\_reflow/test, sp4\_reflow/test |

输出：

- `11_analyse/recall_comparison_v3_v4.txt` — 文本报告，包含：
  - 各数据集的总样本数、两版本召回率、分类统计
  - v0.3.0 独有召回样本详情（标签 + 两版本识别结果）
  - v0.4.0 独有召回样本详情
  - 两者都未召回样本详情

控制台同时打印各数据集的召回率摘要。

### 11\_analyse/compare\_v3\_v5（v0.3.0 vs v0.5.0/ep23 完整对比报告）

> 依赖：仅 Python 标准库（`json`, `os`, `collections`）
>
> 前置：`eval_output/v0.3.0/model.pt.best/` 和 `eval_output/v0.5.0/model.pt.ep23/` 下已有 `reflow_label/{s10_reflow,sp4_reflow}/test/eval_detail.jsonl`

生成 v0.3.0 与 v0.5.0/ep23 的完整对比报告，包含两部分：

1. **错误对比报告**：去除标签和识别结果开头唤醒词后重新计算 CER，按"两者都错/仅v0.3.0错/仅v0.5.0错"分类展示错误样本，按 CER 降序排列
2. **指令召回率专项报告**：去除开头唤醒词后判断标签是否作为子串出现在识别结果中，聚焦"v0.3.0独有召回"、"v0.5.0独有召回"、"两者都未召回"三类样本，按标签长度排序

```bash
python 11_analyse/compare_v3_v5.py
```

无命令行参数，配置通过脚本内变量修改：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `base` | eval\_output 根目录 | `e:\8.音频\03 s10\eval_output` |
| `output_dir` | 输出根目录（报告输出到 `{output_dir}/reports/v3_vs_v5/`） | `e:\8.音频\03 s10\11_analyse` |
| `datasets` | 数据集名称→eval\_detail.jsonl 相对路径映射 | s10\_reflow/test, sp4\_reflow/test |

输出到 `11_analyse/reports/v3_vs_v5/`：

- `errors_v3_v5_{dataset}_stripped.txt` — 错误对比报告（每个数据集一份），包含：
  - 总样本数、准确率/召回率/CER 对比（含差异值）
  - Sub/Ins/Del 操作数对比
  - 错误分布统计（两者都错/仅v0.3.0错/仅v0.5.0错）
  - 各分类下的逐条错误详情（标签 + 两版本识别结果 + CER + 召回状态）
- `recall_comparison_v3_v5.txt` — 指令召回率专项报告，包含：
  - 各数据集的总样本数、两版本召回率、分类统计
  - v0.3.0 独有召回样本详情（按标签长度升序）
  - v0.5.0 独有召回样本详情（按标签长度升序）
  - 两者都未召回样本详情（按标签长度升序）

控制台同时打印各数据集的准确率、召回率、CER 摘要及召回差异统计。

### check\_eval\_data（评估数据完整性检查）

> 依赖：仅 Python 标准库（`json`, `os`）
>
> 前置：`eval_output/` 目录下已有评估结果（由 `10_eval/batch_eval_models.py` 等脚本生成）

快速检查 `eval_output/` 目录下各版本评估数据是否齐全，用于在运行 `extract_best_model_cer.py` 或进行模型对比分析前确认数据完整性。检查四类文件：

1. **各版本 combined\_summary.json** — `eval_output/{version}/{best_model}/combined_summary.json`（v0.1.0~v0.6.0）
2. **base 基线数据** — `eval_output/base/{reflow,speech,augment}_eval_result/{seaco,online_model}/combined_summary.json`（6 个文件）
3. **全量业务数据** — `eval_output/{s10_results,sp4_results,testset_results}/comparison_summary.json`
4. **model\_comparison\_summary.json（备选）** — `eval_output/{version}/model_comparison_summary.json`（部分旧版本使用此格式）

```bash
python check_eval_data.py
```

无命令行参数，配置通过脚本内变量修改：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `BASE` | eval\_output 根目录 | `E:\8.音频\03 s10\eval_output` |
| `BEST_MODELS` | 各版本最佳模型权重名称映射 | `{"v0.1.0": "model.pt.ep16.4000", "v0.2.0": "model.pt.best", ...}` |

输出：直接打印到控制台，每项显示 `OK` 或 `MISSING` 状态。无文件输出。

### extract\_best\_model\_cer（各版本最佳模型 CER 对比表）

> 依赖：仅 Python 标准库（`json`, `os`）
>
> 前置：`eval_output/` 下各版本评估结果已生成（`combined_summary.json` 或 `model_comparison_summary.json`），以及 `s10_results/`、`sp4_results/` 下的 `comparison_summary.json`

从 `eval_output/` 目录读取各版本最佳模型（v0.1.0~v0.6.0）和基线模型（seaco 原始 / online）在所有 test 数据集上的 CER，生成三张对比表：

1. **S10 全量业务数据 CER 对比** — 各版本在 S10 全量数据上的 CER / SER / 句准确率
2. **SP4 全量业务数据 CER 对比** — 各版本在 SP4 全量数据上的 CER / SER / 句准确率
3. **19 个数据集 test 集 CER 对比** — 3 业务 + 8 公开 + 8 增强数据集的 test CER，标注每行最优模型，末尾统计各模型最优数量

```bash
python extract_best_model_cer.py
```

无命令行参数，配置通过脚本内变量修改：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `BASE` | eval\_output 根目录 | `E:\8.音频\03 s10\eval_output` |
| `BEST_MODELS` | 各版本最佳模型权重名称映射 | `{"v0.1.0": "model.pt.ep16.4000", "v0.2.0": "model.pt.best", ...}` |
| `TEST_DATASETS` | 19 个 test 数据集的名称→相对路径映射 | 3 业务 + 8 公开(speech) + 8 增强(augment) |

数据来源：

| 数据 | 文件路径 |
|------|---------|
| seaco / online 基线 | `eval_output/base/{reflow,speech,augment}_eval_result/{seaco,online_model}/combined_summary.json` |
| 各版本 test CER | `eval_output/{version}/{model_name}/combined_summary.json` 或 `eval_output/{version}/model_comparison_summary.json` |
| S10 全量业务数据 | `eval_output/s10_results/comparison_summary.json` |
| SP4 全量业务数据 | `eval_output/sp4_results/comparison_summary.json` |

输出：直接打印到控制台（标准输出），可重定向到文件保存：

```bash
python extract_best_model_cer.py > eval_output/best_model_cer_report.txt
```

### extract\_performance（全版本性能对比表生成）

> 依赖：仅 Python 标准库（`json`, `os`）
>
> 前置：`eval_output/` 目录下已有评估结果（由 `10_eval/batch_eval_models.py` 等脚本生成）

从 `eval_output/` 目录下读取各版本最佳模型的评估数据，生成三张完整对比表：

1. **S10 全量业务数据 CER 对比** — seaco 原始 / finetuned(原online) / 各版本最佳权重的 CER / SER / 句准确率
2. **SP4 全量业务数据 CER 对比** — 同上
3. **19 个数据集 test 集 CER 对比** — 3 业务 + 8 公开(speech) + 8 增强(augment) 数据集的 test CER，标注每行最优模型，末尾统计各模型最优数量

与 `extract_best_model_cer.py` 的区别：本脚本优先从 `s10_results/` 和 `sp4_results/` 下的 `comparison_summary.json` 读取 seaco/online 基线的完整指标（CER/SER/句准确率），若 comparison_summary 中不包含基线模型则回退到 `base/reflow_eval_result/` 按字符数加权合计 train+dev+test 计算 CER（此时 SER/句准确率标注 N/A）。各版本全量业务指标同样从 comparison_summary.json 读取，数据来源更完整。

```bash
python extract_performance.py
```

无命令行参数，配置通过脚本内变量修改：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `BASE` | eval\_output 根目录 | `E:\8.音频\03 s10\eval_output` |
| `BEST_MODELS` | 各版本最佳模型权重名称映射 | `{"v0.1.0": "model.pt.ep16.4000", ..., "v0.5.0": "model.pt.ep35", "v0.6.0": "model.pt.ep33"}` |
| `TEST_DATASETS` | 19 个 test 数据集的名称→相对路径映射 | 3 业务 + 8 公开(speech) + 8 增强(augment) |
| `MODEL_COLS` | 对比表列顺序 | `["seaco", "online", "v0.1.0", ..., "v0.6.0"]` |

数据来源：

| 数据 | 文件路径 |
|------|---------|
| seaco / online 基线（19 数据集） | `eval_output/base/{reflow,speech,augment}_eval_result/{seaco,online_model}/combined_summary.json` |
| seaco / online 全量业务 CER/SER/句准确率 | 优先从 `eval_output/{s10,sp4}_results/comparison_summary.json` 的 `models["base/model.pt.seaco"]` / `models["base/model.pt.online"]` 读取（含 SER 和句准确率）；若不存在则回退到 `eval_output/base/reflow_eval_result/{seaco,online}/combined_summary.json`（仅 CER，SER/句准确率标注 N/A） |
| 各版本 test CER | `eval_output/{version}/{model_name}/combined_summary.json` |
| S10 全量业务数据（各版本） | `eval_output/s10_results/comparison_summary.json` |
| SP4 全量业务数据（各版本） | `eval_output/sp4_results/comparison_summary.json` |

输出：直接打印到控制台（标准输出），包含三张对比表和缺失数据报告。可重定向到文件保存：

```bash
python extract_performance.py > eval_output/performance_report.txt
```

### calc\_recall\_and\_stripped\_cer（SP4 指令召回率 + 剔除唤醒词 CER）

> 依赖：仅 Python 标准库（`json`, `os`, `re`）
>
> 前置：`eval_output/` 目录下已有各版本评估结果（`sp4_results/` 和 `testset_results/` 子目录，含 `eval_detail.jsonl` 和 `comparison_summary.json`）

计算两项指标：

1. **SP4 指令召回率** — 对 `eval_detail.jsonl` 中每条样本，剔除 ref 和 hyp 头尾的唤醒词变体（"你好萤石"及 23 种常见误识别变体），判断剔除后的 ref\_stripped 是否为 hyp\_stripped 的子串，统计召回比例
2. **独立测试集剔除唤醒词前后 CER 对比** — 对 testset 的 `eval_detail.jsonl`，分别计算原始 CER（从 `comparison_summary.json` 读取）和剔除唤醒词后的字符级编辑距离 CER，展示去唤醒词对 CER 的改善

唤醒词剥离逻辑：按长度降序匹配头尾，最多 3 轮迭代，确保嵌套唤醒词（如"你好萤萤石"）也能完整剥离。

```bash
python calc_recall_and_stripped_cer.py
```

无命令行参数，配置通过脚本内变量修改：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `BASE` | eval\_output 根目录 | `E:\8.音频\03 s10\eval_output` |
| `WAKEWORDS` | 唤醒词变体列表（按长度降序排列） | 23 个变体（"你好萤萤石"~"萤石"） |
| `BEST_MODELS` | 各版本最佳模型权重名称映射 | `{"seaco": "base_model.pt.seaco", ..., "v0.6.0": "v0.6.0_model.pt.ep33"}` |

数据来源：

| 数据 | 文件路径 |
|------|---------|
| SP4 指令召回率 | `eval_output/sp4_results/{dir_name}/eval_detail.jsonl` |
| testset 原始 CER | `eval_output/testset_results/comparison_summary.json` |
| testset 剔除唤醒词 CER | `eval_output/testset_results/{dir_name}/eval_detail.jsonl` |

输出：直接打印到控制台（标准输出），包含两张对比表：

- **SP4 指令召回率表** — 各版本模型的召回率（↑越高越好）和有效样本数
- **独立测试集 CER 对比表** — 各版本模型的原始 CER 和剔除唤醒词后 CER（↓越低越好）

可重定向到文件保存：

```bash
python calc_recall_and_stripped_cer.py > eval_output/recall_and_stripped_cer_report.txt
```

### analyze\_testset\_gap（testset 剔除唤醒词 CER 差异归因分析）

> 依赖：仅 Python 标准库（`json`, `os`）
>
> 前置：`eval_output/testset_results/` 目录下已有 `base_model.pt.seaco/eval_detail.jsonl` 和 `v0.5.0_model.pt.ep35/eval_detail.jsonl`

分析独立测试集上 seaco 原始模型 vs v0.5.0/ep35 剔除唤醒词后 CER 差异的具体原因。对两个模型的 `eval_detail.jsonl` 逐条剔除唤醒词后重新计算字符级编辑距离，按 index 对齐逐条对比，输出以下分析维度：

1. **总体 CER 对比** — 两模型剔除唤醒词后的总 CER 和差距
2. **逐条对比统计** — v0.5.0 更差/更好/相同的样本数量和占比
3. **典型 bad case** — v0.5.0 比 seaco 差距 ≥2 的样本，按差距降序展示前 20 条，标注错误原因（插入过多/删除过多/替换错误）和唤醒词残留
4. **唤醒词残留分析** — 两模型剔除唤醒词后 hyp 中仍包含唤醒词变体的样本数
5. **输出长度分析** — 两模型 hyp 与 ref 的平均长度差异，以及 hyp 比 ref 多出 >3 字符的样本占比

```bash
python analyze_testset_gap.py
```

无命令行参数，配置通过脚本内变量修改：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `BASE` | testset\_results 根目录 | `E:\8.音频\03 s10\eval_output\testset_results` |
| `WAKEWORDS` | 唤醒词变体列表（按长度降序排列，23 个变体） | "你好萤萤石"~"萤石" |

数据来源：

| 数据 | 文件路径 |
|------|---------|
| seaco 逐条结果 | `eval_output/testset_results/base_model.pt.seaco/eval_detail.jsonl` |
| v0.5.0/ep35 逐条结果 | `eval_output/testset_results/v0.5.0_model.pt.ep35/eval_detail.jsonl` |

输出：直接打印到控制台（标准输出），包含 5 个分析模块。可重定向到文件保存：

```bash
python analyze_testset_gap.py > eval_output/testset_gap_analysis.txt
```

### find\_wakeword\_residual（testset 唤醒词残留样本定位）

> 依赖：仅 Python 标准库（`json`, `os`）
>
> 前置：`eval_output/testset_results/` 目录下已有 `base_model.pt.seaco/eval_detail.jsonl` 和 `v0.5.0_model.pt.ep35/eval_detail.jsonl`

找出独立测试集中 v0.5.0/ep35 模型在剔除唤醒词后，识别结果（hyp）中仍然包含唤醒词变体的样本。这些样本说明模型在识别过程中"幻觉"出了唤醒词文本（标签中并不包含该唤醒词）。对每条残留样本，同时展示 seaco 原始模型的识别结果作为对比参考。

分析逻辑：

1. 加载 v0.5.0/ep35 和 seaco 的 `eval_detail.jsonl`
2. 对每条样本的 `ref_norm` 和 `hyp_norm` 执行唤醒词剔除（头尾各最多 3 轮）
3. 检查剔除后的 `hyp_stripped` 中是否仍包含唤醒词变体，且该唤醒词不在 `ref_stripped` 中
4. 输出所有残留样本的详细信息

```bash
python find_wakeword_residual.py
```

无命令行参数，配置通过脚本内变量修改：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `BASE` | testset\_results 根目录 | `E:\8.音频\03 s10\eval_output\testset_results` |
| `WAKEWORDS` | 唤醒词变体列表（按长度降序排列，23 个变体） | "你好萤萤石"~"萤石" |

数据来源：

| 数据 | 文件路径 |
|------|---------|
| v0.5.0/ep35 逐条结果 | `eval_output/testset_results/v0.5.0_model.pt.ep35/eval_detail.jsonl` |
| seaco 逐条结果 | `eval_output/testset_results/base_model.pt.seaco/eval_detail.jsonl` |

输出：直接打印到控制台（标准输出），包含：

- 两模型样本数统计
- 唤醒词残留样本总数
- 逐条残留样本详情（音频 key、残留的唤醒词、原始标签、seaco 识别结果、v0.5.0 识别结果、剔除唤醒词后的标签和识别结果）

可重定向到文件保存：

```bash
python find_wakeword_residual.py > eval_output/wakeword_residual_cases.txt
```

### analyze\_child\_gap（ChildMandarin 退化归因分析）

> 依赖：仅 Python 标准库（`json`, `os`, `collections`）
>
> 前置：`eval_output/child/ChildMandarin/` 目录下已有 `base_model.pt.seaco/eval_detail.jsonl` 和 `v0.5.0_model.pt.ep35/eval_detail.jsonl`

分析 ChildMandarin 儿童语音数据集上 seaco 原始模型 vs v0.5.0/ep35 的退化表现。通过解析 uid 中编码的说话人元信息（年龄、性别、方言、城市、设备），从多个维度对比两模型的 CER 差异，定位退化集中的人群和场景。

分析维度：

1. **逐条对比统计** — v0.5.0 更差/更好/相同的样本数量、占比和错误增减总量
2. **按年龄分析 CER** — 各年龄段两模型 CER 和差距
3. **按性别分析 CER** — 男/女两模型 CER 和差距
4. **按方言/地区分析 CER** — 退化最严重的前 15 个方言/地区
5. **按句子长度分析 CER** — 1-5字/6-10字/11-20字/20+字各桶的 CER 对比
6. **错误类型对比** — 替换(sub)/插入(ins)/删除(del) 总量对比
7. **典型 bad case** — v0.5.0 退化最严重的前 15 条样本详情

uid 解析规则：`ID_年龄_性别_方言_城市_设备_序号`（下划线分隔，至少 6 段）

```bash
python analyze_child_gap.py
```

无命令行参数，配置通过脚本内变量修改：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `BASE` | ChildMandarin 评估结果根目录 | `E:\8.音频\03 s10\eval_output\child\ChildMandarin` |

数据来源：

| 数据 | 文件路径 |
|------|---------|
| seaco 逐条结果 | `eval_output/child/ChildMandarin/base_model.pt.seaco/eval_detail.jsonl` |
| v0.5.0/ep35 逐条结果 | `eval_output/child/ChildMandarin/v0.5.0_model.pt.ep35/eval_detail.jsonl` |

输出：直接打印到控制台（标准输出），包含 7 个分析模块。可重定向到文件保存：

```bash
python analyze_child_gap.py > eval_output/child_gap_analysis.txt
```

### analyze\_child\_error\_patterns（ChildMandarin 退化场景分类）

> 依赖：仅 Python 标准库（`json`, `os`, `re`, `collections`）
>
> 前置：`eval_output/child/ChildMandarin/` 目录下已有 `base_model.pt.seaco/eval_detail.jsonl` 和 `v0.5.0_model.pt.ep35/eval_detail.jsonl`

对 ChildMandarin 数据集上 v0.5.0 比 seaco 退化的样本进行错误模式归类。筛选出 v0.5.0 错误数多于 seaco 的样本，根据 ref/hyp 特征自动分类到以下场景：

| 场景标签 | 判定规则 |
|---------|---------|
| 长句截断 | ref ≥ 10 字且 v0.5 输出长度 < ref 的 60% |
| 唤醒词插入 | v0.5 输出中含唤醒词变体但 ref 中无 |
| 重复/结巴语音 | ref 中存在连续重复字符或重复片段 |
| 语气词/拟声词 | ref 中语气词/拟声词占比 > 40% |
| 含英文 | ref 中含连续英文字母（≥ 2 字符） |
| 模型重复输出 | v0.5 输出长度 > ref 的 130% 且含高频重复片段 |
| 口语化/多填充词 | ref 中口头禅（"那个"、"就是"、"然后"等）出现 ≥ 3 次 |
| 一般识别退化 | 不匹配以上任何场景 |

```bash
python analyze_child_error_patterns.py
```

无命令行参数，配置通过脚本内变量修改：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `BASE` | ChildMandarin 评估结果根目录 | `E:\8.音频\03 s10\eval_output\child\ChildMandarin` |

数据来源：

| 数据 | 文件路径 |
|------|---------|
| seaco 逐条结果 | `eval_output/child/ChildMandarin/base_model.pt.seaco/eval_detail.jsonl` |
| v0.5.0/ep35 逐条结果 | `eval_output/child/ChildMandarin/v0.5.0_model.pt.ep35/eval_detail.jsonl` |

输出：直接打印到控制台（标准输出），包含：

1. **场景统计表** — 各场景的样本数、占比、多出错误总数、平均多出错误
2. **典型 case** — 每个场景最多展示 5 条典型退化样本（uid / ref / seaco 识别 / v0.5 识别 / 各自错误数）

可重定向到文件保存：

```bash
python analyze_child_error_patterns.py > eval_output/child_error_patterns.txt
```

### 12\_test/gen\_test\_text（测试集标注整合）

> 依赖：仅 Python 标准库（`os`, `pathlib`）
>
> 前置：`wavs/` 目录下各日期文件夹中已有 `labelled/label/label.txt` 标注文件和 `labelled/source/*.wav` 音频文件

从 `E:\8.音频\03 s10\wavs` 目录下各日期范围文件夹中整合 `label.txt` 标注文件，生成统一的测试集 `text` 和 `wav.scp`。遍历所有日期文件夹（按名称排序），逐个读取 `labelled/label/label.txt`，合并为完整的测试集标注；同时检查对应音频文件是否存在，生成 `wav.scp` 路径映射。

输入目录结构约定：

```
wavs/
├── <日期范围1>/
│   └── labelled/
│       ├── label/label.txt    ← 标注文件（每行: utterance_id 文本）
│       └── source/*.wav       ← 音频文件
├── <日期范围2>/
│   └── labelled/
│       ├── label/label.txt
│       └── source/*.wav
└── ...
```

使用前可修改脚本内的配置变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `WAVS_ROOT` | wavs 根目录（含各日期文件夹） | `E:\8.音频\03 s10\wavs` |
| `OUTPUT_DIR` | 输出目录 | `E:\8.音频\03 s10\12_test` |

```bash
python 12_test/gen_test_text.py
```

工作流程：

1. 扫描 `WAVS_ROOT` 下所有子目录（按名称排序）
2. 对每个日期文件夹，检查 `labelled/label/label.txt` 是否存在（不存在则跳过）
3. 读取 `label.txt`，逐行解析 `utterance_id` 和文本
4. 检查 `labelled/source/{utterance_id}.wav` 是否存在，存在则加入 `wav.scp`
5. 汇总所有文件夹的标注，写出 `text` 和 `wav.scp`

输出到 `E:\8.音频\03 s10\12_test\`：

- `text` — 合并后的 Kaldi 格式文本标注（`<utterance_id> <text>`，每行一条）
- `wav.scp` — Kaldi 格式音频路径映射（`<utterance_id> <wav_absolute_path>`，仅包含实际存在的音频）

控制台输出：找到的日期文件夹数量、每个文件夹的标注条数（或跳过原因）、最终汇总（总标注数、匹配音频数、缺失音频数、输出文件路径）。

### 12\_test/trim\_wakeword（唤醒词位置检测 - 服务器端）

> 依赖：`pip install funasr torch soundfile`
>
> 前置：`12_test/` 目录下已有 `wav.scp`（由 `12_test/gen_test_text.py` 生成）

【服务器端脚本】使用 FunASR 带时间戳的 ASR 模型识别每条音频，检测开头/结尾是否包含唤醒词"你好萤石"，输出检测结果到 `trim_log.jsonl`。**只做识别和检测，不做裁剪**。后续在本地运行 `trim_wakeword_local.py` 进行实际裁剪。

工作原理：

1. 加载 `wav.scp` 后自动过滤损坏文件（不存在、0 字节、无法读取 wav 头、时长 < 0.1s），跳过无效条目并打印统计
2. 用带时间戳的 ASR 模型（SeACo-Paraformer）识别每条有效音频，获取逐字时间戳（支持热词增强，将唤醒词作为热词传入以提高检测准确率）
3. 检测识别文本开头/结尾是否包含唤醒词字符序列（支持模糊匹配：前两字为"你好"即视为唤醒词变体）
4. 输出检测结果（唤醒词边界时间戳）到 `trim_log.jsonl`

```bash
# 默认配置运行
python 12_test/trim_wakeword.py

# 指定输入
python 12_test/trim_wakeword.py --input-scp /data1/12_test/wav.scp

# 自定义唤醒词
python 12_test/trim_wakeword.py --wakeword "你好小萤"

# 采样快速验证
python 12_test/trim_wakeword.py --sample 20

# 指定设备和 batch 大小
python 12_test/trim_wakeword.py --device cuda:0 --batch-size 16
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input-scp` | 输入 wav.scp 路径 | `/data1/12_test/wav.scp` |
| `--output-dir` | 输出目录（存放 `trim_log.jsonl`） | `/data1/12_test` |
| `--wakeword` | 唤醒词文本 | `你好萤石` |
| `--model` | ASR 模型路径（需支持时间戳输出） | SeACo-Paraformer 本地路径 |
| `--device` | 推理设备 | `cuda:0` |
| `--batch-size` | 推理 batch 大小 | `8` |
| `--sample` | 随机采样 N 条快速验证 | 全部 |

输出到 `{output_dir}/`（默认 `/data1/12_test/`）：

- `trim_log.jsonl` — 检测结果日志，每行一条 JSON 记录，包含：
  - `uid` — 音频唯一标识
  - `wav_path` — 原始音频路径
  - `recognized_text` — ASR 识别文本
  - `head_boundary` — 开头唤醒词边界 `[start_ms, end_ms]`（未检测到为 `null`）
  - `tail_boundary` — 结尾唤醒词边界 `[start_ms, end_ms]`（未检测到为 `null`）

后续步骤：

```bash
# 将 trim_log.jsonl 拷贝到本地，运行本地裁剪脚本
python 12_test/trim_wakeword_local.py
```

### 12\_test/verify\_trim\_energy（能量校验唤醒词边界）

> 依赖：`pip install numpy soundfile matplotlib`（matplotlib 仅 `--plot` 时需要）
>
> 前置：`12_test/output/trim_log.jsonl`（由 `trim_wakeword.py` 生成）；本地音频目录（`12_test/wavs2/`）

基于短时能量分析校验 `trim_log.jsonl` 中唤醒词时间戳边界的合理性。唤醒词和正文之间通常有一个短暂的能量低谷（微停顿/换气），本脚本在 `head_boundary[1]`（唤醒词结束时间）附近搜索能量最低点，验证 ASR 时间戳是否落在合理位置。

校验逻辑：

1. 读取 `trim_log.jsonl` 中检测到开头唤醒词的条目
2. 递归扫描 `--wav-dir` 目录，构建 uid → 本地 wav 路径的索引（支持嵌套子目录）
3. 对每条音频计算短时能量曲线（帧长 20ms，帧移 10ms）
4. 在 `head_boundary[1]` 附近 ±300ms 范围内寻找能量最低点
5. 根据低谷与 boundary 的偏差和能量下降幅度判定：pass / suspect / fail

```bash
# 默认配置运行
python 12_test/verify_trim_energy.py

# 指定 trim_log 路径
python 12_test/verify_trim_energy.py --trim-log path/to/trim_log.jsonl

# 只校验前 50 条
python 12_test/verify_trim_energy.py --sample 50

# 生成可视化图片（前 20 条）
python 12_test/verify_trim_energy.py --plot

# 调整容许偏差
python 12_test/verify_trim_energy.py --tolerance 200
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--trim-log` | `trim_log.jsonl` 路径 | `E:\8.音频\03 s10\12_test\output\trim_log.jsonl` |
| `--wav-dir` | 本地音频目录（按 uid 查找 wav） | `E:\8.音频\03 s10\12_test\wavs2` |
| `--output-dir` | 输出目录 | `E:\8.音频\03 s10\12_test\output` |
| `--sample` | 只校验前 N 条 | 全部 |
| `--plot` | 生成能量曲线+边界标注的可视化图片（前 20 条） | — |
| `--tolerance` | 低谷与 boundary 的容许偏差（ms） | `150` |

判定规则：

| 状态 | 条件 |
|------|------|
| `pass` | 偏差 ≤ tolerance 且能量下降 > 3dB |
| `suspect` | 偏差 ≤ tolerance×2，或能量下降不明显 |
| `fail` | 音频过短 / 无法找到能量低谷 |

输出到 `{output_dir}/`（默认 `12_test/output/`）：

- `verify_result.jsonl` — 逐条校验结果，每行包含：
  - `uid` — 音频唯一标识
  - `status` — 判定结果（`pass` / `suspect` / `fail`）
  - `boundary_end_ms` — 原始 boundary 结束时间
  - `valley_ms` — 检测到的能量低谷位置（ms）
  - `offset_ms` — 低谷与 boundary 的偏差（ms）
  - `avg_energy_before_db` — 唤醒词段平均能量（dB）
  - `avg_energy_after_db` — 正文段前 500ms 平均能量（dB）
  - `energy_drop_db` — boundary 处的能量下降幅度（dB）
- `energy_plots/` — 可视化图片目录（仅 `--plot` 时生成，含波形+能量曲线+边界标注）


### 12\_test/ChildMandarin/prepare\_for\_eval（ChildMandarin 数据集预处理）

> 依赖：`pip install soundfile`
>
> 前置：ChildMandarin 数据集已下载，目录结构为 `{split}/{speaker_id}/{uid}.wav` + `{uid}.json`

ChildMandarin 儿童普通话数据集预处理脚本。递归扫描指定 split（默认 test）下所有 speaker 子目录，从每个 `.json` 标注文件读取 `id` 和 `text`，检查音频采样率和声道数（该数据集已是 16kHz 单声道），生成 `eval_on_testset.py` 所需的 `wav.scp` 和 `text` 文件。

目录结构：

```
ChildMandarin/
├── test/
│   ├── 002/
│   │   ├── 002_5_M_L_ZIBO_Android_001.wav
│   │   ├── 002_5_M_L_ZIBO_Android_001.json  (含 id, text)
│   │   └── ...
│   ├── 008/
│   └── ...
├── dev/
├── train/
├── wav.scp         -- 输出
└── text            -- 输出
```

```bash
# 默认处理 test 集
python 12_test/ChildMandarin/prepare_for_eval.py

# 处理 dev 集
python 12_test/ChildMandarin/prepare_for_eval.py --split dev

# 处理所有 split (train+dev+test)
python 12_test/ChildMandarin/prepare_for_eval.py --split all

# 指定输出文件前缀
python 12_test/ChildMandarin/prepare_for_eval.py --output-prefix test
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--split` | 处理哪个 split（`train` / `dev` / `test` / `all`） | `test` |
| `--output-prefix` | 输出文件前缀（如 `test` → `test_wav.scp`） | 无前缀 |

工作流程：

1. 扫描指定 split 目录下所有 speaker 子目录
2. 对每个 speaker 目录，找到所有 `.json` 标注文件
3. 从 JSON 中读取 `id`（uid）和 `text`（标注文本），检查对应 `.wav` 是否存在
4. 抽样检查音频格式（采样率、声道数），过滤格式不符的文件
5. 按 uid 排序并去重
6. 写出 `wav.scp` 和 `text`

输出到 `12_test/ChildMandarin/`：

- `wav.scp` — Kaldi 格式音频路径映射（`<uid> <wav_absolute_path>`）
- `text` — Kaldi 格式文本标注（`<uid> <text>`）

后续评估：

```bash
# 在 Linux 服务器上评估
python eval_on_testset.py \
    --test-scp /data1/12_test/ChildMandarin/wav.scp \
    --test-text /data1/12_test/ChildMandarin/text \
    --model-dir /data1/eval_output
```
