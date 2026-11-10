# 需求文档：音频增强场景多样性

## 简介

本需求文档针对 `07_augment/` 目录下的音频数据增强系统，旨在提升模拟智能音箱场景的丰富度和多样性。"场景多样性"不仅指音箱放置的房间类型，还包括不同的说话场景组合：单人不同距离、单人不同噪声条件、多人说话（一个清晰主说话人 + 一个背景人，背景人从"不太清晰"到"很模糊"连续变化）。当前系统已具备基础的多说话人混合、合成混响、距离关联采样等能力，但在场景组合的系统化覆盖方面存在不足。

此外，增强参数的设定应以真实业务数据的声学特征为基准。系统需要先分析真实数据（`s10_data_reflow/labeled/`）的声学特征分布，再基于分析结果校准增强参数，最后执行增强并验证输出与真实数据的分布一致性。为支持这一"分析→校准→增强→验证"的迭代流程，系统需要提供一系列可视化分析和调试工具。整个开发过程通过 Git 版本控制管理，并维护详细的开发日志。

## 术语表

- **增强引擎 (Augmentation_Engine)**：`augment_engine.py` 中的 `AudioAugmentor` 类，负责执行所有增强策略
- **增强链 (Augmentation_Chain)**：配置文件中 `chain` 字段定义的策略执行顺序
- **场景预设 (Scene_Preset)**：预定义的一组增强参数组合，对应一种典型的智能音箱使用场景
- **说话场景 (Speaking_Scenario)**：描述说话人数量、距离、背景条件的组合，分为单人场景和多人场景
- **背景音源 (Background_Source)**：混入主音频的背景声音，包括人声、电视/广播声、环境噪声等
- **距离因子 (Distance_Factor)**：`t ∈ [0, 1]` 的统一采样值，驱动 SNR/RT60/低通等参数的关联变化
- **房间声学参数 (Room_Acoustics)**：描述房间声学特性的参数集合，包括 RT60、EQ 特征等
- **真实 RIR (Real_RIR)**：从真实房间录制的房间脉冲响应文件（如 OpenSLR-26/28 数据集）
- **环境噪声库 (Noise_Library)**：包含真实环境噪声录音的文件目录（如 MUSAN、DEMAND 数据集）
- **场景多样性报告 (Diversity_Report)**：统计增强输出中各场景维度分布情况的报告
- **主说话人 (Main_Speaker)**：对智能音箱发出指令的目标说话人
- **背景说话人 (Background_Speaker)**：主说话人之外的其他说话人，清晰度从"不太清晰"到"很模糊"连续变化
- **配置文件 (Config_File)**：YAML 格式的增强参数配置文件
- **背景说话人目录 (BG_Speaker_Dir)**：独立于主说话人输入目录的背景音频来源目录，指定后背景说话人将从该目录选取而非从主说话人候选池中选取
- **分析工具集 (Analysis_Toolkit)**：用于可视化和分析增强效果的脚本工具集合
- **频谱图 (Spectrogram)**：音频信号的时频域可视化表示
- **业务数据 (Business_Data)**：`s10_data_reflow/labeled/` 目录下的真实智能音箱录音数据
- **声学基线 (Acoustic_Baseline)**：从真实业务数据中提取的声学特征分布统计，作为增强参数校准的参考基准
- **参数推荐 (Param_Recommendation)**：基于声学基线自动生成的增强参数建议值，用于初始化或校准配置文件中的参数范围
- **开发日志 (Dev_Log)**：记录开发过程中各阶段性成果和决策的 Markdown 文档

## 需求

### 需求 1：房间类型场景预设

**用户故事：** 作为数据工程师，我希望能通过场景预设快速配置不同房间类型的声学参数，以便生成覆盖多种家庭房间的增强数据。

#### 验收标准

1. WHEN 配置文件中定义了 `scene_presets` 字段，THE Augmentation_Engine SHALL 根据预设名称加载对应的房间声学参数（RT60 范围、环境噪声 SNR 范围、EQ 特征等）
2. THE Augmentation_Engine SHALL 内置 3 种家庭场景预设：客厅（living_room）、卧室（bedroom）、书房（study），每种预设具有不同的 RT60 范围和环境噪声特征
3. WHEN 配置文件中设置 `scene_mode: "random"` 时，THE Augmentation_Engine SHALL 对每条增强音频随机选择一种场景预设
4. WHEN 配置文件中设置 `scene_mode: "balanced"` 时，THE Augmentation_Engine SHALL 在所有增强副本中均匀分配各场景预设，使每种场景的样本数偏差不超过 1
5. IF 配置文件中引用了未定义的场景预设名称，THEN THE Augmentation_Engine SHALL 在启动时记录错误日志并终止处理

### 需求 2：说话场景组合覆盖

**用户故事：** 作为数据工程师，我希望增强系统能系统化地覆盖不同的说话场景组合（单人/多人、不同距离、不同噪声条件），以确保训练数据涵盖智能音箱的各种真实使用情况。

#### 验收标准

1. WHEN 配置文件中定义了 `speaking_scenarios` 字段，THE Augmentation_Engine SHALL 支持以下说话场景类型：`single_clean`（单人安静环境）、`single_noisy`（单人有噪声环境）、`multi_speaker`（多人说话场景）
2. WHEN 说话场景为 `single_clean` 时，THE Augmentation_Engine SHALL 仅对主说话人应用房间声学和距离变化，不混入背景人声或环境噪声
3. WHEN 说话场景为 `single_noisy` 时，THE Augmentation_Engine SHALL 对主说话人应用房间声学和距离变化，并混入环境噪声（高斯噪声或真实噪声文件）
4. WHEN 说话场景为 `multi_speaker` 时，THE Augmentation_Engine SHALL 混入 1 个背景说话人，背景说话人的清晰度由距离因子控制，覆盖从"不太清晰"（距离因子 0.3-0.5，SNR 较低、轻度混响）到"很模糊"（距离因子 0.7-1.0，SNR 较高、重度混响和低通）的连续范围
5. WHEN 配置了 `scenario_mode: "balanced"` 时，THE Augmentation_Engine SHALL 在所有增强副本中均匀分配各说话场景类型
6. WHEN 配置了 `scenario_mode: "weighted"` 且提供了 `scenario_weights` 时，THE Augmentation_Engine SHALL 按权重比例分配各说话场景类型

### 需求 3：主说话人距离多样性

**用户故事：** 作为数据工程师，我希望同一条原始音频的不同增强副本能模拟主说话人在不同距离的效果，以覆盖近场到远场的连续变化。

#### 验收标准

1. WHEN 配置文件中设置 `main_speaker_distance: true` 时，THE Augmentation_Engine SHALL 为每条增强副本独立采样一个主说话人距离因子 `d ∈ [0, 1]`（`d=0` 最近，`d=1` 最远）
2. THE Augmentation_Engine SHALL 使用主说话人距离因子 `d` 关联驱动以下参数：主说话人 RT60（近小远大）、音量衰减（近 0dB 远负值）、高频衰减（近保留远衰减）
3. WHEN `main_speaker_distance` 启用时，THE Augmentation_Engine SHALL 覆盖 `reverb` 和 `volume_perturb` 策略中的独立随机采样，改为由距离因子 `d` 统一驱动
4. WHEN `main_speaker_distance` 未设置或设置为 `false` 时，THE Augmentation_Engine SHALL 保持当前行为（各参数独立随机采样）

### 需求 4：背景说话人清晰度连续控制

**用户故事：** 作为数据工程师，我希望背景说话人的清晰度能在"不太清晰"到"很模糊"之间连续变化，以覆盖真实场景中背景人声的各种干扰程度。

#### 验收标准

1. WHEN `speaker_mix` 策略启用且 `bg_distance_correlated: true` 时，THE Augmentation_Engine SHALL 使用距离因子 `t` 统一控制背景说话人的 SNR、RT60 和低通截止频率
2. THE Augmentation_Engine SHALL 支持配置 `bg_clarity_range: [min, max]`（默认 `[0.0, 1.0]`），限制背景说话人距离因子的采样范围，其中较小值对应"不太清晰"，较大值对应"很模糊"
3. WHEN `multi_speaker` 说话场景中背景说话人数量固定为 1 时，THE Augmentation_Engine SHALL 确保该背景说话人的距离因子在 `bg_clarity_range` 范围内均匀采样
4. THE Augmentation_Engine SHALL 保持当前的距离因子关联逻辑：`t=0`（近）→ SNR 低 + RT60 小 + 低通高；`t=1`（远）→ SNR 高 + RT60 大 + 低通低

### 需求 5：真实环境噪声集成

**用户故事：** 作为数据工程师，我希望能将真实环境噪声（如空调声、电视声等）混入增强音频，以替代单一的高斯白噪声。

#### 验收标准

1. WHEN `noise_from_file` 策略启用且 `noise_dir` 指向有效目录时，THE Augmentation_Engine SHALL 从该目录随机选取噪声文件并按配置的 SNR 范围混入主音频
2. WHEN `noise_dir` 包含子目录时，THE Augmentation_Engine SHALL 将每个子目录视为一种噪声类别（如 `hvac/`、`kitchen/`、`traffic/`）
3. WHEN 配置了 `noise_category_weights` 时，THE Augmentation_Engine SHALL 按权重比例从各噪声类别中采样
4. IF `noise_dir` 路径不存在或目录为空，THEN THE Augmentation_Engine SHALL 记录警告日志并跳过该策略，继续执行增强链中的后续策略

### 需求 6：真实 RIR 集成

**用户故事：** 作为数据工程师，我希望能使用真实房间脉冲响应（RIR）替代合成混响，以获得更逼真的房间声学效果。

#### 验收标准

1. WHEN `real_rir` 策略启用且 `rir_dir` 指向有效目录时，THE Augmentation_Engine SHALL 从该目录随机选取 RIR 文件并与主音频进行 FFT 卷积
2. WHEN `rir_dir` 包含按房间类型命名的子目录时，THE Augmentation_Engine SHALL 支持按子目录名称与场景预设关联
3. WHEN `real_rir` 和 `reverb`（合成混响）同时出现在增强链中时，THE Augmentation_Engine SHALL 仅执行链中排在前面的那个，跳过后面的，避免双重混响
4. THE Augmentation_Engine SHALL 在卷积后保持主音频的原始峰值电平
5. IF RIR 文件读取失败，THEN THE Augmentation_Engine SHALL 记录错误日志并回退到合成混响

### 需求 7：背景说话人多样性控制

**用户故事：** 作为数据工程师，我希望能控制背景说话人的选取策略，避免同一说话人反复出现导致训练数据缺乏多样性。

#### 验收标准

1. WHEN 配置文件中设置 `bg_speaker_diversity: true` 时，THE Augmentation_Engine SHALL 在同一批次增强中追踪已使用的背景音频路径
2. WHILE `bg_speaker_diversity` 启用时，THE Augmentation_Engine SHALL 优先选择尚未被使用过的候选音频作为背景说话人
3. WHEN 所有候选音频均已被使用过时，THE Augmentation_Engine SHALL 重置使用记录并重新开始选取
4. WHEN 配置文件中设置 `bg_max_reuse` 为正整数时，THE Augmentation_Engine SHALL 限制每条候选音频在整个增强批次中被选为背景的最大次数

### 需求 8：背景说话人音频源分离

**用户故事：** 作为数据工程师，我希望能灵活指定背景说话人的音频来源——既可以从主说话人同一目录中选取（自混合），也可以指定一个独立的背景说话人目录，以便使用不同录音条件或不同说话人群体的音频作为背景声。

#### 验收标准

1. WHEN 仅指定 `--input_dir`（或 manifest 中的音频路径）且未配置 `bg_speaker_dir` 时，THE Augmentation_Engine SHALL 从主说话人的同一候选池中选取背景音频（当前行为，自混合模式）
2. WHEN 配置文件或命令行中指定了 `bg_speaker_dir` 指向一个有效目录时，THE Augmentation_Engine SHALL 从该目录预加载音频作为背景说话人候选池，不再从主说话人目录中选取
3. WHEN `bg_speaker_dir` 指定时，THE Augmentation_Engine SHALL 在启动时独立扫描并预加载该目录下的所有 wav 文件到背景音频缓存中
4. WHEN `bg_speaker_dir` 指定时，THE Augmentation_Engine SHALL 确保主说话人音频不会出现在背景候选池中，背景候选池也不会出现在主说话人的增强输出中
5. IF `bg_speaker_dir` 路径不存在或目录中无 wav 文件，THEN THE Augmentation_Engine SHALL 在启动时记录错误日志并终止处理
6. THE `augment_dir.py` SHALL 支持 `--bg_speaker_dir` 命令行参数，THE `augment_manifest.py` SHALL 支持相同参数，两者行为一致

### 需求 9：场景多样性统计报告

**用户故事：** 作为数据工程师，我希望在增强完成后能看到各场景维度的分布统计，以验证增强数据的多样性是否达标。

#### 验收标准

1. WHEN 增强处理完成时，THE Augmentation_Engine SHALL 输出一份场景多样性统计报告到输出目录
2. THE Diversity_Report SHALL 包含以下维度的分布统计：房间场景预设分布、说话场景类型分布、overlap_mode 分布、背景说话人清晰度分布（距离因子直方图分桶）、主说话人距离因子分布
3. THE Diversity_Report SHALL 以人类可读的文本格式输出，文件名为 `diversity_report.txt`
4. THE Diversity_Report SHALL 同时输出 JSON 格式的机器可读版本，文件名为 `diversity_summary.json`
5. IF 某个维度的分布严重不均匀（最大类别占比超过 60%），THEN THE Diversity_Report SHALL 在报告中标注警告信息

### 需求 10：增强链动态组合

**用户故事：** 作为数据工程师，我希望不同的增强副本能使用不同的增强链组合，而非所有副本都走同一条固定链，以进一步提升场景多样性。

#### 验收标准

1. WHEN 配置文件中定义了 `chain_variants` 列表时，THE Augmentation_Engine SHALL 对每条增强副本从 `chain_variants` 中选择一条增强链执行
2. WHEN `chain_variants` 未定义时，THE Augmentation_Engine SHALL 保持当前行为（使用单一 `chain` 字段）
3. WHEN 配置了 `chain_variant_mode: "round_robin"` 时，THE Augmentation_Engine SHALL 按轮询方式依次使用各增强链变体
4. WHEN 配置了 `chain_variant_mode: "random"` 时，THE Augmentation_Engine SHALL 随机选择增强链变体
5. THE Augmentation_Engine SHALL 在 Diversity_Report 中记录各增强链变体的使用次数分布

### 需求 11：配置文件向后兼容

**用户故事：** 作为数据工程师，我希望现有的配置文件无需修改即可继续正常工作，新功能仅在显式配置时生效。

#### 验收标准

1. WHEN 配置文件中未包含 `scene_presets`、`scene_mode`、`speaking_scenarios`、`main_speaker_distance`、`bg_speaker_diversity`、`chain_variants` 等新字段时，THE Augmentation_Engine SHALL 保持与当前版本完全一致的行为
2. THE Augmentation_Engine SHALL 对所有新增配置字段提供合理的默认值，使未显式配置时等价于功能关闭
3. WHEN 使用现有的 `augment_config_speaker.yaml`、`augment_config_speaker_multi.yaml`、`augment_config_speaker_multi_2.yaml` 配置文件时，THE Augmentation_Engine SHALL 产生与当前版本相同的增强结果（在相同随机种子下）
4. IF 配置文件中同时存在新旧冲突字段（如同时设置 `scene_presets` 中的 RT60 和 `reverb` 策略中的 `rt60_range`），THEN THE Augmentation_Engine SHALL 优先使用场景预设中的参数，并记录信息日志说明覆盖关系

### 需求 12：音频频谱可视化分析工具

**用户故事：** 作为数据工程师，我希望能通过频谱图直观对比原始音频和增强音频的差异，以便快速判断增强效果是否合理并调整参数。

#### 验收标准

1. THE Analysis_Toolkit SHALL 提供 `plot_spectrogram.py` 脚本，支持对单条音频生成 Mel 频谱图和线性频谱图
2. WHEN 同时指定原始音频路径和增强音频路径时，THE Analysis_Toolkit SHALL 生成并排对比的频谱图（上下或左右排列），标注各自的文件名
3. WHEN 指定一个目录时，THE Analysis_Toolkit SHALL 批量生成该目录下所有 wav 文件的频谱图，输出到指定的图片目录
4. THE Analysis_Toolkit SHALL 在频谱图上标注关键声学指标：采样率、时长、峰值电平、RMS 能量
5. THE Analysis_Toolkit SHALL 支持通过命令行参数控制频谱图的分辨率、频率范围、色彩映射方案

### 需求 13：增强参数信息打印与审计工具

**用户故事：** 作为数据工程师，我希望能查看每条增强音频实际使用的参数值（而非配置范围），以便精确定位参数问题。

#### 验收标准

1. WHEN 配置文件中设置 `log_augment_params: true` 时，THE Augmentation_Engine SHALL 为每条增强音频记录实际采样的参数值（包括距离因子、SNR、RT60、低通频率、音量增益、overlap_mode 等）
2. THE Augmentation_Engine SHALL 将参数记录输出为 JSONL 格式文件（`augment_params.jsonl`），每行对应一条增强音频
3. THE Analysis_Toolkit SHALL 提供 `print_augment_params.py` 脚本，支持按文件名查询、按参数范围过滤、按参数维度统计分布
4. THE Analysis_Toolkit SHALL 提供 `plot_param_distribution.py` 脚本，将参数分布可视化为直方图和散点图（如 SNR vs RT60 的联合分布）
5. WHEN 指定 `--summary` 参数时，THE `print_augment_params.py` SHALL 输出所有参数维度的统计摘要（均值、标准差、最小值、最大值、四分位数）

### 需求 14：真实业务数据声学基线分析（增强前置步骤）

**用户故事：** 作为数据工程师，我希望在配置增强参数之前，先分析真实业务数据（`s10_data_reflow/labeled/`）的声学特征分布，以此作为增强参数校准的基准，避免凭经验盲目设定参数。

#### 验收标准

1. THE Analysis_Toolkit SHALL 提供 `analyze_real_data.py` 脚本，支持对指定目录下的真实音频进行批量声学特征提取
2. THE `analyze_real_data.py` SHALL 提取以下特征：采样率、时长分布、RMS 能量分布、峰值电平分布、频谱质心分布、信噪比估计、频谱包络特征
3. THE `analyze_real_data.py` SHALL 输出声学基线报告（文本格式 `acoustic_baseline.txt` + JSON 格式 `acoustic_baseline.json`），包含各特征的分布统计（均值、标准差、百分位数）
4. THE `analyze_real_data.py` SHALL 支持随机采样模式（`--sample_n`），避免对全量 20 万条数据进行分析时耗时过长
5. THE 声学基线报告 SHALL 作为增强参数配置的输入依据，在增强流程启动前生成

### 需求 15：基于基线的增强参数推荐

**用户故事：** 作为数据工程师，我希望系统能基于真实数据的声学基线自动推荐增强参数范围，减少手动调参的工作量，并确保增强数据与真实数据的声学特征分布一致。

#### 验收标准

1. THE Analysis_Toolkit SHALL 提供 `recommend_params.py` 脚本，读取声学基线报告（`acoustic_baseline.json`）并输出推荐的增强参数配置
2. THE `recommend_params.py` SHALL 基于真实数据的 SNR 分布推荐 `snr_range`，基于频谱包络推荐 `eq_random` 参数，基于能量分布推荐 `volume_perturb` 的 `gain_range`
3. THE `recommend_params.py` SHALL 输出推荐参数为 YAML 片段格式，可直接复制到增强配置文件中
4. THE `recommend_params.py` SHALL 同时输出推荐理由，说明每个参数值是如何从基线数据推导的
5. WHEN 指定 `--apply` 参数和目标配置文件路径时，THE `recommend_params.py` SHALL 将推荐参数自动合并到目标配置文件中（保留用户已有的自定义配置，仅填充未设置的参数）

### 需求 16：增强后分布验证

**用户故事：** 作为数据工程师，我希望在增强完成后能对比增强数据与真实数据的声学特征分布，验证增强效果是否与真实场景匹配。

#### 验收标准

1. THE Analysis_Toolkit SHALL 提供 `compare_distributions.py` 脚本，支持对比真实数据和增强数据的特征分布，生成对比图表
2. WHEN 真实数据和增强数据的某项特征分布差异显著（KL 散度超过阈值）时，THE `compare_distributions.py` SHALL 在报告中标注该特征并建议参数调整方向
3. THE `compare_distributions.py` SHALL 支持指定声学基线报告路径（`--baseline`）和增强数据目录路径（`--augmented_dir`），自动提取增强数据特征并与基线对比
4. THE `compare_distributions.py` SHALL 输出对比报告（文本格式 + 图表），包含各特征维度的分布对比图和 KL 散度值

### 需求 17：增强效果快速试听与 A/B 对比工具

**用户故事：** 作为数据工程师，我希望能快速试听不同参数配置下的增强效果，以便通过听觉直观判断参数是否合理。

#### 验收标准

1. THE Analysis_Toolkit SHALL 提供 `quick_augment.py` 脚本，支持对单条音频应用指定配置并输出增强结果，无需运行完整的批量增强流程
2. WHEN 指定多个配置文件时，THE `quick_augment.py` SHALL 对同一条原始音频分别应用各配置，输出多个增强版本，文件名包含配置名称标识
3. THE `quick_augment.py` SHALL 支持通过命令行参数临时覆盖配置文件中的特定参数（如 `--override snr_range=[5,15]`），无需修改配置文件
4. THE Analysis_Toolkit SHALL 提供 `generate_ab_report.py` 脚本，生成包含原始音频和多个增强版本的 HTML 对比页面，支持在浏览器中直接播放和对比
5. THE `generate_ab_report.py` SHALL 在 HTML 页面中展示每个音频的频谱图缩略图和关键参数信息

### 需求 18：开发过程版本控制与日志记录

**用户故事：** 作为数据工程师，我希望整个开发过程通过 Git 管理，并有详细的开发日志记录每个阶段的成果和决策，以便追溯和复现。

#### 验收标准

1. THE 项目根目录 SHALL 初始化为 Git 仓库（如尚未初始化）
2. WHEN 完成一个阶段性成果（如新增工具脚本、参数调优、架构修改）时，THE 开发过程 SHALL 创建一个 Git 提交，提交信息包含简要的变更描述
3. THE 项目根目录 SHALL 包含 `DEVLOG.md` 文件，记录每个阶段性成果的详细信息，包括：变更内容、变更原因、关键决策、参数调整记录
4. THE `DEVLOG.md` SHALL 按时间倒序排列，最新的记录在最前面
5. THE 项目根目录 SHALL 包含合理的 `.gitignore` 文件，排除大型音频数据文件、临时输出、Python 缓存等不应纳入版本控制的内容
