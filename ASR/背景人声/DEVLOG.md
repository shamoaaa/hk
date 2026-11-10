# 开发日志 (Dev Log)

本文件记录音频增强系统各阶段性成果和关键决策，按时间倒序排列（最新在前）。

---

## 2026-04-21 — 场景多样性功能实现完成

### 变更内容

实现了音频增强场景多样性功能的完整实现计划，包括以下模块：

**核心引擎组件：**
- `07_augment/scene_resolver.py` — SceneResolver 类：根据场景预设、说话场景和增强链变体解析每条增强样本的完整参数
- `07_augment/diversity_controller.py` — DiversityController 类：管理均衡/加权分配和背景说话人多样性追踪
- `07_augment/param_logger.py` — ParamLogger 类：将实际增强参数记录到 JSONL 文件
- `07_augment/diversity_reporter.py` — DiversityReporter 类：生成多样性统计报告（文本 + JSON）

**引擎集成：**
- `07_augment/augment_engine.py` — 扩展 AudioAugmentor，集成 SceneResolver、DiversityController、ParamLogger
- `07_augment/augment_dir.py` — 扩展支持 `--bg_speaker_dir` 和批次生命周期
- `07_augment/augment_manifest.py` — 扩展支持 `--bg_speaker_dir` 和批次生命周期

**分析工具集（07_augment/tools/）：**
- `analyze_real_data.py` — 真实数据声学基线分析
- `recommend_params.py` — 基于基线的增强参数推荐
- `compare_distributions.py` — 真实 vs 增强数据分布对比
- `plot_spectrogram.py` — 频谱图可视化
- `print_augment_params.py` — 增强参数查询与审计
- `plot_param_distribution.py` — 参数分布可视化
- `quick_augment.py` — 单文件快速增强
- `generate_ab_report.py` — HTML A/B 对比报告

**配置与文档：**
- `augment_config_scene_diversity_example.yaml` — 完整示例配置，包含所有新字段的文档注释
- `.gitignore` — 排除大型音频文件和临时输出

### 变更原因

当前增强系统缺乏系统化的场景组合覆盖。本次变更引入三层场景多样性：
1. 房间级多样性：场景预设（客厅、卧室、书房）
2. 场景级多样性：说话场景类型（单人安静、单人噪声、多人说话）
3. 参数级多样性：距离关联参数采样

同时引入"分析→校准→增强→验证"的迭代工作流，确保增强参数以真实数据为基准。

### 关键决策

1. **SceneResolver 作为独立组件**：不嵌入 augment_single()，保持增强方法的纯净和可测试性
2. **DiversityController 管理均衡分配**：使用 least-used-first 策略保证 max_count - min_count ≤ 1
3. **所有新字段均为可选**：省略时等价于功能关闭，保证向后兼容
4. **背景说话人源分离在调用层处理**：augment_dir.py / augment_manifest.py 决定传入哪个候选池
5. **分析工具集为独立脚本**：不依赖引擎运行时状态，可独立使用

---
