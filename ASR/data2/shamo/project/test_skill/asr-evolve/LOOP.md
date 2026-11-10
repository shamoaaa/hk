# LOOP.md — ASR-Evolve 环路运维说明

## 设计原则

1. **Kiro 决策，代码执行**：所有判断类决策（质量阈值、配方选择、是否继续迭代）由 Kiro Agent 做出
2. **阶段化 + 幂等 + 可断点**：每个阶段产物存在则跳过，`--force` 可重跑
3. **渐进信任（L1→L3）**：从报告开始，逐步放权
4. **maker/checker 分离**：训练产出由独立验证者判定，Agent 不自评

## 正在运行的环路

三个环路对应 `.kiro/hooks/` 下的三个 hook 文件。

### 1. Session Triage（会话分诊，L1）
- Hook：`.kiro/hooks/asr-session-triage.json`
- 触发：`SessionStart`
- 技能：`.kiro/steering/asr-optimization.md`
- 状态：`STATE.md`
- 做什么：读 STATE.md / 实验树 / DEAD_ENDS / INSIGHTS / loop-budget，输出精简简报、提醒待办

### 2. Post-Train Verify（训练后验证，L2）
- Hook：`.kiro/hooks/asr-post-train-verify.json`
- 触发：`PostTaskExec`（Spec 任务完成后）
- 做什么（maker/checker 中的 checker，默认立场 FAIL）：检查输出目录 → `.pt` 存在 → train.log 无 OOM/RuntimeError → 若有评估结果则 CER 改善 > 0.1% 才算有效 → 判定 PASS/FAIL/NEED_HUMAN
- 失败处理：连续 2 次无改善 → 追加 DEAD_ENDS + STATE.md 标注"需换方向"
- 注意：train 阶段代码（`pipeline/stages/train.py`）**内置**了轻量健康检查（`.pt` 存在性 / 静默崩溃拦截）；本 hook 是在任务级别做的独立复核，两者互补。

### 3. Config Guard（配置守卫，L1）
- Hook：`.kiro/hooks/asr-config-guard.json`
- 触发：`PostFileSave`（matcher 匹配 `.yaml` 文件）
- 做什么：检查配方变更是否违反业务约束（seaco_id=8377、freeze_param 含 hotword_output_layer/seaco_decoder、lr、autonomy_level 合法性等）；无问题时保持沉默

## 预算与急停

- Token 预算：`loop-budget.md`
- GPU 预算：每日最多 3 次训练
- 急停：STATE.md 顶部加 `## ⚠️ LOOPS PAUSED`
- 连续失败：2 次 CER 无改善 → 暂停并升级给人

## 安全闸门

- 训练启动需人工确认（L2 模式）
- 破坏性阶段（clean）默认关闭
- `.env` / 密钥文件 / 远程凭证不可自动修改
- seaco_id 启动前校验

## 多环路协调

优先级：Post-Train Verify > Config Guard > Session Triage
（三者天然不冲突：分别在不同事件触发）

## 自治级别配置

```yaml
# configs/asr_evolve.yaml → autonomy_level
autonomy_level: L2  # L1=只报告 L2=需确认训练 L3=全自主
```
