"""Badcase 模式 → 前置数据优化手段 的映射表。

这是本项目"闭环优化"的核心关联表：evaluation 阶段发现的 badcase 错误模式，
对应到 data / post_process 优化阶段的具体手段。Kiro 和 LLMSkillEngine 都
引用这张表来生成优化建议。

设计原则：
- 每个映射是一个 Strategy 字典，含触发条件 + 推荐手段 + CLI 模板
- 触发条件基于 EvaluationReport 的统计量（错误类型分布、CER、场景指标等）
- 推荐手段对应具体的脚本和参数
- 新增/修改映射 = 添加新 Strategy 字典，无需改 Kiro 代码

使用方式：
    from scripts.decision.badcase_strategy_map import match_strategies, DATA_STRATEGIES

    # 给定评估报告，返回匹配的前置手段列表
    matches = match_strategies(evaluation_report, badcase_list)
    for m in matches:
        print(m["name"], m["commands"])

同时这张表会在系统首次初始化时通过 manage_knowledge.py 注入到知识库作为
seed 知识，随着反馈积累自动调整置信度（参见 Skill Evolution 流程）。
"""

from __future__ import annotations

from typing import Any, Callable, Optional


# =====================================================================
# 前置数据优化手段字典
#
# 每条策略格式：
# {
#     "name": 策略名（唯一标识，也用作知识库 key）
#     "dimension": "data" | "post_process" | "train_config"
#     "category": "augmentation" | "preprocess" | "hotword" | "spec_aug"
#     "trigger": 触发条件的 Python 表达式（字符串），以 rep/badcases 为变量
#     "description": 人类可读的说明
#     "commands": 推荐执行的 CLI 命令列表（顺序执行）
#     "config_changes": 可选，建议同时修改的配置字段
#     "priority": 1-5，数字越大优先级越高
#     "confidence": 0.0-1.0，初始置信度，随反馈调整
#     "rationale": 为什么这个手段能解决对应模式
# }
# =====================================================================

DATA_STRATEGIES: list[dict[str, Any]] = [
    # ---- 插入错误为主：对应背景人声/噪声干扰 ----
    {
        "name": "babble_augment_for_insertion",
        "dimension": "data",
        "category": "augmentation",
        "trigger": "rep.error_type_distribution.get('insertion', 0) > rep.error_type_distribution.get('substitution', 0)",
        "description": "插入错误占主导，通常由背景人声/噪声被误识别为内容造成",
        "commands": [
            "python scripts/run_augmentation.py --all --strategy babble",
            "python scripts/run_augmentation.py --all --strategy noise",
        ],
        "config_changes": {
            "augmentation.strategies[babble].probability": 0.5,
        },
        "priority": 5,
        "confidence": 0.80,
        "rationale": "babble 增强让模型在训练时见过大量背景人声，学会忽略它们。noise 增强同理提升噪声鲁棒性。",
    },

    # ---- 替换错误为主：热词纠正 + SpecAugment ----
    {
        "name": "hotwords_from_substitution",
        "dimension": "post_process",
        "category": "hotword",
        "trigger": "rep.error_type_distribution.get('substitution', 0) >= max(rep.error_type_distribution.get('insertion', 0), rep.error_type_distribution.get('deletion', 0))",
        "description": "替换错误占主导，通常是同音字/近音字误识或领域词错误",
        "commands": [
            "python scripts/extract_hotwords.py --eval-report {eval_report} --merge",
            "python scripts/apply_postprocess.py update-config --hotword-file configs/post_process/hotwords.txt",
        ],
        "priority": 5,
        "confidence": 0.90,
        "rationale": "替换错误最直接的修复手段就是热词纠正 — 从 badcase 里挖掘高频替换对，加到热词表立刻生效",
    },
    {
        "name": "specaug_for_substitution",
        "dimension": "train_config",
        "category": "spec_aug",
        "trigger": "rep.overall_cer > 0.10 and rep.error_type_distribution.get('substitution', 0) > 0",
        "description": "替换错误伴随整体 CER 偏高，模型过拟合特定频谱模式",
        "commands": [
            "# 在 configs/funasr_config.yaml 启用 spec_aug",
            "# training.spec_aug.enabled: true",
        ],
        "config_changes": {
            "training.spec_aug.enabled": True,
            "training.spec_aug.num_t_mask": 2,
            "training.spec_aug.num_f_mask": 2,
        },
        "priority": 3,
        "confidence": 0.70,
        "rationale": "SpecAugment 在训练时随机遮挡时间/频率块，强迫模型学习鲁棒特征，缓解过拟合导致的替换错误",
    },

    # ---- 删除错误为主：混响 + 语速增强 ----
    {
        "name": "reverb_speed_for_deletion",
        "dimension": "data",
        "category": "augmentation",
        "trigger": "rep.error_type_distribution.get('deletion', 0) > rep.error_type_distribution.get('insertion', 0) and rep.error_type_distribution.get('deletion', 0) > rep.error_type_distribution.get('substitution', 0) * 0.5",
        "description": "删除错误占主导，通常由混响/快语速导致短字词被吞掉",
        "commands": [
            "python scripts/run_augmentation.py --all --strategy reverb",
            "python scripts/run_augmentation.py --all --strategy speed",
        ],
        "priority": 4,
        "confidence": 0.75,
        "rationale": "reverb 增强让模型在混响环境下训练，speed 增强覆盖不同语速，两者都针对'字被吞'问题",
    },

    # ---- 整体 CER 高：综合增强 ----
    {
        "name": "full_augment_for_high_cer",
        "dimension": "data",
        "category": "augmentation",
        "trigger": "rep.overall_cer > 0.20",
        "description": "整体 CER 过高（> 20%），需要综合增强扩充训练数据多样性",
        "commands": [
            "python scripts/run_augmentation.py --all",
        ],
        "priority": 4,
        "confidence": 0.70,
        "rationale": "CER 过高时单一策略收益有限，用配置中的所有策略（babble+music+noise+reverb）全量增强",
    },

    # ---- 场景不足：基于 scene_distribution 判断 ----
    {
        "name": "balance_scene_augment",
        "dimension": "data",
        "category": "augmentation",
        "trigger": "any(sm.sample_count < 20 for sm in rep.scene_metrics) if rep.scene_metrics else False",
        "description": "某些场景样本数不足（< 20），需要针对性增强",
        "commands": [
            "# 先对增强后数据追加元数据",
            "python scripts/preprocess_metadata.py --all",
            "# 再对不足场景做针对性增强",
            "python scripts/run_augmentation.py --all --strategy babble",
        ],
        "priority": 4,
        "confidence": 0.65,
        "rationale": "模型在样本不足的场景上性能差。先做场景标注，再对该场景加强增强",
    },

    # ---- 音量扰动：音量敏感时使用 ----
    {
        "name": "volume_augment_for_sensitivity",
        "dimension": "data",
        "category": "augmentation",
        "trigger": "False",  # 默认不触发，需要人工判断或更复杂检测
        "description": "模型对音量变化敏感（需要人工观察后启用）",
        "commands": [
            "python scripts/run_augmentation.py --all --strategy volume",
        ],
        "priority": 2,
        "confidence": 0.60,
        "rationale": "volume 增强让模型对不同录音音量鲁棒，但通常不是首要手段",
    },
]


# =====================================================================
# Matching engine
# =====================================================================

def match_strategies(
    evaluation_report, badcases: Optional[list] = None
) -> list[dict[str, Any]]:
    """给定评估报告，返回所有匹配触发条件的策略。

    返回按 priority 降序排列的策略列表（副本，含格式化后的 commands）。
    """
    # 把 EvaluationReport 当作 rep 变量暴露给 trigger 表达式
    ctx = {
        "rep": evaluation_report,
        "badcases": badcases or [],
    }

    matched: list[dict[str, Any]] = []
    for strategy in DATA_STRATEGIES:
        trigger = strategy.get("trigger", "False")
        try:
            if eval(trigger, {"__builtins__": {}}, ctx):
                # 复制一份返回，避免调用者修改原始表
                out = {k: v for k, v in strategy.items() if k != "trigger"}
                matched.append(out)
        except Exception as e:
            # 触发表达式有问题不应阻塞其他策略
            import logging
            logging.getLogger(__name__).debug(
                "Strategy '%s' trigger failed: %s", strategy.get("name"), e,
            )

    # 按 priority 降序排
    matched.sort(key=lambda s: -s.get("priority", 0))
    return matched


def list_all_strategies() -> list[dict[str, Any]]:
    """返回所有注册策略的副本（用于文档/查询）。"""
    return [dict(s) for s in DATA_STRATEGIES]


def seed_knowledge_base(kb) -> int:
    """把 DATA_STRATEGIES 作为 seed 知识批量写入知识库。

    Parameters
    ----------
    kb : SkillKnowledgeBase
        已初始化的知识库实例

    Returns
    -------
    int
        写入的条目数
    """
    from scripts.core.models import KnowledgeEntry
    count = 0
    for strategy in DATA_STRATEGIES:
        # 映射到 knowledge_type
        dim = strategy.get("dimension", "data")
        if dim == "post_process":
            ktype = "scene_solution"
        elif dim == "train_config":
            ktype = "param_effect"
        else:
            ktype = "data_strategy"

        entry = KnowledgeEntry(
            knowledge_type=ktype,
            content={
                "name": strategy["name"],
                "category": strategy.get("category"),
                "description": strategy.get("description"),
                "trigger": strategy.get("trigger"),
                "commands": strategy.get("commands", []),
                "rationale": strategy.get("rationale"),
            },
            confidence=strategy.get("confidence", 0.7),
            source_iteration_id="seed",
        )
        try:
            kb.add_knowledge(entry)
            count += 1
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(
                "Failed to seed strategy '%s': %s", strategy["name"], e,
            )
    return count
