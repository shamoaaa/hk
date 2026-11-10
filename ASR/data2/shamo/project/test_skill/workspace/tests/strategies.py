"""Hypothesis custom strategies for generating ASR Training Optimizer data classes.

Each strategy function returns a ``hypothesis.strategies.SearchStrategy`` that
produces valid, random instances of the corresponding data class.  These are
used in property-based tests throughout the project.
"""

from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

from scripts.core.models import (
    BadCase,
    EffectFeedback,
    EvaluationReport,
    HotwordEntry,
    KnowledgeEntry,
    LRScheduleType,
    ModelConfig,
    PostProcessConfig,
    RegexRule,
    SceneMetrics,
    StrategyWeight,
    TrainingConfig,
    TrainingMetrics,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_SCENE_TYPES = ["clean", "background_voice", "background_noise"]


# ---------------------------------------------------------------------------
# Training_Monitor strategies
# ---------------------------------------------------------------------------


def training_metrics_strategy() -> SearchStrategy[TrainingMetrics]:
    """Generate a ``TrainingMetrics`` with valid ranges.

    - step: 0–10 000
    - epoch: 0–100
    - train_loss: 0.001–10.0 (finite)
    - val_loss: optional 0.001–10.0
    - learning_rate: 1e-7–0.1
    - gradient_norm: 0.0–100.0
    - cer: optional 0.0–1.0
    - wer: optional 0.0–1.0
    """
    return st.builds(
        TrainingMetrics,
        step=st.integers(min_value=0, max_value=10_000),
        epoch=st.integers(min_value=0, max_value=100),
        train_loss=st.floats(
            min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        val_loss=st.one_of(
            st.none(),
            st.floats(
                min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False
            ),
        ),
        learning_rate=st.floats(
            min_value=1e-7, max_value=0.1, allow_nan=False, allow_infinity=False
        ),
        gradient_norm=st.floats(
            min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
        cer=st.one_of(
            st.none(),
            st.floats(
                min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
            ),
        ),
        wer=st.one_of(
            st.none(),
            st.floats(
                min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Result_Analyzer strategies
# ---------------------------------------------------------------------------


def scene_metrics_strategy() -> SearchStrategy[SceneMetrics]:
    """Generate a ``SceneMetrics`` instance.

    - scene_type: one of ``clean``, ``background_voice``, ``background_noise``
    - cer / wer: 0.0–1.0
    - sample_count: 1–10 000
    """
    return st.builds(
        SceneMetrics,
        scene_type=st.sampled_from(_SCENE_TYPES),
        cer=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        wer=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        sample_count=st.integers(min_value=1, max_value=10_000),
    )


def badcase_strategy() -> SearchStrategy[BadCase]:
    """Generate a ``BadCase`` instance.

    - audio_path: arbitrary text
    - reference_text / hypothesis_text: non-empty text
    - cer / wer: 0.0–1.0
    - error_details: list of dicts
    - scene_type: one of the three scene types
    """
    error_detail = st.fixed_dictionaries(
        {
            "type": st.sampled_from(["substitution", "insertion", "deletion"]),
            "ref": st.text(min_size=0, max_size=5),
            "hyp": st.text(min_size=0, max_size=5),
            "position": st.integers(min_value=0, max_value=100),
        }
    )
    return st.builds(
        BadCase,
        audio_path=st.text(min_size=1, max_size=100),
        reference_text=st.text(min_size=1, max_size=50),
        hypothesis_text=st.text(min_size=1, max_size=50),
        cer=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        wer=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        error_details=st.lists(error_detail, min_size=0, max_size=5),
        scene_type=st.sampled_from(_SCENE_TYPES),
    )


def evaluation_report_strategy() -> SearchStrategy[EvaluationReport]:
    """Generate an ``EvaluationReport``.

    - overall_cer / overall_wer: 0.0–1.0
    - scene_metrics: 1–5 items
    - badcases: 0–10 items
    - error_type_distribution: substitution / insertion / deletion counts
    """
    return st.builds(
        EvaluationReport,
        overall_cer=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        overall_wer=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        scene_metrics=st.lists(scene_metrics_strategy(), min_size=1, max_size=5),
        badcases=st.lists(badcase_strategy(), min_size=0, max_size=10),
        error_type_distribution=st.fixed_dictionaries(
            {
                "substitution": st.integers(min_value=0, max_value=1000),
                "insertion": st.integers(min_value=0, max_value=1000),
                "deletion": st.integers(min_value=0, max_value=1000),
            }
        ),
    )


# ---------------------------------------------------------------------------
# Model_Optimizer strategies
# ---------------------------------------------------------------------------


def model_config_strategy() -> SearchStrategy[ModelConfig]:
    """Generate a ``ModelConfig`` with valid structural constraints.

    - attention_heads: power of 2 (1, 2, 4, 8, 16)
    - hidden_dim: a multiple of attention_heads (ensures divisibility)
    - seaco_params: dict with reasonable values
    - speech_enhance_config: optional dict
    """

    def _build_model_config(heads: int, dim_multiplier: int) -> ModelConfig:
        return ModelConfig(
            attention_heads=heads,
            hidden_dim=heads * dim_multiplier,
            seaco_params={
                "co_decoder_layers": dim_multiplier % 6 + 1,
                "attention_weight": round(dim_multiplier / 100.0, 2),
                "vocab_size": 8404,
            },
            speech_enhance_config=(
                {"type": "conv_tasnet", "channels": heads * 8}
                if dim_multiplier % 2 == 0
                else None
            ),
            extra_params={},
        )

    return st.builds(
        _build_model_config,
        heads=st.sampled_from([1, 2, 4, 8, 16]),
        dim_multiplier=st.integers(min_value=1, max_value=128),
    )


# ---------------------------------------------------------------------------
# Param_Tuner strategies
# ---------------------------------------------------------------------------


def training_config_strategy() -> SearchStrategy[TrainingConfig]:
    """Generate a ``TrainingConfig`` with valid ranges.

    - learning_rate: 1e-6–0.01
    - batch_size: 1–128
    - warmup_steps: 0–10 000
    - weight_decay: 0.0–0.1
    - lr_schedule: one of the three ``LRScheduleType`` values
    - max_epochs: 1–200
    - gradient_clip: 0.1–10.0
    """
    return st.builds(
        TrainingConfig,
        learning_rate=st.floats(
            min_value=1e-6, max_value=0.01, allow_nan=False, allow_infinity=False
        ),
        batch_size=st.integers(min_value=1, max_value=128),
        warmup_steps=st.integers(min_value=0, max_value=10_000),
        weight_decay=st.floats(
            min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False
        ),
        lr_schedule=st.sampled_from(list(LRScheduleType)),
        max_epochs=st.integers(min_value=1, max_value=200),
        gradient_clip=st.floats(
            min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        extra_params=st.just({}),
    )


# ---------------------------------------------------------------------------
# Post_Processor strategies
# ---------------------------------------------------------------------------


def hotword_entry_strategy() -> SearchStrategy[HotwordEntry]:
    """Generate a ``HotwordEntry``."""
    return st.builds(
        HotwordEntry,
        wrong_text=st.text(min_size=1, max_size=20),
        correct_text=st.text(min_size=1, max_size=20),
        priority=st.integers(min_value=0, max_value=10),
    )


def regex_rule_strategy() -> SearchStrategy[RegexRule]:
    """Generate a ``RegexRule`` with simple patterns."""
    # Use simple literal patterns to avoid invalid regex
    return st.builds(
        RegexRule,
        pattern=st.from_regex(r"[a-z]{1,10}", fullmatch=True),
        replacement=st.text(min_size=0, max_size=20),
        description=st.text(min_size=0, max_size=50),
        priority=st.integers(min_value=0, max_value=10),
    )


def post_process_config_strategy() -> SearchStrategy[PostProcessConfig]:
    """Generate a ``PostProcessConfig``."""
    return st.builds(
        PostProcessConfig,
        hotword_entries=st.lists(hotword_entry_strategy(), min_size=0, max_size=5),
        regex_rules=st.lists(regex_rule_strategy(), min_size=0, max_size=5),
        enable_lm_rerank=st.booleans(),
        lm_model_path=st.one_of(st.none(), st.text(min_size=1, max_size=50)),
        pipeline_order=st.just(["hotword", "regex", "lm_rerank"]),
    )


# ---------------------------------------------------------------------------
# Skill_Evolution_Engine strategies
# ---------------------------------------------------------------------------


def effect_feedback_strategy() -> SearchStrategy[EffectFeedback]:
    """Generate an ``EffectFeedback``.

    ``is_positive`` is derived from ``cer_before > cer_after``.
    """

    def _build(
        iteration_id: str,
        suggestion_id: str,
        cer_before: float,
        cer_after: float,
        wer_before: float,
        wer_after: float,
    ) -> EffectFeedback:
        return EffectFeedback(
            iteration_id=iteration_id,
            suggestion_id=suggestion_id,
            suggestion_content="auto-generated suggestion",
            input_analysis={},
            param_changes={},
            cer_before=cer_before,
            cer_after=cer_after,
            wer_before=wer_before,
            wer_after=wer_after,
            is_positive=cer_after < cer_before,
            timestamp="2024-01-01T00:00:00",
        )

    return st.builds(
        _build,
        iteration_id=st.text(min_size=1, max_size=20),
        suggestion_id=st.text(min_size=1, max_size=20),
        cer_before=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        cer_after=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        wer_before=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        wer_after=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )


def knowledge_entry_strategy() -> SearchStrategy[KnowledgeEntry]:
    """Generate a ``KnowledgeEntry``."""
    return st.builds(
        KnowledgeEntry,
        knowledge_type=st.sampled_from(
            ["scene_solution", "param_effect", "data_strategy"]
        ),
        content=st.fixed_dictionaries(
            {
                "description": st.text(min_size=1, max_size=50),
                "solution": st.text(min_size=1, max_size=50),
            }
        ),
        confidence=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        source_iteration_id=st.text(min_size=1, max_size=20),
        created_at=st.just("2024-01-01T00:00:00"),
        updated_at=st.just("2024-01-01T00:00:00"),
    )


def strategy_weight_strategy() -> SearchStrategy[StrategyWeight]:
    """Generate a ``StrategyWeight``."""
    return st.builds(
        StrategyWeight,
        strategy_name=st.text(min_size=1, max_size=30),
        weight=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        success_rate=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        sample_count=st.integers(min_value=0, max_value=1000),
    )
