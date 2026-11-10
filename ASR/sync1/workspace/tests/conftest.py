"""Public fixtures for the ASR Training Optimizer test suite.

Provides reusable fixtures for temporary workspaces, sample data instances,
and database access used across all test modules.
"""

import os

import pytest
from hypothesis import settings

from scripts.core.database import Database
from scripts.core.models import (
    BadCase,
    EvaluationReport,
    HotwordEntry,
    LRScheduleType,
    ModelConfig,
    PostProcessConfig,
    RegexRule,
    SceneMetrics,
    TrainingConfig,
    TrainingMetrics,
)

# ---------------------------------------------------------------------------
# Hypothesis configuration
# ---------------------------------------------------------------------------

settings.register_profile("default", max_examples=100)
settings.register_profile("ci", max_examples=200)
settings.load_profile("default")


# ---------------------------------------------------------------------------
# Temporary workspace fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary workspace directory mirroring the real structure.

    Directories created::

        scripts/
        configs/model/
        configs/training/
        configs/post_process/
        data/original/
        data/augmented/
        models/baseline/
        models/iterations/
        logs/training/
        logs/optimization/
        reports/evaluation/
        reports/dataset/
        reports/augmentation/
        knowledge/skill_versions/
    """
    dirs = [
        "scripts",
        "configs/model",
        "configs/training",
        "configs/post_process",
        "data/original",
        "data/augmented",
        "models/baseline",
        "models/iterations",
        "logs/training",
        "logs/optimization",
        "reports/evaluation",
        "reports/dataset",
        "reports/augmentation",
        "knowledge/skill_versions",
    ]
    for d in dirs:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return tmp_path


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_training_metrics():
    """Return a list of sample TrainingMetrics instances."""
    return [
        TrainingMetrics(
            step=0,
            epoch=0,
            train_loss=2.5,
            val_loss=2.8,
            learning_rate=1e-4,
            gradient_norm=1.2,
            cer=0.45,
            wer=0.55,
            timestamp="2024-01-01T00:00:00",
        ),
        TrainingMetrics(
            step=100,
            epoch=0,
            train_loss=1.8,
            val_loss=2.1,
            learning_rate=1e-4,
            gradient_norm=0.9,
            cer=0.35,
            wer=0.45,
            timestamp="2024-01-01T00:10:00",
        ),
        TrainingMetrics(
            step=200,
            epoch=1,
            train_loss=1.2,
            val_loss=1.5,
            learning_rate=5e-5,
            gradient_norm=0.7,
            cer=0.25,
            wer=0.35,
            timestamp="2024-01-01T00:20:00",
        ),
    ]


@pytest.fixture
def sample_badcases():
    """Return a list of sample BadCase instances."""
    return [
        BadCase(
            audio_path="/data/audio_001.wav",
            reference_text="今天天气很好",
            hypothesis_text="今天天气很号",
            cer=0.167,
            wer=0.2,
            error_details=[
                {"type": "substitution", "ref": "好", "hyp": "号", "position": 5}
            ],
            scene_type="clean",
        ),
        BadCase(
            audio_path="/data/audio_002.wav",
            reference_text="请打开空调",
            hypothesis_text="请打开空",
            cer=0.2,
            wer=0.25,
            error_details=[
                {"type": "deletion", "ref": "调", "hyp": "", "position": 4}
            ],
            scene_type="background_voice",
        ),
        BadCase(
            audio_path="/data/audio_003.wav",
            reference_text="明天下午开会",
            hypothesis_text="明天下午来开会",
            cer=0.167,
            wer=0.2,
            error_details=[
                {"type": "insertion", "ref": "", "hyp": "来", "position": 4}
            ],
            scene_type="background_noise",
        ),
    ]


@pytest.fixture
def sample_scene_metrics():
    """Return a list of sample SceneMetrics (one per scene type)."""
    return [
        SceneMetrics(scene_type="clean", cer=0.05, wer=0.08, sample_count=500),
        SceneMetrics(
            scene_type="background_voice", cer=0.15, wer=0.22, sample_count=300
        ),
        SceneMetrics(
            scene_type="background_noise", cer=0.10, wer=0.15, sample_count=200
        ),
    ]


@pytest.fixture
def sample_evaluation_report(sample_scene_metrics, sample_badcases):
    """Return a sample EvaluationReport."""
    return EvaluationReport(
        overall_cer=0.09,
        overall_wer=0.14,
        scene_metrics=sample_scene_metrics,
        badcases=sample_badcases,
        error_type_distribution={
            "substitution": 45,
            "insertion": 20,
            "deletion": 35,
        },
    )


@pytest.fixture
def sample_model_config():
    """Return a sample ModelConfig."""
    return ModelConfig(
        attention_heads=8,
        hidden_dim=512,
        seaco_params={
            "co_decoder_layers": 2,
            "attention_weight": 0.5,
            "vocab_size": 8404,
        },
        speech_enhance_config={
            "type": "conv_tasnet",
            "channels": 64,
            "layers": 4,
        },
        extra_params={"dropout": 0.1, "activation": "relu"},
    )


@pytest.fixture
def sample_training_config():
    """Return a sample TrainingConfig."""
    return TrainingConfig(
        learning_rate=1e-4,
        batch_size=32,
        warmup_steps=2000,
        weight_decay=0.01,
        lr_schedule=LRScheduleType.WARMUP_DECAY,
        max_epochs=50,
        gradient_clip=5.0,
        extra_params={"label_smoothing": 0.1},
    )


@pytest.fixture
def sample_post_process_config():
    """Return a sample PostProcessConfig."""
    return PostProcessConfig(
        hotword_entries=[
            HotwordEntry(wrong_text="号", correct_text="好", priority=1),
            HotwordEntry(wrong_text="以经", correct_text="已经", priority=2),
        ],
        regex_rules=[
            RegexRule(
                pattern=r"\s+",
                replacement="",
                description="Remove extra whitespace",
                priority=1,
            ),
        ],
        enable_lm_rerank=True,
        lm_model_path="/models/lm/zh_lm.bin",
        pipeline_order=["hotword", "regex", "lm_rerank"],
    )


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path):
    """Return a temporary database file path."""
    return str(tmp_path / "test_metadata.db")


@pytest.fixture
def db(tmp_path):
    """Return an initialized Database instance with a temp SQLite file.

    The database is automatically closed after the test.
    """
    path = str(tmp_path / "test_metadata.db")
    database = Database(path)
    database.initialize()
    yield database
    database.close()
