"""Tests for the DatasetAnalyzer class.

Covers:
- analyze_basic_stats computes total samples, duration stats, empty dataset
- analyze_snr filters by threshold, empty dataset, all above/below threshold
- analyze_text computes character distribution, word frequency, sentence length
- detect_anomalies detects empty labels, duration mismatch, corrupted audio
- analyze_scene_distribution computes counts, ratios, underrepresented scenes
- generate_quality_report aggregates issues sorted by severity
- CLI argument parsing and output format
"""

import json
import os

import pytest

from scripts.analysis.dataset_analyzer import DatasetAnalyzer
from scripts.core.models import (
    AnomalySample,
    DatasetQualityReport,
    DatasetStats,
    SceneDistribution,
    TextStats,
)


# =====================================================================
# Helper: create manifest.jsonl in a temp directory
# =====================================================================


def _write_manifest(tmp_path, records: list[dict]) -> str:
    """Write a manifest.jsonl file and return the dataset directory path."""
    dataset_dir = str(tmp_path / "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    manifest_path = os.path.join(dataset_dir, "manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return dataset_dir


# =====================================================================
# Sample data
# =====================================================================

_SAMPLE_RECORDS = [
    {
        "audio_path": "audio/001.wav",
        "text": "今天天气很好",
        "duration": 3.0,
        "scene_type": "clean",
        "snr": 25.0,
    },
    {
        "audio_path": "audio/002.wav",
        "text": "请打开空调",
        "duration": 2.5,
        "scene_type": "background_voice",
        "snr": 8.0,
    },
    {
        "audio_path": "audio/003.wav",
        "text": "明天下午开会",
        "duration": 4.0,
        "scene_type": "background_noise",
        "snr": 15.0,
    },
    {
        "audio_path": "audio/004.wav",
        "text": "这是一个测试句子用来验证功能",
        "duration": 6.0,
        "scene_type": "clean",
        "snr": 30.0,
    },
    {
        "audio_path": "audio/005.wav",
        "text": "你好世界",
        "duration": 1.5,
        "scene_type": "background_voice",
        "snr": 5.0,
    },
]


# =====================================================================
# TestAnalyzeBasicStats
# =====================================================================


class TestAnalyzeBasicStats:
    """Tests for DatasetAnalyzer.analyze_basic_stats."""

    def test_total_samples(self):
        """Total samples count matches input length."""
        analyzer = DatasetAnalyzer()
        stats = analyzer.analyze_basic_stats(samples=_SAMPLE_RECORDS)
        assert stats.total_samples == 5

    def test_total_duration_hours(self):
        """Total duration in hours is computed correctly."""
        analyzer = DatasetAnalyzer()
        stats = analyzer.analyze_basic_stats(samples=_SAMPLE_RECORDS)
        total_seconds = 3.0 + 2.5 + 4.0 + 6.0 + 1.5
        expected_hours = total_seconds / 3600.0
        assert stats.total_duration_hours == pytest.approx(expected_hours)

    def test_avg_duration_seconds(self):
        """Average duration in seconds is computed correctly."""
        analyzer = DatasetAnalyzer()
        stats = analyzer.analyze_basic_stats(samples=_SAMPLE_RECORDS)
        total_seconds = 3.0 + 2.5 + 4.0 + 6.0 + 1.5
        expected_avg = total_seconds / 5
        assert stats.avg_duration_seconds == pytest.approx(expected_avg)

    def test_duration_distribution(self):
        """Duration distribution bins are populated correctly."""
        analyzer = DatasetAnalyzer()
        stats = analyzer.analyze_basic_stats(samples=_SAMPLE_RECORDS)
        # 1.5, 2.5, 3.0, 4.0 → "0-5s"; 6.0 → "5-10s"
        assert stats.duration_distribution["0-5s"] == 4
        assert stats.duration_distribution["5-10s"] == 1
        assert stats.duration_distribution["10-20s"] == 0
        assert stats.duration_distribution["20-30s"] == 0
        assert stats.duration_distribution["30+s"] == 0

    def test_empty_dataset(self):
        """Empty dataset returns zero stats."""
        analyzer = DatasetAnalyzer()
        stats = analyzer.analyze_basic_stats(samples=[])
        assert stats.total_samples == 0
        assert stats.total_duration_hours == 0.0
        assert stats.avg_duration_seconds == 0.0

    def test_from_dataset_path(self, tmp_path):
        """Stats can be loaded from a manifest file on disk."""
        dataset_dir = _write_manifest(tmp_path, _SAMPLE_RECORDS)
        analyzer = DatasetAnalyzer()
        stats = analyzer.analyze_basic_stats(dataset_path=dataset_dir)
        assert stats.total_samples == 5

    def test_single_sample(self):
        """Single sample dataset computes correctly."""
        records = [{"audio_path": "a.wav", "text": "hello", "duration": 7.0}]
        analyzer = DatasetAnalyzer()
        stats = analyzer.analyze_basic_stats(samples=records)
        assert stats.total_samples == 1
        assert stats.avg_duration_seconds == pytest.approx(7.0)
        assert stats.duration_distribution["5-10s"] == 1


# =====================================================================
# TestAnalyzeSNR
# =====================================================================


class TestAnalyzeSNR:
    """Tests for DatasetAnalyzer.analyze_snr."""

    def test_filters_below_threshold(self):
        """Samples with SNR below threshold are returned."""
        analyzer = DatasetAnalyzer(snr_threshold=10.0)
        low_quality = analyzer.analyze_snr(samples=_SAMPLE_RECORDS)
        # SNR 8.0 and 5.0 are below 10.0
        assert len(low_quality) == 2
        assert "audio/002.wav" in low_quality
        assert "audio/005.wav" in low_quality

    def test_custom_threshold(self):
        """Custom threshold overrides default."""
        analyzer = DatasetAnalyzer(snr_threshold=10.0)
        low_quality = analyzer.analyze_snr(samples=_SAMPLE_RECORDS, threshold=20.0)
        # SNR 8.0, 5.0, 15.0 are below 20.0
        assert len(low_quality) == 3

    def test_empty_dataset(self):
        """Empty dataset returns empty list."""
        analyzer = DatasetAnalyzer()
        low_quality = analyzer.analyze_snr(samples=[])
        assert low_quality == []

    def test_all_above_threshold(self):
        """No samples returned when all are above threshold."""
        records = [
            {"audio_path": "a.wav", "snr": 20.0},
            {"audio_path": "b.wav", "snr": 30.0},
        ]
        analyzer = DatasetAnalyzer(snr_threshold=10.0)
        low_quality = analyzer.analyze_snr(samples=records)
        assert low_quality == []

    def test_all_below_threshold(self):
        """All samples returned when all are below threshold."""
        records = [
            {"audio_path": "a.wav", "snr": 3.0},
            {"audio_path": "b.wav", "snr": 5.0},
        ]
        analyzer = DatasetAnalyzer(snr_threshold=10.0)
        low_quality = analyzer.analyze_snr(samples=records)
        assert len(low_quality) == 2

    def test_missing_snr_field(self):
        """Samples without SNR field are not flagged."""
        records = [
            {"audio_path": "a.wav"},
            {"audio_path": "b.wav", "snr": 5.0},
        ]
        analyzer = DatasetAnalyzer(snr_threshold=10.0)
        low_quality = analyzer.analyze_snr(samples=records)
        assert len(low_quality) == 1
        assert "b.wav" in low_quality


# =====================================================================
# TestAnalyzeText
# =====================================================================


class TestAnalyzeText:
    """Tests for DatasetAnalyzer.analyze_text."""

    def test_char_distribution(self):
        """Character distribution counts each character correctly."""
        records = [
            {"text": "aab"},
            {"text": "bc"},
        ]
        analyzer = DatasetAnalyzer()
        text_stats = analyzer.analyze_text(samples=records)
        assert text_stats.char_distribution["a"] == 2
        assert text_stats.char_distribution["b"] == 2
        assert text_stats.char_distribution["c"] == 1

    def test_word_frequency(self):
        """Word frequency counts words split on whitespace."""
        records = [
            {"text": "hello world"},
            {"text": "hello python"},
        ]
        analyzer = DatasetAnalyzer()
        text_stats = analyzer.analyze_text(samples=records)
        assert text_stats.word_frequency["hello"] == 2
        assert text_stats.word_frequency["world"] == 1
        assert text_stats.word_frequency["python"] == 1

    def test_sentence_length_distribution(self):
        """Sentence length distribution bins are correct."""
        records = [
            {"text": "short"},           # 5 chars → "0-10"
            {"text": "a" * 15},          # 15 chars → "10-20"
            {"text": "b" * 30},          # 30 chars → "20-50"
            {"text": "c" * 75},          # 75 chars → "50-100"
            {"text": "d" * 120},         # 120 chars → "100+"
        ]
        analyzer = DatasetAnalyzer()
        text_stats = analyzer.analyze_text(samples=records)
        assert text_stats.sentence_length_distribution["0-10"] == 1
        assert text_stats.sentence_length_distribution["10-20"] == 1
        assert text_stats.sentence_length_distribution["20-50"] == 1
        assert text_stats.sentence_length_distribution["50-100"] == 1
        assert text_stats.sentence_length_distribution["100+"] == 1

    def test_empty_dataset(self):
        """Empty dataset returns empty distributions."""
        analyzer = DatasetAnalyzer()
        text_stats = analyzer.analyze_text(samples=[])
        assert text_stats.char_distribution == {}
        assert text_stats.word_frequency == {}

    def test_chinese_text(self):
        """Chinese text characters are counted individually."""
        records = [{"text": "你好世界"}]
        analyzer = DatasetAnalyzer()
        text_stats = analyzer.analyze_text(samples=records)
        assert text_stats.char_distribution["你"] == 1
        assert text_stats.char_distribution["好"] == 1
        # Chinese without spaces → single word
        assert text_stats.word_frequency["你好世界"] == 1

    def test_char_count_conservation(self):
        """Sum of all char counts equals total characters across all texts."""
        analyzer = DatasetAnalyzer()
        text_stats = analyzer.analyze_text(samples=_SAMPLE_RECORDS)
        total_chars = sum(len(s["text"]) for s in _SAMPLE_RECORDS)
        assert sum(text_stats.char_distribution.values()) == total_chars

    def test_word_count_conservation(self):
        """Sum of all word frequencies equals total words across all texts."""
        analyzer = DatasetAnalyzer()
        text_stats = analyzer.analyze_text(samples=_SAMPLE_RECORDS)
        total_words = sum(len(s["text"].split()) for s in _SAMPLE_RECORDS)
        assert sum(text_stats.word_frequency.values()) == total_words


# =====================================================================
# TestDetectAnomalies
# =====================================================================


class TestDetectAnomalies:
    """Tests for DatasetAnalyzer.detect_anomalies."""

    def test_empty_label(self):
        """Empty text is detected as empty_label anomaly."""
        records = [
            {"audio_path": "a.wav", "text": "", "duration": 3.0},
        ]
        analyzer = DatasetAnalyzer()
        anomalies = analyzer.detect_anomalies(samples=records)
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == "empty_label"

    def test_whitespace_only_label(self):
        """Whitespace-only text is detected as empty_label anomaly."""
        records = [
            {"audio_path": "a.wav", "text": "   ", "duration": 3.0},
        ]
        analyzer = DatasetAnalyzer()
        anomalies = analyzer.detect_anomalies(samples=records)
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == "empty_label"

    def test_duration_mismatch_too_fast(self):
        """Too many chars per second is detected as duration_mismatch."""
        records = [
            {"audio_path": "a.wav", "text": "a" * 100, "duration": 1.0},
        ]
        analyzer = DatasetAnalyzer(chars_per_second_range=(1.0, 25.0))
        anomalies = analyzer.detect_anomalies(samples=records)
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == "duration_mismatch"

    def test_duration_mismatch_too_slow(self):
        """Too few chars per second is detected as duration_mismatch."""
        records = [
            {"audio_path": "a.wav", "text": "a", "duration": 100.0},
        ]
        analyzer = DatasetAnalyzer(chars_per_second_range=(1.0, 25.0))
        anomalies = analyzer.detect_anomalies(samples=records)
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == "duration_mismatch"

    def test_corrupted_audio_zero_duration(self):
        """Zero duration is detected as corrupted_audio."""
        records = [
            {"audio_path": "a.wav", "text": "hello", "duration": 0.0},
        ]
        analyzer = DatasetAnalyzer()
        anomalies = analyzer.detect_anomalies(samples=records)
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == "corrupted_audio"

    def test_corrupted_audio_negative_duration(self):
        """Negative duration is detected as corrupted_audio."""
        records = [
            {"audio_path": "a.wav", "text": "hello", "duration": -1.0},
        ]
        analyzer = DatasetAnalyzer()
        anomalies = analyzer.detect_anomalies(samples=records)
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == "corrupted_audio"

    def test_corrupted_audio_missing_path(self):
        """Missing audio_path is detected as corrupted_audio."""
        records = [
            {"text": "hello", "duration": 3.0},
        ]
        analyzer = DatasetAnalyzer()
        anomalies = analyzer.detect_anomalies(samples=records)
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == "corrupted_audio"

    def test_no_anomalies(self):
        """Normal samples produce no anomalies."""
        records = [
            {"audio_path": "a.wav", "text": "hello world test", "duration": 3.0},
        ]
        analyzer = DatasetAnalyzer(chars_per_second_range=(1.0, 25.0))
        anomalies = analyzer.detect_anomalies(samples=records)
        assert len(anomalies) == 0

    def test_multiple_anomaly_types(self):
        """Multiple anomaly types are detected in a single pass."""
        records = [
            {"audio_path": "a.wav", "text": "", "duration": 3.0},
            {"audio_path": "b.wav", "text": "x" * 100, "duration": 1.0},
            {"audio_path": "", "text": "hello", "duration": 3.0},
        ]
        analyzer = DatasetAnalyzer(chars_per_second_range=(1.0, 25.0))
        anomalies = analyzer.detect_anomalies(samples=records)
        types = {a.anomaly_type for a in anomalies}
        assert "empty_label" in types
        assert "duration_mismatch" in types
        assert "corrupted_audio" in types


# =====================================================================
# TestAnalyzeSceneDistribution
# =====================================================================


class TestAnalyzeSceneDistribution:
    """Tests for DatasetAnalyzer.analyze_scene_distribution."""

    def test_scene_counts(self):
        """Scene counts match the input data."""
        analyzer = DatasetAnalyzer()
        dist = analyzer.analyze_scene_distribution(samples=_SAMPLE_RECORDS)
        assert dist.scene_counts["clean"] == 2
        assert dist.scene_counts["background_voice"] == 2
        assert dist.scene_counts["background_noise"] == 1

    def test_scene_ratios_sum_to_one(self):
        """Scene ratios sum to approximately 1.0."""
        analyzer = DatasetAnalyzer()
        dist = analyzer.analyze_scene_distribution(samples=_SAMPLE_RECORDS)
        total_ratio = sum(dist.scene_ratios.values())
        assert total_ratio == pytest.approx(1.0)

    def test_scene_ratios_correct(self):
        """Individual scene ratios are computed correctly."""
        analyzer = DatasetAnalyzer()
        dist = analyzer.analyze_scene_distribution(samples=_SAMPLE_RECORDS)
        assert dist.scene_ratios["clean"] == pytest.approx(2 / 5)
        assert dist.scene_ratios["background_voice"] == pytest.approx(2 / 5)
        assert dist.scene_ratios["background_noise"] == pytest.approx(1 / 5)

    def test_underrepresented_scenes(self):
        """Scenes with fewer samples than threshold are flagged."""
        analyzer = DatasetAnalyzer(min_scene_samples=3)
        dist = analyzer.analyze_scene_distribution(samples=_SAMPLE_RECORDS)
        # background_noise has 1 sample, clean has 2, background_voice has 2
        # All are below 3
        assert "background_noise" in dist.underrepresented_scenes
        assert "clean" in dist.underrepresented_scenes
        assert "background_voice" in dist.underrepresented_scenes

    def test_no_underrepresented(self):
        """No underrepresented scenes when all meet threshold."""
        analyzer = DatasetAnalyzer(min_scene_samples=1)
        dist = analyzer.analyze_scene_distribution(samples=_SAMPLE_RECORDS)
        assert dist.underrepresented_scenes == []

    def test_empty_dataset(self):
        """Empty dataset returns empty distribution."""
        analyzer = DatasetAnalyzer()
        dist = analyzer.analyze_scene_distribution(samples=[])
        assert dist.scene_counts == {}
        assert dist.scene_ratios == {}
        assert dist.underrepresented_scenes == []

    def test_missing_scene_type(self):
        """Samples without scene_type are grouped as 'unknown'."""
        records = [
            {"audio_path": "a.wav", "text": "hello", "duration": 3.0},
        ]
        analyzer = DatasetAnalyzer()
        dist = analyzer.analyze_scene_distribution(samples=records)
        assert "unknown" in dist.scene_counts
        assert dist.scene_counts["unknown"] == 1

    def test_counts_sum_equals_total(self):
        """Sum of scene counts equals total samples."""
        analyzer = DatasetAnalyzer()
        dist = analyzer.analyze_scene_distribution(samples=_SAMPLE_RECORDS)
        assert sum(dist.scene_counts.values()) == len(_SAMPLE_RECORDS)


# =====================================================================
# TestGenerateQualityReport
# =====================================================================


class TestGenerateQualityReport:
    """Tests for DatasetAnalyzer.generate_quality_report."""

    def test_report_contains_all_sections(self):
        """Report includes stats, text_stats, anomalies, scene_distribution."""
        analyzer = DatasetAnalyzer()
        report = analyzer.generate_quality_report(samples=_SAMPLE_RECORDS)
        assert report.stats.total_samples == 5
        assert isinstance(report.text_stats, TextStats)
        assert isinstance(report.anomalies, list)
        assert isinstance(report.scene_distribution, SceneDistribution)

    def test_issues_sorted_by_severity(self):
        """Issues are sorted by severity: critical > warning > info."""
        records = [
            {"audio_path": "", "text": "hello", "duration": 3.0},           # corrupted
            {"audio_path": "a.wav", "text": "", "duration": 3.0},           # empty label
            {"audio_path": "b.wav", "text": "x" * 100, "duration": 1.0},   # mismatch
            {"audio_path": "c.wav", "text": "hello world", "duration": 3.0, "snr": 5.0},  # low snr
            {"audio_path": "d.wav", "text": "test text here", "duration": 3.0,
             "scene_type": "rare", "snr": 20.0},  # underrepresented
        ]
        analyzer = DatasetAnalyzer(
            snr_threshold=10.0,
            min_scene_samples=100,
            chars_per_second_range=(1.0, 25.0),
        )
        report = analyzer.generate_quality_report(samples=records)

        severity_order = {"critical": 0, "warning": 1, "info": 2}
        severities = [issue["severity"] for issue in report.issues_by_severity]
        for i in range(len(severities) - 1):
            assert severity_order[severities[i]] <= severity_order[severities[i + 1]]

    def test_low_quality_samples_in_stats(self):
        """Low quality samples are populated in stats."""
        analyzer = DatasetAnalyzer(snr_threshold=10.0)
        report = analyzer.generate_quality_report(samples=_SAMPLE_RECORDS)
        assert len(report.stats.low_quality_samples) == 2

    def test_snr_distribution_in_stats(self):
        """SNR distribution is populated in stats."""
        analyzer = DatasetAnalyzer()
        report = analyzer.generate_quality_report(samples=_SAMPLE_RECORDS)
        assert isinstance(report.stats.snr_distribution, dict)
        assert sum(report.stats.snr_distribution.values()) == len(_SAMPLE_RECORDS)

    def test_empty_dataset_report(self):
        """Empty dataset produces a valid report with zero stats."""
        analyzer = DatasetAnalyzer()
        report = analyzer.generate_quality_report(samples=[])
        assert report.stats.total_samples == 0
        assert report.anomalies == []
        assert report.issues_by_severity == []

    def test_from_dataset_path(self, tmp_path):
        """Report can be generated from a manifest file on disk."""
        dataset_dir = _write_manifest(tmp_path, _SAMPLE_RECORDS)
        analyzer = DatasetAnalyzer()
        report = analyzer.generate_quality_report(dataset_path=dataset_dir)
        assert report.stats.total_samples == 5


# =====================================================================
# TestAnalyzeDatasetCLI
# =====================================================================


class TestAnalyzeDatasetCLI:
    """Tests for the analyze_dataset CLI script."""

    def test_cli_output_format(self, tmp_path, capsys):
        """CLI outputs valid JSON with expected keys."""
        dataset_dir = _write_manifest(tmp_path, _SAMPLE_RECORDS)

        from scripts.analyze_dataset import main

        main(["--dataset-path", dataset_dir])

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert "stats" in output
        assert "text_stats" in output
        assert "anomalies" in output
        assert "scene_distribution" in output
        assert "issues_by_severity" in output

    def test_cli_custom_thresholds(self, tmp_path, capsys):
        """CLI respects custom threshold arguments."""
        dataset_dir = _write_manifest(tmp_path, _SAMPLE_RECORDS)

        from scripts.analyze_dataset import main

        main([
            "--dataset-path", dataset_dir,
            "--snr-threshold", "20.0",
            "--min-scene-samples", "50",
        ])

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        # With SNR threshold 20.0, samples with SNR 8.0, 5.0, 15.0 are low quality
        assert len(output["stats"]["low_quality_samples"]) == 3

    def test_cli_stats_values(self, tmp_path, capsys):
        """CLI output stats match expected values."""
        dataset_dir = _write_manifest(tmp_path, _SAMPLE_RECORDS)

        from scripts.analyze_dataset import main

        main(["--dataset-path", dataset_dir])

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output["stats"]["total_samples"] == 5
        assert output["stats"]["total_duration_hours"] > 0
