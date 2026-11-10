"""Tests for the ResultAnalyzer class.

Covers:
- evaluate_model computes overall and per-scene CER/WER
- generate_badcases filters by CER threshold
- classify_errors returns correct error-type counts
- compare_with_baseline computes correct deltas
- Edge cases: empty samples, identical texts, empty texts
"""

import pytest

from scripts.core.models import (
    BadCase,
    EvaluationReport,
    SceneMetrics,
)
from scripts.analysis.result_analyzer import (
    EvalSample,
    ResultAnalyzer,
    TestSample,
    _compute_cer,
    _compute_wer,
    _edit_distance_ops,
)


# =====================================================================
# Helper CER/WER computation tests
# =====================================================================


class TestComputeCER:
    """Tests for the _compute_cer helper."""

    def test_identical_strings(self):
        """CER is 0.0 when reference and hypothesis are identical."""
        assert _compute_cer("你好世界", "你好世界") == 0.0

    def test_single_substitution(self):
        """CER for one substitution in a 6-char string."""
        # "今天天气很好" vs "今天天气很号" → 1 sub / 6 chars
        cer = _compute_cer("今天天气很好", "今天天气很号")
        assert cer == pytest.approx(1 / 6)

    def test_single_deletion(self):
        """CER for one deletion."""
        # "请打开空调" vs "请打开空" → 1 del / 5 chars
        cer = _compute_cer("请打开空调", "请打开空")
        assert cer == pytest.approx(1 / 5)

    def test_single_insertion(self):
        """CER for one insertion."""
        # "明天下午开会" vs "明天下午来开会" → 1 ins / 6 chars
        cer = _compute_cer("明天下午开会", "明天下午来开会")
        assert cer == pytest.approx(1 / 6)

    def test_empty_reference_empty_hypothesis(self):
        """CER is 0.0 when both are empty."""
        assert _compute_cer("", "") == 0.0

    def test_empty_reference_nonempty_hypothesis(self):
        """CER is 1.0 when reference is empty but hypothesis is not."""
        assert _compute_cer("", "abc") == 1.0

    def test_nonempty_reference_empty_hypothesis(self):
        """CER equals 1.0 when hypothesis is empty (all deletions)."""
        assert _compute_cer("abc", "") == pytest.approx(1.0)

    def test_completely_different(self):
        """CER for completely different strings of same length."""
        # "abc" vs "xyz" → 3 subs / 3 chars = 1.0
        assert _compute_cer("abc", "xyz") == pytest.approx(1.0)


class TestComputeWER:
    """Tests for the _compute_wer helper."""

    def test_identical_sentences(self):
        """WER is 0.0 for identical word sequences."""
        assert _compute_wer("hello world", "hello world") == 0.0

    def test_one_word_error(self):
        """WER for one substitution in a 3-word sentence."""
        wer = _compute_wer("the cat sat", "the dog sat")
        assert wer == pytest.approx(1 / 3)

    def test_chinese_no_spaces(self):
        """For Chinese without spaces, WER falls back to single-token comparison."""
        # "你好" vs "你好" → 0 errors / 1 word = 0.0
        assert _compute_wer("你好", "你好") == 0.0

    def test_empty_both(self):
        """WER is 0.0 when both are empty."""
        assert _compute_wer("", "") == 0.0


# =====================================================================
# Edit distance operations tests
# =====================================================================


class TestEditDistanceOps:
    """Tests for _edit_distance_ops."""

    def test_identical_returns_empty(self):
        """No operations for identical strings."""
        assert _edit_distance_ops("abc", "abc") == []

    def test_single_substitution(self):
        """One substitution operation."""
        ops = _edit_distance_ops("abc", "axc")
        assert len(ops) == 1
        assert ops[0]["type"] == "substitution"
        assert ops[0]["ref"] == "b"
        assert ops[0]["hyp"] == "x"

    def test_single_deletion(self):
        """One deletion operation."""
        ops = _edit_distance_ops("abc", "ac")
        assert len(ops) == 1
        assert ops[0]["type"] == "deletion"
        assert ops[0]["ref"] == "b"

    def test_single_insertion(self):
        """One insertion operation."""
        ops = _edit_distance_ops("ac", "abc")
        assert len(ops) == 1
        assert ops[0]["type"] == "insertion"
        assert ops[0]["hyp"] == "b"

    def test_op_count_equals_edit_distance(self):
        """Number of operations equals the edit distance."""
        ref = "kitten"
        hyp = "sitting"
        ops = _edit_distance_ops(ref, hyp)
        # Edit distance between "kitten" and "sitting" is 3
        assert len(ops) == 3

    def test_empty_ref(self):
        """All insertions when reference is empty."""
        ops = _edit_distance_ops("", "abc")
        assert len(ops) == 3
        assert all(op["type"] == "insertion" for op in ops)

    def test_empty_hyp(self):
        """All deletions when hypothesis is empty."""
        ops = _edit_distance_ops("abc", "")
        assert len(ops) == 3
        assert all(op["type"] == "deletion" for op in ops)


# =====================================================================
# ResultAnalyzer.evaluate_model
# =====================================================================


class TestEvaluateModel:
    """Tests for ResultAnalyzer.evaluate_model."""

    def test_overall_metrics(self):
        """Overall CER/WER are computed as macro-averages."""
        samples = [
            TestSample("a1.wav", "abc", "abc", "clean"),       # CER=0
            TestSample("a2.wav", "abc", "axc", "clean"),       # CER=1/3
        ]
        analyzer = ResultAnalyzer(samples)
        report = analyzer.evaluate_model()

        expected_cer = (0.0 + 1 / 3) / 2
        assert report.overall_cer == pytest.approx(expected_cer)

    def test_per_scene_metrics(self):
        """Per-scene CER/WER are grouped correctly."""
        samples = [
            TestSample("a1.wav", "abc", "abc", "clean"),
            TestSample("a2.wav", "abc", "axc", "clean"),
            TestSample("a3.wav", "abc", "xyz", "background_voice"),
        ]
        analyzer = ResultAnalyzer(samples)
        report = analyzer.evaluate_model()

        scene_map = {sm.scene_type: sm for sm in report.scene_metrics}
        assert "clean" in scene_map
        assert "background_voice" in scene_map
        assert scene_map["clean"].sample_count == 2
        assert scene_map["background_voice"].sample_count == 1
        # "abc" vs "xyz" → 3 subs / 3 chars = 1.0
        assert scene_map["background_voice"].cer == pytest.approx(1.0)

    def test_empty_samples(self):
        """Empty sample list produces a zero report."""
        analyzer = ResultAnalyzer()
        report = analyzer.evaluate_model()

        assert report.overall_cer == 0.0
        assert report.overall_wer == 0.0
        assert report.scene_metrics == []
        assert report.badcases == []

    def test_error_distribution(self):
        """Error-type distribution is computed across all samples."""
        samples = [
            TestSample("a1.wav", "abc", "axc", "clean"),   # 1 sub
            TestSample("a2.wav", "abc", "ac", "clean"),    # 1 del
            TestSample("a3.wav", "ac", "abc", "clean"),    # 1 ins
        ]
        analyzer = ResultAnalyzer(samples)
        report = analyzer.evaluate_model()

        assert report.error_type_distribution["substitution"] == 1
        assert report.error_type_distribution["deletion"] == 1
        assert report.error_type_distribution["insertion"] == 1

    def test_samples_override(self):
        """Passing samples to evaluate_model overrides internal list."""
        analyzer = ResultAnalyzer()
        samples = [TestSample("a1.wav", "abc", "axc", "clean")]
        report = analyzer.evaluate_model(samples=samples)

        assert report.overall_cer > 0
        assert len(analyzer.samples) == 1

    def test_badcases_in_report(self):
        """Badcases in the report include samples with any error."""
        samples = [
            TestSample("a1.wav", "abc", "abc", "clean"),   # perfect
            TestSample("a2.wav", "abc", "axc", "clean"),   # has error
        ]
        analyzer = ResultAnalyzer(samples)
        report = analyzer.evaluate_model()

        # Only the sample with errors should be a badcase
        assert len(report.badcases) == 1
        assert report.badcases[0].audio_path == "a2.wav"


# =====================================================================
# ResultAnalyzer.generate_badcases
# =====================================================================


class TestGenerateBadcases:
    """Tests for ResultAnalyzer.generate_badcases."""

    def test_filters_by_threshold(self):
        """Only samples with CER > threshold are returned."""
        samples = [
            TestSample("a1.wav", "abcdef", "abcdef", "clean"),     # CER=0
            TestSample("a2.wav", "abcdef", "abcdex", "clean"),     # CER=1/6≈0.167
            TestSample("a3.wav", "abc", "xyz", "background_voice"),  # CER=1.0
        ]
        analyzer = ResultAnalyzer(samples)
        badcases = analyzer.generate_badcases(threshold=0.5)

        assert len(badcases) == 1
        assert badcases[0].audio_path == "a3.wav"

    def test_all_above_threshold(self):
        """All samples returned when threshold is 0."""
        samples = [
            TestSample("a1.wav", "abc", "axc", "clean"),
            TestSample("a2.wav", "abc", "xyz", "clean"),
        ]
        analyzer = ResultAnalyzer(samples)
        badcases = analyzer.generate_badcases(threshold=0.0)

        assert len(badcases) == 2

    def test_none_above_threshold(self):
        """Empty list when no samples exceed threshold."""
        samples = [
            TestSample("a1.wav", "abc", "abc", "clean"),
        ]
        analyzer = ResultAnalyzer(samples)
        badcases = analyzer.generate_badcases(threshold=0.5)

        assert len(badcases) == 0

    def test_badcase_has_error_details(self):
        """Each badcase includes error_details from edit distance."""
        samples = [
            TestSample("a1.wav", "abc", "axc", "clean"),
        ]
        analyzer = ResultAnalyzer(samples)
        badcases = analyzer.generate_badcases(threshold=0.0)

        assert len(badcases) == 1
        assert len(badcases[0].error_details) == 1
        assert badcases[0].error_details[0]["type"] == "substitution"

    def test_completeness(self):
        """All samples above threshold are included (no false negatives)."""
        samples = [
            TestSample(f"a{i}.wav", "abc", "xyz", "clean")
            for i in range(10)
        ]
        analyzer = ResultAnalyzer(samples)
        badcases = analyzer.generate_badcases(threshold=0.0)

        assert len(badcases) == 10

    def test_samples_override(self):
        """Passing samples parameter overrides internal list."""
        analyzer = ResultAnalyzer()
        samples = [TestSample("a1.wav", "abc", "axc", "clean")]
        badcases = analyzer.generate_badcases(threshold=0.0, samples=samples)

        assert len(badcases) == 1


# =====================================================================
# ResultAnalyzer.classify_errors
# =====================================================================


class TestClassifyErrors:
    """Tests for ResultAnalyzer.classify_errors."""

    def test_substitution_only(self):
        """Single substitution is classified correctly."""
        analyzer = ResultAnalyzer()
        counts = analyzer.classify_errors("abc", "axc")

        assert counts["substitution"] == 1
        assert counts["insertion"] == 0
        assert counts["deletion"] == 0

    def test_deletion_only(self):
        """Single deletion is classified correctly."""
        analyzer = ResultAnalyzer()
        counts = analyzer.classify_errors("abc", "ac")

        assert counts["substitution"] == 0
        assert counts["insertion"] == 0
        assert counts["deletion"] == 1

    def test_insertion_only(self):
        """Single insertion is classified correctly."""
        analyzer = ResultAnalyzer()
        counts = analyzer.classify_errors("ac", "abc")

        assert counts["substitution"] == 0
        assert counts["insertion"] == 1
        assert counts["deletion"] == 0

    def test_mixed_errors(self):
        """Mixed error types sum to edit distance."""
        analyzer = ResultAnalyzer()
        counts = analyzer.classify_errors("kitten", "sitting")

        total = counts["substitution"] + counts["insertion"] + counts["deletion"]
        assert total == 3  # edit distance between kitten and sitting

    def test_identical_strings(self):
        """No errors for identical strings."""
        analyzer = ResultAnalyzer()
        counts = analyzer.classify_errors("hello", "hello")

        assert counts["substitution"] == 0
        assert counts["insertion"] == 0
        assert counts["deletion"] == 0

    def test_empty_both(self):
        """No errors when both strings are empty."""
        analyzer = ResultAnalyzer()
        counts = analyzer.classify_errors("", "")

        assert counts["substitution"] == 0
        assert counts["insertion"] == 0
        assert counts["deletion"] == 0

    def test_sum_equals_edit_distance(self):
        """Total error count equals edit distance for Chinese text."""
        analyzer = ResultAnalyzer()
        ref = "今天天气很好"
        hyp = "今天天气很号啊"
        counts = analyzer.classify_errors(ref, hyp)

        # "好"→"号" (sub) + "啊" (ins) = 2 errors
        total = counts["substitution"] + counts["insertion"] + counts["deletion"]
        assert total == 2

    def test_classify_from_badcases(self):
        """classify_errors_from_badcases aggregates across multiple badcases."""
        analyzer = ResultAnalyzer()
        badcases = [
            BadCase(
                audio_path="a1.wav",
                reference_text="abc",
                hypothesis_text="axc",
                cer=1 / 3,
                wer=1 / 3,
            ),
            BadCase(
                audio_path="a2.wav",
                reference_text="abc",
                hypothesis_text="ac",
                cer=1 / 3,
                wer=1.0,
            ),
        ]
        counts = analyzer.classify_errors_from_badcases(badcases)

        assert counts["substitution"] == 1
        assert counts["deletion"] == 1
        assert counts["insertion"] == 0


# =====================================================================
# ResultAnalyzer.compare_with_baseline
# =====================================================================


class TestCompareWithBaseline:
    """Tests for ResultAnalyzer.compare_with_baseline."""

    def test_overall_deltas(self):
        """Overall CER/WER deltas are computed correctly."""
        baseline = EvaluationReport(overall_cer=0.10, overall_wer=0.15)
        current = EvaluationReport(overall_cer=0.08, overall_wer=0.12)

        analyzer = ResultAnalyzer()
        result = analyzer.compare_with_baseline(baseline, current)

        assert result["overall_cer_delta"] == pytest.approx(-0.02)
        assert result["overall_wer_delta"] == pytest.approx(-0.03)

    def test_scene_deltas(self):
        """Per-scene deltas are computed for matching scene types."""
        baseline = EvaluationReport(
            overall_cer=0.10,
            overall_wer=0.15,
            scene_metrics=[
                SceneMetrics(scene_type="clean", cer=0.05, wer=0.08, sample_count=100),
                SceneMetrics(scene_type="background_voice", cer=0.20, wer=0.30, sample_count=50),
            ],
        )
        current = EvaluationReport(
            overall_cer=0.08,
            overall_wer=0.12,
            scene_metrics=[
                SceneMetrics(scene_type="clean", cer=0.04, wer=0.06, sample_count=100),
                SceneMetrics(scene_type="background_voice", cer=0.15, wer=0.22, sample_count=50),
            ],
        )

        analyzer = ResultAnalyzer()
        result = analyzer.compare_with_baseline(baseline, current)

        assert len(result["scene_deltas"]) == 2
        scene_map = {d["scene_type"]: d for d in result["scene_deltas"]}
        assert scene_map["clean"]["cer_delta"] == pytest.approx(-0.01)
        assert scene_map["background_voice"]["cer_delta"] == pytest.approx(-0.05)
        assert scene_map["background_voice"]["wer_delta"] == pytest.approx(-0.08)

    def test_mismatched_scenes(self):
        """Only scenes present in both reports appear in scene_deltas."""
        baseline = EvaluationReport(
            overall_cer=0.10,
            overall_wer=0.15,
            scene_metrics=[
                SceneMetrics(scene_type="clean", cer=0.05, wer=0.08, sample_count=100),
            ],
        )
        current = EvaluationReport(
            overall_cer=0.08,
            overall_wer=0.12,
            scene_metrics=[
                SceneMetrics(scene_type="clean", cer=0.04, wer=0.06, sample_count=100),
                SceneMetrics(scene_type="background_noise", cer=0.10, wer=0.15, sample_count=50),
            ],
        )

        analyzer = ResultAnalyzer()
        result = analyzer.compare_with_baseline(baseline, current)

        # Only "clean" is in both
        assert len(result["scene_deltas"]) == 1
        assert result["scene_deltas"][0]["scene_type"] == "clean"

    def test_no_scene_metrics(self):
        """Empty scene_deltas when neither report has scene metrics."""
        baseline = EvaluationReport(overall_cer=0.10, overall_wer=0.15)
        current = EvaluationReport(overall_cer=0.08, overall_wer=0.12)

        analyzer = ResultAnalyzer()
        result = analyzer.compare_with_baseline(baseline, current)

        assert result["scene_deltas"] == []

    def test_worsening_metrics(self):
        """Positive deltas indicate worsening performance."""
        baseline = EvaluationReport(overall_cer=0.05, overall_wer=0.08)
        current = EvaluationReport(overall_cer=0.10, overall_wer=0.15)

        analyzer = ResultAnalyzer()
        result = analyzer.compare_with_baseline(baseline, current)

        assert result["overall_cer_delta"] == pytest.approx(0.05)
        assert result["overall_wer_delta"] == pytest.approx(0.07)
