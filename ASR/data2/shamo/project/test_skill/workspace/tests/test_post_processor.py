"""Tests for the PostProcessor module and apply_postprocess CLI.

Covers:
- Hotword loading and correction (Task 12.1)
- Regex rule application (Task 12.1)
- LM N-best reranking (Task 12.3)
- Full pipeline and effect evaluation (Task 12.5)
- CLI subcommands (Task 12.8)
"""

import json
import os

import pytest

from scripts.core.models import (
    HotwordEntry,
    PostProcessConfig,
    PostProcessResult,
    RegexRule,
)
from scripts.execution.post_processor import PostProcessor, _compute_cer, _edit_distance


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def hotword_file(tmp_path):
    """Create a temporary hotword file."""
    path = tmp_path / "hotwords.txt"
    path.write_text(
        "号\t好\t1\n"
        "以经\t已经\t2\n"
        "# comment line\n"
        "\n"
        "在见\t再见\t3\n",
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture
def basic_config():
    """Return a basic PostProcessConfig with hotwords and regex rules."""
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
        enable_lm_rerank=False,
        pipeline_order=["hotword", "regex", "lm_rerank"],
    )


@pytest.fixture
def full_config():
    """Return a PostProcessConfig with LM rerank enabled."""
    return PostProcessConfig(
        hotword_entries=[
            HotwordEntry(wrong_text="号", correct_text="好", priority=1),
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


# =====================================================================
# Helper function tests
# =====================================================================


class TestEditDistance:
    def test_identical_strings(self):
        assert _edit_distance("abc", "abc") == 0

    def test_empty_strings(self):
        assert _edit_distance("", "") == 0

    def test_one_empty(self):
        assert _edit_distance("abc", "") == 3
        assert _edit_distance("", "abc") == 3

    def test_substitution(self):
        assert _edit_distance("abc", "axc") == 1

    def test_insertion(self):
        assert _edit_distance("abc", "abxc") == 1

    def test_deletion(self):
        assert _edit_distance("abc", "ac") == 1

    def test_chinese(self):
        assert _edit_distance("今天天气很好", "今天天气很号") == 1


class TestComputeCER:
    def test_identical(self):
        assert _compute_cer("hello", "hello") == 0.0

    def test_both_empty(self):
        assert _compute_cer("", "") == 0.0

    def test_empty_reference_nonempty_hypothesis(self):
        assert _compute_cer("", "abc") == 1.0

    def test_one_error(self):
        # "好" -> "号": 1 edit / 6 chars
        cer = _compute_cer("今天天气很好", "今天天气很号")
        assert abs(cer - 1 / 6) < 1e-6


# =====================================================================
# Task 12.1 – Hotword loading and correction
# =====================================================================


class TestLoadHotwords:
    def test_load_from_file(self, hotword_file):
        config = PostProcessConfig()
        processor = PostProcessor(config=config)
        entries = processor.load_hotwords(hotword_file)

        assert len(entries) == 3
        assert entries[0].wrong_text == "号"
        assert entries[0].correct_text == "好"
        assert entries[0].priority == 1
        assert entries[1].wrong_text == "以经"
        assert entries[1].correct_text == "已经"
        assert entries[1].priority == 2
        assert entries[2].wrong_text == "在见"
        assert entries[2].correct_text == "再见"
        assert entries[2].priority == 3

    def test_load_via_constructor(self, hotword_file):
        config = PostProcessConfig()
        processor = PostProcessor(config=config, hotword_path=hotword_file)
        # Entries should be loaded into internal state
        assert len(processor._hotword_entries) == 3

    def test_merge_with_config_entries(self, hotword_file):
        config = PostProcessConfig(
            hotword_entries=[
                HotwordEntry(wrong_text="测试", correct_text="测试OK", priority=5),
            ]
        )
        processor = PostProcessor(config=config, hotword_path=hotword_file)
        # 1 from config + 3 from file
        assert len(processor._hotword_entries) == 4

    def test_skip_comment_and_blank_lines(self, tmp_path):
        path = tmp_path / "hw.txt"
        path.write_text(
            "# This is a comment\n"
            "\n"
            "错\t对\t1\n"
            "   \n",
            encoding="utf-8",
        )
        config = PostProcessConfig()
        processor = PostProcessor(config=config)
        entries = processor.load_hotwords(str(path))
        assert len(entries) == 1

    def test_skip_malformed_lines(self, tmp_path):
        path = tmp_path / "hw.txt"
        path.write_text(
            "only_one_field\n"
            "two\tfields\n"
            "correct\tformat\t1\n",
            encoding="utf-8",
        )
        config = PostProcessConfig()
        processor = PostProcessor(config=config)
        entries = processor.load_hotwords(str(path))
        assert len(entries) == 1

    def test_invalid_priority_defaults_to_zero(self, tmp_path):
        path = tmp_path / "hw.txt"
        path.write_text("错\t对\tnot_a_number\n", encoding="utf-8")
        config = PostProcessConfig()
        processor = PostProcessor(config=config)
        entries = processor.load_hotwords(str(path))
        assert len(entries) == 1
        assert entries[0].priority == 0


class TestApplyHotwordCorrection:
    def test_basic_replacement(self, basic_config):
        processor = PostProcessor(config=basic_config)
        result = processor.apply_hotword_correction("今天天气很号")
        assert result == "今天天气很好"

    def test_multiple_replacements(self, basic_config):
        processor = PostProcessor(config=basic_config)
        result = processor.apply_hotword_correction("以经很号了")
        assert result == "已经很好了"

    def test_no_match(self, basic_config):
        processor = PostProcessor(config=basic_config)
        result = processor.apply_hotword_correction("今天天气很好")
        assert result == "今天天气很好"

    def test_priority_ordering(self):
        """Higher priority entries are applied first."""
        config = PostProcessConfig(
            hotword_entries=[
                HotwordEntry(wrong_text="AB", correct_text="X", priority=1),
                HotwordEntry(wrong_text="ABC", correct_text="Y", priority=2),
            ]
        )
        processor = PostProcessor(config=config)
        # "ABC" has higher priority, so it should match first
        result = processor.apply_hotword_correction("ABC")
        assert result == "Y"

    def test_longer_match_preferred_at_same_priority(self):
        """At the same priority, longer wrong_text is matched first."""
        config = PostProcessConfig(
            hotword_entries=[
                HotwordEntry(wrong_text="AB", correct_text="X", priority=1),
                HotwordEntry(wrong_text="ABC", correct_text="Y", priority=1),
            ]
        )
        processor = PostProcessor(config=config)
        result = processor.apply_hotword_correction("ABC")
        assert result == "Y"

    def test_empty_wrong_text_skipped(self):
        config = PostProcessConfig(
            hotword_entries=[
                HotwordEntry(wrong_text="", correct_text="X", priority=1),
                HotwordEntry(wrong_text="号", correct_text="好", priority=1),
            ]
        )
        processor = PostProcessor(config=config)
        result = processor.apply_hotword_correction("号")
        assert result == "好"

    def test_empty_text_input(self, basic_config):
        processor = PostProcessor(config=basic_config)
        result = processor.apply_hotword_correction("")
        assert result == ""


# =====================================================================
# Task 12.1 – Regex rules
# =====================================================================


class TestApplyRegexRules:
    def test_whitespace_removal(self, basic_config):
        processor = PostProcessor(config=basic_config)
        result = processor.apply_regex_rules("今天  天气  很好")
        assert result == "今天天气很好"

    def test_no_rules(self):
        config = PostProcessConfig(regex_rules=[])
        processor = PostProcessor(config=config)
        result = processor.apply_regex_rules("hello world")
        assert result == "hello world"

    def test_priority_ordering(self):
        """Higher priority rules are applied first."""
        config = PostProcessConfig(
            regex_rules=[
                RegexRule(pattern=r"a", replacement="b", priority=1),
                RegexRule(pattern=r"b", replacement="c", priority=2),
            ]
        )
        processor = PostProcessor(config=config)
        # Priority 2 rule (b->c) runs first on "ab" -> "ac",
        # then priority 1 (a->b) runs on "ac" -> "bc"
        result = processor.apply_regex_rules("ab")
        assert result == "bc"

    def test_empty_pattern_skipped(self):
        config = PostProcessConfig(
            regex_rules=[
                RegexRule(pattern="", replacement="X", priority=1),
                RegexRule(pattern=r"\d+", replacement="NUM", priority=2),
            ]
        )
        processor = PostProcessor(config=config)
        result = processor.apply_regex_rules("abc123")
        assert result == "abcNUM"

    def test_empty_text(self, basic_config):
        processor = PostProcessor(config=basic_config)
        result = processor.apply_regex_rules("")
        assert result == ""


# =====================================================================
# Task 12.3 – LM N-best reranking
# =====================================================================


class TestApplyLmRerank:
    def test_returns_highest_score(self):
        config = PostProcessConfig()
        processor = PostProcessor(config=config)
        nbest = [
            ("候选一", 0.5),
            ("候选二", 0.9),
            ("候选三", 0.3),
        ]
        result = processor.apply_lm_rerank(nbest)
        assert result == "候选二"

    def test_single_candidate(self):
        config = PostProcessConfig()
        processor = PostProcessor(config=config)
        nbest = [("唯一候选", 0.7)]
        result = processor.apply_lm_rerank(nbest)
        assert result == "唯一候选"

    def test_empty_list(self):
        config = PostProcessConfig()
        processor = PostProcessor(config=config)
        result = processor.apply_lm_rerank([])
        assert result == ""

    def test_negative_scores(self):
        config = PostProcessConfig()
        processor = PostProcessor(config=config)
        nbest = [
            ("a", -1.0),
            ("b", -0.5),
            ("c", -2.0),
        ]
        result = processor.apply_lm_rerank(nbest)
        assert result == "b"

    def test_equal_scores_returns_first_max(self):
        config = PostProcessConfig()
        processor = PostProcessor(config=config)
        nbest = [
            ("first", 0.9),
            ("second", 0.9),
        ]
        result = processor.apply_lm_rerank(nbest)
        # max() returns the first maximum element
        assert result == "first"


# =====================================================================
# Task 12.5 – Full pipeline (process)
# =====================================================================


class TestProcess:
    def test_hotword_and_regex_pipeline(self, basic_config):
        processor = PostProcessor(config=basic_config)
        result = processor.process(
            text="以经 很号了",
            reference_text="已经很好了",
        )
        assert result.processed_text == "已经很好了"
        assert "hotword" in result.applied_rules
        assert "regex" in result.applied_rules
        assert result.cer_after < result.cer_before

    def test_pipeline_order_respected(self, basic_config):
        processor = PostProcessor(config=basic_config)
        result = processor.process(
            text="以经 很号了",
            reference_text="已经很好了",
        )
        # hotword should come before regex in applied_rules
        if "hotword" in result.applied_rules and "regex" in result.applied_rules:
            assert result.applied_rules.index("hotword") < result.applied_rules.index("regex")

    def test_lm_rerank_in_pipeline(self):
        """LM rerank picks the best candidate when it differs from current text."""
        config = PostProcessConfig(
            hotword_entries=[],
            regex_rules=[],
            enable_lm_rerank=True,
            lm_model_path="/models/lm/zh_lm.bin",
            pipeline_order=["hotword", "regex", "lm_rerank"],
        )
        processor = PostProcessor(config=config)
        nbest = [
            ("今天天气很号", 0.3),
            ("今天天气很好", 0.9),
        ]
        result = processor.process(
            text="今天天气很号",
            reference_text="今天天气很好",
            nbest=nbest,
        )
        # The lm_rerank step should pick the best candidate
        assert "lm_rerank" in result.applied_rules
        assert result.processed_text == "今天天气很好"

    def test_no_reference_text(self, basic_config):
        processor = PostProcessor(config=basic_config)
        result = processor.process(text="以经很号了")
        assert result.processed_text == "已经很好了"
        assert result.cer_before == 0.0
        assert result.cer_after == 0.0

    def test_no_changes_applied(self):
        config = PostProcessConfig()
        processor = PostProcessor(config=config)
        result = processor.process(text="完美文本", reference_text="完美文本")
        assert result.processed_text == "完美文本"
        assert result.applied_rules == []
        assert result.cer_before == 0.0
        assert result.cer_after == 0.0

    def test_cer_computation(self, basic_config):
        processor = PostProcessor(config=basic_config)
        result = processor.process(
            text="今天天气很号",
            reference_text="今天天气很好",
        )
        # Before: 1 error / 6 chars
        assert abs(result.cer_before - 1 / 6) < 1e-6
        # After: should be 0 (corrected)
        assert result.cer_after == 0.0

    def test_pipeline_with_disabled_steps(self):
        """Only enabled steps run."""
        config = PostProcessConfig(
            hotword_entries=[
                HotwordEntry(wrong_text="号", correct_text="好", priority=1),
            ],
            regex_rules=[],
            enable_lm_rerank=False,
            pipeline_order=["hotword", "regex", "lm_rerank"],
        )
        processor = PostProcessor(config=config)
        result = processor.process(
            text="号",
            reference_text="好",
        )
        assert result.processed_text == "好"
        assert result.applied_rules == ["hotword"]


# =====================================================================
# Task 12.5 – Effect evaluation
# =====================================================================


class TestEvaluateEffect:
    def test_basic_evaluation(self):
        config = PostProcessConfig()
        processor = PostProcessor(config=config)
        results = [
            PostProcessResult(
                original_text="a",
                processed_text="b",
                applied_rules=["hotword"],
                cer_before=0.5,
                cer_after=0.3,
            ),
            PostProcessResult(
                original_text="c",
                processed_text="d",
                applied_rules=["regex"],
                cer_before=0.4,
                cer_after=0.2,
            ),
        ]
        effect = processor.evaluate_effect(results)
        assert effect["total_samples"] == 2
        assert abs(effect["avg_cer_before"] - 0.45) < 1e-6
        assert abs(effect["avg_cer_after"] - 0.25) < 1e-6
        assert abs(effect["cer_delta"] - (-0.20)) < 1e-6

    def test_empty_results(self):
        config = PostProcessConfig()
        processor = PostProcessor(config=config)
        effect = processor.evaluate_effect([])
        assert effect["total_samples"] == 0
        assert effect["avg_cer_before"] == 0.0
        assert effect["avg_cer_after"] == 0.0
        assert effect["cer_delta"] == 0.0

    def test_single_result(self):
        config = PostProcessConfig()
        processor = PostProcessor(config=config)
        results = [
            PostProcessResult(
                original_text="x",
                processed_text="y",
                applied_rules=[],
                cer_before=0.8,
                cer_after=0.6,
            ),
        ]
        effect = processor.evaluate_effect(results)
        assert effect["total_samples"] == 1
        assert abs(effect["avg_cer_before"] - 0.8) < 1e-6
        assert abs(effect["avg_cer_after"] - 0.6) < 1e-6
        assert abs(effect["cer_delta"] - (-0.2)) < 1e-6

    def test_no_improvement(self):
        config = PostProcessConfig()
        processor = PostProcessor(config=config)
        results = [
            PostProcessResult(
                original_text="x",
                processed_text="y",
                applied_rules=[],
                cer_before=0.3,
                cer_after=0.5,
            ),
        ]
        effect = processor.evaluate_effect(results)
        assert effect["cer_delta"] > 0  # Positive delta means degradation


# =====================================================================
# Task 12.8 – CLI tests
# =====================================================================


class TestCLIApply:
    def test_apply_with_text(self, tmp_path):
        from scripts.apply_postprocess import main
        import io
        from contextlib import redirect_stdout

        # Create a config file
        config_path = tmp_path / "config.yaml"
        from scripts.core.config_utils import write_post_process_config

        config = PostProcessConfig(
            hotword_entries=[
                HotwordEntry(wrong_text="号", correct_text="好", priority=1),
            ],
        )
        write_post_process_config(config, str(config_path))

        buf = io.StringIO()
        with redirect_stdout(buf):
            main([
                "apply",
                "--config-file", str(config_path),
                "--text", "今天天气很号",
            ])

        output = json.loads(buf.getvalue())
        assert output["status"] == "success"
        assert output["total"] == 1
        assert output["results"][0]["processed_text"] == "今天天气很好"

    def test_apply_with_hotword_file(self, tmp_path, hotword_file):
        from scripts.apply_postprocess import main
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            main([
                "apply",
                "--hotword-file", hotword_file,
                "--text", "在见",
            ])

        output = json.loads(buf.getvalue())
        assert output["status"] == "success"
        assert output["results"][0]["processed_text"] == "再见"

    def test_apply_with_input_file(self, tmp_path):
        from scripts.apply_postprocess import main
        import io
        from contextlib import redirect_stdout

        input_path = tmp_path / "input.txt"
        input_path.write_text("号\n以经\n", encoding="utf-8")

        config_path = tmp_path / "config.yaml"
        from scripts.core.config_utils import write_post_process_config

        config = PostProcessConfig(
            hotword_entries=[
                HotwordEntry(wrong_text="号", correct_text="好", priority=1),
                HotwordEntry(wrong_text="以经", correct_text="已经", priority=2),
            ],
        )
        write_post_process_config(config, str(config_path))

        buf = io.StringIO()
        with redirect_stdout(buf):
            main([
                "apply",
                "--config-file", str(config_path),
                "--input-file", str(input_path),
            ])

        output = json.loads(buf.getvalue())
        assert output["status"] == "success"
        assert output["total"] == 2
        assert output["results"][0]["processed_text"] == "好"
        assert output["results"][1]["processed_text"] == "已经"

    def test_apply_with_reference_file(self, tmp_path):
        from scripts.apply_postprocess import main
        import io
        from contextlib import redirect_stdout

        input_path = tmp_path / "input.txt"
        input_path.write_text("号\n", encoding="utf-8")
        ref_path = tmp_path / "ref.txt"
        ref_path.write_text("好\n", encoding="utf-8")

        config_path = tmp_path / "config.yaml"
        from scripts.core.config_utils import write_post_process_config

        config = PostProcessConfig(
            hotword_entries=[
                HotwordEntry(wrong_text="号", correct_text="好", priority=1),
            ],
        )
        write_post_process_config(config, str(config_path))

        buf = io.StringIO()
        with redirect_stdout(buf):
            main([
                "apply",
                "--config-file", str(config_path),
                "--input-file", str(input_path),
                "--reference-file", str(ref_path),
            ])

        output = json.loads(buf.getvalue())
        assert output["status"] == "success"
        assert output["results"][0]["cer_before"] > 0
        assert output["results"][0]["cer_after"] == 0.0

    def test_apply_no_input(self):
        from scripts.apply_postprocess import main
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["apply"])

        output = json.loads(buf.getvalue())
        assert output["status"] == "error"


class TestCLIEvaluate:
    def test_evaluate(self, tmp_path):
        from scripts.apply_postprocess import main
        import io
        from contextlib import redirect_stdout

        input_path = tmp_path / "input.txt"
        input_path.write_text("号\n以经\n", encoding="utf-8")
        ref_path = tmp_path / "ref.txt"
        ref_path.write_text("好\n已经\n", encoding="utf-8")

        config_path = tmp_path / "config.yaml"
        from scripts.core.config_utils import write_post_process_config

        config = PostProcessConfig(
            hotword_entries=[
                HotwordEntry(wrong_text="号", correct_text="好", priority=1),
                HotwordEntry(wrong_text="以经", correct_text="已经", priority=2),
            ],
        )
        write_post_process_config(config, str(config_path))

        buf = io.StringIO()
        with redirect_stdout(buf):
            main([
                "evaluate",
                "--config-file", str(config_path),
                "--input-file", str(input_path),
                "--reference-file", str(ref_path),
            ])

        output = json.loads(buf.getvalue())
        assert output["status"] == "success"
        assert "evaluation" in output
        assert output["evaluation"]["total_samples"] == 2
        assert output["evaluation"]["cer_delta"] <= 0  # Should improve or stay same

    def test_evaluate_missing_input(self):
        from scripts.apply_postprocess import main
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["evaluate"])

        output = json.loads(buf.getvalue())
        assert output["status"] == "error"

    def test_evaluate_missing_reference(self, tmp_path):
        from scripts.apply_postprocess import main
        import io
        from contextlib import redirect_stdout

        input_path = tmp_path / "input.txt"
        input_path.write_text("test\n", encoding="utf-8")

        buf = io.StringIO()
        with redirect_stdout(buf):
            main([
                "evaluate",
                "--input-file", str(input_path),
            ])

        output = json.loads(buf.getvalue())
        assert output["status"] == "error"

    def test_evaluate_line_count_mismatch(self, tmp_path):
        from scripts.apply_postprocess import main
        import io
        from contextlib import redirect_stdout

        input_path = tmp_path / "input.txt"
        input_path.write_text("line1\nline2\n", encoding="utf-8")
        ref_path = tmp_path / "ref.txt"
        ref_path.write_text("ref1\n", encoding="utf-8")

        buf = io.StringIO()
        with redirect_stdout(buf):
            main([
                "evaluate",
                "--input-file", str(input_path),
                "--reference-file", str(ref_path),
            ])

        output = json.loads(buf.getvalue())
        assert output["status"] == "error"
        assert "mismatch" in output["message"]


class TestCLIShowConfig:
    def test_show_default_config(self):
        from scripts.apply_postprocess import main
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["show-config"])

        output = json.loads(buf.getvalue())
        assert output["status"] == "success"
        assert "config" in output

    def test_show_config_from_file(self, tmp_path):
        from scripts.apply_postprocess import main
        import io
        from contextlib import redirect_stdout
        from scripts.core.config_utils import write_post_process_config

        config = PostProcessConfig(
            hotword_entries=[
                HotwordEntry(wrong_text="号", correct_text="好", priority=1),
            ],
            enable_lm_rerank=True,
        )
        config_path = tmp_path / "config.yaml"
        write_post_process_config(config, str(config_path))

        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["show-config", "--config-file", str(config_path)])

        output = json.loads(buf.getvalue())
        assert output["status"] == "success"
        assert output["config"]["enable_lm_rerank"] is True
        assert len(output["config"]["hotword_entries"]) == 1

    def test_show_config_file_not_found(self, tmp_path):
        from scripts.apply_postprocess import main
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["show-config", "--config-file", str(tmp_path / "nonexistent.yaml")])

        output = json.loads(buf.getvalue())
        assert output["status"] == "error"
