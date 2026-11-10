"""Post-processing module for ASR recognition results.

Provides hotword correction, regex-based replacement, language-model
N-best reranking, and a full pipeline that chains these steps together.
Evaluates the effect of post-processing by computing CER before and after.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6
"""

import re
from typing import Optional

from scripts.core.models import (
    HotwordEntry,
    PostProcessConfig,
    PostProcessResult,
    RegexRule,
)


# =====================================================================
# Helper – edit-distance based CER
# =====================================================================


def _edit_distance(ref: str, hyp: str) -> int:
    """Compute the Levenshtein edit distance between *ref* and *hyp*."""
    n = len(ref)
    m = len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],
                    dp[i - 1][j],
                    dp[i][j - 1],
                )
    return dp[n][m]


def _compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate.

    CER = edit_distance(ref, hyp) / len(ref)

    Returns 0.0 when both strings are empty, 1.0 when reference is empty
    but hypothesis is not.
    """
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    return _edit_distance(reference, hypothesis) / len(reference)


# =====================================================================
# PostProcessor
# =====================================================================


class PostProcessor:
    """Post-processing pipeline for ASR recognition results.

    Parameters
    ----------
    config : PostProcessConfig
        Pipeline configuration including hotword entries, regex rules,
        LM rerank toggle, and pipeline execution order.
    hotword_path : str, optional
        Path to a hotword file.  If provided the file is loaded on
        construction and its entries are **merged** with those already
        present in *config*.
    """

    def __init__(
        self,
        config: PostProcessConfig,
        hotword_path: Optional[str] = None,
    ) -> None:
        self.config = config
        # Internal mutable copy of hotword entries
        self._hotword_entries: list[HotwordEntry] = list(config.hotword_entries)
        if hotword_path is not None:
            self._hotword_entries.extend(self.load_hotwords(hotword_path))

    # -----------------------------------------------------------------
    # Hotword loading
    # -----------------------------------------------------------------

    def load_hotwords(self, hotword_path: str) -> list[HotwordEntry]:
        """Load hotword entries from a text file.

        Expected format (tab-separated, one entry per line)::

            wrong_text\\tcorrect_text\\tpriority

        Blank lines and lines starting with ``#`` are skipped.

        Returns
        -------
        list[HotwordEntry]
            The parsed entries.  They are also appended to the
            processor's internal hotword table.
        """
        entries: list[HotwordEntry] = []
        with open(hotword_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                wrong_text = parts[0]
                correct_text = parts[1]
                try:
                    priority = int(parts[2])
                except ValueError:
                    priority = 0
                entries.append(
                    HotwordEntry(
                        wrong_text=wrong_text,
                        correct_text=correct_text,
                        priority=priority,
                    )
                )
        return entries

    # -----------------------------------------------------------------
    # Hotword correction
    # -----------------------------------------------------------------

    def apply_hotword_correction(self, text: str) -> str:
        """Replace wrong words with correct words from the hotword table.

        Entries are applied in order of:
        1. Higher priority first.
        2. Longer ``wrong_text`` first (to avoid partial-match issues).

        Replacement is word-boundary-aware: we use ``re.sub`` with the
        wrong_text escaped so that it matches as a literal substring.
        For Chinese text (where there are no whitespace word boundaries),
        simple substring replacement is the correct behaviour.
        """
        sorted_entries = sorted(
            self._hotword_entries,
            key=lambda e: (-e.priority, -len(e.wrong_text)),
        )
        for entry in sorted_entries:
            if not entry.wrong_text:
                continue
            # Use re.escape to treat wrong_text as a literal pattern.
            pattern = re.escape(entry.wrong_text)
            text = re.sub(pattern, entry.correct_text, text)
        return text

    # -----------------------------------------------------------------
    # Regex rules
    # -----------------------------------------------------------------

    def apply_regex_rules(self, text: str) -> str:
        """Apply regex replacement rules sorted by priority (higher first)."""
        sorted_rules = sorted(
            self.config.regex_rules,
            key=lambda r: -r.priority,
        )
        for rule in sorted_rules:
            if not rule.pattern:
                continue
            text = re.sub(rule.pattern, rule.replacement, text)
        return text

    # -----------------------------------------------------------------
    # LM N-best reranking
    # -----------------------------------------------------------------

    def apply_lm_rerank(self, nbest: list[tuple[str, float]]) -> str:
        """Return the text with the highest score from *nbest*.

        Parameters
        ----------
        nbest : list[tuple[str, float]]
            A list of ``(text, score)`` tuples.

        Returns
        -------
        str
            The text corresponding to the highest score.  Returns an
            empty string when *nbest* is empty.
        """
        if not nbest:
            return ""
        return max(nbest, key=lambda item: item[1])[0]

    # -----------------------------------------------------------------
    # Full pipeline
    # -----------------------------------------------------------------

    def process(
        self,
        text: str,
        reference_text: str = "",
        nbest: Optional[list[tuple[str, float]]] = None,
    ) -> PostProcessResult:
        """Execute the full post-processing pipeline.

        The pipeline steps are executed in the order specified by
        ``self.config.pipeline_order`` (default:
        ``["hotword", "regex", "lm_rerank"]``).

        Parameters
        ----------
        text : str
            The ASR recognition text to post-process.
        reference_text : str
            The ground-truth reference text (used for CER computation).
        nbest : list[tuple[str, float]], optional
            N-best candidates for LM reranking.

        Returns
        -------
        PostProcessResult
        """
        cer_before = _compute_cer(reference_text, text) if reference_text else 0.0

        applied_rules: list[str] = []
        current_text = text

        for step in self.config.pipeline_order:
            if step == "hotword" and self._hotword_entries:
                new_text = self.apply_hotword_correction(current_text)
                if new_text != current_text:
                    applied_rules.append("hotword")
                current_text = new_text

            elif step == "regex" and self.config.regex_rules:
                new_text = self.apply_regex_rules(current_text)
                if new_text != current_text:
                    applied_rules.append("regex")
                current_text = new_text

            elif step == "lm_rerank" and self.config.enable_lm_rerank and nbest:
                reranked = self.apply_lm_rerank(nbest)
                if reranked and reranked != current_text:
                    applied_rules.append("lm_rerank")
                    current_text = reranked

        cer_after = _compute_cer(reference_text, current_text) if reference_text else 0.0

        return PostProcessResult(
            original_text=text,
            processed_text=current_text,
            applied_rules=applied_rules,
            cer_before=cer_before,
            cer_after=cer_after,
        )

    # -----------------------------------------------------------------
    # Effect evaluation
    # -----------------------------------------------------------------

    def evaluate_effect(self, results: list[PostProcessResult]) -> dict:
        """Compute average CER before/after and the delta.

        Parameters
        ----------
        results : list[PostProcessResult]
            A list of post-processing results (each with ``cer_before``
            and ``cer_after``).

        Returns
        -------
        dict
            ``{"avg_cer_before": float, "avg_cer_after": float,
            "cer_delta": float, "total_samples": int}``
        """
        if not results:
            return {
                "avg_cer_before": 0.0,
                "avg_cer_after": 0.0,
                "cer_delta": 0.0,
                "total_samples": 0,
            }

        total = len(results)
        avg_before = sum(r.cer_before for r in results) / total
        avg_after = sum(r.cer_after for r in results) / total

        return {
            "avg_cer_before": avg_before,
            "avg_cer_after": avg_after,
            "cer_delta": avg_after - avg_before,
            "total_samples": total,
        }
