"""Result analyzer module for the ASR Training Optimizer system.

Provides the ``ResultAnalyzer`` class which evaluates ASR model performance
on test sets, computes per-scene CER/WER metrics, generates badcase lists,
classifies errors by type, and compares models against baselines.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from scripts.core.models import (
    BadCase,
    ErrorType,
    EvaluationReport,
    SceneMetrics,
)


# =====================================================================
# Edit-distance helpers
# =====================================================================


def _edit_distance_ops(ref: str, hyp: str) -> list[dict]:
    """Compute the minimum-edit-distance alignment between *ref* and *hyp*.

    Returns a list of error operations.  Each operation is a dict with:

    - ``type``: one of ``"substitution"``, ``"insertion"``, ``"deletion"``
    - ``ref``: the reference character (empty string for insertions)
    - ``hyp``: the hypothesis character (empty string for deletions)
    - ``position``: index in the reference string where the error occurs

    The implementation uses the standard dynamic-programming algorithm
    with back-tracking to recover the alignment path.
    """
    n = len(ref)
    m = len(hyp)

    # dp[i][j] = edit distance between ref[:i] and hyp[:j]
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
                    dp[i - 1][j - 1],  # substitution
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                )

    # Back-track to recover operations
    ops: list[dict] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            # Match – no error
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            # Substitution
            ops.append({
                "type": ErrorType.SUBSTITUTION.value,
                "ref": ref[i - 1],
                "hyp": hyp[j - 1],
                "position": i - 1,
            })
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            # Deletion (character in ref missing from hyp)
            ops.append({
                "type": ErrorType.DELETION.value,
                "ref": ref[i - 1],
                "hyp": "",
                "position": i - 1,
            })
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            # Insertion (extra character in hyp)
            ops.append({
                "type": ErrorType.INSERTION.value,
                "ref": "",
                "hyp": hyp[j - 1],
                "position": i,
            })
            j -= 1
        else:
            # Safety fallback – should not happen with a correct DP table
            break  # pragma: no cover

    ops.reverse()
    return ops


def _compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate between *reference* and *hypothesis*.

    CER = edit_distance(ref, hyp) / len(ref)

    Returns 0.0 when the reference is empty and the hypothesis is also
    empty.  Returns 1.0 when the reference is empty but the hypothesis
    is not (all characters are insertions, capped at 1.0 for consistency).
    """
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    ops = _edit_distance_ops(reference, hypothesis)
    return len(ops) / len(reference)


def _compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between *reference* and *hypothesis*.

    Splits on whitespace.  For Chinese text where words are not
    space-separated, this falls back to character-level (equivalent to
    CER).

    Returns 0.0 when both are empty, 1.0 when reference is empty but
    hypothesis is not.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    # Reuse the same DP logic but on word tokens
    ref_str = ref_words
    hyp_str = hyp_words
    n = len(ref_str)
    m = len(hyp_str)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_str[i - 1] == hyp_str[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],
                    dp[i - 1][j],
                    dp[i][j - 1],
                )

    return dp[n][m] / n


# =====================================================================
# Test-set sample representation
# =====================================================================


class EvalSample:
    """A single sample from a test set used for evaluation.

    Parameters
    ----------
    audio_path : str
        Path to the audio file.
    reference_text : str
        Ground-truth transcription.
    hypothesis_text : str
        Model-predicted transcription.
    scene_type : str
        Scene label (e.g. ``"clean"``, ``"background_voice"``,
        ``"background_noise"``).
    """

    __slots__ = ("audio_path", "reference_text", "hypothesis_text", "scene_type")

    def __init__(
        self,
        audio_path: str,
        reference_text: str,
        hypothesis_text: str,
        scene_type: str = "",
    ) -> None:
        self.audio_path = audio_path
        self.reference_text = reference_text
        self.hypothesis_text = hypothesis_text
        self.scene_type = scene_type


# Backward-compatible alias
TestSample = EvalSample


# =====================================================================
# ResultAnalyzer
# =====================================================================


class ResultAnalyzer:
    """Analyzes ASR model evaluation results.

    The analyzer works on a list of :class:`TestSample` instances that
    pair reference transcriptions with model hypotheses.  It computes
    overall and per-scene CER/WER, generates badcase lists, classifies
    errors, and compares evaluation reports.

    Parameters
    ----------
    samples : list[TestSample] | None
        Pre-loaded test samples.  Can also be set later via
        :meth:`load_samples` or :meth:`evaluate_model`.
    """

    def __init__(self, samples: list[TestSample] | None = None) -> None:
        self._samples: list[TestSample] = samples or []
        self._last_report: Optional[EvaluationReport] = None

    # ------------------------------------------------------------------
    # Sample management
    # ------------------------------------------------------------------

    def load_samples(self, samples: list[TestSample]) -> None:
        """Replace the internal sample list."""
        self._samples = list(samples)

    @property
    def samples(self) -> list[TestSample]:
        """Read-only access to the current sample list."""
        return list(self._samples)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate_model(
        self,
        model_path: str = "",
        test_set_path: str = "",
        samples: list[TestSample] | None = None,
    ) -> EvaluationReport:
        """Evaluate the model on a test set and return an :class:`EvaluationReport`.

        If *samples* is provided they are used directly; otherwise the
        analyzer uses its internal sample list.  The *model_path* and
        *test_set_path* parameters are recorded for provenance but the
        actual inference is expected to have been performed externally
        (the samples already contain hypothesis texts).

        The report includes:

        - Overall CER and WER (macro-average across all samples).
        - Per-scene CER and WER.
        - Badcases (samples with CER > 0, i.e. any error).
        - Error-type distribution across all samples.

        Parameters
        ----------
        model_path : str
            Path to the model (informational).
        test_set_path : str
            Path to the test set (informational).
        samples : list[TestSample] | None
            Optional override for the sample list.

        Returns
        -------
        EvaluationReport
        """
        if samples is not None:
            self._samples = list(samples)

        if not self._samples:
            report = EvaluationReport(overall_cer=0.0, overall_wer=0.0)
            self._last_report = report
            return report

        # Compute per-sample CER/WER
        sample_cer: list[float] = []
        sample_wer: list[float] = []
        for s in self._samples:
            sample_cer.append(_compute_cer(s.reference_text, s.hypothesis_text))
            sample_wer.append(_compute_wer(s.reference_text, s.hypothesis_text))

        overall_cer = sum(sample_cer) / len(sample_cer)
        overall_wer = sum(sample_wer) / len(sample_wer)

        # Per-scene metrics
        scene_metrics = self._compute_scene_metrics(sample_cer, sample_wer)

        # Badcases (all samples with any error, threshold=0)
        badcases = self._build_badcases(sample_cer, sample_wer, threshold=0.0)

        # Error-type distribution
        error_dist = self._compute_error_distribution()

        report = EvaluationReport(
            overall_cer=overall_cer,
            overall_wer=overall_wer,
            scene_metrics=scene_metrics,
            badcases=badcases,
            error_type_distribution=error_dist,
        )
        self._last_report = report
        return report

    # ------------------------------------------------------------------
    # Per-scene grouping
    # ------------------------------------------------------------------

    def _compute_scene_metrics(
        self,
        sample_cer: list[float],
        sample_wer: list[float],
    ) -> list[SceneMetrics]:
        """Group samples by scene type and compute per-scene CER/WER."""
        scene_data: dict[str, dict] = {}
        for idx, s in enumerate(self._samples):
            scene = s.scene_type or "unknown"
            if scene not in scene_data:
                scene_data[scene] = {"cer_sum": 0.0, "wer_sum": 0.0, "count": 0}
            scene_data[scene]["cer_sum"] += sample_cer[idx]
            scene_data[scene]["wer_sum"] += sample_wer[idx]
            scene_data[scene]["count"] += 1

        result: list[SceneMetrics] = []
        for scene, data in sorted(scene_data.items()):
            count = data["count"]
            result.append(
                SceneMetrics(
                    scene_type=scene,
                    cer=data["cer_sum"] / count,
                    wer=data["wer_sum"] / count,
                    sample_count=count,
                )
            )
        return result

    # ------------------------------------------------------------------
    # Badcase generation
    # ------------------------------------------------------------------

    def generate_badcases(
        self,
        threshold: float,
        samples: list[TestSample] | None = None,
    ) -> list[BadCase]:
        """Return samples whose CER exceeds *threshold*.

        Each returned :class:`BadCase` includes the audio path, reference
        and hypothesis texts, CER, WER, and detailed error operations.

        Parameters
        ----------
        threshold : float
            CER threshold.  Only samples with ``cer > threshold`` are
            included.
        samples : list[TestSample] | None
            Optional sample list override.

        Returns
        -------
        list[BadCase]
        """
        target_samples = samples if samples is not None else self._samples
        badcases: list[BadCase] = []
        for s in target_samples:
            cer = _compute_cer(s.reference_text, s.hypothesis_text)
            if cer > threshold:
                wer = _compute_wer(s.reference_text, s.hypothesis_text)
                error_details = _edit_distance_ops(
                    s.reference_text, s.hypothesis_text
                )
                badcases.append(
                    BadCase(
                        audio_path=s.audio_path,
                        reference_text=s.reference_text,
                        hypothesis_text=s.hypothesis_text,
                        cer=cer,
                        wer=wer,
                        error_details=error_details,
                        scene_type=s.scene_type,
                    )
                )
        return badcases

    # ------------------------------------------------------------------
    # Error classification
    # ------------------------------------------------------------------

    def classify_errors(
        self,
        reference: str,
        hypothesis: str,
    ) -> dict[str, int]:
        """Classify errors between *reference* and *hypothesis* by type.

        Returns a dict mapping each :class:`ErrorType` value to its count.
        The sum of all counts equals the edit distance between the two
        strings.

        Parameters
        ----------
        reference : str
            Ground-truth text.
        hypothesis : str
            Predicted text.

        Returns
        -------
        dict[str, int]
            Counts keyed by ``"substitution"``, ``"insertion"``,
            ``"deletion"``.
        """
        ops = _edit_distance_ops(reference, hypothesis)
        counts = {
            ErrorType.SUBSTITUTION.value: 0,
            ErrorType.INSERTION.value: 0,
            ErrorType.DELETION.value: 0,
        }
        for op in ops:
            counts[op["type"]] += 1
        return counts

    def classify_errors_from_badcases(
        self,
        badcases: list[BadCase],
    ) -> dict[str, int]:
        """Aggregate error-type counts across a list of badcases.

        Parameters
        ----------
        badcases : list[BadCase]
            Badcase instances (each must have ``reference_text`` and
            ``hypothesis_text``).

        Returns
        -------
        dict[str, int]
        """
        totals = {
            ErrorType.SUBSTITUTION.value: 0,
            ErrorType.INSERTION.value: 0,
            ErrorType.DELETION.value: 0,
        }
        for bc in badcases:
            counts = self.classify_errors(bc.reference_text, bc.hypothesis_text)
            for k, v in counts.items():
                totals[k] += v
        return totals

    # ------------------------------------------------------------------
    # Baseline comparison
    # ------------------------------------------------------------------

    def compare_with_baseline(
        self,
        baseline_report: EvaluationReport,
        current_report: EvaluationReport,
    ) -> dict:
        """Compare *current_report* against *baseline_report*.

        Returns a dict with:

        - ``overall_cer_delta``: current CER − baseline CER
        - ``overall_wer_delta``: current WER − baseline WER
        - ``scene_deltas``: list of per-scene dicts with ``scene_type``,
          ``cer_delta``, and ``wer_delta`` for scenes present in both
          reports.

        Parameters
        ----------
        baseline_report : EvaluationReport
        current_report : EvaluationReport

        Returns
        -------
        dict
        """
        overall_cer_delta = current_report.overall_cer - baseline_report.overall_cer
        overall_wer_delta = current_report.overall_wer - baseline_report.overall_wer

        # Build lookup for baseline scenes
        baseline_scenes: dict[str, SceneMetrics] = {
            sm.scene_type: sm for sm in baseline_report.scene_metrics
        }

        scene_deltas: list[dict] = []
        for sm in current_report.scene_metrics:
            if sm.scene_type in baseline_scenes:
                bsm = baseline_scenes[sm.scene_type]
                scene_deltas.append({
                    "scene_type": sm.scene_type,
                    "cer_delta": sm.cer - bsm.cer,
                    "wer_delta": sm.wer - bsm.wer,
                })

        return {
            "overall_cer_delta": overall_cer_delta,
            "overall_wer_delta": overall_wer_delta,
            "scene_deltas": scene_deltas,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_badcases(
        self,
        sample_cer: list[float],
        sample_wer: list[float],
        threshold: float,
    ) -> list[BadCase]:
        """Build badcase list from pre-computed CER/WER arrays."""
        badcases: list[BadCase] = []
        for idx, s in enumerate(self._samples):
            if sample_cer[idx] > threshold:
                error_details = _edit_distance_ops(
                    s.reference_text, s.hypothesis_text
                )
                badcases.append(
                    BadCase(
                        audio_path=s.audio_path,
                        reference_text=s.reference_text,
                        hypothesis_text=s.hypothesis_text,
                        cer=sample_cer[idx],
                        wer=sample_wer[idx],
                        error_details=error_details,
                        scene_type=s.scene_type,
                    )
                )
        return badcases

    def _compute_error_distribution(self) -> dict[str, int]:
        """Compute error-type distribution across all samples."""
        totals = {
            ErrorType.SUBSTITUTION.value: 0,
            ErrorType.INSERTION.value: 0,
            ErrorType.DELETION.value: 0,
        }
        for s in self._samples:
            ops = _edit_distance_ops(s.reference_text, s.hypothesis_text)
            for op in ops:
                totals[op["type"]] += 1
        return totals
