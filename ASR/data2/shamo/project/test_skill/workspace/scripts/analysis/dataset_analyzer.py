"""Dataset analyzer module for the ASR Training Optimizer system.

Provides the ``DatasetAnalyzer`` class which analyzes ASR dataset quality
based on FunASR jsonl format. Each sample is expected to have:

    {"key": ..., "source": ..., "source_len": N, "target": ..., "target_len": M}

Where ``source_len`` is the number of feature frames (each ~60ms at 16kHz),
used to approximate audio duration. ``target`` is the transcription text.

NOTE: This analyzer operates purely on metadata in the jsonl file. It does
NOT load or analyze the actual audio files. Audio-dependent analysis (SNR,
actual duration, corruption detection, scene classification) requires a
richer manifest format with pre-computed metadata, which this project's
FunASR-format data does not provide.

Available analyses:
- Basic stats: sample count, estimated total duration, duration distribution
- Text stats: character distribution, word frequency, sentence length
- Anomaly detection: empty annotations, missing audio files, chars/frame mismatch
- Quality report: issues sorted by severity
"""

from __future__ import annotations

import os
from typing import Optional

from scripts.core.config_utils import read_jsonl
from scripts.core.models import (
    AnomalySample,
    DatasetQualityReport,
    DatasetStats,
    SceneDistribution,
    TextStats,
)


# =====================================================================
# Constants
# =====================================================================

_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}

# FunASR feature frame duration: each frame represents ~60ms of audio
# (empirically validated: source_len=72 corresponds to ~4.3s audio)
_FRAME_DURATION_SECONDS = 0.06


class DatasetAnalyzer:
    """Analyzes ASR dataset quality from FunASR jsonl manifests.

    Reads a jsonl file (FunASR format) and computes metadata-based
    statistics. Audio-dependent analyses (SNR, true duration, corruption)
    are NOT performed — they require a richer manifest with pre-computed
    audio metadata.

    Parameters
    ----------
    min_chars_per_frame_ratio : float
        Minimum character-to-frame ratio below which samples are flagged
        as potentially mis-annotated (text too short for audio length).
        Default 0.03 (roughly 1 char per 2 seconds of audio).
    max_chars_per_frame_ratio : float
        Maximum character-to-frame ratio above which samples are flagged
        as potentially mis-annotated (text too long for audio length).
        Default 0.5 (roughly 8 chars per second of audio).
    check_audio_exists : bool
        If True, verify that audio files referenced by ``source`` exist
        on disk. Default False (skip filesystem checks for speed).
    """

    def __init__(
        self,
        min_chars_per_frame_ratio: float = 0.03,
        max_chars_per_frame_ratio: float = 0.5,
        check_audio_exists: bool = False,
        chars_per_second_range: tuple[float, float] = (1.0, 25.0),
        snr_threshold: Optional[float] = None,
        min_scene_samples: Optional[int] = None,
    ) -> None:
        self.min_chars_per_frame_ratio = min_chars_per_frame_ratio
        self.max_chars_per_frame_ratio = max_chars_per_frame_ratio
        self.check_audio_exists = check_audio_exists
        self.snr_threshold = snr_threshold
        self.min_scene_samples = min_scene_samples
        self.chars_per_second_range = chars_per_second_range

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_manifest(self, dataset_path: str) -> list[dict]:
        """Read the JSONL manifest from *dataset_path*.

        Parameters
        ----------
        dataset_path : str
            Path to the dataset directory containing ``manifest.jsonl``.

        Returns
        -------
        list[dict]
        """
        manifest_path = os.path.join(dataset_path, "manifest.jsonl")
        return read_jsonl(manifest_path)

    @staticmethod
    def _normalize_funasr_sample(sample: dict) -> dict:
        """Normalize a FunASR-format sample for analysis.

        FunASR format: {"key", "source", "source_len", "target", "target_len"}

        Returns a dict with:
        - ``audio_path``: from ``source``
        - ``text``: from ``target``
        - ``source_len``: feature frame count (kept as-is)
        - ``duration_estimated``: source_len * frame_duration (approximate)
        - ``char_count``: len(target) or ``target_len`` if provided

        Does NOT fabricate ``snr``, ``scene_type``, or true duration —
        those fields are not available in FunASR format.
        """
        if "source" in sample and "audio_path" not in sample:
            source_len = sample.get("source_len", 0)
            target = sample.get("target", "")
            char_count = sample.get("target_len") or len(target)
            return {
                "audio_path": sample.get("source", ""),
                "text": target,
                "source_len": source_len,
                "duration_estimated": round(source_len * _FRAME_DURATION_SECONDS, 3),
                "char_count": char_count,
                "key": sample.get("key", ""),
            }
        return sample

    @classmethod
    def _get_samples(
        cls, dataset_path: Optional[str], samples: Optional[list[dict]]
    ) -> list[dict]:
        """Resolve samples from either a dataset path or a pre-loaded list.

        Supports:
        - Pre-loaded sample list
        - Direct path to a .jsonl file (preferred for FunASR format)
        - Directory path containing manifest.jsonl (legacy)

        Automatically detects FunASR format and normalizes it.
        """
        if samples is not None:
            data = samples
        elif dataset_path is not None:
            if dataset_path.endswith(".jsonl"):
                data = read_jsonl(dataset_path)
            else:
                manifest_path = os.path.join(dataset_path, "manifest.jsonl")
                data = read_jsonl(manifest_path)
        else:
            return []

        # Auto-detect FunASR format and normalize
        if data and "source" in data[0] and "audio_path" not in data[0]:
            data = [cls._normalize_funasr_sample(s) for s in data]

        return data

    # ------------------------------------------------------------------
    # Basic statistics
    # ------------------------------------------------------------------

    def analyze_basic_stats(
        self,
        dataset_path: Optional[str] = None,
        samples: Optional[list[dict]] = None,
    ) -> DatasetStats:
        """Compute basic dataset statistics.

        Calculates total samples, total duration (hours), average
        duration (seconds), and a duration distribution histogram with
        bins ``"0-5s"``, ``"5-10s"``, ``"10-20s"``, ``"20-30s"``,
        ``"30+s"``.

        Parameters
        ----------
        dataset_path : str or None
            Path to dataset directory with ``manifest.jsonl``.
        samples : list[dict] or None
            Pre-loaded list of sample dicts.

        Returns
        -------
        DatasetStats
        """
        data = self._get_samples(dataset_path, samples)

        total_samples = len(data)
        if total_samples == 0:
            return DatasetStats(
                total_samples=0,
                total_duration_hours=0.0,
                avg_duration_seconds=0.0,
            )

        # Use estimated duration from source_len (FunASR) or explicit
        # duration field (legacy manifest format).
        durations = [
            s.get("duration_estimated") or s.get("duration", 0.0)
            for s in data
        ]
        total_seconds = sum(durations)
        total_duration_hours = total_seconds / 3600.0
        avg_duration_seconds = total_seconds / total_samples

        # Duration distribution histogram
        duration_distribution = {
            "0-5s": 0,
            "5-10s": 0,
            "10-20s": 0,
            "20-30s": 0,
            "30+s": 0,
        }
        for d in durations:
            if d < 5:
                duration_distribution["0-5s"] += 1
            elif d < 10:
                duration_distribution["5-10s"] += 1
            elif d < 20:
                duration_distribution["10-20s"] += 1
            elif d < 30:
                duration_distribution["20-30s"] += 1
            else:
                duration_distribution["30+s"] += 1

        return DatasetStats(
            total_samples=total_samples,
            total_duration_hours=total_duration_hours,
            avg_duration_seconds=avg_duration_seconds,
            duration_distribution=duration_distribution,
        )

    # ------------------------------------------------------------------
    # SNR analysis
    # ------------------------------------------------------------------

    def analyze_snr(
        self,
        dataset_path: Optional[str] = None,
        threshold: Optional[float] = None,
        samples: Optional[list[dict]] = None,
    ) -> list[str]:
        """Analyze SNR distribution and return low-quality sample paths.

        Samples whose ``snr`` field is below *threshold* (or
        ``self.snr_threshold`` if not specified) are considered low
        quality.

        Parameters
        ----------
        dataset_path : str or None
            Path to dataset directory.
        threshold : float or None
            SNR threshold override.
        samples : list[dict] or None
            Pre-loaded sample list.

        Returns
        -------
        list[str]
            Audio paths of samples with SNR below the threshold.
        """
        data = self._get_samples(dataset_path, samples)
        snr_thresh = threshold if threshold is not None else self.snr_threshold

        low_quality: list[str] = []
        if snr_thresh is None:
            return low_quality

        for s in data:
            snr = s.get("snr")
            if snr is not None and snr < snr_thresh:
                low_quality.append(s.get("audio_path", ""))

        return low_quality

    def _compute_snr_distribution(
        self, data: list[dict]
    ) -> dict[str, int]:
        """Compute SNR distribution histogram.

        Bins: ``"<0dB"``, ``"0-10dB"``, ``"10-20dB"``, ``"20-30dB"``,
        ``"30+dB"``, ``"unknown"``.
        """
        dist: dict[str, int] = {
            "<0dB": 0,
            "0-10dB": 0,
            "10-20dB": 0,
            "20-30dB": 0,
            "30+dB": 0,
            "unknown": 0,
        }
        for s in data:
            snr = s.get("snr")
            if snr is None:
                dist["unknown"] += 1
            elif snr < 0:
                dist["<0dB"] += 1
            elif snr < 10:
                dist["0-10dB"] += 1
            elif snr < 20:
                dist["10-20dB"] += 1
            elif snr < 30:
                dist["20-30dB"] += 1
            else:
                dist["30+dB"] += 1
        return dist

    # ------------------------------------------------------------------
    # Text analysis
    # ------------------------------------------------------------------

    def analyze_text(
        self,
        dataset_path: Optional[str] = None,
        samples: Optional[list[dict]] = None,
    ) -> TextStats:
        """Compute text annotation statistics.

        Calculates character frequency distribution, word frequency
        (split on whitespace), and sentence length distribution (by
        character count, binned).

        Parameters
        ----------
        dataset_path : str or None
            Path to dataset directory.
        samples : list[dict] or None
            Pre-loaded sample list.

        Returns
        -------
        TextStats
        """
        data = self._get_samples(dataset_path, samples)

        char_distribution: dict[str, int] = {}
        word_frequency: dict[str, int] = {}
        sentence_length_distribution: dict[str, int] = {
            "0-10": 0,
            "10-20": 0,
            "20-50": 0,
            "50-100": 0,
            "100+": 0,
        }

        for s in data:
            text = s.get("text", "")

            # Character distribution
            for ch in text:
                char_distribution[ch] = char_distribution.get(ch, 0) + 1

            # Word frequency (split on whitespace)
            words = text.split()
            for word in words:
                word_frequency[word] = word_frequency.get(word, 0) + 1

            # Sentence length distribution (by character count)
            length = len(text)
            if length < 10:
                sentence_length_distribution["0-10"] += 1
            elif length < 20:
                sentence_length_distribution["10-20"] += 1
            elif length < 50:
                sentence_length_distribution["20-50"] += 1
            elif length < 100:
                sentence_length_distribution["50-100"] += 1
            else:
                sentence_length_distribution["100+"] += 1

        return TextStats(
            char_distribution=char_distribution,
            word_frequency=word_frequency,
            sentence_length_distribution=sentence_length_distribution,
        )

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def detect_anomalies(
        self,
        dataset_path: Optional[str] = None,
        samples: Optional[list[dict]] = None,
    ) -> list[AnomalySample]:
        """Detect anomalous samples in the dataset.

        Anomaly types:

        - **empty_label**: empty or whitespace-only text
        - **duration_mismatch**: ``len(text) / duration`` outside
          ``chars_per_second_range`` — suggests mis-annotation
        - **corrupted_audio**: ``duration`` <= 0, missing, or
          ``audio_path`` missing/empty (proxy for unusable audio)
        - **missing_audio** (optional, when ``check_audio_exists=True``):
          audio file referenced by ``audio_path`` does not exist on disk

        Parameters
        ----------
        dataset_path : str or None
            Path to dataset jsonl file or directory.
        samples : list[dict] or None
            Pre-loaded sample list.

        Returns
        -------
        list[AnomalySample]
        """
        data = self._get_samples(dataset_path, samples)
        anomalies: list[AnomalySample] = []

        min_cps, max_cps = self.chars_per_second_range

        for s in data:
            audio_path = s.get("audio_path", "")
            text = s.get("text", "")
            duration = s.get("duration") or s.get("duration_estimated")
            source_len = s.get("source_len")

            # Empty or whitespace-only label
            if not text or text.strip() == "":
                anomalies.append(
                    AnomalySample(
                        audio_path=audio_path,
                        anomaly_type="empty_label",
                        details="Text label is empty or whitespace-only",
                    )
                )
                continue

            # Missing audio_path or unusable audio (zero / negative / missing duration)
            # Note: source_len = -1 is a valid marker for augmented samples (speed perturb
            # changes duration, so original source_len is invalidated). Skip this check
            # when source_len is explicitly -1.
            if not audio_path:
                anomalies.append(
                    AnomalySample(
                        audio_path=audio_path,
                        anomaly_type="corrupted_audio",
                        details="audio_path is missing or empty",
                    )
                )
                continue
            if source_len == -1:
                # Augmented sample with invalidated source_len — skip duration check
                duration = None
            if duration is not None and duration <= 0:
                anomalies.append(
                    AnomalySample(
                        audio_path=audio_path,
                        anomaly_type="corrupted_audio",
                        details=f"duration is missing or non-positive: {duration}",
                    )
                )
                continue
            if duration is None and source_len != -1:
                # No duration info and not an augmented sample
                if source_len is None or source_len <= 0:
                    anomalies.append(
                        AnomalySample(
                            audio_path=audio_path,
                            anomaly_type="corrupted_audio",
                            details=f"duration unavailable and source_len={source_len}",
                        )
                    )
                    continue

            # Duration / char-count mismatch (chars per second)
            char_count = len(text)
            if duration and duration > 0:
                cps = char_count / duration
                if cps < min_cps or cps > max_cps:
                    anomalies.append(
                        AnomalySample(
                            audio_path=audio_path,
                            anomaly_type="duration_mismatch",
                            details=(
                                f"chars-per-second ({cps:.2f}) outside expected range "
                                f"[{min_cps}, {max_cps}] (chars={char_count}, duration={duration}s)"
                            ),
                        )
                    )

                # Too short audio (< 0.5s) — likely noise or annotation error (Ch3/Ch5)
                if duration < 0.5:
                    anomalies.append(
                        AnomalySample(
                            audio_path=audio_path,
                            anomaly_type="too_short",
                            details=f"Audio too short ({duration:.2f}s < 0.5s), likely noise or mis-cut",
                        )
                    )

                # Too long audio (> 30s) — may cause OOM or alignment issues (Ch5)
                if duration > 30.0:
                    anomalies.append(
                        AnomalySample(
                            audio_path=audio_path,
                            anomaly_type="too_long",
                            details=f"Audio too long ({duration:.1f}s > 30s), may cause OOM or poor alignment",
                        )
                    )

            # Very short text (< 2 chars) — suspicious annotation (Ch3)
            if char_count < 2:
                anomalies.append(
                    AnomalySample(
                        audio_path=audio_path,
                        anomaly_type="short_text",
                        details=f"Text too short ({char_count} chars), possibly mis-annotated",
                    )
                )

            # Optional: verify audio file exists on disk
            if self.check_audio_exists and audio_path:
                if not os.path.isfile(audio_path):
                    anomalies.append(
                        AnomalySample(
                            audio_path=audio_path,
                            anomaly_type="missing_audio",
                            details=f"Audio file does not exist: {audio_path}",
                        )
                    )

        return anomalies

    # ------------------------------------------------------------------
    # Scene distribution
    # ------------------------------------------------------------------

    def analyze_scene_distribution(
        self,
        dataset_path: Optional[str] = None,
        samples: Optional[list[dict]] = None,
    ) -> SceneDistribution:
        """Analyze scene type distribution in the dataset.

        NOTE: FunASR jsonl format does not include a ``scene_type`` field.
        When scene labels are unavailable, this method returns an empty
        distribution — it does NOT fabricate scene assignments. To enable
        scene analysis, provide a manifest with explicit ``scene_type``
        labels per sample.

        Counts samples per scene, computes ratios, and identifies
        scenes with fewer samples than ``self.min_scene_samples``
        (when set; ``None`` disables the underrepresentation check).

        Parameters
        ----------
        dataset_path : str or None
            Path to dataset directory.
        samples : list[dict] or None
            Pre-loaded sample list.

        Returns
        -------
        SceneDistribution
        """
        data = self._get_samples(dataset_path, samples)

        # Count samples per scene; samples without an explicit scene_type
        # are grouped as "unknown" to keep counts consistent with total.
        scene_counts: dict[str, int] = {}
        for s in data:
            scene = s.get("scene_type") or "unknown"
            scene_counts[scene] = scene_counts.get(scene, 0) + 1

        if not scene_counts:
            return SceneDistribution(
                scene_counts={},
                scene_ratios={},
                underrepresented_scenes=[],
            )

        total = sum(scene_counts.values())
        scene_ratios: dict[str, float] = {}
        if total > 0:
            for scene, count in scene_counts.items():
                scene_ratios[scene] = count / total
        else:
            for scene in scene_counts:
                scene_ratios[scene] = 0.0

        # Only flag underrepresented scenes if a threshold is configured
        if self.min_scene_samples is None:
            underrepresented: list[str] = []
        else:
            underrepresented = [
                scene
                for scene, count in scene_counts.items()
                if count < self.min_scene_samples
            ]

        return SceneDistribution(
            scene_counts=scene_counts,
            scene_ratios=scene_ratios,
            underrepresented_scenes=underrepresented,
        )

    # ------------------------------------------------------------------
    # Quality report generation
    # ------------------------------------------------------------------

    def generate_quality_report(
        self,
        dataset_path: Optional[str] = None,
        samples: Optional[list[dict]] = None,
    ) -> DatasetQualityReport:
        """Generate a comprehensive dataset quality report.

        Calls all analysis methods and aggregates issues sorted by
        severity (critical > warning > info).

        Parameters
        ----------
        dataset_path : str or None
            Path to dataset directory.
        samples : list[dict] or None
            Pre-loaded sample list.

        Returns
        -------
        DatasetQualityReport
        """
        data = self._get_samples(dataset_path, samples)

        # Run all analyses
        stats = self.analyze_basic_stats(samples=data)
        low_quality = self.analyze_snr(samples=data)
        stats.low_quality_samples = low_quality
        stats.snr_distribution = self._compute_snr_distribution(data)

        text_stats = self.analyze_text(samples=data)
        anomalies = self.detect_anomalies(samples=data)
        scene_dist = self.analyze_scene_distribution(samples=data)

        # Aggregate issues by severity
        issues: list[dict] = []

        # Missing source_len → critical (can't analyze)
        missing_len = [a for a in anomalies if a.anomaly_type == "missing_source_len"]
        if missing_len:
            issues.append({
                "severity": "critical",
                "type": "missing_source_len",
                "count": len(missing_len),
                "message": f"Found {len(missing_len)} sample(s) with missing source_len",
            })

        # Empty labels → critical
        empty_labels = [a for a in anomalies if a.anomaly_type == "empty_label"]
        if empty_labels:
            issues.append({
                "severity": "critical",
                "type": "empty_label",
                "count": len(empty_labels),
                "message": f"Found {len(empty_labels)} sample(s) with empty labels",
            })

        # Missing audio files (if check_audio_exists enabled) → critical
        missing_audio = [a for a in anomalies if a.anomaly_type == "missing_audio"]
        if missing_audio:
            issues.append({
                "severity": "critical",
                "type": "missing_audio",
                "count": len(missing_audio),
                "message": f"Found {len(missing_audio)} sample(s) with missing audio files",
            })

        # Length mismatch (char/frame ratio anomaly) → warning
        mismatches = [a for a in anomalies if a.anomaly_type == "length_mismatch"]
        if mismatches:
            issues.append({
                "severity": "warning",
                "type": "length_mismatch",
                "count": len(mismatches),
                "message": f"Found {len(mismatches)} sample(s) with char/frame ratio anomaly",
            })

        # Low quality SNR → warning (only if data has snr field)
        if low_quality:
            issues.append({
                "severity": "warning",
                "type": "low_snr",
                "count": len(low_quality),
                "message": f"Found {len(low_quality)} sample(s) with SNR below threshold",
            })

        # Underrepresented scenes → info (only if data has scene_type field)
        if scene_dist.underrepresented_scenes:
            issues.append({
                "severity": "info",
                "type": "underrepresented_scenes",
                "scenes": scene_dist.underrepresented_scenes,
                "message": (
                    f"Found {len(scene_dist.underrepresented_scenes)} "
                    f"underrepresented scene(s): "
                    f"{', '.join(scene_dist.underrepresented_scenes)}"
                ),
            })

        # Sort by severity (critical first, then warning, then info)
        issues.sort(key=lambda x: _SEVERITY_ORDER.get(x["severity"], 99))

        return DatasetQualityReport(
            stats=stats,
            text_stats=text_stats,
            anomalies=anomalies,
            scene_distribution=scene_dist,
            issues_by_severity=issues,
        )
