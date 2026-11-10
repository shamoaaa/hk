"""FunASR model inference module.

Responsibilities:
- Load test data (auto-detect FunASR / project JSONL format)
- Run FunASR AutoModel inference
- Optionally apply post-processing (hotword, regex, LM rerank)
- Write per-sample results to JSONL
- Compute summary CER/WER (with and without post-processing)

CER/WER scene-level analysis is handled by analyze_results.py (via ResultAnalyzer).
"""

import json
import logging
from pathlib import Path
from typing import Optional

from scripts.analysis.result_analyzer import _compute_cer, _compute_wer
from scripts.execution.post_processor import PostProcessor

logger = logging.getLogger(__name__)


# ======================================================================
# Public API
# ======================================================================


class Evaluator:
    """ASR model evaluator using FunASR AutoModel.

    Parameters
    ----------
    config : dict
        Full config dict (funasr_config.yaml contents).
    base_dir : str
        Workspace root directory.
    """

    def __init__(self, config: dict, base_dir: str = ".") -> None:
        self.config = config
        self.base_dir = Path(base_dir)
        self._post_processor = _build_post_processor(config)

    def run_inference_only(
        self,
        model_dir: str,
        test_jsonl: str,
        output_path: str,
        output_format: str = "project",
        apply_postprocess: bool = True,
    ) -> dict:
        """Run inference and write per-sample results to JSONL.

        Parameters
        ----------
        model_dir : str
            Path to trained model directory.
        test_jsonl : str
            Path to test data JSONL file.
        output_path : str
            Where to write inference results.
        output_format : str
            "project" or "funasr" output schema.
        apply_postprocess : bool
            If True, apply post-processing. Output always contains both
            hypothesis_text (processed) and hypothesis_raw (original).

        Returns
        -------
        dict with status, num_samples, cer, wer, and optional postprocess delta.
        """
        entries = _load_test_data(test_jsonl)
        if not entries:
            return {"status": "error", "message": "No test data loaded"}

        # --- Inference ---
        results = self._infer(model_dir, entries)
        if not results:
            return {"status": "error", "message": "Inference produced no results"}

        # --- Post-processing ---
        if apply_postprocess and self._post_processor is not None:
            _apply_postprocess(results, self._post_processor)
        else:
            logger.info("Post-processing skipped")

        # --- Write JSONL ---
        _write_results(results, output_path, output_format)

        # --- Compute metrics ---
        return _compute_summary(results, output_path, apply_postprocess,
                                has_postprocessor=self._post_processor is not None)

    # ------------------------------------------------------------------
    # Private: model inference
    # ------------------------------------------------------------------

    def _infer(self, model_dir: str, entries: list[dict]) -> list[dict]:
        """Load model and run inference on all entries."""
        try:
            from funasr import AutoModel
        except ImportError:
            logger.error("funasr not installed. Run: pip install funasr")
            return []

        logger.info("Loading model from %s", model_dir)
        model = AutoModel(model=model_dir)

        results = []
        for i, entry in enumerate(entries):
            wav_path = entry["audio_path"]
            ref_text = entry["reference"]

            try:
                res = model.generate(input=wav_path)
                hyp_text = res[0].get("text", "") if res else ""
            except Exception as e:
                logger.debug("Inference failed for %s: %s", entry.get("key", i), e)
                hyp_text = ""

            results.append({
                "key": entry.get("key", f"sample_{i}"),
                "audio_path": wav_path,
                "reference": ref_text,
                "hypothesis_raw": hyp_text,
                "hypothesis": hyp_text,
                "scene_type": entry.get("scene_type", "unknown"),
            })

            if (i + 1) % 50 == 0 or (i + 1) == len(entries):
                import time as _time
                if not hasattr(self, '_infer_start_time'):
                    self._infer_start_time = _time.time()
                elapsed = _time.time() - self._infer_start_time
                done = i + 1
                total = len(entries)
                pct = done / total * 100
                speed = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / speed if speed > 0 else 0
                eta_min, eta_sec = divmod(int(eta), 60)
                logger.info(
                    "Inference: %d/%d (%.1f%%) | %.1f samples/s | ETA %d:%02d",
                    done, total, pct, speed, eta_min, eta_sec,
                )

        return results


# ======================================================================
# Module-level helpers (stateless, testable)
# ======================================================================


def _build_post_processor(config: dict) -> Optional[PostProcessor]:
    """Build PostProcessor from config. Returns None if not configured."""
    from scripts.core.models import PostProcessConfig, HotwordEntry, RegexRule

    pp_cfg = config.get("post_process", {})
    if not pp_cfg:
        return None

    hotword_entries = [
        HotwordEntry(**e) for e in pp_cfg.get("hotword_entries", [])
        if isinstance(e, dict)
    ]
    regex_rules = [
        RegexRule(**r) for r in pp_cfg.get("regex_rules", [])
        if isinstance(r, dict)
    ]

    pp_config = PostProcessConfig(
        hotword_entries=hotword_entries,
        regex_rules=regex_rules,
        enable_lm_rerank=pp_cfg.get("enable_lm_rerank", False),
        lm_model_path=pp_cfg.get("lm_model_path"),
        pipeline_order=pp_cfg.get("pipeline_order", ["hotword", "regex", "lm_rerank"]),
    )

    hotword_file = pp_cfg.get("hotword_file")
    hotword_path = hotword_file if hotword_file and Path(hotword_file).exists() else None

    return PostProcessor(config=pp_config, hotword_path=hotword_path)


def _load_test_data(test_jsonl: str) -> list[dict]:
    """Load test data JSONL, auto-detecting FunASR or project format.

    Returns list of dicts with keys: key, audio_path, reference, scene_type.
    """
    entries = []
    with open(test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        return []

    first = entries[0]

    # FunASR format: {"source": "...", "target": "..."}
    if "source" in first and "target" in first:
        return [
            {
                "key": e.get("key", f"sample_{i}"),
                "audio_path": e["source"],
                "reference": e["target"],
                "scene_type": e.get("scene_type", "unknown"),
            }
            for i, e in enumerate(entries)
        ]

    # Project format: {"audio_path": "...", "reference_text": "..."}
    if "audio_path" in first and "reference_text" in first:
        return [
            {
                "key": e.get("key", f"sample_{i}"),
                "audio_path": e["audio_path"],
                "reference": e["reference_text"],
                "scene_type": e.get("scene_type", "unknown"),
            }
            for i, e in enumerate(entries)
        ]

    logger.warning("Unknown test data format in %s", test_jsonl)
    return entries


def _apply_postprocess(results: list[dict], processor: PostProcessor) -> None:
    """Apply post-processing in-place. Overwrites results[i]['hypothesis']."""
    pp_count = 0
    for r in results:
        pp_result = processor.process(
            text=r["hypothesis_raw"],
            reference_text=r["reference"],
        )
        r["hypothesis"] = pp_result.processed_text
        if pp_result.applied_rules:
            pp_count += 1
    logger.info("Post-processing applied to %d/%d samples", pp_count, len(results))


def _write_results(results: list[dict], output_path: str, fmt: str) -> None:
    """Write inference results to JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            if fmt == "funasr":
                record = {
                    "key": r.get("key", ""),
                    "source": r["audio_path"],
                    "target": r["reference"],
                    "hypothesis": r["hypothesis"],
                    "hypothesis_raw": r["hypothesis_raw"],
                }
            else:
                record = {
                    "audio_path": r["audio_path"],
                    "reference_text": r["reference"],
                    "hypothesis_text": r["hypothesis"],
                    "hypothesis_raw": r["hypothesis_raw"],
                    "scene_type": r.get("scene_type", "unknown"),
                }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _compute_summary(
    results: list[dict],
    output_path: str,
    apply_postprocess: bool,
    has_postprocessor: bool,
) -> dict:
    """Compute CER/WER summary. Optionally include raw vs processed delta."""
    n = len(results)

    cer = sum(_compute_cer(r["reference"], r["hypothesis"]) for r in results) / n if n else 0.0
    wer = sum(_compute_wer(r["reference"], r["hypothesis"]) for r in results) / n if n else 0.0

    summary = {
        "status": "completed",
        "output_path": str(output_path),
        "num_samples": n,
        "postprocess_applied": apply_postprocess,
        "cer": cer,
        "wer": wer,
    }

    # When post-processing was applied, also report raw metrics for comparison
    if apply_postprocess and has_postprocessor:
        cer_raw = sum(_compute_cer(r["reference"], r["hypothesis_raw"]) for r in results) / n if n else 0.0
        wer_raw = sum(_compute_wer(r["reference"], r["hypothesis_raw"]) for r in results) / n if n else 0.0
        summary["cer_raw"] = cer_raw
        summary["wer_raw"] = wer_raw
        summary["postprocess_cer_delta"] = cer_raw - cer
        summary["postprocess_wer_delta"] = wer_raw - wer

    return summary
