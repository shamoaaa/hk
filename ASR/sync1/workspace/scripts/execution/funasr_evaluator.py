"""FunASR model evaluation module for the ASR Training Optimizer system.

Uses FunASR's ``AutoModel`` for inference. CER/WER computation and
scene-level metrics are delegated to ``result_analyzer.py`` to avoid
code duplication.

Supports both FunASR jsonl format and the project's test_set.jsonl format.

Output: workspace/reports/evaluation/
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from scripts.core.models import SceneMetrics
from scripts.analysis.result_analyzer import (
    ResultAnalyzer,
    EvalSample,
    _compute_cer,
    _compute_wer,
)
from scripts.execution.post_processor import PostProcessor
from scripts.core.config_utils import load_config

logger = logging.getLogger(__name__)


class Evaluator:
    """ASR model evaluator using FunASR AutoModel.

    Inference is done here; CER/WER/scene metrics are computed by
    ``ResultAnalyzer`` from ``result_analyzer.py``.

    Parameters
    ----------
    config : dict
        Configuration dict with ``evaluation`` section.
    base_dir : str
        Workspace root directory.
    """

    def __init__(self, config: dict, base_dir: str = ".") -> None:
        self.config = config
        self.base_dir = Path(base_dir)
        self.eval_cfg = config.get("evaluation", {})
        self.report_dir = self.base_dir / "reports" / "evaluation"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._post_processor = self._build_post_processor(config)

    @staticmethod
    def _build_post_processor(config: dict) -> Optional[PostProcessor]:
        """Build a PostProcessor from config if post_process section exists."""
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
        hotword_path = None
        if hotword_file and Path(hotword_file).exists():
            hotword_path = hotword_file

        return PostProcessor(config=pp_config, hotword_path=hotword_path)

    def evaluate(
        self,
        model_dir: str,
        test_jsonl: str,
        tag: str = "",
    ) -> dict:
        """Run inference + evaluation. Returns CER, WER, scene metrics."""
        logger.info("Starting evaluation: model=%s", model_dir)

        test_entries = self._load_test_data(test_jsonl)
        if not test_entries:
            logger.error("No test data loaded from %s", test_jsonl)
            return {"cer": 1.0, "wer": 1.0, "num_samples": 0}

        results = self._run_inference(model_dir, test_entries)
        if not results:
            logger.error("Inference produced no results")
            return {"cer": 1.0, "wer": 1.0, "num_samples": 0}

        # Delegate CER/WER/scene computation to ResultAnalyzer
        samples = [
            EvalSample(
                audio_path=r["audio_path"],
                reference_text=r["reference"],
                hypothesis_text=r["hypothesis"],
                scene_type=r.get("scene_type", "unknown"),
            )
            for r in results
        ]
        analyzer = ResultAnalyzer(samples)
        report = analyzer.evaluate_model(model_path=model_dir, test_set_path=test_jsonl)

        eval_result = {
            "cer": report.overall_cer,
            "wer": report.overall_wer,
            "num_samples": len(results),
            "model_dir": model_dir,
            "tag": tag,
            "scene_metrics": [
                {
                    "scene_type": sm.scene_type,
                    "cer": sm.cer,
                    "wer": sm.wer,
                    "sample_count": sm.sample_count,
                }
                for sm in report.scene_metrics
            ],
            "error_type_distribution": report.error_type_distribution,
            "samples": results,
        }

        if self.eval_cfg.get("generate_report", True):
            self._generate_report(eval_result, tag)

        return eval_result

    def run_inference_only(
        self,
        model_dir: str,
        test_jsonl: str,
        output_path: str,
        output_format: str = "project",
    ) -> dict:
        """Run inference and write results to a file."""
        test_entries = self._load_test_data(test_jsonl)
        if not test_entries:
            return {"status": "error", "message": "No test data loaded"}

        results = self._run_inference(model_dir, test_entries)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                if output_format == "funasr":
                    record = {
                        "key": r.get("key", ""),
                        "source": r["audio_path"],
                        "target": r["reference"],
                        "hypothesis": r["hypothesis"],
                    }
                else:
                    record = {
                        "audio_path": r["audio_path"],
                        "reference_text": r["reference"],
                        "hypothesis_text": r["hypothesis"],
                        "scene_type": r.get("scene_type", "unknown"),
                    }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Compute summary CER/WER using result_analyzer
        cer_sum = 0.0
        for r in results:
            cer_sum += _compute_cer(r["reference"], r["hypothesis"])
        avg_cer = cer_sum / len(results) if results else 0.0

        wer_sum = 0.0
        for r in results:
            wer_sum += _compute_wer(r["reference"], r["hypothesis"])
        avg_wer = wer_sum / len(results) if results else 0.0

        return {
            "status": "completed",
            "output_path": str(output_path),
            "num_samples": len(results),
            "cer": avg_cer,
            "wer": avg_wer,
        }

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_test_data(test_jsonl: str) -> list[dict]:
        """Load test data, auto-detecting FunASR or project format."""
        entries = []
        with open(test_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))

        if not entries:
            return []

        first = entries[0]
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
        elif "audio_path" in first and "reference_text" in first:
            return [
                {
                    "key": e.get("key", f"sample_{i}"),
                    "audio_path": e["audio_path"],
                    "reference": e["reference_text"],
                    "scene_type": e.get("scene_type", "unknown"),
                }
                for i, e in enumerate(entries)
            ]
        else:
            logger.warning("Unknown test data format")
            return entries

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_inference(
        self, model_dir: str, test_entries: list[dict]
    ) -> list[dict]:
        """Run FunASR AutoModel inference."""
        try:
            from funasr import AutoModel
        except ImportError:
            logger.error("funasr not installed. Run: pip install funasr")
            return []

        logger.info("Loading model from %s", model_dir)
        model = AutoModel(model=model_dir)

        results = []
        for i, entry in enumerate(test_entries):
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
                "hypothesis": hyp_text,  # will be overwritten by post-processing
                "scene_type": entry.get("scene_type", "unknown"),
            })

            if (i + 1) % 100 == 0:
                logger.info("Inference progress: %d/%d", i + 1, len(test_entries))

        # Apply post-processing to all hypotheses
        if self._post_processor is not None:
            pp_count = 0
            for r in results:
                pp_result = self._post_processor.process(
                    text=r["hypothesis_raw"],
                    reference_text=r["reference"],
                )
                r["hypothesis"] = pp_result.processed_text
                if pp_result.applied_rules:
                    pp_count += 1
            logger.info(
                "Post-processing applied to %d/%d samples",
                pp_count, len(results),
            )

        return results

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def _generate_report(self, results: dict, tag: str) -> None:
        """Generate JSON + text evaluation reports."""
        tag_name = tag or "default"

        json_report = self.report_dir / f"eval_report_{tag_name}.json"
        report_data = {
            "model_dir": results["model_dir"],
            "num_samples": results["num_samples"],
            "cer": results["cer"],
            "wer": results["wer"],
            "scene_metrics": results.get("scene_metrics", []),
            "error_type_distribution": results.get("error_type_distribution", {}),
            "tag": results.get("tag", ""),
        }
        with open(json_report, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        txt_report = self.report_dir / f"eval_report_{tag_name}.txt"
        with open(txt_report, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("FunASR Paraformer Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: {results['model_dir']}\n")
            f.write(f"Samples: {results['num_samples']}\n")
            f.write(f"CER: {results['cer']:.4f} ({results['cer']*100:.2f}%)\n")
            f.write(f"WER: {results['wer']:.4f} ({results['wer']*100:.2f}%)\n\n")

            scene_metrics = results.get("scene_metrics", [])
            if scene_metrics:
                f.write("-" * 60 + "\n")
                f.write("Scene-level Metrics:\n")
                f.write("-" * 60 + "\n\n")
                for sm in scene_metrics:
                    f.write(
                        f"  {sm['scene_type']:20s}  "
                        f"CER={sm['cer']:.4f}  "
                        f"WER={sm['wer']:.4f}  "
                        f"(n={sm['sample_count']})\n"
                    )
                f.write("\n")

            f.write("-" * 60 + "\n")
            f.write("Error Samples (CER > 0):\n")
            f.write("-" * 60 + "\n\n")

            error_count = 0
            for r in results.get("samples", []):
                if r["reference"] != r["hypothesis"]:
                    error_count += 1
                    if error_count <= 50:
                        f.write(f"[{r['key']}] REF: {r['reference']}\n")
                        f.write(f"           HYP: {r['hypothesis']}\n\n")

            f.write(f"\nTotal errors: {error_count}/{results['num_samples']}\n")

        logger.info("Reports saved: %s, %s", json_report, txt_report)
