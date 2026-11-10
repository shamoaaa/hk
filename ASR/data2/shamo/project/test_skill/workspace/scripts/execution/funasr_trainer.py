"""FunASR training wrapper for the ASR Training Optimizer system.

Wraps FunASR's ``funasr.bin.train_ds`` training script via subprocess.
During training, FunASR's stdout is parsed line-by-line in real time
and converted to the project's ``TrainingMetrics`` format, which is
written to ``logs/training/training_metrics.jsonl``. This means
downstream tools (``collect_metrics.py``, ``TrainingMonitor``) work
directly — no separate log conversion step needed.

The raw FunASR log is also saved to ``logs/training/<tag>/log.txt``
for reference.

Output directories:
    - Models: workspace/models/iterations/<tag>/
    - Logs:   workspace/logs/training/ (training_metrics.jsonl + <tag>/log.txt)
"""

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from scripts.core.config_utils import append_jsonl
from scripts.analysis.funasr_log_analyzer import FunASRLogParser
from scripts.core.models import TrainingMetrics

logger = logging.getLogger(__name__)


class Trainer:
    """FunASR Paraformer training wrapper.

    During training, each line of FunASR output is:
    1. Written to ``logs/training/<tag>/log.txt`` (raw log)
    2. Parsed by ``FunASRLogParser`` regex
    3. Converted to ``TrainingMetrics`` and appended to
       ``logs/training/training_metrics.jsonl``

    Parameters
    ----------
    config : dict
        Full configuration dict (model, training, data sections).
    base_dir : str
        Workspace root directory.
    """

    def __init__(self, config: dict, base_dir: str = ".") -> None:
        self.config = config
        self.base_dir = Path(base_dir)
        self.model_cfg = config.get("model", {})
        self.train_cfg = config.get("training", {})
        self._log_parser = FunASRLogParser()

        config_output = self.train_cfg.get("output_dir", "")
        if config_output:
            output_path = Path(config_output)
            try:
                output_path = output_path.relative_to(self.base_dir)
            except ValueError:
                pass
            self.models_dir = self.base_dir / output_path
        else:
            self.models_dir = self.base_dir / "models" / "iterations"

        self.logs_dir = self.base_dir / "logs" / "training"

    def train(
        self,
        train_jsonl: str,
        val_jsonl: str,
        override_params: Optional[dict] = None,
        tag: str = "",
    ) -> dict:
        """Launch training. Logs are written to training_metrics.jsonl in real time.

        Returns
        -------
        dict
            ``{"status", "output_dir", "log_file", "metrics_file", "params"}``.
        """
        output_dir = self.models_dir / tag if tag else self.models_dir / "default"
        tag_log_dir = self.logs_dir / tag if tag else self.logs_dir / "default"
        output_dir.mkdir(parents=True, exist_ok=True)
        tag_log_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        raw_log_file = tag_log_dir / "log.txt"
        metrics_file = self.logs_dir / "training_metrics.jsonl"

        params = self._build_params(train_jsonl, val_jsonl, output_dir)
        if override_params:
            params.update(override_params)

        gpu_ids = self.train_cfg.get("gpu_ids", "4")
        gpu_list = gpu_ids.split(",")
        gpu_num = len(gpu_list)

        cmd = self._build_command(params, gpu_num, gpu_ids)

        logger.info("Starting training (tag=%s)...", tag or "default")
        logger.info("Raw log: %s", raw_log_file)
        logger.info("Metrics: %s", metrics_file)
        logger.info("GPU: %s (%d card(s))", gpu_ids, gpu_num)
        logger.info("Command: %s", " ".join(cmd))

        metrics_count = 0
        _train_start_time = None

        with open(raw_log_file, "w", encoding="utf-8") as raw_f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu_ids},
                text=True,
                bufsize=1,
            )

            for line in process.stdout:
                # Write raw log
                raw_f.write(line)
                raw_f.flush()

                # Parse and convert to TrainingMetrics
                parsed = self._log_parser.parse_line(line)
                if parsed and parsed.rank == 0:
                    if _train_start_time is None:
                        import time as _time
                        _train_start_time = _time.time()

                    approx_cer = (
                        1.0 - parsed.acc_avg_epoch
                        if parsed.acc_avg_epoch > 0
                        else None
                    )
                    metrics = TrainingMetrics(
                        step=parsed.total_step,
                        epoch=parsed.epoch,
                        train_loss=parsed.loss_avg_epoch,
                        val_loss=None,
                        learning_rate=parsed.lr,
                        gradient_norm=0.0,
                        cer=approx_cer,
                        wer=None,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                    append_jsonl(str(metrics_file), asdict(metrics))
                    metrics_count += 1

                    if metrics_count % 50 == 0 or metrics_count == 1:
                        import time as _time
                        elapsed = _time.time() - _train_start_time
                        speed = metrics_count / elapsed if elapsed > 0 else 0
                        max_epoch = self.train_cfg.get("max_epoch", 10)
                        # Rough ETA: assume linear progress within epochs
                        epoch_progress = f"epoch {parsed.epoch}/{max_epoch}"
                        elapsed_min = int(elapsed) // 60
                        elapsed_sec = int(elapsed) % 60
                        logger.info(
                            "[Training] step=%d %s | loss=%.4f acc=%.4f lr=%.2e | %.1f steps/s | elapsed %d:%02d",
                            parsed.total_step,
                            epoch_progress,
                            parsed.loss_avg_epoch,
                            parsed.acc_avg_epoch,
                            parsed.lr,
                            speed,
                            elapsed_min,
                            elapsed_sec,
                        )

            process.wait()
            return_code = process.returncode

        if return_code != 0:
            logger.error("Training failed with return code %d", return_code)
            return {
                "status": "failed",
                "output_dir": str(output_dir),
                "log_file": str(raw_log_file),
                "metrics_file": str(metrics_file),
                "metrics_count": metrics_count,
                "params": params,
                "return_code": return_code,
            }

        logger.info(
            "Training completed. %d metrics written to %s",
            metrics_count,
            metrics_file,
        )
        return {
            "status": "completed",
            "output_dir": str(output_dir),
            "log_file": str(raw_log_file),
            "metrics_file": str(metrics_file),
            "metrics_count": metrics_count,
            "params": params,
        }

    def get_training_command(
        self,
        train_jsonl: str,
        val_jsonl: str,
        override_params: Optional[dict] = None,
        tag: str = "",
    ) -> dict:
        """Return the training command without executing (dry-run)."""
        output_dir = self.models_dir / tag if tag else self.models_dir / "default"
        tag_log_dir = self.logs_dir / tag if tag else self.logs_dir / "default"

        params = self._build_params(train_jsonl, val_jsonl, output_dir)
        if override_params:
            params.update(override_params)

        gpu_ids = self.train_cfg.get("gpu_ids", "4")
        gpu_list = gpu_ids.split(",")
        gpu_num = len(gpu_list)

        cmd = self._build_command(params, gpu_num, gpu_ids)

        return {
            "command": " ".join(cmd),
            "command_list": cmd,
            "env": {"CUDA_VISIBLE_DEVICES": gpu_ids},
            "output_dir": str(output_dir),
            "log_file": str(tag_log_dir / "log.txt"),
            "metrics_file": str(self.logs_dir / "training_metrics.jsonl"),
            "params": params,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_params(
        self, train_jsonl: str, val_jsonl: str, output_dir: Path
    ) -> dict:
        cfg = self.train_cfg
        params = {
            "++model": self.model_cfg.get(
                "name",
                "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            ),
            "++model_revision": self.model_cfg.get("model_revision", "v2.0.4"),
            "++train_data_set_list": str(Path(train_jsonl).resolve()),
            "++valid_data_set_list": str(Path(val_jsonl).resolve()),
            "++dataset": "AudioDatasetHotword",
            "++dataset_conf.seaco_id": 1,
            "++dataset_conf.batch_size": cfg.get("batch_size", 5000),
            "++dataset_conf.batch_type": cfg.get("batch_type", "token"),
            "++dataset_conf.num_workers": cfg.get("num_workers", 4),
            "++train_conf.max_epoch": cfg.get("max_epoch", 1),
            "++train_conf.log_interval": cfg.get("log_interval", 1),
            "++train_conf.resume": str(cfg.get("resume", False)).lower(),
            "++train_conf.validate_interval": cfg.get("validate_interval", 2000),
            "++train_conf.save_checkpoint_interval": cfg.get(
                "save_checkpoint_interval", 2000
            ),
            "++train_conf.keep_nbest_models": cfg.get("keep_nbest_models", 20),
            "++train_conf.avg_nbest_model": cfg.get("avg_nbest_model", 10),
            "++optim_conf.lr": cfg.get("learning_rate", 0.0002),
            "++output_dir": str(output_dir),
        }
        max_token_length = cfg.get("max_token_length")
        if max_token_length:
            params["++dataset_conf.max_token_length"] = max_token_length

        # SpecAugment 特征增强（训练时应用到 log-mel 特征）
        # SeACo-Paraformer 使用 LFR 前端，模型自带 specaug_conf 含 lfr_rate，
        # 因此使用 SpecAugLFR 类（兼容 lfr_rate 参数）
        spec_aug = cfg.get("spec_aug", {})
        if spec_aug.get("enabled", False):
            params["++specaug"] = "SpecAugLFR"
            params["++specaug_conf.apply_time_warp"] = str(
                spec_aug.get("apply_time_warp", True)
            ).lower()
            params["++specaug_conf.time_warp_window"] = spec_aug.get("max_w", 5)
            params["++specaug_conf.apply_freq_mask"] = str(
                spec_aug.get("apply_freq_mask", True)
            ).lower()
            params["++specaug_conf.freq_mask_width_range"] = str(
                [0, spec_aug.get("max_f", 10)]
            )
            params["++specaug_conf.num_freq_mask"] = spec_aug.get("num_f_mask", 2)
            params["++specaug_conf.apply_time_mask"] = str(
                spec_aug.get("apply_time_mask", True)
            ).lower()
            params["++specaug_conf.time_mask_width_range"] = str(
                [0, spec_aug.get("max_t", 50)]
            )
            params["++specaug_conf.num_time_mask"] = spec_aug.get("num_t_mask", 2)

        return params

    def _build_command(self, params: dict, gpu_num: int, gpu_ids: str) -> list:
        if gpu_num > 1:
            cmd = [
                "torchrun",
                "--nnodes", "1",
                "--nproc_per_node", str(gpu_num),
            ]
        else:
            cmd = [sys.executable]
        cmd.extend(["-m", "funasr.bin.train_ds"])
        for key, value in params.items():
            cmd.append(f"{key}={value}")
        return cmd
