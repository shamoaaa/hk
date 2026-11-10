"""Training monitor module for the ASR Training Optimizer system.

Provides the ``TrainingMonitor`` class which collects training metrics at
specified step intervals, persists them to JSONL files, and computes
epoch-level aggregate statistics (average CER and WER).
"""

import datetime
import math
from dataclasses import asdict
from typing import Optional

from scripts.core.config_utils import append_jsonl, read_jsonl
from scripts.core.models import AlertType, TrainingAlert, TrainingMetrics

import os


class TrainingMonitor:
    """Monitors ASR model training by collecting and aggregating metrics.

    Parameters
    ----------
    log_dir : str
        Directory where JSONL metric files are stored.
    collect_interval : int
        Step interval at which metrics are collected. A call to
        ``collect_metrics`` is only recorded when ``step`` is a multiple
        of this interval.
    """

    def __init__(self, log_dir: str, collect_interval: int = 100) -> None:
        self.log_dir = log_dir
        self.collect_interval = collect_interval
        self._metrics: list[TrainingMetrics] = []
        self._jsonl_path = os.path.join(log_dir, "training_metrics.jsonl")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_metrics(
        self, step: int, epoch: int, raw_metrics: dict
    ) -> Optional[TrainingMetrics]:
        """Collect training metrics if *step* aligns with the collect interval.

        Parameters
        ----------
        step : int
            Current global training step.
        epoch : int
            Current training epoch.
        raw_metrics : dict
            Dictionary that may contain the keys ``train_loss``,
            ``val_loss``, ``learning_rate``, ``gradient_norm``, ``cer``,
            and ``wer``.

        Returns
        -------
        TrainingMetrics or None
            The recorded metrics instance when the step is collected,
            otherwise ``None``.
        """
        if step % self.collect_interval != 0:
            return None

        timestamp = raw_metrics.get(
            "timestamp", datetime.datetime.now().isoformat()
        )

        metrics = TrainingMetrics(
            step=step,
            epoch=epoch,
            train_loss=raw_metrics.get("train_loss", 0.0),
            val_loss=raw_metrics.get("val_loss"),
            learning_rate=raw_metrics.get("learning_rate", 0.0),
            gradient_norm=raw_metrics.get("gradient_norm", 0.0),
            cer=raw_metrics.get("cer"),
            wer=raw_metrics.get("wer"),
            timestamp=timestamp,
        )

        # Store in memory
        self._metrics.append(metrics)

        # Persist to JSONL
        append_jsonl(self._jsonl_path, asdict(metrics))

        return metrics

    def compute_epoch_metrics(self, epoch: int) -> dict:
        """Compute average CER and WER for a given *epoch*.

        Only metrics whose ``cer`` / ``wer`` values are not ``None`` are
        included in the averages.

        Parameters
        ----------
        epoch : int
            The epoch number to aggregate.

        Returns
        -------
        dict
            A dictionary with keys ``epoch``, ``avg_cer``, ``avg_wer``,
            and ``sample_count``.  If no metrics match the epoch, both
            averages are ``None`` and ``sample_count`` is ``0``.
        """
        epoch_metrics = [m for m in self._metrics if m.epoch == epoch]

        cer_values = [m.cer for m in epoch_metrics if m.cer is not None]
        wer_values = [m.wer for m in epoch_metrics if m.wer is not None]

        avg_cer = sum(cer_values) / len(cer_values) if cer_values else None
        avg_wer = sum(wer_values) / len(wer_values) if wer_values else None

        return {
            "epoch": epoch,
            "avg_cer": avg_cer,
            "avg_wer": avg_wer,
            "sample_count": len(epoch_metrics),
        }

    # ------------------------------------------------------------------
    # Stubs – implemented in later tasks (2.3, 2.6)
    # ------------------------------------------------------------------

    def check_stagnation(
        self, threshold: float, window: int = 5
    ) -> Optional[TrainingAlert]:
        """Detect training stagnation over the last *window* epochs.

        Computes the per-epoch average ``train_loss`` from stored metrics.
        If the loss decrease between every consecutive pair of epochs in
        the last *window* epochs is less than *threshold*, a
        ``STAGNATION`` alert is returned.

        Parameters
        ----------
        threshold : float
            Minimum loss decrease between consecutive epochs to be
            considered "progressing".
        window : int
            Number of recent epochs to examine (default ``5``).

        Returns
        -------
        TrainingAlert or None
            A ``STAGNATION`` alert when stagnation is detected, otherwise
            ``None``.
        """
        # Group metrics by epoch and compute average train_loss per epoch.
        epoch_losses: dict[int, list[float]] = {}
        for m in self._metrics:
            epoch_losses.setdefault(m.epoch, []).append(m.train_loss)

        # Need at least `window` distinct epochs.
        unique_epochs = sorted(epoch_losses.keys())
        if len(unique_epochs) < window:
            return None

        # Take the last `window` epochs' average losses.
        recent_epochs = unique_epochs[-window:]
        avg_losses = [
            sum(epoch_losses[e]) / len(epoch_losses[e]) for e in recent_epochs
        ]

        # Check if ALL consecutive decreases are below threshold.
        for i in range(len(avg_losses) - 1):
            decrease = avg_losses[i] - avg_losses[i + 1]
            if decrease >= threshold:
                return None

        # All decreases are below threshold → stagnation detected.
        last_epoch = recent_epochs[-1]
        last_metrics = [m for m in self._metrics if m.epoch == last_epoch]
        last_step = max(m.step for m in last_metrics) if last_metrics else 0

        return TrainingAlert(
            alert_type=AlertType.STAGNATION,
            message=(
                f"Training stagnation detected: loss decrease below "
                f"{threshold} for the last {window} epochs"
            ),
            context={
                "window": window,
                "threshold": threshold,
                "recent_epochs": recent_epochs,
                "avg_losses": avg_losses,
            },
            epoch=last_epoch,
            step=last_step,
        )

    def check_numerical_anomaly(self, loss: float) -> Optional[TrainingAlert]:
        """Detect NaN or Inf values in the given loss.

        Parameters
        ----------
        loss : float
            The loss value to check.

        Returns
        -------
        TrainingAlert or None
            A ``NAN_INF`` alert when the loss is NaN or Inf, otherwise
            ``None``.
        """
        if math.isnan(loss) or math.isinf(loss):
            return TrainingAlert(
                alert_type=AlertType.NAN_INF,
                message=f"Numerical anomaly detected: loss is {loss}",
                context={"loss_value": str(loss)},
            )
        return None

    def get_training_curves(self) -> dict:
        """Return training curve data for visualization.

        Returns
        -------
        dict
            A dictionary with three keys:

            ``"loss"``
                Dict with ``"steps"`` (list of step numbers),
                ``"train_loss"`` (list of train loss values), and
                ``"val_loss"`` (list of validation loss values, ``None``
                kept for steps where validation was not performed).

            ``"cer_wer"``
                Dict with ``"steps"``, ``"cer"``, and ``"wer"`` lists.
                Only metrics where both CER and WER are not ``None`` are
                included.

            ``"learning_rate"``
                Dict with ``"steps"`` and ``"values"`` lists containing
                the learning rate at each collected step.
        """
        loss_steps: list[int] = []
        train_loss_values: list[float] = []
        val_loss_values: list[float | None] = []

        cer_wer_steps: list[int] = []
        cer_values: list[float] = []
        wer_values: list[float] = []

        lr_steps: list[int] = []
        lr_values: list[float] = []

        for m in self._metrics:
            # Loss curves: include all metrics
            loss_steps.append(m.step)
            train_loss_values.append(m.train_loss)
            val_loss_values.append(m.val_loss)

            # CER/WER curves: only where both are not None
            if m.cer is not None and m.wer is not None:
                cer_wer_steps.append(m.step)
                cer_values.append(m.cer)
                wer_values.append(m.wer)

            # Learning rate: include all metrics
            lr_steps.append(m.step)
            lr_values.append(m.learning_rate)

        return {
            "loss": {
                "steps": loss_steps,
                "train_loss": train_loss_values,
                "val_loss": val_loss_values,
            },
            "cer_wer": {
                "steps": cer_wer_steps,
                "cer": cer_values,
                "wer": wer_values,
            },
            "learning_rate": {
                "steps": lr_steps,
                "values": lr_values,
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def metrics(self) -> list[TrainingMetrics]:
        """Return the in-memory metrics list (read-only access)."""
        return list(self._metrics)
