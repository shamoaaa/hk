"""FunASR training log line parser.

Parses FunASR's native log output line-by-line using regex, extracting
training metrics (loss, accuracy, learning rate, GPU memory, etc.).

Used by ``funasr_trainer.py`` to convert FunASR stdout into the project's
``TrainingMetrics`` format in real time during training.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class FunASRTrainStep:
    """Single training step parsed from a FunASR log line."""

    rank: int = 0
    epoch: int = 0
    step: int = 0
    total_step: int = 0
    loss_avg_rank: float = 0.0
    loss_avg_epoch: float = 0.0
    ppl_avg_epoch: float = 0.0
    acc_avg_epoch: float = 0.0
    lr: float = 0.0
    loss_att: float = 0.0
    acc_att: float = 0.0
    loss_seaco: float = 0.0
    loss: float = 0.0
    total_time: float = 0.0
    gpu_memory_usage: float = 0.0
    gpu_memory_peak: float = 0.0


class FunASRLogParser:
    """Parse individual FunASR log lines into structured data.

    Supports two FunASR log formats:
    - Old: ``step: x/y``
    - New: ``data_slice: x/y, step_in_slice: x/y, step_in_epoch: x``
    """

    _LOG_PATTERN = re.compile(
        r"train, rank: (\d+), epoch: (\d+)/(\d+), "
        r"(?:step: (\d+)/\d+|data_slice: \d+/\d+, step_in_slice: (\d+)/\d+, step_in_epoch: (\d+)), "
        r"total step: (\d+), "
        r"\(loss_avg_rank: ([\d.]+)\), "
        r"\(loss_avg_(?:epoch|slice): ([\d.]+)\), "
        r"\(ppl_avg_(?:epoch|slice): ([\d.e+-]+)\), "
        r"\(acc_avg_(?:epoch|slice): ([\d.]+)\), "
        r"\(lr: ([\d.e+-]+)\)"
    )

    _DETAIL_PATTERN = re.compile(
        r"\[.*?['\"]loss_att['\"],\s*([\d.]+)\).*?"
        r"['\"]acc_att?['\"],\s*([\d.]+)\).*?"
        r"['\"]loss(?:_pre|_seaco)['\"],\s*([\d.]+)\).*?"
        r"['\"]loss['\"],\s*([\d.]+)\)"
    )

    _TIME_PATTERN = re.compile(r"['\"]total_time['\"]:\s*['\"]?([\d.]+)['\"]?")

    _GPU_PATTERN = re.compile(
        r"memory: usage: ([\d.]+) GB, peak: ([\d.]+) GB"
    )

    def parse_line(self, line: str) -> Optional[FunASRTrainStep]:
        """Parse a single FunASR log line.

        Returns ``None`` if the line is not a training step record.
        """
        match = self._LOG_PATTERN.search(line)
        if not match:
            return None

        step_num = match.group(4) or match.group(6) or match.group(5)

        step = FunASRTrainStep(
            rank=int(match.group(1)),
            epoch=int(match.group(2)),
            step=int(step_num),
            total_step=int(match.group(7)),
            loss_avg_rank=float(match.group(8)),
            loss_avg_epoch=float(match.group(9)),
            ppl_avg_epoch=float(match.group(10)),
            acc_avg_epoch=float(match.group(11)),
            lr=float(match.group(12)),
        )

        detail_match = self._DETAIL_PATTERN.search(line)
        if detail_match:
            step.loss_att = float(detail_match.group(1))
            step.acc_att = float(detail_match.group(2))
            step.loss_seaco = float(detail_match.group(3))
            step.loss = float(detail_match.group(4))

        if step.acc_avg_epoch == 0.0 and step.acc_att > 0.0:
            step.acc_avg_epoch = step.acc_att

        time_match = self._TIME_PATTERN.search(line)
        if time_match:
            step.total_time = float(time_match.group(1))

        gpu_match = self._GPU_PATTERN.search(line)
        if gpu_match:
            step.gpu_memory_usage = float(gpu_match.group(1))
            step.gpu_memory_peak = float(gpu_match.group(2))

        return step
