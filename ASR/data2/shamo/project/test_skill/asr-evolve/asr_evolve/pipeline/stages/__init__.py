"""流水线各阶段实现（8 阶段）.

每个阶段实现 Stage 协议（artifact_exists / plan / execute）。
阶段之间通过 Kaldi 文件（wav.scp / text / manifest.jsonl）衔接。
"""

from asr_evolve.pipeline.stages.prepare import PrepareStage
from asr_evolve.pipeline.stages.quality import QualityScanner
from asr_evolve.pipeline.stages.clean import Cleaner
from asr_evolve.pipeline.stages.split import DataSplitter
from asr_evolve.pipeline.stages.augment import Augmentor
from asr_evolve.pipeline.stages.merge import Merger
from asr_evolve.pipeline.stages.jsonl import ManifestGenerator
from asr_evolve.pipeline.stages.train import RecipeTrainer

ALL_STAGES = [
    PrepareStage, QualityScanner, Cleaner, DataSplitter,
    Augmentor, Merger, ManifestGenerator, RecipeTrainer,
]
