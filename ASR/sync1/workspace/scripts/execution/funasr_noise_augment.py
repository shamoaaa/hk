"""Real audio noise augmentation module for the ASR Training Optimizer system.

Adapted from funasr_finetune_tool/src/noise_augment.py. Performs REAL audio
augmentation using soundfile, numpy, and scipy — not numpy simulation.

Supports babble (background voice), music, noise, and reverb augmentation
using the MUSAN noise dataset.

Output: workspace/data/augmented/
"""

import json
import logging
import os
import random
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Augmentation type → scene_type label mapping
_AUG_TYPE_TO_SCENE: dict[str, str] = {
    "babble": "background_voice",
    "music": "background_music",
    "noise": "background_noise",
    "reverb": "reverb",
    "speed": "speed_perturb",
    "volume": "volume_perturb",
}


class NoiseAugmentor:
    """Real audio noise/voice augmentor using MUSAN dataset.

    Parameters
    ----------
    config : dict
        Configuration dict with ``augmentation`` section.
    base_dir : str
        Workspace root directory.
    """

    def __init__(self, config: dict, base_dir: str = ".") -> None:
        self.config = config
        self.base_dir = Path(base_dir)
        self.aug_cfg = config.get("augmentation", {})
        self.noise_dir = Path(
            self.aug_cfg.get("noise_dir", str(self.base_dir / "data" / "musan"))
        )
        self.output_dir = Path(
            self.aug_cfg.get("augmented_dir", str(self.base_dir / "data" / "augmented"))
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.noise_files: dict[str, list[Path]] = {
            "speech": [],
            "music": [],
            "noise": [],
        }
        self._seed = 42

    def prepare_noise_data(self) -> None:
        """Prepare noise data: download MUSAN if needed, then load file lists."""
        if self.aug_cfg.get("download_musan", False) and not self.noise_dir.exists():
            self._download_musan()
        self._load_noise_files()

    def _download_musan(self) -> None:
        """Download MUSAN noise dataset."""
        self.noise_dir.mkdir(parents=True, exist_ok=True)
        url = "https://www.openslr.org/resources/17/musan.tar.gz"
        tgz_file = self.noise_dir.parent / "musan.tar.gz"

        logger.info("Downloading MUSAN noise dataset to %s...", self.noise_dir)

        if not tgz_file.exists():
            subprocess.run(["wget", "-c", url, "-O", str(tgz_file)], check=True)

        logger.info("Extracting MUSAN dataset...")
        subprocess.run(
            ["tar", "-xzf", str(tgz_file), "-C", str(self.noise_dir.parent)],
            check=True,
        )
        logger.info("MUSAN dataset download complete")

    def _load_noise_files(self) -> None:
        """Load noise file lists from the MUSAN directory."""
        if not self.noise_dir.exists():
            logger.warning("Noise directory not found: %s, skipping augmentation", self.noise_dir)
            return

        for category in ["speech", "music", "noise"]:
            cat_dir = self.noise_dir / category
            if cat_dir.exists():
                files = list(cat_dir.rglob("*.wav"))
                self.noise_files[category] = files
                logger.info("Loaded %d %s noise files", len(files), category)

    def augment_dataset(
        self,
        jsonl_path: str,
        output_jsonl: str,
        strategies: Optional[list[dict]] = None,
    ) -> dict:
        """Augment an entire dataset with real audio noise.

        Parameters
        ----------
        jsonl_path : str
            Path to the original FunASR jsonl data.
        output_jsonl : str
            Path for the augmented jsonl output.
        strategies : list[dict] or None
            Augmentation strategies. If None, uses config defaults.

        Returns
        -------
        dict
            Summary with original_count, augmented_count, output_path.
        """
        import soundfile as sf

        if not self.aug_cfg.get("enabled", True):
            logger.info("Augmentation disabled, skipping")
            return {
                "status": "skipped",
                "original_count": 0,
                "augmented_count": 0,
            }

        # Reuse existing augmented data if available
        output_path = Path(output_jsonl)
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info("Augmented data already exists, reusing: %s", output_jsonl)
            count = sum(1 for _ in open(output_jsonl))
            return {
                "status": "reused",
                "output_path": str(output_jsonl),
                "augmented_count": count,
            }

        random.seed(self._seed)
        np.random.seed(self._seed)

        # Read original data
        entries = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        logger.info("Starting augmentation, original data: %d entries", len(entries))

        if strategies is None:
            strategies = self.aug_cfg.get("strategies", [])

        augmented_entries = []

        for i, entry in enumerate(entries):
            # Keep original data with scene_type = "clean"
            orig_entry = dict(entry)
            orig_entry.setdefault("scene_type", "clean")
            augmented_entries.append(orig_entry)

            wav_path = entry["source"]
            utt_id = entry["key"]
            text = entry["target"]

            for strategy in strategies:
                if random.random() > strategy.get("probability", 0.3):
                    continue

                aug_type = strategy["type"]
                try:
                    aug_wav_path = self._apply_augmentation(
                        wav_path, utt_id, aug_type, strategy
                    )
                    if aug_wav_path:
                        aug_entry = {
                            "key": f"{utt_id}_aug_{aug_type}",
                            "source": str(aug_wav_path),
                            "source_len": -1,
                            "target": text,
                            "target_len": len(text),
                            "scene_type": _AUG_TYPE_TO_SCENE.get(aug_type, aug_type),
                        }
                        augmented_entries.append(aug_entry)
                except Exception as e:
                    logger.debug("Augmentation failed %s (%s): %s", utt_id, aug_type, e)

            if (i + 1) % 1000 == 0:
                logger.info("Augmentation progress: %d/%d", i + 1, len(entries))

        # Write augmented jsonl
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for entry in augmented_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        original_count = len(entries)
        augmented_count = len(augmented_entries) - original_count

        logger.info(
            "Augmentation complete: %d -> %d entries (+%d augmented), saved to %s",
            original_count,
            len(augmented_entries),
            augmented_count,
            output_jsonl,
        )

        return {
            "status": "completed",
            "output_path": str(output_jsonl),
            "original_count": original_count,
            "augmented_count": augmented_count,
            "total_count": len(augmented_entries),
        }

    def augment_single_strategy(
        self,
        jsonl_path: str,
        output_jsonl: str,
        strategy_type: str,
        snr_range: Optional[list[float]] = None,
        probability: float = 1.0,
    ) -> dict:
        """Augment dataset with a single strategy type.

        Parameters
        ----------
        jsonl_path : str
            Input jsonl path.
        output_jsonl : str
            Output jsonl path.
        strategy_type : str
            One of: ``"babble"``, ``"music"``, ``"noise"``, ``"reverb"``.
        snr_range : list[float] or None
            SNR range in dB (default depends on type).
        probability : float
            Probability of applying augmentation per sample.

        Returns
        -------
        dict
            Augmentation summary.
        """
        defaults = {
            "babble": {"snr_range": [5, 20], "num_speakers": [1, 3]},
            "music": {"snr_range": [5, 25]},
            "noise": {"snr_range": [5, 25]},
            "reverb": {"rt60_range": [0.1, 1.0]},
            "speed": {"speed_factors": [0.9, 1.0, 1.1]},
            "volume": {"gain_db_range": [-6.0, 6.0]},
        }

        strategy = defaults.get(strategy_type, {})
        strategy["type"] = strategy_type
        strategy["probability"] = probability
        if snr_range is not None:
            strategy["snr_range"] = snr_range

        return self.augment_dataset(jsonl_path, output_jsonl, strategies=[strategy])

    # ------------------------------------------------------------------
    # Core augmentation methods
    # ------------------------------------------------------------------

    def _apply_augmentation(
        self, wav_path: str, utt_id: str, aug_type: str, strategy: dict
    ) -> Optional[str]:
        """Apply a single augmentation to one audio file."""
        import soundfile as sf

        speech, sr = sf.read(wav_path)
        if len(speech.shape) > 1:
            speech = speech[:, 0]

        if aug_type == "babble":
            augmented = self._add_babble(speech, sr, strategy)
        elif aug_type == "music":
            augmented = self._add_background(speech, sr, "music", strategy)
        elif aug_type == "noise":
            augmented = self._add_background(speech, sr, "noise", strategy)
        elif aug_type == "reverb":
            augmented = self._add_reverb(speech, sr, strategy)
        elif aug_type == "speed":
            augmented = self._apply_speed(speech, sr, strategy)
        elif aug_type == "volume":
            augmented = self._apply_volume(speech, sr, strategy)
        else:
            return None

        if augmented is None:
            return None

        out_path = self.output_dir / f"{utt_id}_aug_{aug_type}.wav"
        sf.write(str(out_path), augmented, sr)
        return str(out_path)

    def _add_babble(
        self, speech: np.ndarray, sr: int, strategy: dict
    ) -> Optional[np.ndarray]:
        """Add babble noise (multiple background speakers)."""
        import soundfile as sf

        speech_files = self.noise_files.get("speech", [])
        if not speech_files:
            return None

        num_speakers_range = strategy.get("num_speakers", [1, 3])
        num_speakers = random.randint(*num_speakers_range)
        snr_range = strategy.get("snr_range", [5, 20])
        snr_db = random.uniform(*snr_range)

        babble = np.zeros_like(speech, dtype=np.float64)
        for _ in range(num_speakers):
            noise_file = random.choice(speech_files)
            noise, noise_sr = sf.read(str(noise_file))
            if len(noise.shape) > 1:
                noise = noise[:, 0]

            if noise_sr != sr:
                noise = self._resample(noise, noise_sr, sr)

            noise = self._match_length(noise, len(speech))
            babble += noise

        return self._mix_at_snr(speech, babble, snr_db)

    def _add_background(
        self, speech: np.ndarray, sr: int, category: str, strategy: dict
    ) -> Optional[np.ndarray]:
        """Add background noise or music."""
        import soundfile as sf

        noise_files = self.noise_files.get(category, [])
        if not noise_files:
            return None

        snr_range = strategy.get("snr_range", [5, 25])
        snr_db = random.uniform(*snr_range)

        noise_file = random.choice(noise_files)
        noise, noise_sr = sf.read(str(noise_file))
        if len(noise.shape) > 1:
            noise = noise[:, 0]

        if noise_sr != sr:
            noise = self._resample(noise, noise_sr, sr)

        noise = self._match_length(noise, len(speech))
        return self._mix_at_snr(speech, noise, snr_db)

    def _add_reverb(
        self, speech: np.ndarray, sr: int, strategy: dict
    ) -> Optional[np.ndarray]:
        """Add reverberation effect using scipy convolution."""
        try:
            from scipy.signal import fftconvolve
        except ImportError:
            logger.warning("scipy not available, skipping reverb augmentation")
            return None

        rt60_range = strategy.get("rt60_range", [0.1, 1.0])
        rt60 = random.uniform(*rt60_range)

        rir = self._generate_simple_rir(sr, rt60)
        reverbed = fftconvolve(speech, rir, mode="full")[: len(speech)]

        max_val = np.max(np.abs(reverbed))
        if max_val > 0:
            reverbed = reverbed * (np.max(np.abs(speech)) / max_val)

        return reverbed.astype(np.float32)

    def _apply_speed(
        self, speech: np.ndarray, sr: int, strategy: dict
    ) -> Optional[np.ndarray]:
        """Apply speed perturbation by resampling.

        Speed up (factor > 1.0) = shorter, faster; slow down (< 1.0) = longer, slower.
        Uses scipy resample_poly for high-quality anti-aliased resampling.
        Speech transcript stays the same — listeners perceive the same words
        at a different rate.
        """
        try:
            from scipy.signal import resample_poly
        except ImportError:
            logger.warning("scipy not available, skipping speed augmentation")
            return None

        factors = strategy.get("speed_factors", [0.9, 1.0, 1.1])
        factor = random.choice(factors)
        if abs(factor - 1.0) < 1e-6:
            return speech.astype(np.float32)

        # factor > 1.0 (faster) → fewer samples.  resample_poly(speech, up, down)
        # changes output length to len * up / down. To get factor-times-faster,
        # we want up/down ≈ 1/factor.  Use integer ratio via Fraction for clean
        # arithmetic.
        from fractions import Fraction
        ratio = Fraction(1.0 / factor).limit_denominator(100)
        up, down = ratio.numerator, ratio.denominator
        if up == 0 or down == 0:
            return speech.astype(np.float32)

        out = resample_poly(speech.astype(np.float64), up, down)
        return out.astype(np.float32)

    def _apply_volume(
        self, speech: np.ndarray, sr: int, strategy: dict
    ) -> Optional[np.ndarray]:
        """Apply volume gain perturbation in dB.

        Positive dB = louder; negative = quieter. Clip to [-1, 1] range to
        avoid int16 overflow when saving back.
        """
        gain_db_range = strategy.get("gain_db_range", [-6.0, 6.0])
        gain_db = random.uniform(*gain_db_range)
        gain_linear = 10 ** (gain_db / 20.0)
        out = speech.astype(np.float32) * gain_linear
        # soft-clip to prevent saturation on loud gain
        max_val = float(np.max(np.abs(out)))
        if max_val > 0.99:
            out = out * (0.99 / max_val)
        return out

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_simple_rir(sr: int, rt60: float) -> np.ndarray:
        """Generate a simple exponential-decay room impulse response."""
        length = int(sr * rt60)
        if length < 1:
            length = 1
        t = np.arange(length) / sr
        rir = np.random.randn(length) * np.exp(-6.9 * t / rt60)
        rir[0] = 1.0
        rir = rir / np.sqrt(np.sum(rir ** 2))
        return rir

    @staticmethod
    def _mix_at_snr(
        speech: np.ndarray, noise: np.ndarray, snr_db: float
    ) -> np.ndarray:
        """Mix speech and noise at a specified SNR level."""
        speech_power = np.mean(speech ** 2)
        noise_power = np.mean(noise ** 2)

        if noise_power == 0:
            return speech.astype(np.float32)

        snr_linear = 10 ** (snr_db / 10)
        scale = np.sqrt(speech_power / (snr_linear * noise_power))
        mixed = speech + scale * noise

        max_val = np.max(np.abs(mixed))
        if max_val > 0.99:
            mixed = mixed * 0.99 / max_val

        return mixed.astype(np.float32)

    @staticmethod
    def _match_length(signal: np.ndarray, target_length: int) -> np.ndarray:
        """Adjust signal length to match target (trim or loop-pad)."""
        if len(signal) >= target_length:
            start = random.randint(0, len(signal) - target_length)
            return signal[start : start + target_length]
        else:
            repeats = target_length // len(signal) + 1
            signal = np.tile(signal, repeats)
            return signal[:target_length]

    @staticmethod
    def _resample(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio signal to target sample rate."""
        try:
            import librosa

            return librosa.resample(signal, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            ratio = target_sr / orig_sr
            new_length = int(len(signal) * ratio)
            indices = np.linspace(0, len(signal) - 1, new_length)
            return np.interp(indices, np.arange(len(signal)), signal)
