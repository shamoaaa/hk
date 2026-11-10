
import random
import pickle
import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi


class WavReader(object):
    def __init__(self,
        sample_rate = 16000,
        duration: float = 3.0,
        speed_pertub: bool = False,
        lm: bool = True,
    ):
        self.duration = duration
        self.sample_rate = sample_rate
        self.speed_pertub = speed_pertub
        self.lm = lm

    def __call__(self, wav_path):
        wav, sr = torchaudio.load(wav_path)
        assert sr == self.sample_rate
        wav = wav[0]

        if self.speed_pertub and self.lm:
            speeds = [1.0, 0.9, 1.1]
            speed_idx = random.randint(0, 2)
            if speed_idx > 0:
                wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                    wav.unsqueeze(0), self.sample_rate, [['speed', str(speeds[speed_idx])], ['rate', str(self.sample_rate)]])
        else:
            speed_idx = 0

        wav = wav.squeeze(0)
        data_len = wav.shape[0]

        chunk_len = int(self.duration * sr)
        if data_len >= chunk_len:
            start = random.randint(0, data_len - chunk_len)
            end = start + chunk_len
            wav = wav[start:end]
        else:
            wav = F.pad(wav, (0, chunk_len - data_len))

        return wav, speed_idx


class FBank(object):
    def __init__(self,
        n_mels,
        sample_rate,
        mean_nor: bool = False,
    ):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mean_nor = mean_nor

    def __call__(self, wav, dither=0):
        sr = 16000
        assert sr==self.sample_rate
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        # select single channel
        if wav.shape[0] > 1:
            wav = wav[0, :]
            wav = wav.unsqueeze(0)
        assert len(wav.shape) == 2 and wav.shape[0]==1
        feat = Kaldi.fbank(wav, num_mel_bins=self.n_mels,
            sample_frequency=sr, dither=dither)
        # feat: [T, N]
        if self.mean_nor:
            feat = feat - feat.mean(0, keepdim=True)
        return feat
