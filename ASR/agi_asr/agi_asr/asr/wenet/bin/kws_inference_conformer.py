
import os
import torch
import torchaudio.compliance.kaldi as kaldi
import yaml
import numpy as np
from ms_common.configs import logger

from asr.wenet.utils.file_utils import read_symbol_table
from asr.wenet.transformer.asr_model import init_asr_model
from asr.wenet.utils.checkpoint import load_checkpoint
from asr.wenet.transformer.search import ctc_prefix_beam_search


def load_bytes(input):
    middle_data = np.frombuffer(input, dtype=np.int16)
    middle_data = np.asarray(middle_data)
    dtype = np.dtype('float32')
    i = np.iinfo(middle_data.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    array = np.frombuffer((middle_data.astype(dtype) - offset) / abs_max, dtype=np.float32)
    return array

class inference:

    def __init__(self,
                 model_dir: str,
                 device: str = 'cuda:0',
                 beam: int = 1,
                 resample_rate: int = 16000):

        self.device = torch.device(device)
        self.audio_cache = bytes()
        # Read config and override some config
        config_file = os.path.join(model_dir, 'config.yaml')
        with open(config_file, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
        token_file = os.path.join(model_dir, 'units.txt')
        configs['tokenizer_conf']['symbol_table_path'] = token_file
        cmvn_file = os.path.join(model_dir, 'global_cmvn')
        if os.path.exists(cmvn_file):
            configs['cmvn_conf']['cmvn_file'] = cmvn_file
            configs['cmvn_file'] = cmvn_file
        # Read model
        pt_file = os.path.join(model_dir, 'model.pt')
        self.model = init_asr_model(configs)
        load_checkpoint(self.model, pt_file)
        self.model.eval()
        self.model.to(self.device)

        self.resample_rate = resample_rate
        self.symbol_table = read_symbol_table(token_file)
        self.char_dict = {v: k for k, v in self.symbol_table.items()}
        self.beam = beam
        self.context_graph = None

    def compute_feats(self, audio_bytes: bytes) -> torch.Tensor:
        waveform = load_bytes(audio_bytes)
        waveform = torch.tensor(waveform, dtype=torch.float).to(self.device)
        waveform = waveform * (1 << 15)
        feats = kaldi.fbank(waveform.unsqueeze(0),
                            num_mel_bins=80,
                            frame_length=25,
                            frame_shift=10,
                            energy_floor=0.0,
                            sample_frequency=self.resample_rate)
        feats = feats.unsqueeze(0)
        return feats

    def post_processing(self, tokens: list):
        text = ''.join([self.char_dict[x] for x in tokens])
        text = text.replace("▁", " ").replace("<unk>", "")
        return text.lower()
    
    @torch.no_grad()
    def __call__(self,
                  audio_bytes) -> dict:

        feats = self.compute_feats(audio_bytes)
        feat_lengths = torch.tensor([feats.size(1)],
                                    dtype=torch.long,
                                    device=feats.device)
        encoder_out, _ = self.model._forward_encoder(feats, feat_lengths, -1, -1, False)
        encoder_lens = torch.tensor([encoder_out.size(1)],
                                    dtype=torch.long,
                                    device=encoder_out.device)
        ctc_probs = self.model.ctc_activation(encoder_out)

        ctc_prefix_results = ctc_prefix_beam_search(
            ctc_probs,
            encoder_lens,
            self.beam,
            context_graph=self.context_graph)

        tokens = ctc_prefix_results[0].tokens
        results = self.post_processing(tokens)
        logger.info(f"kws result output {results}")

        if all(sub in results for sub in ["好","萤","石"]) or all(sub in results for sub in ["你","好","萤"]) or all(sub in results for sub in ["你","萤","石"]):
            # logger.info(f"kws result output {results}")
            return True  
        else:
            # logger.info(f"kws result output {results}")
            return False
