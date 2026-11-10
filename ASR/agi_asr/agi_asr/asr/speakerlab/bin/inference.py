

import torch
from asr.speakerlab.process.processor import FBank
from asr.speakerlab.utils.builder import dynamic_import
from asr.funasr.datasets.iterable_dataset import load_bytes

from ms_common.configs import logger


ERes2NetV2_w24s4ep4_COMMON = {
    'obj': 'asr.speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
        'baseWidth': 24,
        'scale': 4,
        'expansion': 4,
    },
}

ERes2NetV2_COMMON = {
    'obj': 'asr.speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
        'baseWidth': 26,
        'scale': 2,
        'expansion': 2,
    },
}

class speaker_verification():
    def __init__(self,
                 device,
                 model_path):
            self.device = torch.device(device)
            pretrained_state = torch.load(model_path, map_location='cpu')
            # load model
            model = ERes2NetV2_COMMON
            self.embedding_model = dynamic_import(model['obj'])(**model['args'])
            self.embedding_model.load_state_dict(pretrained_state)
            self.embedding_model.to(self.device)
            self.embedding_model.eval()

            self.fs = 16000
            self.feature_extractor = FBank(80, sample_rate=self.fs, mean_nor=True)

            self.similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    def compute_embedding(self, byte_data):
        wav = load_bytes(byte_data)
        feat = self.feature_extractor(torch.tensor(wav)).unsqueeze(0).to(self.device)
        # compute embedding
        with torch.no_grad():
            embedding = self.embedding_model(feat).detach().squeeze(0).cpu().numpy()
        return wav, embedding
    
    def inference(
        self,
        wakeup_byte_data,
        asr_byte_data,
    ):
        wav_wakeup, embedding_wakeup = self.compute_embedding(wakeup_byte_data)
        wav_cmd, embedding_cmd = self.compute_embedding(asr_byte_data)

        # compute similarity score
        scores = self.similarity(torch.from_numpy(embedding_wakeup).unsqueeze(0), torch.from_numpy(embedding_cmd).unsqueeze(0)).item()
        logger.debug(f"Speaker Verification output scores: {scores}")
        return wav_wakeup, wav_cmd, scores   
