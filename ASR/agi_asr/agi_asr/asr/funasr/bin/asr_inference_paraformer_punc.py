#!/usr/bin/env python3
import logging
import sys
import time
import copy
import os
import codecs
import tempfile
import requests
from pathlib import Path
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import Dict
from typing import Any
from typing import List

import numpy as np
import torch
from typeguard import check_argument_types

from asr.funasr.modules.beam_search.beam_search import BeamSearchPara as BeamSearch
from asr.funasr.modules.beam_search.beam_search import Hypothesis
from asr.funasr.modules.scorers.ctc import CTCPrefixScorer
from asr.funasr.modules.scorers.length_bonus import LengthBonus
from asr.funasr.modules.subsampling import TooShortUttError
from asr.funasr.tasks.asr import ASRTaskParaformer as ASRTask
from asr.funasr.tasks.lm import LMTask
from asr.funasr.text.build_tokenizer import build_tokenizer
from asr.funasr.text.token_id_converter import TokenIDConverter
from asr.funasr.torch_utils.device_funcs import to_device
from asr.funasr.torch_utils.set_all_random_seed import set_all_random_seed
from asr.funasr.utils import asr_utils, wav_utils, postprocess_utils
from asr.funasr.models.frontend.wav_frontend import WavFrontend
from asr.funasr.models.e2e_asr_paraformer import BiCifParaformer, ContextualParaformer
from asr.funasr.utils.timestamp_tools import ts_prediction_lfr6_standard
from asr.funasr.bin.tp_inference import SpeechText2Timestamp

from asr.funasr.bin.punctuation_infer import Text2Punc

class Speech2Text:
    """Speech2Text class

    Examples:
            >>> import soundfile
            >>> speech2text = Speech2Text("asr_config.yml", "asr.pb")
            >>> audio, rate = soundfile.read("speech.wav")
            >>> speech2text(audio)
            [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
            self,
            asr_train_config: Union[Path, str] = None,
            asr_model_file: Union[Path, str] = None,
            cmvn_file: Union[Path, str] = None,
            lm_train_config: Union[Path, str] = None,
            lm_file: Union[Path, str] = None,
            token_type: str = None,
            bpemodel: str = None,
            device: str = "cpu",
            maxlenratio: float = 0.0,
            minlenratio: float = 0.0,
            dtype: str = "float32",
            beam_size: int = 20,
            ctc_weight: float = 0.5,
            lm_weight: float = 1.0,
            ngram_weight: float = 0.9,
            penalty: float = 0.0,
            nbest: int = 1,
            frontend_conf: dict = None,
            hotword_list_or_file: str = None,
            **kwargs,
    ):
        assert check_argument_types()

        # 1. Build ASR model
        scorers = {}
        asr_model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, cmvn_file, device
        )
        frontend = None
        if asr_train_args.frontend is not None and asr_train_args.frontend_conf is not None:
            frontend = WavFrontend(cmvn_file=cmvn_file, **asr_train_args.frontend_conf)

        asr_model.to(dtype=getattr(torch, dtype)).eval()

        if asr_model.ctc != None:
            ctc = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)
            scorers.update(
                ctc=ctc
            )
        token_list = asr_model.token_list
        scorers.update(
            length_bonus=LengthBonus(len(token_list)),
        )

        # 2. Build Language model
        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, device
            )
            scorers["lm"] = lm.lm

        # 3. Build ngram model
        # ngram is not supported now
        ngram = None
        scorers["ngram"] = ngram

        # 4. Build BeamSearch object
        # transducer is not supported now
        beam_search_transducer = None

        weights = dict(
            decoder=1.0 - ctc_weight,
            ctc=ctc_weight,
            lm=lm_weight,
            ngram=ngram_weight,
            length_bonus=penalty,
        )
        beam_search = BeamSearch(
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            sos=asr_model.sos,
            eos=asr_model.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None if ctc_weight == 1.0 else "full",
        )

        beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
        for scorer in scorers.values():
            if isinstance(scorer, torch.nn.Module):
                scorer.to(device=device, dtype=getattr(torch, dtype)).eval()

        logging.info(f"Decoding device={device}, dtype={dtype}")

        # 5. [Optional] Build Text converter: e.g. bpe-sym -> Text
        if token_type is None:
            token_type = asr_train_args.token_type
        if bpemodel is None:
            bpemodel = asr_train_args.bpemodel

        if token_type is None:
            tokenizer = None
        elif token_type == "bpe":
            if bpemodel is not None:
                tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)
        converter = TokenIDConverter(token_list=token_list)
        logging.info(f"Text tokenizer: {tokenizer}")

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.converter = converter
        self.tokenizer = tokenizer

        # 6. [Optional] Build hotword list from str, local file or url
        self.hotword_list = None
        self.hotword_list = self.generate_hotwords_list(hotword_list_or_file)

        is_use_lm = lm_weight != 0.0 and lm_file is not None
        if (ctc_weight == 0.0 or asr_model.ctc == None) and not is_use_lm:
            beam_search = None
        self.beam_search = beam_search
        #logging.info(f"Beam_search: {self.beam_search}")
        self.beam_search_transducer = beam_search_transducer
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest
        self.frontend = frontend
        self.encoder_downsampling_factor = 1
        if asr_train_args.encoder == "data2vec_encoder" or asr_train_args.encoder_conf["input_layer"] == "conv2d":
            self.encoder_downsampling_factor = 4

    @torch.no_grad()
    def __call__(
            self, speech: Union[torch.Tensor, np.ndarray], speech_lengths: Union[torch.Tensor, np.ndarray] = None,
            begin_time: int = 0, end_time: int = None,
    ):
        """Inference

        Args:
                speech: Input speech data
        Returns:
                text, token, token_int, hyp

        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        if self.frontend is not None:
            feats, feats_len = self.frontend.forward(speech, speech_lengths)
            feats = to_device(feats, device=self.device)
            feats_len = feats_len.int()
            self.asr_model.frontend = None
        else:
            feats = speech
            feats_len = speech_lengths

        #lfr_factor = max(1, (feats.size()[-1] // 80) - 1)
        batch = {"speech": feats, "speech_lengths": feats_len}

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        enc, enc_len = self.asr_model.encode(**batch)
        if isinstance(enc, tuple):
            enc = enc[0]

        predictor_outs = self.asr_model.calc_predictor(enc, enc_len)
        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = predictor_outs[0], predictor_outs[1], \
                                                                        predictor_outs[2], predictor_outs[3]
        pre_token_length = pre_token_length.round().long()
        if torch.max(pre_token_length) < 1:
            return []
        if not isinstance(self.asr_model, ContextualParaformer):
            if self.hotword_list:
                logging.warning("Hotword is given but asr model is not a ContextualParaformer.")
            decoder_outs = self.asr_model.cal_decoder_with_predictor(enc, enc_len, pre_acoustic_embeds, pre_token_length)
            decoder_out, ys_pad_lens = decoder_outs[0], decoder_outs[1]
        else:
            decoder_outs = self.asr_model.cal_decoder_with_predictor(enc, enc_len, pre_acoustic_embeds, pre_token_length, hw_list=self.hotword_list)
            decoder_out, ys_pad_lens = decoder_outs[0], decoder_outs[1]

        if isinstance(self.asr_model, BiCifParaformer):
            _, _, us_alphas, us_peaks = self.asr_model.calc_predictor_timestamp(enc, enc_len,
                                                                                   pre_token_length) 

        results = []
        b, n, d = decoder_out.size()
        for i in range(b):
            x = enc[i, :enc_len[i], :]
            am_scores = decoder_out[i, :pre_token_length[i], :]
            if self.beam_search is not None:
                nbest_hyps = self.beam_search(
                    x=x, am_scores=am_scores, maxlenratio=self.maxlenratio, minlenratio=self.minlenratio
                )

                nbest_hyps = nbest_hyps[: self.nbest]
            else:
                if pre_token_length[i] == 0:
                    yseq = torch.tensor(
                        [self.asr_model.sos] + [self.asr_model.eos], device=pre_acoustic_embeds.device
                    )
                    score = torch.tensor(0.0, device=pre_acoustic_embeds.device)
                else:
                    yseq = am_scores.argmax(dim=-1)
                    score = am_scores.max(dim=-1)[0]
                    score = torch.sum(score, dim=-1)
                    # pad with mask tokens to ensure compatibility with sos/eos tokens
                    yseq = torch.tensor(
                        [self.asr_model.sos] + yseq.tolist() + [self.asr_model.eos], device=yseq.device
                    )
                nbest_hyps = [Hypothesis(yseq=yseq, score=score)]

            for hyp in nbest_hyps:
                assert isinstance(hyp, (Hypothesis)), type(hyp)

                # remove sos/eos and get results
                last_pos = -1
                if isinstance(hyp.yseq, list):
                    token_int = hyp.yseq[1:last_pos]
                else:
                    token_int = hyp.yseq[1:last_pos].tolist()

                # remove blank symbol id, which is assumed to be 0
                token_int = list(filter(lambda x: x != 0 and x != 2, token_int))

                # Change integer-ids to tokens
                token = self.converter.ids2tokens(token_int)

                if self.tokenizer is not None:
                    text = self.tokenizer.tokens2text(token)
                else:
                    text = None

                if isinstance(self.asr_model, BiCifParaformer):
                    _, timestamp = ts_prediction_lfr6_standard(us_alphas[i], 
                                                            us_peaks[i], 
                                                            copy.copy(token), 
                                                            vad_offset=begin_time)
                    results.append((text, token, token_int, hyp, timestamp))
                else:
                    results.append((text, token, token_int, hyp))

        # assert check_return_type(results)
        return results

    def generate_hotwords_list(self, hotword_list_or_file):
        # for None
        if hotword_list_or_file is None:
            hotword_list = None
        # for local txt inputs
        elif os.path.exists(hotword_list_or_file) and hotword_list_or_file.endswith('.txt'):
            logging.info("Attempting to parse hotwords from local txt...")
            hotword_list = []
            hotword_str_list = []
            with codecs.open(hotword_list_or_file, 'r') as fin:
                for line in fin.readlines():
                    hw = line.strip()
                    hotword_str_list.append(hw)
                    hotword_list.append(self.converter.tokens2ids([i for i in hw]))
                hotword_list.append([self.asr_model.sos])
                hotword_str_list.append('<s>')
            logging.info("Initialized hotword list from file: {}, hotword list: {}."
                         .format(hotword_list_or_file, hotword_str_list))
        # for url, download and generate txt
        elif hotword_list_or_file.startswith('http'):
            logging.info("Attempting to parse hotwords from url...")
            work_dir = tempfile.TemporaryDirectory().name
            if not os.path.exists(work_dir):
                os.makedirs(work_dir)
            text_file_path = os.path.join(work_dir, os.path.basename(hotword_list_or_file))
            local_file = requests.get(hotword_list_or_file)
            open(text_file_path, "wb").write(local_file.content)
            hotword_list_or_file = text_file_path
            hotword_list = []
            hotword_str_list = []
            with codecs.open(hotword_list_or_file, 'r') as fin:
                for line in fin.readlines():
                    hw = line.strip()
                    hotword_str_list.append(hw)
                    hotword_list.append(self.converter.tokens2ids([i for i in hw]))
                hotword_list.append([self.asr_model.sos])
                hotword_str_list.append('<s>')
            logging.info("Initialized hotword list from file: {}, hotword list: {}."
                         .format(hotword_list_or_file, hotword_str_list))
        # for text str input
        elif not hotword_list_or_file.endswith('.txt'):
            logging.info("Attempting to parse hotwords as str...")
            hotword_list = []
            hotword_str_list = []
            for hw in hotword_list_or_file.strip().split():
                hotword_str_list.append(hw)
                hotword_list.append(self.converter.tokens2ids([i for i in hw]))
            hotword_list.append([self.asr_model.sos])
            hotword_str_list.append('<s>')
            logging.info("Hotword list: {}.".format(hotword_str_list))
        else:
            hotword_list = None
        return hotword_list


def inference(
        maxlenratio: float,
        minlenratio: float,
        batch_size: int,
        beam_size: int,
        ngpu: int,
        ctc_weight: float,
        lm_weight: float,
        penalty: float,
        log_level: Union[int, str],
        # data_path_and_name_and_type,
        asr_train_config: Optional[str],
        asr_model_file: Optional[str],
        cmvn_file: Optional[str] = None,
        lm_train_config: Optional[str] = None,
        lm_file: Optional[str] = None,
        token_type: Optional[str] = None,
        key_file: Optional[str] = None,
        word_lm_train_config: Optional[str] = None,
        bpemodel: Optional[str] = None,
        allow_variable_data_keys: bool = False,
        dtype: str = "float32",
        seed: int = 0,
        ngram_weight: float = 0.9,
        nbest: int = 1,
        num_workers: int = 1,
        device = "cpu",
        timestamp_infer_config: Union[Path, str] = None,
        timestamp_model_file: Union[Path, str] = None,
        punc_infer_config: Optional[str] = None,
        punc_model_file: Optional[str] = None,
        param_dict: dict = None,
        **kwargs,
):
    assert check_argument_types()
    ncpu = kwargs.get("ncpu", 1)
    torch.set_num_threads(ncpu)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
      
    if param_dict is not None:
        hotword_list_or_file = param_dict.get('hotword')
    else:
        hotword_list_or_file = None
    
    if ngpu >= 0 and torch.cuda.is_available():
        device = device
    else:
        device = "cpu"
        batch_size = 1
    logging.warning("device is given as: {}".format(device))

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2text
    speech2text_kwargs = dict(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        cmvn_file=cmvn_file,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        lm_weight=lm_weight,
        ngram_weight=ngram_weight,
        penalty=penalty,
        nbest=nbest,
        hotword_list_or_file=hotword_list_or_file,
    )

    speech2text = Speech2Text(**speech2text_kwargs)

    if timestamp_model_file is not None:
        speechtext2timestamp = SpeechText2Timestamp(
            timestamp_cmvn_file=cmvn_file,
            timestamp_model_file=timestamp_model_file,
            timestamp_infer_config=timestamp_infer_config,
        )
    else:
        speechtext2timestamp = None

    text2punc = None
    if punc_model_file is not None:
        text2punc = Text2Punc(punc_infer_config, punc_model_file, device=device, dtype=dtype)

    def _forward(
            data_path_and_name_and_type,
            raw_inputs: Union[np.ndarray, torch.Tensor] = None,
            fs: dict = None,
            param_dict: dict = None,
            **kwargs,
    ):
        # hotword also support online correction
        hotword_list_or_file = None
        if param_dict is not None:
            hotword_list_or_file = param_dict.get('hotword')
        if hotword_list_or_file is not None:
            speech2text.hotword_list = speech2text.generate_hotwords_list(hotword_list_or_file)

        # 3. Build data-iterator
        if data_path_and_name_and_type is None and raw_inputs is not None:
            if isinstance(raw_inputs, torch.Tensor):
                raw_inputs = raw_inputs.numpy()
            data_path_and_name_and_type = [raw_inputs, "speech", "waveform"]
        loader = ASRTask.build_streaming_iterator(
            data_path_and_name_and_type,
            dtype=dtype,
            fs=fs,
            batch_size=batch_size,
            key_file=key_file,
            num_workers=num_workers,
            preprocess_fn=ASRTask.build_preprocess_fn(speech2text.asr_train_args, False),
            collate_fn=ASRTask.build_collate_fn(speech2text.asr_train_args, False),
            allow_variable_data_keys=allow_variable_data_keys,
            inference=True,
        )

        if param_dict is not None:
            use_timestamp = param_dict.get('use_timestamp', True)
        else:
            use_timestamp = False

        # 7 .Start for-loop
        asr_result_list = []

        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            # batch = {k: v for k, v in batch.items() if not k.endswith("_lengths")}

            # logging.debug("decoding, utt_id: {}".format(keys))
            # N-best list of (text, token, token_int, hyp_object)

            results = speech2text(**batch)
            if len(results) < 1:
                hyp = Hypothesis(score=0.0, scores={}, states={}, yseq=[])
                results = [[" ", ["sil"], [2], hyp]] * nbest

            for batch_id in range(_bs):
                result = [results[batch_id]]
                key = keys[batch_id]
                for n, result in zip(range(1, nbest + 1), result):
                    text, token, token_int, hyp = result[0], result[1], result[2], result[3]
                    timestamp = None if len(result) < 5 else result[4]

                    # conduct timestamp prediction here
                    # timestamp inference requires token length
                    # thus following inference cannot be conducted in batch
                    if timestamp is None and speechtext2timestamp:
                        ts_batch = {}
                        ts_batch['speech'] = batch['speech'][batch_id].unsqueeze(0)
                        ts_batch['speech_lengths'] = torch.tensor([batch['speech_lengths'][batch_id]])
                        ts_batch['text_lengths'] = torch.tensor([len(token)])
                        us_alphas, us_peaks = speechtext2timestamp(**ts_batch)
                        ts_str, timestamp = ts_prediction_lfr6_standard(us_alphas[0], us_peaks[0], token, force_time_shift=-3.0)

                    if text is not None:
                        if use_timestamp and timestamp is not None:
                            postprocessed_result = postprocess_utils.sentence_postprocess(token, timestamp)
                        else:
                            postprocessed_result = postprocess_utils.sentence_postprocess(token)
                        timestamp_postprocessed = ""
                        if len(postprocessed_result) == 3:
                            text_postprocessed, timestamp_postprocessed, word_lists = postprocessed_result[0], \
                                                                                       postprocessed_result[1], \
                                                                                       postprocessed_result[2]
                        else:
                            text_postprocessed, word_lists = postprocessed_result[0], postprocessed_result[1]

                        text_postprocessed_punc = text_postprocessed

                        if len(word_lists) > 0 and text2punc is not None:
                           text_postprocessed_punc, _ = text2punc(word_lists, 20)
             
                        item = {'key': key, 'value': text_postprocessed_punc}

                        if timestamp_postprocessed != "":
                            item['timestamp'] = timestamp_postprocessed
                        asr_result_list.append(item)

        return asr_result_list

    return _forward


