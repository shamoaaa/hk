# v 1.3.0
# 2024/01/30
# seaco-paraformer

import os
import re
import logging
import time
import copy
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

from asr.funasr.modules.beam_search.beam_search import BeamSearchPara as BeamSearch
from asr.funasr.modules.beam_search.beam_search import Hypothesis
from asr.funasr.modules.scorers.ctc import CTCPrefixScorer
from asr.funasr.modules.scorers.length_bonus import LengthBonus
from asr.funasr.tasks.asr import ASRTaskParaformer as ASRTask
from asr.funasr.tasks.lm import LMTask
from asr.funasr.text.build_tokenizer import build_tokenizer
from asr.funasr.text.token_id_converter import TokenIDConverter
from asr.funasr.torch_utils.device_funcs import to_device
from asr.funasr.torch_utils.set_all_random_seed import set_all_random_seed
from asr.funasr.utils import postprocess_utils
from asr.funasr.models.frontend.wav_frontend import WavFrontend
from asr.funasr.models.e2e_asr_paraformer import BiCifParaformer, ContextualParaformer, SeacoParaformer
from asr.funasr.models.e2e_asr_contextual_paraformer import NeatContextualParaformer
from asr.funasr.utils.timestamp_tools import ts_prediction_lfr6_standard
from asr.funasr.bin.tp_inference import SpeechText2Timestamp

from asr.funasr.bin.punctuation_infer import Text2Punc
from asr.textProcessing.itn.chinese.inverse_normalizer import InverseNormalizer
from asr.funasr.bin.vad_inference import Speech2VadSegment, Speech2VadSegmentOnline

from collections import defaultdict
from asr.funasr.utils.context_fst import ContextGraph, log_add

class Speech2Text:
    """Speech2Text class

    Examples:
            >>> import soundfile
            >>> speech2text = Speech2Text("asr_config.yml", "asr.pb")
            >>> audio, rate = soundfile.read("speech.wav")
            >>> speech2text(audio)
            [(token, hypothesis object), ...]

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
            hotword_list_or_file: Union[list, str] = None,
            eable_fst_search: bool = False,
            eable_time_stamp: bool = False,
            clas_scale_score: float = 1.0,
            **kwargs,
    ):
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
        # ngram = None
        # scorers["ngram"] = ngram

        # 4. Build BeamSearch object if model.ctc or lm_file exists [NOTE]
        # transducer is not supported now
        beam_search_transducer = None

        beam_search = None
        is_use_lm = lm_weight != 0.0 and lm_file is not None
        if (ctc_weight != 0.0 and asr_model.ctc != None) or is_use_lm:
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
        # if token_type is None:
        #     token_type = asr_train_args.token_type
        # if bpemodel is None:
        #     bpemodel = asr_train_args.bpemodel

        # if token_type is None:
        #     tokenizer = None
        # elif token_type == "bpe":
        #     if bpemodel is not None:
        #         tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
        #     else:
        #         tokenizer = None
        # else:
        #     tokenizer = build_tokenizer(token_type=token_type)
        # logging.info(f"Text tokenizer: {tokenizer}")

        converter = TokenIDConverter(token_list=token_list)
        
        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.converter = converter
        self.cmvn_file = cmvn_file
        # self.tokenizer = tokenizer

        self.seg_dict = None
        if self.cmvn_file is not None:
            model_dir = os.path.dirname(self.cmvn_file)
            seg_dict_file = os.path.join(model_dir, 'seg_dict')
            if os.path.exists(seg_dict_file):
                seg_dict = {}
                with open(seg_dict_file, "r", encoding="utf8") as f:
                    lines = f.readlines()
                    for line in lines:
                        s = line.strip().split()
                        key = s[0]
                        value = s[1:]
                        seg_dict[key] = " ".join(value)
                self.seg_dict = seg_dict
            else:
                self.seg_dict = None

        # 6. [Optional] Build hotword list from str, local file or url
        self.hotword_list = None
        self.hotword_list = self.generate_hotwords_list(hotword_list_or_file)
       
        self.beam_search = beam_search
        logging.info(f"Beam_search: {self.beam_search}")
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
        self.eable_fst_search = eable_fst_search
        self.eable_time_stamp = eable_time_stamp
        self.clas_scale = clas_scale_score

    @torch.no_grad()
    def __call__(
            self, speech: Union[torch.Tensor, np.ndarray], speech_lengths: Union[torch.Tensor, np.ndarray] = None,
            begin_time: int = 0, end_time: int = None, eable_vad: bool = False
    ):
        """Inference

        Args:
                speech: Input speech data
        Returns:
                token, hyp

        """

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        if self.frontend is not None:
            if eable_vad:
                feats, feats_len = self.frontend.forward_lfr_cmvn(speech, speech_lengths)
            else:
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
        
        decoder_out = self.asr_model._seaco_decode_with_ASF(enc, enc_len,
                                                    pre_acoustic_embeds,
                                                    pre_token_length,
                                                    hw_list=self.hotword_list,
                                                    seaco_weight=self.clas_scale)
        # decoder_out, _ = decoder_outs[0], decoder_outs[1]
        if self.eable_time_stamp:
            _, _, us_alphas, us_peaks = self.asr_model.calc_predictor_timestamp(enc, enc_len, pre_token_length)

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
                    if self.hotword_list is not None and self.eable_fst_search:
                        # start_time = time.time()
                        context_graph = ContextGraph(self.hotword_list)
                        # end_time = time.time()
                        # print("======>", float(end_time-start_time))
                        beam_size = 10
                        cur_hyps = [(tuple(), (0.0, -float('inf'), 0, 0.0))]
                        # 2. CTC beam search step by step
                        for t in range(0, am_scores.size()[0]):
                            logp = am_scores[t]  # (vocab_size,)
                            # key: prefix, value (pb, pnb, context_state, context_score),
                            # default value(-inf, -inf, 0, 0.0)
                            next_hyps = defaultdict(lambda: (-float('inf'), -float('inf'), 0, 0.0))
                            # 2.1 First beam prune: select topk best
                            top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
                            for s in top_k_index:
                                s = s.item()
                                ps = logp[s].item()
                                for prefix, (pb, pnb, c_state, c_score) in cur_hyps:
                                    # last = prefix[-1] if len(prefix) > 0 else None
                                    if s == 0:  # blank
                                        n_pb, n_pnb, _, _ = next_hyps[prefix]
                                        n_pb = log_add([n_pb, pb + ps, pnb + ps])
                                        next_hyps[prefix] = (n_pb, n_pnb, c_state, c_score)
                                    # elif s == last:
                                    #     #  Update *ss -> *s;
                                    #     n_pb, n_pnb, _, _ = next_hyps[prefix]
                                    #     n_pnb = log_add([n_pnb, pnb + ps])
                                    #     next_hyps[prefix] = (n_pb, n_pnb, c_state, c_score)
                                    #     # Update *s-s -> *ss, - is for blank
                                    #     n_prefix = prefix + (s, )
                                    #     n_pb, n_pnb, _, _ = next_hyps[n_prefix]
                                    #     new_c_state, new_c_score = context_graph. \
                                    #         find_next_state(c_state, s)
                                    #     n_pnb = log_add([n_pnb, pb + ps])
                                    #     next_hyps[n_prefix] = (n_pb, n_pnb, new_c_state,
                                    #                         c_score + new_c_score)
                                    else:
                                        n_prefix = prefix + (s, )
                                        n_pb, n_pnb, _, _ = next_hyps[n_prefix]
                                        new_c_state, new_c_score = context_graph. \
                                            find_next_state(c_state, s)
                                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                                        next_hyps[n_prefix] = (n_pb, n_pnb, new_c_state,
                                                            c_score + new_c_score)

                            # 2.2 Second beam prune
                            next_hyps = sorted(next_hyps.items(),
                                            key=lambda x: log_add([x[1][0], x[1][1]]) + x[1][3],
                                            reverse=True)
                            cur_hyps = next_hyps[:beam_size]
                        hyps = [(y[0], log_add([y[1][0], y[1][1]]) + y[1][3]) for y in cur_hyps]
                  
                        token_int = list(filter(lambda x: x != 0 and x != 2 and x != 8403, hyps[0][0]))
                        token = self.converter.ids2tokens(token_int)
                        if self.eable_time_stamp:
                            _, timestamp = ts_prediction_lfr6_standard(us_alphas[i][:enc_len[i] * 3], 
                                                                    us_peaks[i][:enc_len[i] * 3], 
                                                                    copy.copy(token), 
                                                                    vad_offset=begin_time)
                            results.append((token, hyps, timestamp))
                        else:
                            results.append((token, hyps))
                        
                        return results

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
                # remove sos/eos and get results
                last_pos = -1
                if isinstance(hyp.yseq, list):
                    token_int = hyp.yseq[1:last_pos]
                else:
                    token_int = hyp.yseq[1:last_pos].tolist()

                # remove blank symbol id, which is assumed to be 0
                token_int = list(filter(lambda x: x != 0 and x != 2 and x != 1, token_int))

                # Change integer-ids to tokens
                token = self.converter.ids2tokens(token_int)


                # _, timestamp = ts_prediction_lfr6_standard(us_alphas[i][:enc_len[i] * 3],
                #                                             us_peaks[i][:enc_len[i] * 3],
                #                                             copy.copy(token),
                #                                             vad_offset=kwargs.get("begin_time", 0))
                    
                # if self.tokenizer is not None:
                #     text = self.tokenizer.tokens2text(token)
                # else:
                #     text = None

                if self.eable_time_stamp:
                    _, timestamp = ts_prediction_lfr6_standard(us_alphas[i][:enc_len[i] * 3], 
                                                            us_peaks[i][:enc_len[i] * 3], 
                                                            copy.copy(token), 
                                                            vad_offset=begin_time)
                    results.append((token, hyp, timestamp))
                else:
                    results.append((token, hyp))

        return results

    def generate_hotwords_list(self, hotword_list_or_file):

        def seg_tokenize(txt, seg_dict):
            # pattern = re.compile(r'^[\u4E00-\u9FA50-9]+$')
            out_txt = ""
            for word in txt:
                if word in seg_dict:
                    out_txt += seg_dict[word] + " "
                else:
                    for char in word:
                        if char in seg_dict:
                            out_txt += seg_dict[char] + " "
                        else:
                            return []
            return out_txt.strip().split()

        def add_space(text):
            text = re.sub(r'([a-z]+)([\u4e00-\u9fa5]|[0-9])', r'\1 \2', text)
            text = re.sub(r'([\u4e00-\u9fa5]|[0-9])([a-z]+)', r'\1 \2', text)
            return text

        # for None
        if hotword_list_or_file is None:
            hotword_list = None
        # for text str input
        else:
            hotword_list = []
            for hw in hotword_list_or_file:
                hw = hw.strip().lower()
                filterP = re.findall(r'\d{7,}|[a-z]{1,2}[0-9]{1,2}[a-z]{1,2}[0-9]{2,}', hw.replace(" ",""))
                if filterP:
                    continue
                hw = add_space(hw)
                hw_list = hw.split() 
                if self.seg_dict is not None:
                    hw_list = seg_tokenize(hw_list, self.seg_dict)
                if hw_list and len(hw_list) <= 10:
                    hotword_list.append(self.converter.tokens2ids(hw_list))
            hotword_list = hotword_list[:500]
            hotword_list.append([self.asr_model.sos])
        return hotword_list


def inference_tp(
        maxlenratio: float,
        minlenratio: float,
        batch_size: int,
        beam_size: int,
        ngpu: int,
        ctc_weight: float,
        lm_weight: float,
        penalty: float,
        log_level: Union[int, str],
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
        is_use_itn: bool = False,
        eable_fst_search: bool = False,
        param_dict: dict = None,
        **kwargs,
):
 
    ncpu = kwargs.get("ncpu", 1)
    torch.set_num_threads(ncpu)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
      
    if param_dict is not None:
        hotword_list_or_file = param_dict.get('hotword')
        clas_scale_score = param_dict.get('clas_scale_score', 1.0)
        use_timestamp = param_dict.get('use_timestamp', False)
    else:
        hotword_list_or_file = None
        clas_scale_score = 1.0
        use_timestamp = False
    
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
        eable_fst_search=eable_fst_search,
        eable_time_stamp=use_timestamp,
        clas_scale_score=clas_scale_score,
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
        logging.warning("Text2punc is used as: {}".format(device))
        text2punc = Text2Punc(punc_infer_config, punc_model_file, device=device, dtype=dtype)

    itnormalizer = None
    if is_use_itn:
        logging.warning("ITN is used as: {}".format(device))
        itnormalizer = InverseNormalizer(
            cache_dir='asr.textProcessing.itn', 
            overwrite_cache=False,
            enable_standalone_number=True,
            enable_0_to_9=True
        )

    # init_preprocess_fn=ASRTask.build_preprocess_fn(speech2text.asr_train_args, False)
    init_collate_fn=ASRTask.build_collate_fn(speech2text.asr_train_args, False)

    def _forward(
            data_path_and_name_and_type,
            raw_inputs: Union[np.ndarray, torch.Tensor] = None,
            fs: dict = None,
            param_dict: dict = None,
            **kwargs,
    ):
        # hotword also support online correction
        hotword_list_or_file = None
        use_timestamp = speech2text.eable_time_stamp
        if param_dict is not None:
            hotword_list_or_file = param_dict.get('hotword')
            speech2text.clas_scale = param_dict.get('clas_scale_score', clas_scale_score)
            speech2text.eable_time_stamp = param_dict.get('use_timestamp', False)
            use_timestamp = speech2text.eable_time_stamp
        if hotword_list_or_file is not None:
            speech2text.hotword_list = speech2text.generate_hotwords_list(hotword_list_or_file)

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
            preprocess_fn=None,
            collate_fn=init_collate_fn,
            allow_variable_data_keys=allow_variable_data_keys,
            inference=True,
        )

        asr_result_list = []

        for keys, batch in loader:
            _bs = len(next(iter(batch.values())))

            # N-best list of (token, hyp_object)
            try:
                results = speech2text(**batch)  
            except:
                results = []    

            if len(results) < 1:
                hyp = Hypothesis(score=0.0, scores={}, states={}, yseq=[])
                results = [[["sil"], hyp]] * nbest

            for batch_id in range(_bs):
                result = [results[batch_id]]
                key = keys[batch_id]
                for n, result in zip(range(1, nbest + 1), result):
                    token = result[0]
                    timestamp = None if len(result) < 3 else result[2]

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
                    if itnormalizer is not None:
                        text_postprocessed_punc = itnormalizer.normalize(text_postprocessed)

                    if len(word_lists) > 0 and text2punc is not None:
                        text_postprocessed_punc, _ = text2punc(text_postprocessed_punc, 20)
            
                    item = {'key': key, 'value': text_postprocessed_punc}

                    if timestamp_postprocessed != "":
                        item['timestamp'] = timestamp_postprocessed
                    asr_result_list.append(item)

        # torch.cuda.empty_cache()
        return asr_result_list

    return _forward


def inference_vad_tp(
        maxlenratio: float,
        minlenratio: float,
        batch_size: int,
        beam_size: int,
        ngpu: int,
        ctc_weight: float,
        lm_weight: float,
        penalty: float,
        log_level: Union[int, str],
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
        vad_infer_config: Optional[str] = None,
        vad_model_file: Optional[str] = None,
        vad_cmvn_file: Optional[str] = None,
        timestamp_infer_config: Union[Path, str] = None,
        timestamp_model_file: Union[Path, str] = None,
        punc_infer_config: Optional[str] = None,
        punc_model_file: Optional[str] = None,
        is_use_itn: bool = False,
        eable_fst_search: bool = False,
        param_dict: dict = None,
        **kwargs,
):
    ncpu = kwargs.get("ncpu", 1)
    torch.set_num_threads(ncpu)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if param_dict is not None:
        hotword_list_or_file = param_dict.get('hotword')
        clas_scale_score = param_dict.get('clas_scale_score', 1.0)
        use_timestamp = param_dict.get('use_timestamp', False)
    else:
        hotword_list_or_file = None
        clas_scale_score = 1.0
        use_timestamp = False 
    
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
        eable_fst_search=eable_fst_search,
        eable_time_stamp=use_timestamp,
        clas_scale_score=clas_scale_score,
    )
    speech2text = Speech2Text(**speech2text_kwargs)

    # 3. Build speech2vadsegment
    speech2vadsegment_kwargs = dict(
        vad_infer_config=vad_infer_config,
        vad_model_file=vad_model_file,
        vad_cmvn_file=vad_cmvn_file,
        device=device,
        dtype=dtype,
    )
    speech2vadsegment = Speech2VadSegment(**speech2vadsegment_kwargs)

    text2punc = None
    if punc_model_file is not None:
        logging.warning("Text2punc is used as: {}".format(device))
        text2punc = Text2Punc(punc_infer_config, punc_model_file, device=device, dtype=dtype)

    itnormalizer = None
    if is_use_itn:
        logging.warning("ITN is used as: {}".format(device))
        itnormalizer = InverseNormalizer(
            cache_dir='asr.textProcessing.itn', 
            overwrite_cache=False,
            enable_standalone_number=True,
            enable_0_to_9=True
        )

    # init_preprocess_fn=ASRTask.build_preprocess_fn(speech2text.asr_train_args, False)
    init_collate_fn=ASRTask.build_collate_fn(speech2text.asr_train_args, False)

    def _forward(
            data_path_and_name_and_type,
            raw_inputs: Union[np.ndarray, torch.Tensor] = None,
            fs: dict = None,
            param_dict: dict = None,
            **kwargs,
    ):
        # hotword also support online correction
        hotword_list_or_file = None
        use_timestamp = speech2text.eable_time_stamp
        if param_dict is not None:
            hotword_list_or_file = param_dict.get('hotword')
            speech2text.clas_scale = param_dict.get('clas_scale_score', clas_scale_score)
            speech2text.eable_time_stamp = param_dict.get('use_timestamp', False)
            use_timestamp = speech2text.eable_time_stamp
        if hotword_list_or_file is not None:
            speech2text.hotword_list = speech2text.generate_hotwords_list(hotword_list_or_file)

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
            preprocess_fn=None,
            collate_fn=init_collate_fn,
            allow_variable_data_keys=allow_variable_data_keys,
            inference=True,
        )

        asr_result_list = []
        for keys, batch in loader:
            vad_results = speech2vadsegment(**batch)
            fbanks, vadsegments = vad_results[0], vad_results[1]
            for i, segments in enumerate(vadsegments):
                for j, segment_idx in enumerate(segments):
                    bed_idx, end_idx = int(segment_idx[0] / 10), int(segment_idx[1] / 10)
                    segment = fbanks[:, bed_idx:end_idx, :].to(device)
                    speech_lengths = torch.Tensor([end_idx - bed_idx]).int().to(device)
                    batch = {"speech": segment, "speech_lengths": speech_lengths, "begin_time": vadsegments[i][j][0],
                             "end_time": vadsegments[i][j][1], "eable_vad": True}
                    results = speech2text(**batch)
                    if len(results) < 1:
                        continue

                    if j == 0:
                        result_segments = results
                    else:
                        result_segments = [
                            [result_segments[0][i] + results[0][i] for i in range(len(results[0]))]]

                key = keys[0]
                result = result_segments[0]
                token = result[0]
                timestamp = None if len(result) < 3 else result[2]

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
                if itnormalizer is not None:
                    text_postprocessed_punc = itnormalizer.normalize(text_postprocessed)

                if len(word_lists) > 0 and text2punc is not None:
                    text_postprocessed_punc, _ = text2punc(text_postprocessed_punc, 20)
        
                item = {'key': key, 'value': text_postprocessed_punc}

                if timestamp_postprocessed != "":
                    item['timestamp'] = timestamp_postprocessed
                asr_result_list.append(item)

        # torch.cuda.empty_cache()
        return asr_result_list

    return _forward


def vadOnlineInfer(
        ngpu: int,
        log_level: Union[int, str],
        batch_size: int = 1,
        key_file: Optional[str] = None,
        allow_variable_data_keys: bool = False,
        dtype: str = "float32",
        seed: int = 0,
        num_workers: int = 1,
        device = "cpu",
        vad_infer_config: Optional[str] = None,
        vad_model_file: Optional[str] = None,
        vad_cmvn_file: Optional[str] = None,
        param_dict: dict = None,
        **kwargs,
):
    ncpu = kwargs.get("ncpu", 1)
    torch.set_num_threads(ncpu)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 0 and torch.cuda.is_available():
        device = device
    else:
        device = "cpu"
        batch_size = 1
    logging.warning("device is given as: {}".format(device))

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2vadsegment
    speech2vadsegment_kwargs = dict(
        vad_infer_config=vad_infer_config,
        vad_model_file=vad_model_file,
        vad_cmvn_file=vad_cmvn_file,
        device=device,
        dtype=dtype,
    )
    speech2vadsegment = Speech2VadSegmentOnline(**speech2vadsegment_kwargs)

    # init_preprocess_fn=ASRTask.build_preprocess_fn(speech2text.asr_train_args, False)
    init_collate_fn=ASRTask.build_collate_fn(speech2vadsegment.vad_infer_args, False)

    def _forward(
            data_path_and_name_and_type: list = None,
            raw_inputs: Union[np.ndarray, torch.Tensor] = None,
            fs: dict = None,
            param_dict: dict = None,
            **kwargs,
    ):
        is_final = param_dict.get('is_final', False) 
        if is_final:
            speech2vadsegment.frontend.cache_reset()
            return [], [], {}

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
            preprocess_fn=None,
            collate_fn=init_collate_fn,
            allow_variable_data_keys=allow_variable_data_keys,
            inference=True,
        )

        vad_in_cache = param_dict.get('in_cache', dict())
        max_end_sil = param_dict.get('max_end_sil', 600)
        speech2vadsegment.vad_model.vad_opts.max_single_segment_time = param_dict.get("max_len", 15000) 

        for _, batch in loader:
            batch['in_cache'] = vad_in_cache
            batch['is_final'] = is_final
            batch['max_end_sil'] = max_end_sil
            fbanks, vadsegments, vad_out_cache = speech2vadsegment(**batch)
        # torch.cuda.empty_cache()
        return fbanks, vadsegments, vad_out_cache

    return _forward