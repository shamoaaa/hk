import numpy as np
import torchaudio
from opuslib import Decoder
from pydub import AudioSegment
import json
import pyloudnorm as pyln

from ms_common.configs import logger, model_config, MS_WEIGHTS
from ms_common.model import Model
from ms_common.protocol import CommonRequest, CommonResponse, Completion
from ms_common.utils import base64_decode

from asr.funasr.bin.asr_inference_paraformer_vad_tp import vadOnlineInfer as vad_infer_online
from asr.funasr.bin.asr_inference_paraformer_vad_tp import inference_tp as asr_infer_offline
from asr.funasr.bin.asr_inference_paraformer_streaming import inference_online as asr_infer_online

from asr.semantic_vad.infer import SemanticComplete_Bert
from asr.speakerlab.bin.inference import speaker_verification
from asr.wenet.bin.kws_inference_conformer import inference as kws_infer_offline

class EzvizModelBackend(Model):
    """
    萤石智能设备 ASR 后端服务核心类。
    
    提供两种推理模式：
    - forward(): 非流式推理，一次性传入完整音频，返回识别结果
    - stream_init() + stream_forward() × N: 流式推理，2-Pass 策略
    
    2-Pass 流式策略：
    - Pass-1: ParaformerOnline 每 480ms 出一次中间结果（贪心解码，低延迟）
    - Pass-2: SeacoParaformer 在流结束时对全段音频做离线纠正（BeamSearch + 热词，高精度）
    - 客户端以 stage_flag=1 的结果为准，stage_flag=0 仅供即时反馈
    
    通信协议架构：
    - 外层（客户端 ↔ ms_manager）: REST HTTP/JSON，多次 POST 模拟流式
    - 内层（ms_manager ↔ backend）: gRPC，真正的流式通信
    - 回调（ms_manager → 调用方）: HTTP POST 到 stream_result_url
    - 丢包/乱序：TCP 保证传输层有序可靠，应用层无额外排序逻辑
    
    流式结束的两种触发方式：
    1. 客户端主动结束: rqs_info.sequence_end = True
    2. 服务端 VAD 主动结束: 检测到语音终止点，强制 is_final = True
       此时 ms_manager 通过 early_stop 机制释放 channel，丢弃后续帧
    
    2-Pass 边界情况（流式有结果但最终为空）：
    - 声纹验证触发静默（非同一人 + 响度差大）
    - KWS 唤醒词验证失败
    - 离线模型判定为噪声/无效语音
    - VAD 截断后音频过短（< 100ms）
    """

    def __init__(self):
        
        # check dir exist if backend need
        assert MS_WEIGHTS.exists(), f"MS_WEIGHTS not exist, please export MS_WEIGHTS=/path/to/weights"
        chunk_size = [8, 8, 4]
        # 默认参数配置
        self.model_fs = 16000
        self.param_dict = dict() 
        self.default_hotwords = dict()
        self.default_hotwords['general'] = ["萤石"] 
        self.default_hw_scores = dict()

        self.default_clas_scale_score = 0.54 # 非流式默认热词权重
        self.default_hw_max_length = 20 # 非流式设置允许热词的最大长度，可用于过滤
        self.default_hw_max_num = 1000 # 非流式设置允许热词的最大数量，可用于过滤
        self.param_dict['hotword'] = self.default_hotwords['general']
        self.param_dict['clas_scale_score'] = self.default_clas_scale_score
        self.param_dict['hw_max_length'] = self.default_hw_max_length
        self.param_dict['hw_max_num'] = self.default_hw_max_num
        self.supoort_audio_format = ["base64", "url", "opus", "wav", "pcm", "aac"]

        # 流式pipeline 与 非流式 共用
        # inference_offline: SeacoParaformer 非流式推理引擎
        # - 用于非流式 forward() 的完整推理
        # - 用于流式 stream_forward() 中 Pass-2 离线纠正（is_final=True 时）
        # - 支持 SeACo 热词增强 + CTC-Attention BeamSearch (beam=10)
        self.inference_offline = asr_infer_offline(
            dtype="float32",
            maxlenratio=0.0,
            minlenratio=0.0,
            batch_size=1,
            beam_size=10,
            ngpu=1,
            ctc_weight=0.5,
            lm_weight=0.0,
            penalty=0.0,
            log_level="INFO",
            asr_train_config=str(MS_WEIGHTS / "seaco_paraformer_model/config.yaml"),
            asr_model_file=str(MS_WEIGHTS / "seaco_paraformer_model/model.pt"),
            cmvn_file=str(MS_WEIGHTS / "seaco_paraformer_model/am.mvn"),
            lm_train_config=None,
            lm_file=None,
            streaming=False,
            nbest=1,
            num_workers=0,
            punc_infer_config=str(MS_WEIGHTS / "punc_model/config.yaml"),
            punc_model_file=None,
            is_use_itn=False,
            eable_fst_search=False,
            param_dict=self.param_dict,
            device="cuda:0"
        )

        # inference_online: ParaformerOnline 流式推理引擎
        # - 用于流式 stream_forward() 中 Pass-1 中间结果（每 480ms 触发一次）
        # - chunk_size=[8,8,4]: 左看8帧 + 当前8帧 + 右看4帧（每帧60ms LFR后）
        # - 使用贪心解码（argmax），不用 BeamSearch，保证低延迟
        # - 内部维护跨 chunk 的 cache（encoder/CIF/decoder），需在流结束时传 is_final=True 清空
        self.inference_online = asr_infer_online(
            dtype="float32",
            maxlenratio=0.0,
            minlenratio=0.0,
            batch_size=1,
            beam_size=10,
            ngpu=1,
            ctc_weight=0.5,
            lm_weight=0.0,
            penalty=0.0,
            log_level="INFO",
            asr_train_config=str(MS_WEIGHTS / "paraformer_streaming_model/config.yaml"),
            asr_model_file=str(MS_WEIGHTS / "paraformer_streaming_model/model.pb"),
            cmvn_file=str(MS_WEIGHTS / "paraformer_streaming_model/am.mvn"),
            lm_train_config=None,
            lm_file=None,
            streaming=True,
            nbest=1,
            chunk_size=chunk_size,
            device="cuda:0"
        )

        # vad_online: FSMN-VAD 在线语音活动检测
        # - 流式增量推理，每帧输入后检测语音起止点
        # - 返回 vadsegments: List[List[List[int]]]，结构为 [batch][segment][start_ms, end_ms]
        # - -1 表示未检测到起始/终止点
        # - 配合语义完整性 BERT 模型动态调整 max_end_sil（500~1300ms）
        # - 检测到终止点时强制 is_final=True，触发 Pass-2 纠正
        self.vad_online = vad_infer_online(
            dtype="float32",
            ngpu=1,
            log_level="INFO",
            num_workers=0,
            vad_infer_config=str(MS_WEIGHTS / "model_vad_fsmn/vad.yaml"),
            vad_model_file=str(MS_WEIGHTS / "model_vad_fsmn/vad.pb"),
            vad_cmvn_file=str(MS_WEIGHTS / "model_vad_fsmn/vad.mvn"),
            device="cuda:0"
        )

        # opus 裸流解码decoder
        # Opus: 开源低延迟音频编解码器，IoT 设备用于压缩传输（压缩比约 8x）
        # 固定参数: 16kHz 采样率, 单声道, 20ms 帧长
        # 解码: 每 80 字节 Opus → 320 samples × 2 bytes = 640 字节 PCM
        self.opus_dec = Decoder(fs=self.model_fs, channels=1)
        # 2pass识别相关
        self.data_online = bytes()  # PCM 音频累积缓冲区，所有帧的音频追加到此
        self.decode_beg = 0  # 已处理到的字节偏移（下次流式推理从这里开始读）
        self.decode_end = 0  # 当前数据写入到的字节偏移（总共累积了多少字节）
        # 待处理长度 = decode_end - decode_beg，≥ stride_size 时触发流式推理
        self.sequence_index = 0  # 输出序号（产出了几个有效响应），非输入序号
        self.stride_size = int(60*chunk_size[1]/1000*16000*2)  # = 15360 bytes = 480ms 音频
        self.business_type = ""
        self.rqs_format = ""
        # vad相关
        self.vad_beg_point = 0  # VAD 检测到的语音起始点（字节偏移）
        self.vad_end_sil_init = 800  # VAD 默认尾部静音阈值（ms），可被语义完整性动态调整
        self.vad_param_dict = dict()
        self.vad_param_dict['in_cache'] = dict()  # VAD 跨帧状态缓存
        self.vad_param_dict['max_end_sil'] = self.vad_end_sil_init  
        self.vad_param_dict['max_len'] = 15000  # 单段语音最大长度（ms）
        self.vad_eable_tpye = []  # 启用 VAD 的业务类型列表（通过 Apollo 配置）
        self.set_vad_beg = False  # 是否检测到语音起始点，Pass-2 纠正需要此标记为 True
        # semantic_complete相关
        # 语义完整性判断：用 BERT 模型判断已识别文本是否语义完整
        # 语义完整（如"打开空调"）→ 缩短 max_end_sil → 更快结束
        # 语义不完整（如"帮我把"）→ 延长 max_end_sil → 等用户继续说
        self.last_asr_text = ""  # 累积的识别文本（供语义完整性判断）
        self.semantic_complete_flag = False  # 标记下一轮是否需要计算语义完整性
        self.semantic_complete_prob = SemanticComplete_Bert(
            model_path=str(MS_WEIGHTS / "semantic_vad/model.pt"),
            vocab_path=str(MS_WEIGHTS / "semantic_vad/vocab.txt"),
            device="cuda:0"
        )
        # wakeup相关
        # 唤醒帧处理：客户端在首帧中携带唤醒音频段，用于 KWS 二次验证和声纹比对
        self.wakeup_audio_length = -1  # 唤醒音频字节长度，-1 表示无唤醒帧
        self.audio_wakeup_type = -1  # 唤醒类型: 0=主唤醒词, 1=快捷, 2=自定义, 3=人脸, 4=点击, 5=免唤醒, 6=持续聆听
        self.wakeup_refilter_type = []  # 启用 KWS 二次验证的业务类型列表
        self.kws_refilter_res = True  # KWS 验证结果，False 时强制终止会话不返回结果
        self.kws_infer_model = kws_infer_offline(model_dir=str(MS_WEIGHTS / "kws_model"))
        self.kws_infer_model(bytes(54400)) # warmup
        # sv相关
        # 声纹验证（Speaker Verification）：防误唤醒
        # 比对唤醒音频和命令音频的说话人是否一致
        # 双重判断：声纹相似度 + 响度差（区分"家人正常使用"和"电视背景声误触发"）
        self.sv_sim_thres = 0.3  # 余弦相似度阈值，低于此值判定非同一人
        self.sv_loud_thres = 6.5  # LUFS 响度差阈值，唤醒音比命令音大此值以上触发静默
        self.sv_eable_type = []  # 启用声纹验证的业务类型列表
        self.wakeup_audio_data = bytes()
        self.loudness_meter = pyln.Meter(self.model_fs)
        self.sv_model = speaker_verification(device="cuda:0", model_path=str(MS_WEIGHTS / "sv_model/pretrained_eres2netv2.ckpt"))
        self.sv_model.inference(bytes(54400), bytes(54400)) # warmup
    
    def stream_reset(self,):
        self.inference_online(data_path_and_name_and_type=None, raw_inputs=None, param_dict={"is_final": True})
        self.data_online = bytes()
        self.decode_beg = 0
        self.decode_end = 0
        self.sequence_index = 0
        self.vad_beg_point = 0
        self.vad_param_dict['in_cache'] = dict()
        self.vad_online(param_dict={"is_final": True})
        self.rqs_format = ""
        self.wakeup_audio_length = -1
        self.audio_wakeup_type = -1
        self.set_vad_beg = False
        self.last_asr_text = ""
        self.semantic_complete_flag = False
        self.vad_param_dict['max_end_sil'] = self.vad_end_sil_init
        self.wakeup_audio_data = bytes()
        self.kws_refilter_res = True

    # 非流式process 
    def forward(self, rqs_info:CommonRequest, rsp_msg:CommonResponse):
        
        # 入参检查
        if len(rqs_info.data_info) != 1:
            rsp_msg.InvalidParamError("data_info length != 1")
            return rsp_msg
        
        if rqs_info.data_info[0].modal != "audio" or (rqs_info.data_info[0].type != "base64" and rqs_info.data_info[0].type != "url"):
            rsp_msg.InvalidParamError(f"data_info modal:{rqs_info.data_info[0].modal}, type:{rqs_info.data_info[0].type}, ASR algorithm need modal=audio and type=base64|url")
            return rsp_msg

        # 热词作用
        if rqs_info.data_params:
            if rqs_info.data_params[0].hotwords:
                self.param_dict['hotword'] = self.default_hotwords['general'] + rqs_info.data_params[0].hotwords
            else:
                self.param_dict['hotword'] = self.default_hotwords['general']
        else:
            self.param_dict['hotword'] = self.default_hotwords['general']

        infer_fs = None
        if rqs_info.data_info[0].type == "base64":
            try:
                binary_Data = base64_decode(rqs_info.data_info[0].data) # 把base64数据解码为二进制数据
                data_path_and_name_and_type = [[binary_Data, "speech", "bytes", "id1"]]
            except:
                rsp_msg.InvalidParamError(f"Incorrect padding of base64 in rqs_info.data_info[0].data")
                rsp_msg.sequence_end = True
                return rsp_msg
        else:
            if rqs_info.data_info[0].data.endswith("pcm"):
                with open(rqs_info.data_info[0].data, 'rb') as f:
                    binary_Data = f.read()
                data_path_and_name_and_type = [[binary_Data, "speech", "bytes", "id1"]]
            elif rqs_info.data_info[0].data.endswith(("url", "opus", "wav")):
                waveforms, audio_fs = torchaudio.load(rqs_info.data_info[0].data)
                data_path_and_name_and_type = [[waveforms.numpy()[0], "speech", "waveform", "id1"]]
                if audio_fs != self.model_fs:
                    infer_fs = {"audio_fs":audio_fs, "model_fs": self.model_fs}
            elif rqs_info.data_info[0].data.endswith(("aac")):
                waveraw = AudioSegment.from_file(rqs_info.data_info[0].data, format="aac") 
                waveforms = np.array(waveraw.get_array_of_samples()).astype(np.float32)
                audio_fs = waveraw.frame_rate
                data_path_and_name_and_type = [[waveforms, "speech", "waveform", "id1"]]
                if audio_fs != self.model_fs:
                    infer_fs = {"audio_fs":audio_fs, "model_fs": self.model_fs}
            else:
                rsp_msg.InvalidParamError(f"data type:{rqs_info.data_info[0].type}, ASR algorithm need data type in {self.supoort_audio_format}")
                return rsp_msg
        
        asr_results_text = self.inference_offline(data_path_and_name_and_type, raw_inputs=None, param_dict=self.param_dict, fs=infer_fs)
        logger.debug(f"ASR result:{asr_results_text}")

        if asr_results_text and asr_results_text[0]['value']:        
            rsp_msg.completion = []
            completion = Completion(
                modal= "text",
                type= "raw",
                data=asr_results_text[0]['value'].lower()
            )
            rsp_msg.completion.append(completion)
        return rsp_msg
    
    def stream_init(self, rqs_info: CommonRequest, rsp_msg: CommonResponse):
        # 清空buffer
        self.stream_reset()
        
        # 热词作用
        if rqs_info.statistic:
            self.business_type = rqs_info.statistic.biz_key
            logger.debug(f"asr stream init, biz_key: {self.business_type}")
            if self.business_type in self.default_hotwords:
                business_hws = self.default_hotwords[self.business_type] 
            else:
                business_hws = self.default_hotwords['general']
        else:
            self.business_type = ""
            business_hws = self.default_hotwords['general']
        
        if rqs_info.data_params:
            if rqs_info.data_params[0].modal != "audio":
                rsp_msg.InvalidParamError(f"data_params[0].modal, algorithm need modal=audio")
                return rsp_msg
            # v2.4.0 支持opus格式传入
            self.rqs_format = rqs_info.data_params[0].format
            logger.debug(f"asr stream init, rqs_format: {self.rqs_format}")
            if self.rqs_format and self.rqs_format not in ['pcm', 'opus']:
                rsp_msg.InvalidParamError(f"asr stream init: data_params.format should be in [pcm, opus]")
                return rsp_msg               
            # v2.4.0 支持唤醒帧上传
            self.wakeup_audio_length = rqs_info.data_params[0].wakeup_audio_length
            logger.debug(f"asr stream init, wakeup_audio_length: {self.wakeup_audio_length}")
            self.audio_wakeup_type = rqs_info.data_params[0].audio_wakeup_type
            logger.debug(f"asr stream init, audio_wakeup_type: {self.audio_wakeup_type}")
            # 热词传入
            if rqs_info.data_params[0].hotwords:
                if len(rqs_info.data_params[0].hotwords) > self.param_dict['hw_max_num']:
                    rsp_msg.InvalidParamError(f"hotwords num should be less than {self.param_dict['hw_max_num']}")
                    return rsp_msg
                if max(map(len, rqs_info.data_params[0].hotwords)) > self.param_dict['hw_max_length']:
                    rsp_msg.InvalidParamError(f"hotwords len should be less than {self.param_dict['hw_max_length']}")
                    return rsp_msg
                self.param_dict['hotword'] = business_hws + rqs_info.data_params[0].hotwords 
            else:
                self.param_dict['hotword'] = business_hws
        else:
            self.param_dict['hotword'] = business_hws
        logger.debug(f"asr stream init, hotwords:{self.param_dict['hotword']}")

        if not rqs_info.output_params:
            rsp_msg.success()
            return rsp_msg
        
        if len(rqs_info.output_params) != 1:
            rsp_msg.InvalidParamError("rqs_info.output_params length isn't equal 1")
            return rsp_msg
        
        if rqs_info.output_params[0].modal != 'audio':
            rsp_msg.InvalidParamError("rqs_info.output_params[0].modal != 'audio'")
            return rsp_msg
                        
        return rsp_msg
    
    def stream_forward(self, rqs_info: CommonRequest, rsp_msg: CommonResponse):
        # 入参检查
        if len(rqs_info.data_info) != 1:
            rsp_msg.InvalidParamError("data_info length != 1")
            rsp_msg.sequence_index = self.sequence_index
            rsp_msg.sequence_end = True
            yield rsp_msg
            return rsp_msg
        if rqs_info.data_info[0].modal != "audio" or rqs_info.data_info[0].type != "base64":
            rsp_msg.InvalidParamError(f"data_info modal:{rqs_info.data_info[0].modal}, type:{rqs_info.data_info[0].type}, ASR algorithm need modal=audio and type=base64")
            rsp_msg.sequence_index = self.sequence_index
            rsp_msg.sequence_end = True
            yield rsp_msg
            return rsp_msg
        
        if model_config.debug_mode:
            if rqs_info.sequence_end:
                rsp_msg.stage_flag = 1
                rsp_msg.sequence_index = self.sequence_index
                rsp_msg.sequence_end = rqs_info.sequence_end
                
                rsp_msg.completion = []
                completion = Completion(
                    modal= "text",
                    type= "raw",
                    data="语音识别算法现在是debug模式"
                )
                logger.debug(f"ASR-backend: get response msg, rsp_msg:{rsp_msg.model_dump(exclude_unset=True)}")
                yield rsp_msg
                self.sequence_index += 1

                return rsp_msg
            else:
                logger.debug(f"ASR backend get rqs info, sequence_end:{rqs_info.sequence_end}, sequence_index:{rqs_info.sequence_index}")
                rsp_msg.success()
                yield rsp_msg
                return rsp_msg
        
        try:
            b64Decode_Data = base64_decode(rqs_info.data_info[0].data) # 把base64数据解码为二进制数据
        except:
            rsp_msg.InvalidParamError(f"Incorrect padding of base64 in rqs_info.data_info[0].data")
            rsp_msg.sequence_index = self.sequence_index
            rsp_msg.sequence_end = True
            yield rsp_msg
            return rsp_msg

        if self.rqs_format == 'opus':
            # NOTE: Opus 指裸流解码，固定了帧长 20 ms, 16k采样率，所以固定 320
            try:
                binary_Data = bytes()
                for dec_step in range(0, len(b64Decode_Data), 80):
                    binary_Data += self.opus_dec.decode(b64Decode_Data[dec_step:dec_step+80], frame_size=320)
            except:
                rsp_msg.InvalidParamError(f"Incorrect decoding of opus input")
                rsp_msg.sequence_index = self.sequence_index
                rsp_msg.sequence_end = True
                yield rsp_msg
                return rsp_msg
        else:
            binary_Data = b64Decode_Data

        # 0:主唤醒词唤醒, 1:快捷唤醒, 2:用户自定义唤醒, 3:人脸唤醒, 4:点击唤醒, 5:免唤醒, 6:持续聆听
        if self.audio_wakeup_type in [0,2] and self.wakeup_audio_length > 0 and rqs_info.sequence_index == 0:
            self.wakeup_audio_data = binary_Data[:self.wakeup_audio_length]
            binary_Data = binary_Data[self.wakeup_audio_length:]
            if self.audio_wakeup_type == 0 and self.business_type in self.wakeup_refilter_type and self.wakeup_audio_length >= 48000:
                self.kws_refilter_res = self.kws_infer_model(self.wakeup_audio_data)

        if len(binary_Data) % 2 != 0: # NOTE: 此处是int16的补零要求
            binary_Data += b'\x00'
            logger.warn(f"input data cannot be converted to int16 type, padding here")
        
        # 临时保存输入audio文件
        if model_config.save_out_data:
            save_audio_dir = MS_WEIGHTS.joinpath("../input_audio")
            save_audio_dir.mkdir(exist_ok=True)
            input_audio_p = save_audio_dir.joinpath(f"{rqs_info.request_id}_seq_idx_{rqs_info.sequence_index}.txt")
            logger.debug(f"length of base64 data:{len(rqs_info.data_info[0].data)}")
            with input_audio_p.open("w") as f:
                f.write(rqs_info.data_info[0].data)
            logger.debug(f"saved input audio of ASR in :{input_audio_p}")
                    
        is_final = rqs_info.sequence_end # 流式是否结束
                
        asr_results_text = [{'value': ''}] # 流式的话，该参数length == 1
        self.decode_end += len(binary_Data)
        self.data_online += binary_Data
        stage_flag = 0
        
        if self.business_type in self.vad_eable_tpye:
            if self.semantic_complete_flag:
                self.vad_param_dict['max_end_sil'] = self.semantic_complete_prob.forward(self.last_asr_text)
                self.semantic_complete_flag = False
            logger.debug(f"max end sil as {self.vad_param_dict['max_end_sil']}")
            _, vadsegments, vad_out_cache = self.vad_online([binary_Data, "speech", "bytes"], param_dict = self.vad_param_dict)
            if vadsegments and vadsegments[0]:
                if vadsegments[0][0][0] != -1:
                    self.vad_beg_point = vadsegments[0][0][0] * 32
                    logger.debug("ASR backend: vad begpoint is detected")
                    self.set_vad_beg = True
                if vadsegments[0][0][1] != -1:
                    self.data_online = self.data_online[self.vad_beg_point:vadsegments[0][0][1]*32]
                    logger.debug("ASR backend: vad endpoint is detected")
                    is_final = True
            self.vad_param_dict['in_cache'] = vad_out_cache
        else:
            self.set_vad_beg = True

        param_online = {
            "is_final": is_final,
            'hotwords': self.param_dict['hotword']
            } 

        logger.debug(f"ASR-backend start forward, rqs_id:{rsp_msg.request_id}, is_end:{rqs_info.sequence_end}")
        if (self.decode_end - self.decode_beg) >= self.stride_size:
            if is_final:
                if self.set_vad_beg: # 当启用了vad, 只有检测到有起始点才会纠正
                    data_path_and_name_and_type = [self.data_online, "speech", "bytes"]
                    if self.business_type in self.default_hw_scores:
                        self.param_dict['clas_scale_score'] = self.default_hw_scores[self.business_type]
                    else:
                        self.param_dict['clas_scale_score'] = self.default_clas_scale_score
                    logger.debug(f"[asr hw_score] has been updated as {self.param_dict['clas_scale_score']}")
                    asr_results_text = self.inference_offline(data_path_and_name_and_type, raw_inputs=None, param_dict=self.param_dict)
                    self.inference_online(data_path_and_name_and_type, raw_inputs=None, param_dict=param_online)
                    # self.data_online = bytes()
                    self.decode_beg = 0
                    self.decode_end = 0
                    stage_flag = 1 # 矫正完成 flag
            else:
                data_path_and_name_and_type = [self.data_online[self.decode_beg:self.decode_beg+self.stride_size], "speech", "bytes"]
                asr_results_text = self.inference_online(data_path_and_name_and_type, raw_inputs=None, param_dict=param_online)
                self.decode_beg += self.stride_size
        else:  
            if is_final and self.set_vad_beg: # 当启用了vad, 只有检测到有起始点才会纠正
                data_path_and_name_and_type = [self.data_online, "speech", "bytes"]
                if len(self.data_online) > 3200: # 保证单次解码长度大于100ms
                    if self.business_type in self.default_hw_scores:
                        self.param_dict['clas_scale_score'] = self.default_hw_scores[self.business_type]
                    else:
                        self.param_dict['clas_scale_score'] = self.default_clas_scale_score
                    logger.debug(f"[asr hw_score] has been updated as {self.param_dict['clas_scale_score']}")
                    asr_results_text = self.inference_offline(data_path_and_name_and_type, raw_inputs=None, param_dict=self.param_dict)
                    self.inference_online(data_path_and_name_and_type, raw_inputs=None, param_dict=param_online)
                    # self.data_online = bytes()
                    self.decode_beg = 0
                    self.decode_end = 0
                    stage_flag = 1 # 矫正完成 flag
                else:
                    self.inference_online(data_path_and_name_and_type, raw_inputs=None, param_dict=param_online)
                    # self.data_online = bytes()
                    self.decode_beg = 0
                    self.decode_end = 0
                    stage_flag = 1 # 矫正完成 flag
                    
                    rsp_msg.AudioDecodeFail("request audio is too short(less than 3200byte), ASR result is empty, please check your input audio")
                    rsp_msg.sequence_index = self.sequence_index
                    rsp_msg.sequence_end = True
                    yield rsp_msg
                    return rsp_msg

        if self.business_type in self.sv_eable_type and is_final and len(self.wakeup_audio_data) > 16000:
            if len(self.data_online) > 16000: # 超过 block_size 400ms
                wav_wakeup, wav_cmd, sv_result = self.sv_model.inference(self.wakeup_audio_data, self.data_online)
                if sv_result <= self.sv_sim_thres: # 此时声纹判断并非同一个人
                    loudness_wakeup = self.loudness_meter.integrated_loudness(wav_wakeup)
                    loudness_cmd = self.loudness_meter.integrated_loudness(wav_cmd)
                    logger.debug(f"loudness_wakeup computed as {loudness_wakeup}, loudness_cmd computed as {loudness_cmd}")
                    loudness_diff = loudness_wakeup - loudness_cmd
                    if loudness_diff >= self.sv_loud_thres: 
                        asr_results_text[0]['value'] = "" # 响度差别大，触发静默，不回答
                        logger.info(f"sv and loudness checking lanunch: {loudness_diff}")

        if self.kws_refilter_res and asr_results_text and asr_results_text[0]['value']:
            self.last_asr_text += asr_results_text[0]['value'].lower()
            self.semantic_complete_flag = True
            rsp_msg.stage_flag = stage_flag
            rsp_msg.sequence_index = self.sequence_index
            rsp_msg.sequence_end = is_final
            rsp_msg.completion = []
            completion = Completion(
                modal= "text",
                type= "raw",
                data=asr_results_text[0]['value'].lower()
            )
            rsp_msg.completion.append(completion)
            logger.debug(f"ASR-backend: get response msg, rsp_msg:{rsp_msg.model_dump(exclude_unset=True)}")
            yield rsp_msg
            self.sequence_index += 1
        else:
            if is_final:
                rsp_msg.stage_flag = 1
                rsp_msg.sequence_end = True
                rsp_msg.sequence_index = self.sequence_index
                logger.info(f"ASR-backend: empty asr text")
                yield rsp_msg
            elif not self.kws_refilter_res:
                rsp_msg.stage_flag = 1
                rsp_msg.sequence_end = True
                rsp_msg.sequence_index = self.sequence_index
                logger.info(f"ASR-backend: kws refilter checked")
                yield rsp_msg

        return rsp_msg

    def apollo_callback(self, value:str):
        logger.warn(f"ASR v2.6.0 get call_back vlaue:{value}")
        try:
            hw_info = value.split(',', 2)
            hw_length, hw_num = hw_info[0], hw_info[1]
            self.param_dict['hw_max_length'] = int(hw_length.split(':')[1])
            logger.warn(f"Hw_max_length has been updated as {self.param_dict['hw_max_length']}")
            self.param_dict['hw_max_num'] = int(hw_num.split(':')[1])
            logger.warn(f"Hw_max_num has been updated as {self.param_dict['hw_max_num']}") 
            if len(hw_info) >= 3:
                apo_key = hw_info[2].split(':', 1)
                if apo_key[0] == 'hw_add':
                    hw_business = json.loads(apo_key[1])
                    for business_type in hw_business:
                        self.default_hotwords[business_type] = hw_business[business_type].split("|")
                        logger.warn(f"{business_type} hotwords have been updated as {self.default_hotwords[business_type]}")
                else:
                    extra_info = json.loads(apo_key[1])
                    if 'hw_add' in extra_info:
                        hw_business = extra_info['hw_add']
                        for business_type in hw_business:
                            self.default_hotwords[business_type] = hw_business[business_type].split("|")
                            logger.warn(f"{business_type} hotwords have been updated as {self.default_hotwords[business_type]}")
                    if 'vad_info' in extra_info:
                        if 'vad_end' in extra_info['vad_info']:
                            self.vad_end_sil_init = int(extra_info['vad_info']['vad_end'])
                            logger.warn(f"VAD max_end_sil has been updated as {self.vad_end_sil_init}")
                        if 'vad_len' in extra_info['vad_info']:
                            self.vad_param_dict['max_len'] = int(extra_info['vad_info']['vad_len'])
                            logger.warn(f"VAD max_len has been updated as {self.vad_param_dict['max_len']}")
                        if 'vad_eable' in extra_info['vad_info']:
                            self.vad_eable_tpye = extra_info['vad_info']['vad_eable'].split('|')
                            logger.warn(f"VAD eable type have been updated as {self.vad_eable_tpye}")
                    if 'hw_score' in extra_info:
                        self.default_clas_scale_score = float(extra_info['hw_score'])
                        logger.warn(f"hotwords clas_scale_score has been updated as {self.default_clas_scale_score}") 
                    if 'hw_score_split' in extra_info:
                        for business_type in extra_info['hw_score_split']:
                            self.default_hw_scores[business_type] = float(extra_info['hw_score_split'][business_type])
                            logger.warn(f"{business_type} hw_score have been updated as {self.default_hw_scores[business_type]}")
                    if 'sv_eable' in extra_info:
                        self.sv_eable_type = extra_info['sv_eable'].split('|')
                        logger.warn(f"sv_eable have been updated as {self.sv_eable_type}")
                    if 'kws_eable' in extra_info:
                        self.wakeup_refilter_type = extra_info['kws_eable'].split('|')
                        logger.warn(f"kws_eable have been updated as {self.wakeup_refilter_type}")
                    if "kws_thres" in extra_info:
                        self.sv_sim_thres = float(extra_info["kws_thres"].split('|')[0])
                        logger.warn(f"sv_sim_thres have been updated as {self.sv_sim_thres}")
                        self.sv_loud_thres = float(extra_info["kws_thres"].split('|')[1])
                        logger.warn(f"sv_loud_thres have been updated as {self.sv_loud_thres}")
        except Exception as e:
            logger.error(e)
        