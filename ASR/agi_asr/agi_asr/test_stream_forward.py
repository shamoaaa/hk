# from MS_BE.pb import model_server_pb2_grpc, model_server_pb2
from time import sleep, time
from json import dumps, loads
from requests import post
import random
from pathlib import Path
from uuid import uuid4
import os

from ms_common.configs import logger, model_config, MS_WEIGHTS
from ms_common.protocol import CommonRequest, CommonResponse, DataInfo, DataParams, protocol_ver

import base64

from asr.ezviz_backend import EzvizModelBackend

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
test_model = EzvizModelBackend()
test_model.apollo_callback('hw_length:40,hw_num:1000,extra:{"hw_add":{"ai_voiceAssist_adultWatch":"萤石|血氧|血压|心电|心率|健走","ai_voiceAssist_rk3":"萤石|人脸录入|人脸跟踪|微聊|息屏|横屏|人形跟踪","ai_voiceAssist_smartApp":"萤石|设备|异声|录像|查看|执行|查询|视频锁|布防|撤防","ai_voiceAssist_pm6":"回充|回去充电|逗宠|跟随|z字前进|十字运动|恰恰|扭动|摇摆|跳舞|疯狂舞者|出击|冲击|扭扭扭|转圈|旋转|八字运动|z字形移动|走个猫步","ai_voiceAssist_s10":"萤石|息屏","ai_voiceAssist_y5000":"萤石|掌静脉|密码|指纹|人脸|指静脉|虹膜|感应卡|录入|姥爷|姥姥","ai_voiceAssist_tianji":"萤石|单曲循环"},"vad_info":{"vad_end":"800","vad_len":"15000","vad_eable":"ai_voiceAssist_adultWatch|ai_voiceAssist_rk3|ai_voiceAssist_c6c"},"hw_score":0.5,"hw_score_split":{"ai_voiceAssist_y5000":1.0},"sv_eable":"ai_voiceAssist_rk3|ai_voiceAssist_s10","kws_eable":"ai_voiceAssist_rk3|ai_voiceAssist_s10","kws_thres":"0.3|9.0"}')

stream_process_partern ={
    "version": protocol_ver,
    "request_id":"uuid",
    # "statistic": {'biz_key': 'voice-assist', 'user_id': '7eb9cbdf3d844ebaa0db2b48900f930d'},
    "task_type":"asr",
    "sequence_index":0,
    "sequence_end":False,
 
    "data_info":[{
        "modal":"audio",
        "type":"base64",
        "data":" "
    }]
}

test_audio_path = Path(r"/data1/wangsheng30/main/dataset/kws/train_data/train_wav")
fout = open("/data1/wangsheng30/main/dataset/s10_kws_sv_test/test_res/tmp", "w")
# test case
test_cases=[_ for _ in test_audio_path.glob('*.wav')]

audio_wakeup_length = 54400
num = len(test_cases)
worker = 30

for jj in range(1):
    start = time()
    for index, audio_p in enumerate(test_cases):
        logger.debug(f"audio_p:{audio_p}")
        
        request_id = str(uuid4())
        if index == 1:
            stream_connect_partern = {
                "version":protocol_ver,
                "request_id":"uuid",
                "task_type":"asr",
                "statistic": {'biz_key': 'ai_voiceAssist_s10', 'user_id': '7eb9cbdf3d844ebaa0db2b48900f930d'},
                "data_params":[{
                    "modal": "audio", 
                    # "format": "",
                    "hotwords": ["萤石"],
                    "wakeup_audio_length": audio_wakeup_length
                }]
            }
        else:
            stream_connect_partern = {
                "version":protocol_ver,
                "request_id":"uuid",
                "task_type":"asr",
                "statistic": {'biz_key': 'ai_voiceAssist_s10', 'user_id': '7eb9cbdf3d844ebaa0db2b48900f930d'},
                "data_params":[{
                    "modal": "audio", 
                    # "format": "pcm",
                    "audio_wakeup_type":0,
                    "hotwords": ["窗帘1", "窗帘2"],
                    "wakeup_audio_length": audio_wakeup_length
                }]
            }

        stream_connect_partern["request_id"] = request_id
        rqs_info = CommonRequest(**stream_connect_partern)
        rsp_msg = CommonResponse(request_id=rqs_info.request_id)
        
        rsp_msg = test_model.stream_init(rqs_info, rsp_msg)
        
        if rsp_msg.rest_code != 200:
            raise TypeError(f"error:{rsp_msg.error_msg}")
        
        stream_process_partern["request_id"] = request_id
        
        start_idx = 0
        idx = 0
        
        with audio_p.open("rb") as f:
            data = f.read(44)
            data = f.read()

        audio_wakeup = data[:audio_wakeup_length]
        data = data[audio_wakeup_length:]

        speech_length = len(data)
        chunk_size = [8, 8, 4] # [5, 10, 5]
        stride_size = 1024 * 4 # 注意一个chunk的累计字节，防止掉帧
        chunk_num = (speech_length-1)//stride_size + 1

        for idx in range(0, chunk_num):
            
            beg = idx*stride_size
            if idx == chunk_num-1:
                is_final = True
            else:
                is_final = False
                
            data_bin = data[beg:beg+stride_size if beg+stride_size < speech_length else speech_length]
            if idx == 0: data_bin = audio_wakeup + data_bin
            base64_data = str(base64.b64encode(data_bin), 'utf-8')

            # raise TypeError(f"error:{rsp_msg.error_msg}")
            
            # request json set
            stream_process_partern["sequence_index"] = idx
            stream_process_partern["data_info"][0]["data"] = base64_data
            stream_process_partern["sequence_end"] = is_final
            rqs_info = CommonRequest(**stream_process_partern)
            rsp_msg = CommonResponse(request_id=rqs_info.request_id)
            rsp_msg.success()
            st = time()
            for rsp_msg in test_model.stream_forward(rqs_info, rsp_msg):
                if rsp_msg.rest_code != 200:
                    raise TypeError(f"error:{rsp_msg.error_msg}")
                # print(f"{rsp_msg.model_dump(exclude_unset=True)}")
            idx += 1
            ed = time()
            print(f"this stream_process time cost {float(ed-st)}")
            if rsp_msg.stage_flag == 1:
                break
        
        res = rsp_msg.model_dump(exclude_unset=True)
        key = str(audio_p).split("/")[-1][:-4]
        if 'completion' in res:
            fout.write(key+" "+res['completion'][0]['data']+"\n")
        else:
            fout.write(key+" \n")
    end = time()
    print(f"casts num:{num} -> each case cost time:{(end -start)/num:.3f} s")

fout.close()