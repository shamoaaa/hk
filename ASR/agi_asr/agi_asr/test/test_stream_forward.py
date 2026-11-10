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

stream_connect_partern = {
    "version":protocol_ver,
    "request_id":"uuid",
    "task_type":"asr",
    "data_params":[{
        "modal": "audio", 
        "hotwords": ["萤石", "其实"]
    }]
}

stream_process_partern ={
    "version": protocol_ver,
    "request_id":"uuid",
    "task_type":"asr",
    "sequence_index":0,
    "sequence_end":False,
 
    "data_info":[{
        "modal":"audio",
        "type":"base64",
        "data":" "
    }]
}

test_audio_path = Path(r"test/test_data")
# test case
test_cases=[_ for _ in test_audio_path.glob('*.wav')]

num = len(test_cases)
worker = 30


start = time()
for index, audio_p in enumerate(test_cases):
    logger.debug(f"audio_p:{audio_p}")
    
    request_id = str(uuid4())
    
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


    speech_length = len(data)
    chunk_size = [8, 8, 4] # [5, 10, 5]
    stride_size = int(60*chunk_size[1]/1000*16000*0.8) # 注意一个chunk的累计字节，防止掉帧
    chunk_num = (speech_length-1)//stride_size + 1

    for idx in range(0, chunk_num):
        
        beg = idx*stride_size
        if idx == chunk_num-1:
            is_final = True
        else:
            is_final = False
            
        data_bin = data[beg:beg+stride_size if beg+stride_size < speech_length else speech_length]
        base64_data = str(base64.b64encode(data_bin), 'utf-8')

        # raise TypeError(f"error:{rsp_msg.error_msg}")
        
        # request json set
        stream_process_partern["sequence_index"] = idx
        stream_process_partern["data_info"][0]["data"] = base64_data
        stream_process_partern["sequence_end"] = is_final
        rqs_info = CommonRequest(**stream_process_partern)
        rsp_msg = CommonResponse(request_id=rqs_info.request_id)
        rsp_msg.success()
        
        for rsp_msg in test_model.stream_forward(rqs_info, rsp_msg):
            if rsp_msg.rest_code != 200:
                raise TypeError(f"error:{rsp_msg.error_msg}")
            print(f"{rsp_msg.model_dump(exclude_unset=True)}") if rsp_msg.stage_flag else None
        idx += 1
   
end = time()
print(f"casts num:{num} -> each case cost time:{(end -start)/num:.3f} s")
