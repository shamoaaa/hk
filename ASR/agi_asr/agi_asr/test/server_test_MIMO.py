import random
from uuid import uuid4
import os
from json import dumps, loads
from requests import post,get
from time import time, sleep
from pathlib import Path
import base64
import sys

from ms_manager.common import CommonRequest, CommonResponse, DataInfo, OutputParams, protocol_ver,model_config, port_manager, file2str

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

root_url =f"http://0.0.0.0:{port_manager.FLASK_port}"
connect_url = root_url+"/stream_connect"
process_url = root_url+"/stream_process"
cancel_url = root_url+"/stream_cancel"

stream_connect_partern = {
    "version":protocol_ver,
    "request_id":"uuid",
    "task_type":"asr",
    "data_params":[{
        "modal": "audio", 
        "hotwords": ["萤石", "其实", "经鸡"]
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

stream_cancel_partern ={
    "version":"20230713",
    "task_type":"asr",
    "request_id":"uuid",
}

def run(url, rqs_info):
    '''
    模拟请求服务方法信息
    :return:
    '''
    
    # print("request_info:")
    # print(request_info)
    respnse = post(url, json=rqs_info)
    rsp_msg =  CommonResponse(**loads(respnse.text))
    print(rsp_msg)
    return rsp_msg

# test case
test_audio_path = Path(r"test/test_data")
# test case
test_cases=[_ for _ in test_audio_path.glob('*.wav')][:1]


num = len(test_cases)
cost_time = []
start = time()
for index, audio_p in enumerate(test_cases):
    # request_id = "3d2ced78-5234-11ee-85be-001b576031b0"
    # stream_cancel_partern["request_id"] = request_id
    # rsp_msg = run(cancel_url, stream_cancel_partern)
    # if rsp_msg.rest_code != 200:
    #     raise TypeError(f"error:{rsp_msg.error_msg}")
    print(f"test audio:{audio_p.name}")
    request_id = str(uuid4())
    
    stream_connect_partern["request_id"] = request_id
    rqs_json = stream_connect_partern
    
    rsp_msg = run(connect_url, rqs_json)
    
    if rsp_msg.rest_code != 200:
        raise TypeError(f"error:{rsp_msg.error_msg}")
    
    stream_process_partern["request_id"] = request_id
    
    context = file2str(audio_p)
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

        # request json set
        stream_process_partern["sequence_index"] = idx
        stream_process_partern["data_info"][0]["data"] = base64_data
        stream_process_partern["sequence_end"] = is_final
        
        rsp_msg = run(process_url, stream_process_partern)
        
        if rsp_msg.rest_code != 200:
            raise TypeError(f"error:{rsp_msg.error_msg}")
        
        # if index != len(test_cases) - 1:
        #     if random.random() < 0.1:
        #         print("not end case, try cancel now:{get(root_url+'/tmp')}")
        #         sleep(1)
        #         print("try return normal status:{get(root_url+'/tmp')}")
        #         break
                # sys.exit(-1)

    sleep(2)
   
end = time()
print(f"casts num:{num} -> each case cost time:{(end -start)/num:.3f} s")
