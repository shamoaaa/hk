import random
from uuid import uuid4
from pathlib import Path
import os
from json import dumps, loads
from requests import post
from time import time

from ms_manager.common import CommonRequest, CommonResponse, DataInfo, DataParams, protocol_ver, file2str, port_manager

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

root_url =f"http://0.0.0.0:{port_manager.FLASK_port}"
process_url = root_url+"/process"

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

test_audio_path = Path(r"test/test_data")
# test case
test_cases=[_ for _ in test_audio_path.glob('*.wav')]


audio_format = "aac"
cost_time = []

for audio_p in test_cases[:1]:
    rsq_info = CommonRequest(
        version= protocol_ver,
        request_id= str(uuid4()),
        task_type="asr"
    )
    
    rsq_info.data_info = []
    data_info = DataInfo(modal="audio", type="base64", data=file2str(audio_p))
    rsq_info.data_info.append(data_info)

    rsq_info.data_params = []
    rsq_info.data_params.append(
        DataParams(
            modal="audio", 
            hotwords=["萤石", "其实", "经鸡"]
        )
    )
    # process
    #for i in range(100):
    start_time = time()
    rsp_msg = run(process_url, rsq_info.model_dump(exclude_unset=True))
    print(rsp_msg.model_dump(exclude_unset=True))
    cost_time.append(time() - start_time)
    print(f"cost time:{time() - start_time:.3f}s")
    print(f">>>>>>>>> ")