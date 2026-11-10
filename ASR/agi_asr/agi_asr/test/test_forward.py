import random
from pathlib import Path
from uuid import uuid4
import os
from time import time

from ms_common.configs import logger, model_config, MS_WEIGHTS
from ms_common.protocol import CommonRequest, CommonResponse, DataInfo, DataParams, protocol_ver
from ms_common.utils import file2str

from asr.ezviz_backend import EzvizModelBackend

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
test_model = EzvizModelBackend()

test_audio_path = Path(r"test/test_data")
# test case
test_cases=[_ for _ in test_audio_path.glob('*.wav')]

cost_time = []

for audio_p in test_cases:
    logger.debug(f"test audio_p:{audio_p}")
    rsq_info = CommonRequest(
        version= protocol_ver,
        request_id= str(uuid4()),
        task_type="asr"
    )
    rsp_msg = CommonResponse(
        version=protocol_ver,
        request_id=rsq_info.request_id
    )
    rsp_msg.success()
    
    rsq_info.data_info = []
    if audio_p.suffix == '.pcm':
        data_info = DataInfo(modal="audio", type="base64", data=file2str(audio_p))
    else:
        data_info = DataInfo(modal="audio", type="url", data=str(audio_p))
    rsq_info.data_info.append(data_info)

    rsq_info.data_params.append(
        DataParams(
            modal="audio", 
            hotwords=["萤石", "其实"]
        )
    )

    # process
    #for i in range(100):
    start_time = time()
    rsp_msg = test_model.forward(rsq_info, rsp_msg)
    print(rsp_msg.model_dump(exclude_unset=True))
    cost_time.append(time() - start_time)
    print(f"cost time:{time() - start_time:.3f}s")
    print(f">>>>>>>>> ")