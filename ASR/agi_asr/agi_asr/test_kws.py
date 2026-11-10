import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import time
from asr.wenet.bin.kws_inference_conformer import inference as kws_infer_offline
from ms_common.configs import logger, model_config, MS_WEIGHTS

logger.change_log_level("ERROR")

kws_infer_model = kws_infer_offline(model_dir=str(MS_WEIGHTS / "kws_model"))

with open("/data/home/wangsheng30/main/examples/kws/mianban_20221110_1.pcm", "rb") as f:
    data = f.read()

step = int(60*16000/1000*2)
duration = int(1.7*16000*2)
length = len(data)
i = 0
TP, FP = 0, 0
while(i<=length-step):
    wakeup_audio_data = data[i:i+duration]
    if len(wakeup_audio_data) < duration - 100: break
    st = time.time()
    kws_refilter_res = kws_infer_model(wakeup_audio_data)
    et = time.time()
    # print(f"time cost is {float(et-st)}")
    if kws_refilter_res: 
        TP += 1 
        print(f"************{i}")
    else: FP += 1
    i += step

print(f"TP is {TP}, FP is {FP}")