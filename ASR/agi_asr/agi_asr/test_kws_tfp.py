import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import time
from asr.wenet.bin.kws_inference_conformer import inference as kws_infer_offline
from ms_common.configs import logger, model_config, MS_WEIGHTS
from pathlib import Path

logger.change_log_level("ERROR")

kws_infer_model = kws_infer_offline(model_dir=str(MS_WEIGHTS / "kws_model"))
st = time.time()
kws_infer_model(bytes(54400))
et = time.time()
print(float(et-st))

TP, FP = 0, 0
test_audio_path = Path(r"/data1/wangsheng30/main/dataset/s10_kws_sv_test/tp")
# test case
test_cases=[_ for _ in test_audio_path.glob('*.wav')]

for audio_p in test_cases:
    # logger.debug(f"audio_p:{audio_p}")
    with audio_p.open("rb") as f:
        data = f.read(44)
        data = f.read()

    audio_wakeup = data[:54400]

    st = time.time()
    kws_refilter_res = kws_infer_model(audio_wakeup)
    et = time.time()
    print(float(et-st))
    if kws_refilter_res: 
        TP += 1 
        # logger.debug(f"audio_p:{audio_p}")
    else: 
        FP += 1
        logger.debug(f"audio_p:{audio_p}")
    
logger.debug(f"TP is {TP}, FP is {FP}")