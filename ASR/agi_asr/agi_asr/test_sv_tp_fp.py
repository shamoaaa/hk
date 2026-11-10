
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import pyloudnorm as pyln
from asr.speakerlab.bin.inference import speaker_verification
from ms_common.configs import logger, model_config, MS_WEIGHTS
from pathlib import Path

logger.change_log_level("ERROR")

loudness_meter = pyln.Meter(16000)
sv_model = speaker_verification(device="cuda:0", model_path=str(MS_WEIGHTS / "sv_model/pretrained_eres2netv2.ckpt"))
st = time.time()
sv_model.inference(bytes(54400), bytes(54400))
et = time.time()
print(float(et-st))

TP, FP = 0, 0
test_audio_path = Path(r"/data1/wangsheng30/main/dataset/s10_kws_sv_test/tp_1")
# test case
test_cases=[_ for _ in test_audio_path.glob('*.wav')]

for audio_p in test_cases:
    # logger.debug(f"audio_p:{audio_p}")
    with audio_p.open("rb") as f:
        data = f.read(44)
        data = f.read()

    audio_wakeup = data[:54400]
    
    st = time.time()
    wav_wakeup, wav_cmd, sv_result = sv_model.inference(audio_wakeup, data[54400:])
    et = time.time()
    print(float(et-st))
    if sv_result <= 0.3:
        # if wav_wakeup.shape[0] > 8000 and wav_cmd.shape[0] > 8000: # 超过 block_size 400ms
        #     loudness_wakeup = loudness_meter.integrated_loudness(wav_wakeup)
        #     loudness_cmd = loudness_meter.integrated_loudness(wav_cmd)
        #     loudness_diff = loudness_wakeup - loudness_cmd
        #     if loudness_diff >= 9.0: 
                FP += 1 
                logger.debug(f"audio_p:{audio_p}")
    else: 
        TP += 1
        # logger.debug(f"audio_p:{audio_p}")
    
logger.debug(f"TP is {TP}, FP is {FP}")