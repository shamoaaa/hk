
import os
from pathlib import Path

def find_wav_files(directory):
    test_cases=[_ for _ in Path(directory).rglob('*.wav')]
    return test_cases

text_list = []
with open("/data1/asr_offline/label_s10_1215_1231.txt") as f:
    for line in f.readlines():
        if line.split()[0].strip() in  text_list:
            print('[WARN]', line.split()[0].strip())
        else:
            text_list.append(line.split()[0].strip())

in_path = "/dataset/qiannengfeng/wangsheng30/s10_data_reflow/wav/source/20241215_20250115/"
wav_flies = find_wav_files(in_path)
with open('/data1/asr_offline/data/list/wav_s10_20241215_20241231.scp', "w") as f:
    for wav in wav_flies:
        if str(wav.name).replace(".wav", "") in text_list:
            f.write(str(wav.name).replace(".wav", "") + " " + str(wav) +"\n")
            text_list.remove(str(wav.name).replace(".wav", ""))

print(text_list)