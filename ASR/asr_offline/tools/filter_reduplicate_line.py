
text_list = []
with open("/data1/asr_offline/data/list/wav_s10_20241215_20250115.scp") as f:
    for line in f.readlines():
        if line.split()[0].strip() in  text_list:
            print('[WARN]', line.split()[0].strip())
        else:
            text_list.append(line.split()[0].strip())