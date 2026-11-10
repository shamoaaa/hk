import numpy as np


def find_peaks_for_breathe(sig_select, span):
    sig_len = len(sig_select)
    mid = span
    peaks_list = []
    while mid < sig_len - span:
        left = max(0, mid - span)
        right = min(sig_len, mid + span)

        max_ind = np.argmax(sig_select[left:right])
        if left + max_ind == mid:
            peaks_list.append(mid)
            mid += span
        else:
            mid += 1
    return peaks_list


def linear_correlation_succeed(x, max_freq):
    if len(x) < 10:
        return False
    x_arr = np.asarray(x)
    N = len(x_arr)  
    y = np.linspace(x_arr[0], x_arr[-1], N)
    corr_coef = np.corrcoef(x_arr, y)[0, 1]
    
    if max_freq < 0.15:
        thr = 0.5
    elif max_freq < 0.25:
        thr = 0.6
    elif max_freq < 0.35:
        thr = 0.7
    else:
        thr = 0.8
    
    if corr_coef > thr:
        return True
    else:
        return False
            

def calculate_min_value_thr(sig):
    arr = np.asarray(sig)
    segments = np.array_split(arr, 6)
    ranges = [np.max(seg) - np.min(seg) for seg in segments]
    min_range = np.min(ranges)
    target_value = min_range / 3    
    return target_value


def find_onsets_for_breathe(sig_select, peaks_list, max_freq):
    min_value_thr = calculate_min_value_thr(sig_select)
    peaks_cnt = len(peaks_list)
    onsets_list = -np.ones(peaks_cnt)
    for i in range(peaks_cnt):
        if i == 0:
            inds = 0
        else:
            inds = peaks_list[i-1] + 1
        target_sig = sig_select[inds: peaks_list[i]-1]
        min_value = np.min(target_sig)
        min_idx = np.argmin(target_sig)
        flag1 = min_value < 0
        flag2 = sig_select[peaks_list[i]] > 0
        flag3 = sig_select[peaks_list[i]] - min_value > min_value_thr
        flag4 = linear_correlation_succeed(sig_select[min_idx + inds: peaks_list[i]-1], max_freq)
        
        if flag1 and flag2 and flag3 and flag4:
            onsets_list[i] = min_idx + inds
    new_peaks_list = []
    new_onsets_list = []
    for peak, onset in zip(peaks_list, onsets_list):
        if onset > -0.5:
            new_peaks_list.append(int(peak))
            new_onsets_list.append(int(onset))
    return new_peaks_list, new_onsets_list
    

def find_sorted_valleys(sig, half_win):
    data_len = len(sig)
    mid = half_win
    valleys = []
    while (mid < data_len - half_win):
        left = max(0, mid - half_win)
        right = min(data_len, mid + half_win)

        min_ind = np.argmin(sig[left:right])
        if left + min_ind == mid:
            valleys.append(mid)
            mid += half_win
        else:
            mid += 1
    
    sorted_valleys = sorted(valleys, key=lambda x: sig[x])
    return sorted_valleys, len(sorted_valleys)


def find_onsets2_for_breathe(sig, onsets_list, peaks_list, fs, max_freq):
    peaks_cnt = len(peaks_list)
    onsets2_list = -np.ones(peaks_cnt)
    sig_len = len(sig)  
    for i in range(peaks_cnt):
        left_val = sig[onsets_list[i]]
        peak_val = sig[peaks_list[i]]
        left_height = peak_val - left_val
        inds = peaks_list[i]
        if i != peaks_cnt - 1:
            inde = peaks_list[i+1]
        else:
            inde = sig_len
        
        [valleys, valleys_cnt] = find_sorted_valleys(sig[inds:inde], 5)
        if valleys_cnt == 0:
            continue
        valleys = [item + inds for item in valleys]
        
        for j in range(valleys_cnt):    
            hr = 60 * fs / (valleys[j] - onsets_list[i])
            if hr < 6 or hr > 90:
                continue
            if not linear_correlation_succeed(sig[peaks_list[i]: valleys[j]], max_freq):
                continue
            right_val = sig[valleys[j]]
            right_height = peak_val - right_val
            left_right_ratio = left_height / right_height
            right_left_ratio = 1 / left_right_ratio
            ratio_thr = 0.6
            if ((left_right_ratio > ratio_thr and left_right_ratio < 1/ratio_thr) or (right_left_ratio > ratio_thr and right_left_ratio < 1/ratio_thr)):
                onsets2_list[i] = int(valleys[j])
                break
    new_peak_list = []
    new_onset_list = []
    new_onset2_list = []
    hr_list = []
    height_list = []
    for i in range(peaks_cnt):
        if onsets2_list[i] > 0:
            new_peak_list.append(peaks_list[i])
            new_onset_list.append(onsets_list[i])
            new_onset2_list.append(int(onsets2_list[i]))
            hr = 60 * fs / (onsets2_list[i] - onsets_list[i])
            hr_list.append(hr)
            height = (2 * sig[peaks_list[i]] - sig[onsets_list[i]] - sig[int(onsets2_list[i])])/2
            height_list.append(height)

    return new_peak_list, new_onset_list, new_onset2_list, hr_list, height_list


def pseudo_online_normalize(input_data):
    input_data = np.asarray(input_data, dtype=np.float32)
    output_data = np.zeros_like(input_data)
    nevelop = 0.0
    for i in range(input_data.shape[0]):
        rawdata_count = i + 1
        if rawdata_count < 25 * 1:
            scale = 0.01
        elif rawdata_count < 25 * 2:
            scale = 0.2
        elif rawdata_count < 25 * 3:
            scale = 0.4
        elif rawdata_count < 25 * 4:
            scale = 0.6
        else:
            scale = 0.7
        Signal_noise = input_data[i]
        if i == 0:
            nevelop = 2 * Signal_noise ** 2
        else:
            nevelop = 2 * Signal_noise ** 2 * (1 - scale) + nevelop * scale
            output_data[i] = Signal_noise / (np.sqrt(nevelop + 2.2204e-16))
    return output_data


def find_max_fft_frequency(signal, fs):
    N = len(signal)
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    fft_ampl = np.abs(fft_vals[:len(freqs)])
    fft_ampl[0] = 0
    idx = np.argmax(fft_ampl)
    max_freq = freqs[idx]
    max_ampl = fft_ampl[idx]
    return max_freq, max_ampl, freqs, fft_ampl


def calc_sig_span(sig, fs):
    
    normal_sig = pseudo_online_normalize(sig)
    max_freq, max_ampl, freqs, fft_ampl = find_max_fft_frequency(normal_sig, fs)
    
    # import matplotlib.pyplot as plt
    # plt.subplot(311)
    # plt.plot(sig)
    # plt.subplot(312)
    # plt.plot(normal_sig)
    # plt.subplot(313)
    # plt.plot(freqs, fft_ampl)
    # plt.scatter([max_freq], [max_ampl])
    # plt.title("maxf:{}".format(max_freq))
    # plt.show()
    
    if max_freq < 0.08 or max_freq > 1.25:
        return 40, max_freq
    elif max_freq < 0.15:
        return int(0.65*fs/max_freq), max_freq
    elif max_freq < 0.20:
        return int(0.55*fs/max_freq), max_freq
    elif max_freq < 0.25:
        return int(0.40*fs/max_freq), max_freq
    else:
        return int(0.35*fs/max_freq), max_freq
    

def calc_breath(sig, fs=20, give_detail=False):   
    
    span, max_freq = calc_sig_span(sig, fs)
    peaks_list = find_peaks_for_breathe(sig, span)
    if len(peaks_list) == 0:
        return [], [], [], [], [], -1.0
    new_peaks_list, new_onsets_list = find_onsets_for_breathe(sig, peaks_list, max_freq)
    if len(new_peaks_list) == 0:
        return [], [], [], [], [], -1.0
    final_peak_list, final_onset_list, final_onset2_list, hr_list, height_list = find_onsets2_for_breathe(sig, new_onsets_list, new_peaks_list, fs, max_freq)
    if len(final_peak_list) < 4:
        return [], [], [], [], [], -1.0
    
    res_peak = []
    res_onset = []
    res_onset2 = []
    res_hr = []
    res_height = []
    
    mid_height = np.median(height_list)
    
    for (peak, onset, onset2, hr, height) in zip(final_peak_list, final_onset_list, final_onset2_list, hr_list, height_list):
        height_ratio = height/mid_height
        if height_ratio > 2 or height_ratio < 0.5:
            pass
        else:
            res_peak.append(peak)
            res_onset.append(onset)
            res_onset2.append(onset2)
            res_hr.append(hr)
            res_height.append(height)
    
    rr_list = []
    for (peak, onset, onset2, hr, height) in zip(res_peak, res_onset, res_onset2, res_hr, res_height):
        rr_list.append(onset2 - onset)
    
    br = -1.0
    if np.sum(rr_list) > len(sig)/3:
        br = 60*fs/np.median(rr_list)
    
    return res_peak, res_onset, res_onset2, res_hr, res_height, br
