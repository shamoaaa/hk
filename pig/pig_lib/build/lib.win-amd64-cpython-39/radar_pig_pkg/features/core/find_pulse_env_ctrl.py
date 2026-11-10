import numpy as np

from .utils import calculate_envelope, ampd, get_freq_by_pks
from sklearn.cluster import KMeans


def find_peaks_for_breathe(sig_select, span):
    # 寻峰可以优化
    # 找主频然后确认窗口大小，对于呼吸速率变化的情况，兼容效果不好
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


def linear_correlation_succeed(x, max_freq, span):
    # 是否需要根据窗口大小进行调整？
    # 采样率20Hz，过滤波峰小于0.5秒的
    # 可以根据滤波器设置阈值
    if len(x) < span:
        return False

    # 对于滤波效果不好、周期性不好的，这条会过不了
    x_arr = np.asarray(x)
    N = len(x_arr)
    y = np.linspace(x_arr[0], x_arr[-1], N)
    corr_coef = np.corrcoef(x_arr, y)[0, 1]

    if max_freq < 0.15:
        thr = 0.8
    elif max_freq < 0.25:
        thr = 0.7
    elif max_freq < 0.35:
        thr = 0.6
    else:
        thr = 0.5

    if abs(corr_coef) >= thr:
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


def find_onsets_for_breathe(sig_select, peaks_list, max_freq, span):
    min_value_thr = calculate_min_value_thr(sig_select)  # 设置波峰波谷差值的阈值
    peaks_cnt = len(peaks_list)
    onsets_list = -np.ones(peaks_cnt)
    height_list = []
    height_list_1 = np.ones(peaks_cnt)
    for i in range(peaks_cnt):
        if i == 0:
            inds = 0
        else:
            inds = peaks_list[i - 1] + 1
        target_sig = sig_select[inds: peaks_list[i] - 1]  # 截取两个波峰之间的数据
        if len(target_sig) == 0:
            continue

        min_value = np.min(target_sig)
        min_idx = np.argmin(target_sig)

        flag1 = min_value < 0
        flag2 = sig_select[peaks_list[i]] > 0
        flag3 = sig_select[peaks_list[i]] - min_value > min_value_thr
        flag4 = linear_correlation_succeed(sig_select[min_idx + inds: peaks_list[i] - 1], max_freq, span)

        # if flag1 and flag2 and flag3 and flag4:
        if flag1 and flag2 and flag3:
            onsets_list[i] = min_idx + inds  # 波谷的坐标
            height_list.append(sig_select[peaks_list[i]] - sig_select[min_idx + inds])  # 峰值与谷值幅度之差
            # height_list_1[i] = sig_select[peaks_list[i]]-sig_select[min_idx + inds]
    # height_median_value = np.median(height_list)
    new_peaks_list = []
    new_onsets_list = []
    new_height_list = []
    # height_median_value = np.median(height_list)
    # print(height_median_value)
    for peak, onset, height in zip(peaks_list, onsets_list, height_list_1):
        if onset > -0.5:
            # if abs(height-height_median_value) <= 2:
            new_peaks_list.append(int(peak))
            new_onsets_list.append(int(onset))
            # new_height_list.append(int(height))
    return new_peaks_list, new_onsets_list


def find_sorted_valleys(sig, half_win):
    data_len = len(sig)
    mid = half_win
    valleys = []
    while mid < data_len - half_win:
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


def find_onsets2_for_breathe(sig, onsets_list, peaks_list, fs, max_freq, span):
    peaks_cnt = len(peaks_list)
    onsets2_list = -np.ones(peaks_cnt)
    sig_len = len(sig)

    for i in range(peaks_cnt):
        left_val = sig[onsets_list[i]]
        peak_val = sig[peaks_list[i]]
        left_height = peak_val - left_val
        inds = peaks_list[i]
        if i != peaks_cnt - 1:
            inde = peaks_list[i + 1]
        else:
            inde = sig_len

        [valleys, valleys_cnt] = find_sorted_valleys(sig[inds:inde], 3)
        if valleys_cnt == 0:
            continue
        valleys = [item + inds for item in valleys]

        for j in range(valleys_cnt):
            hr = 60 * fs / (valleys[j] - onsets_list[i])
            # print(hr, end=',')
            if hr < 9 or hr > 150:  # 过滤间隔太大或太小
                continue
            if not linear_correlation_succeed(sig[peaks_list[i]: valleys[j]], max_freq, span):
                continue
            right_val = sig[valleys[j]]
            right_height = peak_val - right_val
            left_right_ratio = left_height / right_height
            right_left_ratio = 1 / left_right_ratio
            ratio_thr = 0.6
            if ((ratio_thr < left_right_ratio < 1 / ratio_thr) or (
                    ratio_thr < right_left_ratio < 1 / ratio_thr)):
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
            height = (2 * sig[peaks_list[i]] - sig[onsets_list[i]] - sig[int(onsets2_list[i])]) / 2
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
    freqs = np.fft.rfftfreq(N, d=1 / fs)
    fft_ampl = np.abs(fft_vals[:len(freqs)])
    fft_ampl[0] = 0
    idx = np.argmax(fft_ampl)
    max_freq = freqs[idx]
    max_ampl = fft_ampl[idx]
    return max_freq, max_ampl, freqs, fft_ampl


def calc_sig_span(sig, fs, min_index):
    normal_sig = pseudo_online_normalize(sig)
    max_freq, max_ampl, freqs, fft_ampl = find_max_fft_frequency(normal_sig, fs)

    if max_freq < 0.08:
        return 90, max_freq
    elif max_freq < 0.15:
        return int(0.65 * fs / max_freq), max_freq
    elif max_freq < 0.20:
        return int(0.55 * fs / max_freq), max_freq
    elif max_freq < 0.25:
        return int(0.40 * fs / max_freq), max_freq
    else:
        return int(0.35 * fs / max_freq), max_freq


def calc_breath(sig, act_mean, ideal_channel, min_index, fs=20, give_detail=False):
    # ampd在一些信号不好的地方，也会出值

    # 根据活动量
    if np.mean(act_mean[600:1200]) > 3e2:
        return [], [], [], [], [], -1.0
    # 根据通道
    elif ideal_channel <= 7:
        return [], [], [], [], [], -1.0
    # 根据呼吸
    else:
        # 幅度
        envelope_low, envelope_high = calculate_envelope(sig, 200)
        br_env = np.abs(envelope_high - envelope_low)
        br_amp = np.mean(br_env)

        if br_amp <= 0.1:
            return [], [], [], [], [], -1.0

    span, max_freq = calc_sig_span(sig, fs, min_index)
    peaks_list = find_peaks_for_breathe(sig, span)

    if len(peaks_list) == 0:
        pks = ampd(sig)
        return pks, [], [], [], [], get_freq_by_pks(pks)

    new_peaks_list, new_onsets_list = find_onsets_for_breathe(sig, peaks_list, max_freq, span)
    if len(new_peaks_list) == 0:
        pks = ampd(sig)
        return pks, [], [], [], [], get_freq_by_pks(pks)

    final_peak_list, final_onset_list, final_onset2_list, hr_list, height_list = find_onsets2_for_breathe(sig,
                                                                                                          new_onsets_list,
                                                                                                          new_peaks_list,
                                                                                                          fs, max_freq,
                                                                                                          span)

    # 还是这一部分就忽略了
    # 如果要忽略，就要优化提峰
    if len(final_peak_list) < 4:
        pks = ampd(sig)
        return pks, [], [], [], [], get_freq_by_pks(pks)

    res_peak = []
    res_onset = []
    res_onset2 = []
    res_hr = []
    res_height = []

    mid_height = np.median(height_list)

    for (peak, onset, onset2, hr, height) in zip(final_peak_list, final_onset_list, final_onset2_list, hr_list,
                                                 height_list):

        res_peak.append(peak)
        res_onset.append(onset)
        res_onset2.append(onset2)
        res_hr.append(hr)
        res_height.append(height)

    rr_list = []
    for (peak, onset, onset2, hr, height) in zip(res_peak, res_onset, res_onset2, res_hr, res_height):
        rr_list.append(onset2 - onset)

    # 尝试去掉限制
    if np.sum(rr_list) > len(sig) / 3:

        # 转换为numpy数组并reshape
        X = np.array(rr_list).reshape(-1, 1)

        # 尝试用K-Means分为两类
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_.flatten()

        # 计算两类中心点的距离
        center_distance = abs(centers[0] - centers[1])
        unique, counts = np.unique(labels, return_counts=True)  # 检查两类样本数量是否均衡（避免因极少数离群点误判）

        # 判断条件：两类中心距离远且两类样本数量均较多
        if center_distance > 50 and min(counts) > 0.4 * len(rr_list):  # 阈值可根据数据范围调且少数类至少占40%
            br = -1.0
        # 不满足分类条件则返回中值
        else:
            br = 60 * fs / np.median(rr_list)
    else:
        pks = ampd(sig)
        return pks, [], [], [], [], get_freq_by_pks(pks)

    return res_peak, res_onset, res_onset2, res_hr, res_height, br
