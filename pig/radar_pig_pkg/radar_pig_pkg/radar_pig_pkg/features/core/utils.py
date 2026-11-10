import numpy as np
from scipy.signal import butter, zpk2sos, detrend, sosfiltfilt, firwin, filtfilt


def butter_bandpass(lowcut, highcut, fs=20, order=8):
    z, p, k = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='zpk')
    sos = zpk2sos(z, p, k)
    return butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')


def butter_bandpass2(signal, sample_rate=20, low_cut=0.15, high_cut=2.5):
    nyq = sample_rate * 0.5
    b, a = butter(4, ([low_cut/nyq, high_cut/nyq]), btype='band', analog=False)
    amp = filtfilt(b, a, signal, axis=0)
    return amp


def butter_lowpass(lowcut, fs=20, order=8):
    z, p, k = butter(order, lowcut, btype='lowpass', fs=fs, output='zpk')
    sos = zpk2sos(z, p, k)
    return sos


def dual_bandpass_filter(signal, fs, low_cutoff_1, low_cutoff_2, order=61):
    nyq = 0.5 * fs  # 奈奎斯特频率

    # 设计第一个低通滤波器 (0.1Hz)
    coeff_1 = firwin(
        order + 1,
        low_cutoff_1 / nyq,  # 归一化截止频率
        window='hamming',
        pass_zero='lowpass'
    )

    # 设计第二个低通滤波器 (1.25Hz)
    coeff_2 = firwin(
        order + 1,
        low_cutoff_2 / nyq,  # 归一化截止频率
        window='hamming',
        pass_zero='lowpass'
    )

    # 零相位滤波
    filtered_1 = filtfilt(coeff_1, 1.0, signal)  # 0.1Hz低通结果
    filtered_2 = filtfilt(coeff_2, 1.0, signal)  # 1.25Hz低通结果

    # 逐点相减：1.25Hz结果 - 0.1Hz结果
    return filtered_2 - filtered_1


def br_data_filter2(phase_data, sample_rate=20):
    nyq = 0.5 * sample_rate
    order = 60

    # 设计第一个低通滤波器 (0.1Hz)
    coeff_1 = firwin(
        order + 1,
        0.1 / nyq,  # 归一化截止频率
        window='hamming',
        pass_zero='lowpass'
    )

    b, a = butter(4, (2.5/nyq), btype='low', analog=False)

    # 零相位滤波
    filtered_1 = filtfilt(coeff_1, 1.0, phase_data)  # 0.1Hz低通结果
    filtered_2 = filtfilt(b, a, phase_data)  # 1.25Hz低通结果

    filter_phase = filtered_2 - filtered_1

    reshape_phase = energy_ratio_windowing(filter_phase, window_size=20, energy_threshold=0.5)

    return reshape_phase


def energy_ratio_windowing(x, window_size=101, energy_threshold=2.0):
    """
    基于能量比值的动态加窗

    参数:
        x: 输入信号
        window_size: 分析窗口大小(奇数)
        energy_threshold: 能量比值阈值

    返回:
        加窗后的信号
    """
    # 确保窗口大小为奇数
    if window_size % 2 == 0:
        window_size += 1

    # 计算全局能量
    global_energy = np.sum(x ** 2)

    half_window = window_size // 2
    padded_x = np.pad(x, (half_window, half_window), mode='reflect')
    local_energy = np.zeros_like(x)

    for i in range(len(x)):
        window_data = padded_x[i:i + window_size]
        local_energy[i] = np.sum(window_data ** 2) / window_size

    # 计算能量比值
    avg_local_energy = np.mean(local_energy)
    energy_ratio = local_energy / avg_local_energy

    # 创建动态窗口函数
    window = np.ones_like(x)
    for i in range(len(x)):
        if energy_ratio[i] > energy_threshold:
            # 在高能量区域应用衰减窗口
            window[i] = energy_threshold / energy_ratio[i]

    # return np.multiply(x * window, 2)
    return x * window


def ampd(x, scale=None, debug=False):
    x = detrend(x)
    N = len(x)
    L = N // 2
    if scale:
        L = min(scale, L)

    # create LSM matix
    LSM = np.ones((L, N), dtype=bool)
    for k in np.arange(1, L + 1):
        LSM[k - 1, 0:N - k] &= (x[0:N - k] > x[k:N]
                                )  # compare to right neighbours
        LSM[k - 1, k:N] &= (x[k:N] > x[0:N - k])  # compare to left neighbours

    # Find scale with most maxima
    G = LSM.sum(axis=1)
    G = G * np.arange(
        N // 2, N // 2 - L, -1
    )  # normalize to adjust for new edge regions
    l_scale = np.argmax(G)

    if l_scale == 0:
        return np.array([-1])

    # find peaks that persist on all scales up to l
    pks_logical = np.min(LSM[0:l_scale, :], axis=0)
    pks = np.flatnonzero(pks_logical)
    if debug:
        return pks, LSM, G, l_scale
    return pks


def get_amp(obj, input_data, i_frame):
    weight = 0.95

    if i_frame == 0:
        obj.weight_data = input_data
        diff_data = np.ones_like(input_data)
    else:
        if i_frame == 1:
            diff_data = input_data - obj.weight_data
            obj.weight_data = 0.5 * input_data + 0.5 * obj.weight_data
        else:
            diff_data = input_data - obj.weight_data
            obj.weight_data = (1 - weight) * input_data + weight * obj.weight_data

    diff_data = np.abs(diff_data)

    return diff_data


def get_freq_by_pks(pks):
    rr_list = np.diff(pks)
    return 60*20/np.median(rr_list)


def calculate_envelope(signal, window_size):
    envelope_low = np.zeros_like(signal)
    envelope_high = np.zeros_like(signal)
    half_window = window_size // 2

    for i in range(len(signal)):
        start = max(0, i - half_window)
        end = min(len(signal), i + half_window)
        envelope_low[i] = np.min(signal[start:end])
        envelope_high[i] = np.max(signal[start:end])

    return envelope_low, envelope_high



def row_normalize(matrix):
    """行归一化：每行的值缩放到0-1之间"""
    row_mins = matrix.min(axis=1, keepdims=True)
    row_maxs = matrix.max(axis=1, keepdims=True)
    return (matrix - row_mins) / (row_maxs - row_mins + 1e-10)  # 加小量防止除以

