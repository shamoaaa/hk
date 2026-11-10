import logging
import os
import shutil
import pandas as pd
import h5py
import numpy as np
import scipy.signal as signal
from scipy.signal import firwin, filtfilt
from scipy.signal import butter, zpk2sos, detrend, firwin, lfilter
from datetime import datetime, timedelta

logger = logging.getLogger('__main__')


def get_angle(data):
    result = np.zeros_like(data, dtype=np.float32)
    for i, row in enumerate(data):
        for j, d in enumerate(row):
            result[i, j] = np.angle(d)

    return result


def butter_bandpass(lowcut, highcut, fs=20, order=8):
    z, p, k = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='zpk')
    sos = zpk2sos(z, p, k)
    return butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')


def butter_lowpass(lowcut, fs=20, order=8):
    z, p, k = butter(order, lowcut, btype='lowpass', fs=fs, output='zpk')
    sos = zpk2sos(z, p, k)
    return sos
def fir_bandpass_filter(data, sample_rate=20, lowcut=0.1, highcut=1.25, order=61):
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    b = signal.firwin(order, [low, high], pass_zero=False)
    filtered_data = signal.lfilter(b, 1.0, data)
    return filtered_data


def fir_lowpass_diff(signal_input, fs=20, cutoff=2.0, numtaps=21, diff_order=1):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    fir_coeff = signal.firwin(numtaps, normal_cutoff, window='hamming')
    filtered_signal = signal.lfilter(fir_coeff, 1.0, signal_input)
    if diff_order == 1:
        diff_signal = np.diff(filtered_signal)
    elif diff_order > 1:
        diff_signal = np.diff(filtered_signal, n=diff_order)
    else:
        diff_signal = filtered_signal.copy()
    return filtered_signal, diff_signal

def dual_lowpass_filter(signal, fs, low_cutoff_1, low_cutoff_2, order=61):
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

def calc_br_by_peaks(peaks):
    len_peaks = len(peaks)
    if len(peaks) < 2:
        return None, None
    RR = peaks[1] - peaks[0]
    cur_seq = [peaks[0]]
    final_seqs = []
    for i in range(1, len_peaks):
        new_RR = peaks[i] - peaks[i-1]
        if new_RR > 0.7 * RR and new_RR < 1.3 * RR:
            cur_seq.append(peaks[i])
            RR = new_RR
            if i == len_peaks - 1:
                final_seqs.append(cur_seq)
        else:
            final_seqs.append(cur_seq)
            cur_seq = [peaks[i]]
            RR = new_RR

    if len_peaks > 30:
        num_thr = 0.4
    elif len_peaks > 20:
        num_thr = 0.5
    elif len_peaks > 15:
        num_thr = 0.6
    elif len_peaks > 10:
        num_thr = 0.6
    else:
        num_thr = 0.6

    # print(final_seqs)
    sub_max = None
    max_len = -1
    for item in final_seqs:
        item_len = len(item)
        if item_len > len_peaks * num_thr:
            if item_len > max_len:
                sub_max = item
                max_len = item_len

    if sub_max is not None:
        # print(sub_max)
        secs = (sub_max[-1] - sub_max[0])/20
        periods = max_len - 1 
        br = periods * 60 / secs 
    else:
        br = None
        secs = None
    return br, secs




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


def row_normalize(matrix):
    """行归一化：每行的值缩放到0-1之间"""
    row_mins = matrix.min(axis=1, keepdims=True)
    row_maxs = matrix.max(axis=1, keepdims=True)
    return (matrix - row_mins) / (row_maxs - row_mins + 1e-10)  # 加小量防止除以


def make_mat_folder(mat_dir):
    files = [f.name for f in os.scandir(mat_dir)]

    for file in files:
        file_path = os.path.join(mat_dir, file)

        if os.path.isfile(file_path):  # 只处理文件
            filename = os.path.splitext(file)[0]  # 获取文件名（不包括扩展名）
            parts = filename.split('-')  # 按 '-' 分割文件名

            if len(parts) > 1:
                out_dir = os.path.join(mat_dir, parts[0], parts[1])
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)  # 创建嵌套文件夹

                shutil.move(file_path, os.path.join(out_dir, file))  # 移动文件


def get_amp(data, i_frame, weight_data):
    weight = 0.95

    if i_frame == 0:
        weight_data = data
        diff_data = np.ones_like(data)
    else:
        if i_frame == 1:
            diff_data = data - weight_data
            weight_data = 0.5 * data + 0.5 * weight_data
        else:
            diff_data = data - weight_data
            weight_data = (1 - weight) * data + weight * weight_data

    diff_data = np.abs(diff_data)

    return diff_data, weight_data


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


def fir_lowpass_envelope(signal, sample_rate, cutoff_freq, numtaps=101):
    """
    全波整流法获取单一包络线
    
    参数：
    - signal: 输入信号
    - sample_rate: 采样率 (Hz)
    - cutoff_freq: 截断频率 (Hz)
    - numtaps: FIR滤波器阶数（需为奇数）
    
    返回：
    - envelope: 包络线
    """
    # 设计FIR低通滤波器
    b = firwin(numtaps, cutoff_freq, fs=sample_rate, pass_zero='lowpass')
    
    # 全波整流（取绝对值）
    rectified = np.abs(signal)
    
    # 滤波处理
    envelope = lfilter(b, 1.0, rectified)
    
    # 补偿群延迟
    delay = (numtaps - 1) // 2
    envelope = np.roll(envelope, -delay)
    
    return envelope


def generate_date_strings(data_limit):
    # 解析输入的字符串，将其转换为 datetime 对象
    start_date = datetime.strptime(data_limit[0], '%Y%m%d')
    end_date = datetime.strptime(data_limit[1], '%Y%m%d')

    # 初始化结果数组和当前处理的日期
    date_strings = []
    current_date = start_date

    # 循环直到当前日期超过结束日期
    while current_date <= end_date:
        # 将当前日期格式化为字符串并添加到结果数组
        date_strings.append(current_date.strftime('%Y%m%d'))
        # 在当前日期上加一天
        current_date += timedelta(days=1)
    
    return date_strings


def decimal_to_ascii(decimal_number):
    # 将十进制数转换为十六进制字符串，不带'0x'前缀
    hex_string = format(decimal_number, 'x')
    
    # 如果长度是奇数，在前面补一个零
    if len(hex_string) % 2 != 0:
        hex_string = '0' + hex_string

    # 初始化结果字符串
    ascii_string = ""

    # 以两个字符为一组遍历十六进制字符串
    for i in range(0, len(hex_string), 2):
        hex_char = hex_string[i:i+2]
        # 将每组十六进制字符转换为十进制，再转换为ASCII字符
        ascii_string += chr(int(hex_char, 16))

    return ascii_string


def get_msg(file_path):
    folder = os.path.dirname(file_path)
    full_filename = os.path.basename(file_path)
    filename, _ = os.path.splitext(full_filename)

    dev_id = str.split(filename, '-')[0]
    date = str.split(filename, '-')[1][0:8]
    hour = str.split(filename, '-')[1][8:]

    msg = {
        'folder': folder,
        'full_filename': full_filename,
        'filename': filename,
        'dev_id': dev_id,
        'date': date,
        'hour': hour
    }

    return msg


# def get_merged_data_hdf5(hdf5_file):

#     rows_minutes = 60
#     n_cols = 23
#     merged_data = np.full([rows_minutes * 1200, n_cols], np.nan).astype(np.complex64)

#     with h5py.File(hdf5_file, 'r') as file:
#         cur_data = file['data'][:]
#         frame_ids = file['frame_ids'][:]

#     try:

#         start_id = frame_ids[0]
#         start_index = 0
#         frame_len = 5 * 20
#         for x, id in enumerate(frame_ids):
#             # print(x, id, start_index)
#             if start_index >= 72000:
#                 break

#             if x == 0:
#                 if not frame_ids[0]+1 == frame_ids[1]:
#                     if frame_ids[1]+1 == frame_ids[2]:
#                         start_id = frame_ids[1]
#                         continue

#             if id == start_id:
#                 merged_data[start_index: start_index+frame_len, :] = cur_data[x*frame_len: (x+1)*frame_len, :]

#                 start_id = id + 1
#                 start_index += frame_len

#             elif id > start_id:

#                 if x+1 == len(frame_ids):
#                     merged_data[start_index: start_index+frame_len, :] = cur_data[x*frame_len: (x+1)*frame_len, :]

#                     start_id = id + 1
#                     start_index += frame_len
#                 else:
#                     if frame_ids[x+1] == start_id:
#                         continue
#                     else:
#                         start_index += (id - start_id) * frame_len

#                         merged_data[start_index: start_index+frame_len, :] = cur_data[x*frame_len: (x+1)*frame_len, :]

#                         start_index += frame_len
#                         start_id = id + 1
            
#             elif id < start_id:
#                 if x+1 == len(frame_ids):
#                     merged_data[start_index: start_index+frame_len, :] = cur_data[x*frame_len: (x+1)*frame_len, :]

#                     start_id = id + 1
#                     start_index += frame_len
#                 else:
#                     if frame_ids[x+1] == start_id:
#                         continue
#                     else:
#                         merged_data[start_index: start_index+frame_len, :] = cur_data[x*frame_len: (x+1)*frame_len, :]

#                         start_id = id + 1
#                         start_index += frame_len

#         # 如果丢太多，就用老版本的代码
#         lost_frame = np.sum(np.isnan(merged_data)) // 2300
#         all_frame = len(frame_ids)
#         if lost_frame / all_frame >= 0.5:
#             return get_merged_data_hdf5_old(hdf5_file)
#         else:
#             # 使用前一个有效值填充缺失值
#             merged_data = fill_missing_with_zero_vectorized(merged_data)

#             return merged_data
#     except Exception:
#         return get_merged_data_hdf5_old(hdf5_file)


def get_merged_data_hdf5(hdf5_file_path):

    rows_minutes = 60
    n_cols = 23
    merged_data = np.full([rows_minutes * 1200, n_cols], np.nan).astype(np.complex64)

    with h5py.File(hdf5_file_path, 'r') as file:
        cur_data = file['data'][:]

    if len(cur_data) > rows_minutes * 1200:
        merged_data[:, :] = cur_data[0:rows_minutes * 1200, :]
    else:
        merged_data[0:len(cur_data), :] = cur_data
    merged_data = fill_missing_with_zero_vectorized(merged_data)

    return merged_data


def fill_missing_with_zero_vectorized(arr):
    """
    向量化实现，用前一个非NaN值填充NaN，如果没有则用0
    
    参数:
    arr -- 输入的NumPy数组
    
    返回:
    填充后的数组
    """
    # 创建结果数组副本
    result = arr.copy()
    
    # 按列处理
    for col in range(arr.shape[1]):
        column_data = arr[:, col]
        mask = np.isnan(column_data)
        
        # 找到非NaN值的索引
        valid_indices = np.where(~mask)[0]
        
        if len(valid_indices) == 0:
            # 如果整列都是NaN，全部填充0
            result[:, col] = 0
            continue
        
        # 构建索引数组
        idx = np.arange(len(column_data))
        # 对于每个位置，找到前一个有效值的索引
        idx = np.maximum.accumulate(np.where(~mask, idx, 0))
        
        # 获取填充值
        filled_values = column_data[idx]
        
        # 处理开头的NaN（idx=0的位置且原始值为NaN）
        filled_values[(idx == 0) & mask] = 0 + 0j
        
        result[:, col] = filled_values
    
    return result


def save_feature(analyzer, save_path, msg):
    dev_id = msg['dev_id']
    date = msg['date']
    hour = msg['hour']

    # 保存数据
    np.save(os.path.join(save_path, f"{dev_id}-{date}{hour}-act.npy"), analyzer.act)
    np.save(os.path.join(save_path, f"{dev_id}-{date}{hour}-br.npy"), analyzer.br)
    np.save(os.path.join(save_path, f"{dev_id}-{date}{hour}-hr.npy"), analyzer.hr)
    np.save(os.path.join(save_path, f"{dev_id}-{date}{hour}-up_down.npy"), analyzer.is_up)


def select_files(file_path, select_all_flag=True, select_changdi="", select_data=[]):
    # 选择相关所有文件列表
    all_pig_date_path = []
    for path1 in os.listdir(file_path):
        cur_exp_path = os.path.join(file_path, path1)
        if (os.path.isdir(cur_exp_path) == False):
            continue

        for pid_id in os.listdir(cur_exp_path):
            pig_id_path = os.path.join(cur_exp_path, pid_id)
            if (os.path.isdir(pig_id_path) == False):
                continue
            pig_id_date_list = os.listdir(pig_id_path)
            pig_id_date_list_path = [os.path.join(pig_id_path, item) for item in pig_id_date_list]
            all_pig_date_path.extend(pig_id_date_list_path)

    # 新的符合条件的路径列表
    filtered_pig_date_path = []

    if (select_all_flag):
        # 如果选择了所有数据，则直接使用所有文件列表
        filtered_pig_date_path = all_pig_date_path
    else:
        # 遍历每个文件路径进行筛选
        date_list = generate_date_strings(select_data)
        for file_path in all_pig_date_path:
            dir_levels = file_path.split(os.sep)

            cur_changdi = dir_levels[-3]
            cur_changdi = cur_changdi.split('_')[-2]  # 获取场地名称
            cur_date = dir_levels[-1]
            if (cur_changdi in select_changdi and cur_date in date_list):
                filtered_pig_date_path.append(file_path)

    filtered_pig_files = []
    for date_path in filtered_pig_date_path:

        if os.path.exists(date_path):
            file_list = [os.path.join(date_path, name) for name in os.listdir(date_path) if os.path.isfile(os.path.join(date_path, name))]
        else:
            continue

        filtered_pig_files.extend(file_list)
    
    return filtered_pig_files


def sort_check_date(date_list):
    # 对日期列表排序
    sorted_date_list = sorted(date_list)

    # 转换为 `datetime` 对象列表
    date_objects = [datetime.strptime(date, "%Y%m%d") for date in sorted_date_list]

    # 检查是否相邻的函数
    def check_adjacent_dates(dates):
        for i in range(1, len(dates)):
            if (dates[i] - dates[i - 1]).days != 1:
                return False
        return True

    # 检查日期是否相邻
    are_dates_adjacent = check_adjacent_dates(date_objects)

    return are_dates_adjacent, sorted_date_list

def count_continuous_ones(data):
    # data 是一个由 0 和 1 组成的列表
    count = 0
    in_segment = False
    
    for bit in data:
        if bit == 1:
            if not in_segment:
                count += 1
                in_segment = True
        else:
            in_segment = False
    
    return count

def generage_features(dataset, label_path):
    all_features = []
    all_labels = []
    exp_list = ['changtang_20250505', 'changtang_20250523', 'changtang_20250610', 'shuitai_20241115', 'shuitai_20250109', 'shuitai_20250321', 'shuitai_20250426', 'shuitai_20250514', 'shuitai_20250531']
    
    for item in dataset:
        experiment = item[0]
        
        if experiment not in exp_list:
            continue
        
        pig_id = item[1]
        dates = item[2]
        raw_data = item[3]

        if pig_id == 'changtang_29773_20250505':
            test = 1

        # if 'shuitai' not in experiment:
        #     continue

        # read label data
        label_file = os.path.join(label_path, f'{experiment}.csv')
        if not os.path.exists(label_file):
            print(f'----\tNo label file found for {experiment}')
            continue
        label_data = pd.read_csv(label_file, index_col=None, header=0, dtype=str)

        # print(f'Processing {experiment} - Pig ID: {pig_id}...')
        # out_file = os.path.join(out_path, f'{pig_id}.html')
        # draw.draw_rawdata.draw_pig_features(pig_data, out_file)

        dates_cnt = len(dates)
        hour_step = 2 * 60

        # act_raw = raw_data['act'].values.reshape(-1, hour_step)
        act_raw = raw_data['normalize_act'].values.reshape(-1, hour_step)
        br_raw = raw_data['br'].values.reshape(-1, hour_step)
        up_raw = raw_data['is_up'].values.reshape(-1, hour_step)

        radar_id = pig_id.split('_')[1]

        select_ind = label_data.iloc[:, 0] == radar_id
        if (not select_ind.any()):
            print(f'----\tNo label found for {radar_id} in {experiment}')
            continue
        else:
            pig_label = label_data[select_ind].iloc[0].tolist()

        if len(pig_label[2]) < 8:
            if (pig_label[2].strip() == '0' or pig_label[2].strip() == '-1'):
                label_date = datetime(1970, 1, 1, 0, 0, 0)
            else:
                raise(f"label error {pig_label[2]}")
        else:
            label_date = datetime.strptime(pig_label[2][0:8], '%Y%m%d')

        # act_24h_raw = raw_data['act'].values.reshape(-1, hour_step)
        # br_24h_raw = raw_data['br'].values.reshape(-1, hour_step)
        # up_24h_raw = raw_data['is_up'].values.reshape(-1, hour_step)

        # 从第三天开始计算特征
        for i in range(2, dates_cnt):

            cur_hour_ind = i * 24
            hour_ind_1d = (i - 1) * 24
            hour_ind_2d = (i - 2) * 24

            select_cur_1 = range(cur_hour_ind, (cur_hour_ind + 7))
            select_cur_2 = range((cur_hour_ind + 20), (cur_hour_ind + 24))
            select_1d_1 = range(hour_ind_1d, (hour_ind_1d + 7))
            select_1d_2 = range((hour_ind_1d + 20), (hour_ind_1d + 24))
            select_2d_1 = range(hour_ind_2d, (hour_ind_2d + 7))
            select_2d_2 = range((hour_ind_2d + 20), (hour_ind_2d + 24))

            value_cur_1 = act_raw[select_cur_1, :]
            value_cur_2 = act_raw[select_cur_2, :]
            value_1d_1 = act_raw[select_1d_1, :]
            value_1d_2 = act_raw[select_1d_2, :]
            value_2d_1 = act_raw[select_2d_1, :]
            value_2d_2 = act_raw[select_2d_2, :]

            feature_act_01 = np.nansum(value_cur_1) - np.nansum(value_1d_1)
            feature_act_02 = np.nansum(value_cur_1) - np.nansum(value_2d_1)
            feature_act_03 = np.nansum(value_1d_2) - np.nansum(value_2d_2)

            feature_act_04 = np.sum(value_cur_1 > 4) - np.sum(value_1d_1 > 4)
            feature_act_05 = np.sum(value_cur_1 > 4) - np.sum(value_2d_1 > 4)
            feature_act_06 = np.sum(value_1d_2 > 4) - np.sum(value_2d_2 > 4)

            feature_act_07 = np.sum(value_cur_1 > 15) - np.sum(value_1d_1 > 15)
            feature_act_08 = np.sum(value_cur_1 > 15) - np.sum(value_2d_1 > 15)
            feature_act_09 = np.sum(value_1d_2 > 15) - np.sum(value_2d_2 > 15)

            feature_act_10 = np.nansum(value_cur_1) / (np.nansum(value_1d_1) + 0.1)
            feature_act_11 = np.nansum(value_cur_1) / (np.nansum(value_2d_1) + 0.1)
            feature_act_12 = np.nansum(value_1d_2) / (np.nansum(value_2d_2) + 0.1)

            feature_act_13 = np.sum(value_cur_1 > 4) / (np.sum(value_1d_1 > 4) + 0.1)
            feature_act_14 = np.sum(value_cur_1 > 4) / (np.sum(value_2d_1 > 4) + 0.1)
            feature_act_15 = np.sum(value_1d_2 > 4) / (np.sum(value_2d_2 > 4) + 0.1)

            feature_act_16 = np.sum(value_cur_1 > 15) / (np.sum(value_1d_1 > 15) + 0.1)
            feature_act_17 = np.sum(value_cur_1 > 15) / (np.sum(value_2d_1 > 15) + 0.1)
            feature_act_18 = np.sum(value_1d_2 > 15) / (np.sum(value_2d_2 > 15) + 0.1)

            # 呼吸特征
            value_cur_1 = br_raw[select_cur_1, :]
            value_cur_2 = br_raw[select_cur_2, :]
            value_1d_1 = br_raw[select_1d_1, :]
            value_1d_2 = br_raw[select_1d_2, :]
            value_2d_1 = br_raw[select_2d_1, :]
            value_2d_2 = br_raw[select_2d_2, :]

            feature_br_01 = np.nanmedian(value_cur_1)
            feature_br_02 = np.nanmedian(value_1d_2)
            feature_br_03 = np.nanmedian(value_1d_1)
            feature_br_04 = np.nanmedian(value_2d_2)
            feature_br_05 = np.nanmedian(value_2d_1)
            
            feature_br_06 = feature_br_01 - feature_br_03
            feature_br_07 = feature_br_01 - feature_br_05
            feature_br_08 = feature_br_03 - feature_br_05
            feature_br_09 = feature_br_02 - feature_br_04
            
            feature_br_10 = feature_br_01 / (feature_br_03 + 0.01)
            feature_br_11 = feature_br_01 / (feature_br_05 + 0.01)
            feature_br_12 = feature_br_03 / (feature_br_05 + 0.01)
            feature_br_13 = feature_br_02 / (feature_br_04 + 0.01)
            
            feature_br_14 = (feature_br_01 + feature_br_02) / (feature_br_03 + feature_br_04 + 0.01)
            feature_br_15 = np.nanpercentile(value_cur_1, 75) / (np.nanpercentile(value_cur_1, 25) + 0.01)
            feature_br_16 = np.nanpercentile(value_1d_1, 75) / (np.nanpercentile(value_1d_1, 25) + 0.01)
            feature_br_17 = np.nanpercentile(value_2d_1, 75) / (np.nanpercentile(value_2d_1, 25) + 0.01)
            feature_br_18 = np.nanpercentile(value_1d_2, 75) / (np.nanpercentile(value_1d_2, 25) + 0.01)
            
            feature_br_19 = np.sum(value_cur_1 > 40)
            feature_br_20 = np.sum(value_1d_1 > 40)

            # 起卧特征
            value_cur_1 = up_raw[select_cur_1, :]
            value_cur_2 = up_raw[select_cur_2, :]
            value_1d_1 = up_raw[select_1d_1, :]
            value_1d_2 = up_raw[select_1d_2, :]
            value_2d_1 = up_raw[select_2d_1, :]
            value_2d_2 = up_raw[select_2d_2, :]

            feature_up_01 = np.nansum(value_cur_1)
            feature_up_02 = np.nansum(value_cur_1) - np.nansum(value_1d_1)
            feature_up_03 = np.nansum(value_cur_1) - np.nansum(value_2d_1)
            feature_up_04 = np.nansum(value_1d_2) - np.nansum(value_2d_2)

            up_raw_diff = np.diff(up_raw)
            value_cur_1 = up_raw_diff[select_cur_1, :]
            value_cur_2 = up_raw_diff[select_cur_2, :]
            value_1d_1 = up_raw_diff[select_1d_1, :]
            value_1d_2 = up_raw_diff[select_1d_2, :]
            value_2d_1 = up_raw_diff[select_2d_1, :]
            value_2d_2 = up_raw_diff[select_2d_2, :]

            feature_up_05 = np.sum(value_cur_1 != 0)
            feature_up_06 = np.sum(value_cur_1 != 0) / (np.sum(value_1d_1 != 0) + 0.1)
            feature_up_07 = np.sum(value_cur_1 != 0) / (np.sum(value_2d_1 != 0) + 0.1)
            feature_up_08 = np.sum(value_1d_2 != 0) / (np.sum(value_2d_2 != 0) + 0.1)

            feature_up_09 = count_continuous_ones(value_cur_1.flatten().tolist())
            feature_up_10 = count_continuous_ones(value_1d_1.flatten().tolist())
            feature_up_11 = feature_up_09 >= 30 and feature_up_10 == 0
            feature_up_12 = feature_up_09 / (feature_up_10 + 0.1)
            feature_up_13 = feature_up_09 - feature_up_10
            feature_up_14 = ((feature_up_10==0 and feature_up_12>=30) or 
                                (feature_up_10==1 and feature_up_12>=40) or 
                                (feature_up_10==2 and feature_up_12>=20) or 
                                (feature_up_10==3 and feature_up_12>=15) or 
                                (feature_up_10==4 and feature_up_12>=10) or 
                                (feature_up_10==5 and feature_up_12>=8) or 
                                (feature_up_10==6 and feature_up_12>=6) or 
                                (10>=feature_up_10>=7 and feature_up_12>=4) or 
                                (feature_up_10>10 and feature_up_12>=2.5) or 
                                (feature_up_10>25 and feature_up_12>=2) or 
                                (feature_up_10>35 and feature_up_12>=1.5) or 
                                (feature_up_10>50 and feature_up_13>=25))
            feature_up_15 = count_continuous_ones(value_2d_1.flatten().tolist())
            feature_up_16 = feature_up_09>=feature_up_10>=feature_up_15 and feature_up_09!=0
            
            

            features = np.array([
                feature_act_01, feature_act_02, feature_act_03,
                        feature_act_04, feature_act_05, feature_act_06,
                        feature_act_07, feature_act_08, feature_act_09,
                        feature_act_10, feature_act_11, feature_act_12,
                        feature_act_13, feature_act_14, feature_act_15,
                        feature_act_16, feature_act_17, feature_act_18,
                        feature_br_01, feature_br_02, feature_br_03,
                        feature_br_04, feature_br_05, feature_br_06,
                        feature_br_07, feature_br_08, feature_br_09,
                        feature_br_10, feature_br_11, feature_br_12,
                        feature_br_13, feature_br_14, feature_br_15,
                        feature_br_16, feature_br_17, feature_br_18,
                        feature_br_19, feature_br_20,
                        feature_up_01, feature_up_02, feature_up_03,
                        feature_up_04, feature_up_05, feature_up_06,
                        feature_up_07, feature_up_08, 
                        feature_up_09,
                        feature_up_10, feature_up_11, feature_up_12,
                        feature_up_13, feature_up_14, feature_up_15,
                        feature_up_16])

            all_features.append(features)

            cur_date = datetime.strptime(dates[i], '%Y%m%d')
            date_delta = cur_date - label_date
            if date_delta == timedelta(days=-2):
                cur_label = 0
                eva_label = 1
            elif (date_delta == timedelta(days=-1)):
                cur_label = 0
                eva_label = 1
            elif (date_delta == timedelta(days=0)):
                cur_label = 1
                eva_label = 1
            else:
                cur_label = 0
                eva_label = 0

            all_labels.append([experiment, pig_id, dates[i], cur_label, eva_label])

    f_df = pd.DataFrame(all_features, columns=[
        'act_01', 'act_02', 'act_03',
        'act_04', 'act_05', 'act_06',
        'act_07', 'act_08', 'act_09',
        'act_10', 'act_11', 'act_12',
        'act_13', 'act_14', 'act_15',
        'act_16', 'act_17', 'act_18',
        'br_01', 'br_02', 'br_03',
        'br_04', 'br_05', 'br_06',
        'br_07', 'br_08', 'br_09',
        'br_10', 'br_11', 'br_12',
        'br_13', 'br_14', 'br_15',
        'br_16', 'br_17', 'br_18',
        'br_19', 'br_20',
        'up_01', 'up_02', 'up_03',
        'up_04', 'up_05', 'up_06',
        'up_07', 'up_08',
        'up_09',
        'up_10', 'up_11', 'up_12',
        'up_13', 'up_14', 'up_15',
        'up_16'
    ])

    l_df = pd.DataFrame(all_labels, columns=['experiment', 'pig_id', 'date', 'label', 'eva_label'])

    return f_df, l_df


def get_freq_by_pks(pks):
    rr_list = np.diff(pks)
    return 60*20/np.median(rr_list)