from datetime import datetime, timedelta
import logging

import h5py
import numpy as np


def string_to_combined_hex_to_decimal(input_string):
    # 初始化空字符串用于保存十六进制表示
    combined_hex = ""

    for char in input_string:
        # 将字符转换为两位的十六进制表示并添加到combined_hex中
        combined_hex += format(ord(char), "02x")

    # 将完整的十六进制字符串转换为十进制数值
    decimal_value = int(combined_hex, 16)

    return decimal_value


def sn2id(sn_list):
    str_sn_list = [str(x) for x in sn_list]
    result = []

    for x in str_sn_list:
        if len(x)==2:
            id = string_to_combined_hex_to_decimal(x)
            result.append(str(id))
        elif len(x)==5 and x.isdigit():
            result.append(x)
        else:
            result.append(x)

    return result


def generate_time_strings(start_time_str, end_time_str):
    # 解析输入的字符串，将其转换为 datetime 对象
    start_date = datetime.strptime(start_time_str, '%Y%m%d%H')
    end_date = datetime.strptime(end_time_str, '%Y%m%d%H')

    # 初始化结果数组和当前处理的日期
    date_strings = []
    current_date = start_date

    # 循环直到当前日期超过结束日期
    while current_date <= end_date:
        # 将当前日期格式化为字符串并添加到结果数组
        date_strings.append(current_date.strftime('%Y%m%d%H'))
        # 在当前日期上加一天
        current_date += timedelta(hours=1)

    return date_strings


def generate_date_strings(start_date_str, end_date_str):
    # 解析输入的字符串，将其转换为 datetime 对象
    start_date = datetime.strptime(start_date_str, '%Y%m%d')
    end_date = datetime.strptime(end_date_str, '%Y%m%d')

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


def get_merged_data_hdf5(hdf5_file):
    try:
        rows_minutes = 60
        n_cols = 23
        merged_data = np.full([rows_minutes * 1200, n_cols], np.nan).astype(np.complex64)

        with h5py.File(hdf5_file, 'r') as file:
            cur_data = file['data'][:]
            frame_ids = file['frame_ids'][:]

        start_id = frame_ids[0]
        start_index = 0
        frame_len = 5 * 20
        for x, id in enumerate(frame_ids):
            # print(x, id, start_index)
            if start_index >= 72000:
                break

            if x == 0:
                if not frame_ids[0]+1 == frame_ids[1]:
                    if frame_ids[1]+1 == frame_ids[2]:
                        start_id = frame_ids[1]
                        continue

            if id == start_id:
                merged_data[start_index: start_index+frame_len, :] = cur_data[x*frame_len: (x+1)*frame_len, :]

                start_id = id + 1
                start_index += frame_len

            elif id > start_id:

                if x+1 == len(frame_ids):
                    merged_data[start_index: start_index+frame_len, :] = cur_data[x*frame_len: (x+1)*frame_len, :]

                    start_id = id + 1
                    start_index += frame_len
                else:
                    if frame_ids[x+1] == start_id:
                        continue
                    else:
                        start_index += (id - start_id) * frame_len

                        merged_data[start_index: start_index+frame_len, :] = cur_data[x*frame_len: (x+1)*frame_len, :]

                        start_index += frame_len
                        start_id = id + 1
            
            elif id < start_id:
                if x+1 == len(frame_ids):
                    merged_data[start_index: start_index+frame_len, :] = cur_data[x*frame_len: (x+1)*frame_len, :]

                    start_id = id + 1
                    start_index += frame_len
                else:
                    if frame_ids[x+1] == start_id:
                        continue
                    else:
                        merged_data[start_index: start_index+frame_len, :] = cur_data[x*frame_len: (x+1)*frame_len, :]

                        start_id = id + 1
                        start_index += frame_len

        # 如果丢太多，就用老版本的代码
        lost_frame = np.sum(np.isnan(merged_data)) // 2300
        all_frame = len(frame_ids)
        if lost_frame / all_frame >= 0.5:
            return get_merged_data_hdf5_old(hdf5_file)
        else:
            # 使用前一个有效值填充缺失值
            merged_data = fill_missing_with_zero_vectorized(merged_data)
            return merged_data
    except:
        return get_merged_data_hdf5_old(hdf5_file)
    

def get_merged_data_hdf5_old(hdf5_file_path):

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

