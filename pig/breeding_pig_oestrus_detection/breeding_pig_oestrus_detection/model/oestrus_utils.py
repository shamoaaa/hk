import logging
import os
import shutil
import time

import pandas as pd
import numpy as np
import math
from typing import Union, List, Tuple, Dict, Optional


def generate_oestrus_features_v1(
    dataset: List[Tuple],
    time_window_days: int = 3,
    start_hour: int = 0,
    end_hour: int = 7,
    evening_start_hour: int = 20,
    evening_end_hour: int = 24,
    include_78_features: bool = False,
    shift_date: int = 0,
) -> Tuple[pd.DataFrame, List[int], dict]:
    """
    生成猪只活动特征的函数 (修改版: 返回 f_df, pig_pred_mask, pig_dates)

    参数:
        dataset: 数据集，格式为[(pig_id, 猪特征的dataframe形式), ...]
        time_window_days: 时间窗口天数，默认为3天
        start_hour: 开始小时，默认为0点
        end_hour: 结束小时，默认为7点
        evening_start_hour: 晚间开始小时，默认为20点
        evening_end_hour: 晚间结束小时，默认为24点
        include_78_features: 是否包含7-8小时的特征，默认为False
        shift_date: 用于计算平均特征的偏移日期

    返回:
        Tuple[pd.DataFrame, List[int], dict]: 
            - f_df: 生成的特征DataFrame
            - pig_pred_mask: 预测掩码列表，1表示成功生成特征，-1表示未生成
            - pig_dates: 字典，记录每头猪对应的数据天数
    """
    all_features = []
    pig_pred_mask = []
    pig_dates = {}

    for item in dataset:
        # 解包数据
        pig_id = item[0]
        raw_data = item[1].copy()

        # 数据清洗：将 br 列的 nan 替换为 -1
        raw_data['br'] = raw_data['br'].fillna(-1)
        # 填补Nan
        raw_data = fill_nan(raw_data)

        # 自动计算天数 (假设数据是连续的，每天 24 小时)
        hour_step = 2 * 60  # 2分钟一个点
        total_points = len(raw_data['normalize_act'].values)
        dates_cnt = math.ceil(total_points / (24 * hour_step))
        
        # 记录该猪的天数
        pig_dates[pig_id] = dates_cnt

        # 数据预处理
        # 注意：如果数据长度不是 hour_step 的整数倍，reshape 可能会报错
        # 建议在传入前确保数据完整性，或者在此处添加截断逻辑
        act_raw = raw_data['normalize_act'].values.reshape(-1, hour_step)
        br_raw = raw_data['br'].values.reshape(-1, hour_step)
        up_raw = raw_data['is_up'].values.reshape(-1, hour_step)

        # 循环每一天
        for i in range(dates_cnt):
            # 如果天数不足时间窗口（例如前两天），标记为 -1 并跳过
            if i < time_window_days - 1:
                pig_pred_mask.append(-1)
                continue
            
            # 计算时间索引
            cur_hour_ind = i * 24
            hour_ind_1d = (i - 1) * 24
            hour_ind_2d = (i - 2) * 24

            # 时间窗口选择
            select_cur_1 = range(cur_hour_ind, (cur_hour_ind + end_hour))
            select_cur_2 = range((cur_hour_ind + evening_start_hour), (cur_hour_ind + evening_end_hour))
            select_1d_1 = range(hour_ind_1d, (hour_ind_1d + end_hour))
            select_1d_2 = range((hour_ind_1d + evening_start_hour), (hour_ind_1d + evening_end_hour))
            select_2d_1 = range(hour_ind_2d, (hour_ind_2d + end_hour))
            select_2d_2 = range((hour_ind_2d + evening_start_hour), (hour_ind_2d + evening_end_hour))

            try:
                # 调用原有的特征计算函数 (保持生成特征部分代码不变)
                features = _compute_feature_set(
                    act_raw, br_raw, up_raw,
                    select_cur_1, select_cur_2, select_1d_1, select_1d_2, select_2d_1, select_2d_2,
                    include_78_features, include_mean_features=include_78_features,
                    cur_hour_ind=cur_hour_ind, hour_ind_1d=hour_ind_1d, hour_ind_2d=hour_ind_2d,
                    shift_date=shift_date, raw_data=raw_data, hour_step=hour_step
                )

                # 检查 NaN 值
                if np.isnan(features).sum() != 0:
                    pig_pred_mask.append(-1)
                    continue

                all_features.append(features)
                pig_pred_mask.append(1)

            except Exception as e:
                # 发生异常，标记为 -1
                pig_pred_mask.append(-1)
                continue

    # 创建特征DataFrame
    feature_columns = _get_feature_columns(include_78_features)
    f_df = pd.DataFrame(all_features, columns=feature_columns)
    
    return f_df, pig_pred_mask, pig_dates


# def fill_nan(df, noise_scale: float = 0.02):
#     """处理单个df - 移到模块级别以便pickle"""
#     np.random.seed(42)
#
#     # 预计算统计信息
#     col_std = df.std(skipna=True)
#     col_mean = df.mean(skipna=True)
#     col_min = df.min(skipna=True)  # 新增：每列的最小值
#     col_std[col_std == 0] = 1e-8
#
#     df_filled = df.copy()
#     n_rows = len(df)
#     nan_mask = df.isna()
#
#     # 为每列处理
#     for col_idx, col_name in enumerate(df.columns):
#         if nan_mask[col_name].sum() == 0:
#             continue
#
#         nan_indices = np.where(nan_mask[col_name])[0]
#
#         # 为每个NaN位置寻找填充值 - 从后往前遍历
#         for nan_idx in reversed(nan_indices):  # 修改：使用reversed从后往前遍历
#             if nan_idx < 2400:
#                 # 2400行之前：使用当前逻辑
#                 fill_value = None
#                 for day_offset in range(1, 8):
#                     target_idx = nan_idx + day_offset * 2880
#                     if target_idx >= n_rows:
#                         fill_value = col_mean[col_name]
#                         if pd.isna(fill_value):
#                             fill_value = 0
#                         break
#                     if not pd.isna(df.iloc[target_idx, col_idx]):
#                         fill_value = df.iloc[target_idx, col_idx]
#                         break
#                 else:
#                     fill_value = col_mean[col_name]
#                     if pd.isna(fill_value):
#                         fill_value = 0
#
#                 # 添加噪声
#                 noise = np.random.normal(0, col_std[col_name] * noise_scale)
#                 imputed_value = fill_value + noise
#
#                 # 如果填补后是负数，就用该特征的最小值填补
#                 if imputed_value < 0:
#                     imputed_value = col_min[col_name]
#                     if pd.isna(imputed_value):
#                         imputed_value = 0
#
#                 df_filled.iloc[nan_idx, col_idx] = imputed_value
#             else:
#                 # 2400行之后：使用插值或邻近均值
#                 # 尝试向前和向后查找非NaN值
#                 forward_value = None
#                 backward_value = None
#
#                 # 向前查找（最近的先前非NaN值）
#                 for i in range(nan_idx - 1, max(-1, nan_idx - 10), -1):
#                     if not pd.isna(df.iloc[i, col_idx]):
#                         forward_value = df.iloc[i, col_idx]
#                         break
#
#                 # 向后查找（最近的后续非NaN值）
#                 for i in range(nan_idx + 1, min(n_rows, nan_idx + 10)):
#                     if not pd.isna(df.iloc[i, col_idx]):
#                         backward_value = df.iloc[i, col_idx]
#                         break
#
#                 # 根据找到的值选择填充策略
#                 if forward_value is not None and backward_value is not None:
#                     # 前后都有值：使用线性插值
#                     imputed_value = (forward_value + backward_value) / 2
#                 elif forward_value is not None:
#                     # 只有前值：使用前值
#                     imputed_value = forward_value
#                 elif backward_value is not None:
#                     # 只有后值：使用后值
#                     imputed_value = backward_value
#                 else:
#                     # 前后都找不到：使用列均值
#                     imputed_value = col_mean[col_name]
#                     if pd.isna(imputed_value):
#                         imputed_value = 0
#
#                 df_filled.iloc[nan_idx, col_idx] = imputed_value
#
#     return df_filled

def fill_nan(df, noise_scale: float = 0.02):
    """
    完全向量化的NaN填充方案
    """
    np.random.seed(42)

    # 预计算统计信息
    col_std = df.std(skipna=True).fillna(1e-8)
    col_mean = df.mean(skipna=True).fillna(0)
    col_min = df.min(skipna=True).fillna(0)
    col_std[col_std == 0] = 1e-8

    df_filled = df.copy()

    # 对整个DataFrame进行插值处理
    df_filled = df_filled.interpolate(method='linear', limit_direction='both')
    df_filled = df_filled.fillna(method='ffill').fillna(method='bfill')

    # 最后用列均值填充剩余缺失值
    df_filled = df_filled.fillna(col_mean)

    # 只对原本是NaN的位置添加噪声
    nan_mask = df.isna()
    if nan_mask.any().any():
        noise = np.random.normal(
            0,
            np.repeat(col_std.values.reshape(1, -1), len(df), axis=0) * noise_scale,
            size=df.shape
        )
        # 只在原来为NaN的位置添加噪声
        df_filled = df_filled + pd.DataFrame(
            np.where(nan_mask, noise, 0),
            columns=df.columns,
            index=df.index
        )

        # 确保非负值
        for col in df.columns:
            negative_mask = (df_filled[col] < 0) & nan_mask[col]
            df_filled.loc[negative_mask, col] = col_min[col]

    return df_filled


def kl_divergence(p, q):
    epsilon = 1e-10
    p = np.asarray(p, dtype=np.float32) + epsilon
    q = np.asarray(q, dtype=np.float32) + epsilon
    return np.sum(p * np.log(p / q))


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


def compute_windows(raw_data, window, step, type):
    # 计算滑窗特征
    window *= 2
    step *= 2
    start = 0
    if type=='act':
        result = []
        while start + window <= len(raw_data):
            act_window = raw_data[start:start+window]
            result.append(np.nanmean(act_window))
            start += step
    if type=='up':
        result = []
        while start + window <= len(raw_data):
            up_window = raw_data[start:start+window]
            result.append(np.nansum(up_window))
            start += step
    return result


def compute_meanfeatures(data):
    # 计算最前面两天可用的7~8， 如果整个周期内7~8特征都不存在则返回空，当前样本应舍弃不好预测
    pig_meanfeature = {}
    for item in data:
        pig_id = item[0]
        raw_data = item[1]
        hour_step = 2 * 60
        date = math.ceil(len(raw_data['normalize_act'].values)/2880)
        act_raw = raw_data['normalize_act'].values.reshape(-1, hour_step)
        up_raw = raw_data['is_up'].values.reshape(-1, hour_step)
        
        mean_feature_up01 = [] # 起立时间
        mean_feature_up02 = [] # 起立次数
        mean_feature_up03 = [] # 滑窗起立结果
        mean_feature_act01 = [] # 平均活动量
        mean_feature_act02 = [] # 平均高活动次数
        mean_feature_act03 = [] # 剧烈活动次数
        mean_feature_act04 = [] # 滑窗活动量结果

        # 跟踪前两天的特征
        use_day = 0
        for i in range(date):
            cur_hour_ind = i * 24
            select_today = range(cur_hour_ind+7, cur_hour_ind+8)
            up_value_today = up_raw[select_today, :]
            act_value_today = act_raw[select_today, :]
            if np.isnan(up_value_today).sum()==120:
                continue
            use_day += 1
            mean_feature_up01.append(np.nansum(up_value_today))
            mean_feature_up02.append(count_continuous_ones(up_value_today.flatten().tolist()))
            mean_feature_up03.append(compute_windows(up_value_today.flatten(), window=10, step=10, type='up'))
            mean_feature_act01.append(np.nansum(act_value_today))
            mean_feature_act02.append(np.sum(act_value_today > 4))
            mean_feature_act03.append(np.sum(act_value_today > 15))
            mean_feature_act04.append(compute_windows(act_value_today.flatten(), window=10, step=10, type='act'))
            if use_day==2:
                break
        if use_day < 2:
            pig_meanfeature[pig_id] = {}
        if use_day==2:
            mean_feature_up01 = np.mean(mean_feature_up01)
            mean_feature_up02 = np.mean(mean_feature_up02)
            mean_feature_up03 = np.mean(mean_feature_up03, axis=0)
            mean_feature_act01 = np.mean(mean_feature_act01)
            mean_feature_act02 = np.mean(mean_feature_act02)
            mean_feature_act03 = np.mean(mean_feature_act03)
            mean_feature_act04 = np.mean(mean_feature_act04, axis=0)

        pig_meanfeature[pig_id] =  {'mean_feature_up01': mean_feature_up01,
                'mean_feature_up02': mean_feature_up02,
                'mean_feature_up03': mean_feature_up03,
                'mean_feature_act01': mean_feature_act01,
                'mean_feature_act02': mean_feature_act02,
                'mean_feature_act03': mean_feature_act03,
                'mean_feature_act04': mean_feature_act04
                }
    return pig_meanfeature


def _compute_feature_set(
    act_raw, br_raw, up_raw,
    select_cur_1, select_cur_2, select_1d_1, select_1d_2, select_2d_1, select_2d_2,
    include_78_features, include_mean_features,
    cur_hour_ind, hour_ind_1d, hour_ind_2d, shift_date, raw_data, hour_step
) -> np.ndarray:
    """计算特征集合"""

    # 活动特征
    act_features = _compute_activity_features(
        act_raw, select_cur_1, select_cur_2, select_1d_1, select_1d_2, select_2d_1, select_2d_2
    )

    # 呼吸特征
    br_features = _compute_breathing_features(
        br_raw, select_cur_1, select_cur_2, select_1d_1, select_1d_2, select_2d_1, select_2d_2
    )

    # 起卧特征
    up_features = _compute_movement_features(
        up_raw, select_cur_1, select_cur_2, select_1d_1, select_1d_2, select_2d_1, select_2d_2
    )

    # 睡眠特征
    sleep_features = _compute_sleep_features(up_raw, hour_ind_1d, cur_hour_ind, hour_ind_2d)

    features = np.concatenate([act_features, br_features, up_features, sleep_features])

    # 如果需要7-8小时特征
    if include_78_features:
        additional_features = _compute_78_features(
            act_raw, br_raw, up_raw, cur_hour_ind, hour_ind_1d,
            shift_date, raw_data, include_mean_features
        )
        features = np.concatenate([features, additional_features])

    return features


def _compute_activity_features(act_raw, select_cur_1, select_cur_2, select_1d_1, select_1d_2, select_2d_1, select_2d_2) -> np.ndarray:
    """计算活动特征"""

    value_cur_1 = act_raw[select_cur_1, :]
    value_cur_2 = act_raw[select_cur_2, :]
    value_1d_1 = act_raw[select_1d_1, :]
    value_1d_2 = act_raw[select_1d_2, :]
    value_2d_1 = act_raw[select_2d_1, :]
    value_2d_2 = act_raw[select_2d_2, :]

    # 活动总量趋势
    feature_act_01 = np.nansum(value_cur_1) - np.nansum(value_1d_1)
    feature_act_02 = np.nansum(value_cur_1) - np.nansum(value_2d_1)
    feature_act_03 = np.nansum(value_1d_2) - np.nansum(value_2d_2)

    # 较大幅度活动总量趋势 >4
    feature_act_04 = np.sum(value_cur_1 > 4) - np.sum(value_1d_1 > 4)
    feature_act_05 = np.sum(value_cur_1 > 4) - np.sum(value_2d_1 > 4)
    feature_act_06 = np.sum(value_1d_2 > 4) - np.sum(value_2d_2 > 4)

    # 大幅度活动总量趋势 >15
    feature_act_07 = np.sum(value_cur_1 > 15) - np.sum(value_1d_1 > 15)
    feature_act_08 = np.sum(value_cur_1 > 15) - np.sum(value_2d_1 > 15)
    feature_act_09 = np.sum(value_1d_2 > 15) - np.sum(value_2d_2 > 15)

    # 比例特征
    feature_act_10 = np.nansum(value_cur_1) / (np.nansum(value_1d_1) + 0.1)
    feature_act_11 = np.nansum(value_cur_1) / (np.nansum(value_2d_1) + 0.1)
    feature_act_12 = np.nansum(value_1d_2) / (np.nansum(value_2d_2) + 0.1)

    feature_act_13 = np.sum(value_cur_1 > 4) / (np.sum(value_1d_1 > 4) + 0.1)
    feature_act_14 = np.sum(value_cur_1 > 4) / (np.sum(value_2d_1 > 4) + 0.1)
    feature_act_15 = np.sum(value_1d_2 > 4) / (np.sum(value_2d_2 > 4) + 0.1)

    feature_act_16 = np.sum(value_cur_1 > 15) / (np.sum(value_1d_1 > 15) + 0.1)
    feature_act_17 = np.sum(value_cur_1 > 15) / (np.sum(value_2d_1 > 15) + 0.1)
    feature_act_18 = np.sum(value_1d_2 > 15) / (np.sum(value_2d_2 > 15) + 0.1)

    return np.array([
        feature_act_01, feature_act_02, feature_act_03,
        feature_act_04, feature_act_05, feature_act_06,
        feature_act_07, feature_act_08, feature_act_09,
        feature_act_10, feature_act_11, feature_act_12,
        feature_act_13, feature_act_14, feature_act_15,
        feature_act_16, feature_act_17, feature_act_18
    ])


def _compute_breathing_features(br_raw, select_cur_1, select_cur_2, select_1d_1, select_1d_2, select_2d_1, select_2d_2) -> np.ndarray:
    """计算呼吸特征"""

    value_cur_1 = br_raw[select_cur_1, :]
    value_cur_2 = br_raw[select_cur_2, :]
    value_1d_1 = br_raw[select_1d_1, :]
    value_1d_2 = br_raw[select_1d_2, :]
    value_2d_1 = br_raw[select_2d_1, :]
    value_2d_2 = br_raw[select_2d_2, :]

    # 平均呼吸率,中位数
    feature_br_01 = np.nanmedian(value_cur_1)
    feature_br_02 = np.nanmedian(value_1d_2)
    feature_br_03 = np.nanmedian(value_1d_1)
    feature_br_04 = np.nanmedian(value_2d_2)
    feature_br_05 = np.nanmedian(value_2d_1)

    # 平均呼吸率变化量
    feature_br_06 = feature_br_01 - feature_br_03
    feature_br_07 = feature_br_01 - feature_br_05
    feature_br_08 = feature_br_03 - feature_br_05
    feature_br_09 = feature_br_02 - feature_br_04

    # 比例特征
    feature_br_10 = feature_br_01 / (feature_br_03 + 0.01)
    feature_br_11 = feature_br_01 / (feature_br_05 + 0.01)
    feature_br_12 = feature_br_03 / (feature_br_05 + 0.01)
    feature_br_13 = feature_br_02 / (feature_br_04 + 0.01)

    feature_br_14 = (feature_br_01 + feature_br_02) / (feature_br_03 + feature_br_04 + 0.01)

    # 呼吸率上四分之一位数和下四分之一位数比例
    feature_br_15 = np.nanpercentile(value_cur_1, 75) / (np.nanpercentile(value_cur_1, 25) + 0.01)
    feature_br_16 = np.nanpercentile(value_1d_1, 75) / (np.nanpercentile(value_1d_1, 25) + 0.01)
    feature_br_17 = np.nanpercentile(value_2d_1, 75) / (np.nanpercentile(value_2d_1, 25) + 0.01)
    feature_br_18 = np.nanpercentile(value_1d_2, 75) / (np.nanpercentile(value_1d_2, 25) + 0.01)

    # 呼吸率异常急促次数
    feature_br_19 = np.sum(value_cur_1 > 40)
    feature_br_20 = np.sum(value_1d_1 > 40)

    return np.array([
        feature_br_01, feature_br_02, feature_br_03,
        feature_br_04, feature_br_05, feature_br_06,
        feature_br_07, feature_br_08, feature_br_09,
        feature_br_10, feature_br_11, feature_br_12,
        feature_br_13, feature_br_14, feature_br_15,
        feature_br_16, feature_br_17, feature_br_18,
        feature_br_19, feature_br_20
    ])


def _compute_movement_features(up_raw, select_cur_1, select_cur_2, select_1d_1, select_1d_2, select_2d_1, select_2d_2) -> np.ndarray:
    """计算起卧特征"""

    value_cur_1 = up_raw[select_cur_1, :]
    value_cur_2 = up_raw[select_cur_2, :]
    value_1d_1 = up_raw[select_1d_1, :]
    value_1d_2 = up_raw[select_1d_2, :]
    value_2d_1 = up_raw[select_2d_1, :]
    value_2d_2 = up_raw[select_2d_2, :]

    # 站立时间特征计算
    feature_up_01 = np.nansum(value_cur_1)
    feature_up_02 = np.nansum(value_cur_1) - np.nansum(value_1d_1)
    feature_up_03 = np.nansum(value_cur_1) - np.nansum(value_2d_1)
    feature_up_04 = np.nansum(value_1d_2) - np.nansum(value_2d_2)

    # 计算起立躺下次数变化趋势
    up_raw_diff = np.diff(up_raw)
    value_cur_1 = up_raw_diff[select_cur_1, :]
    value_cur_2 = up_raw_diff[select_cur_2, :]
    value_1d_1 = up_raw_diff[select_1d_1, :]
    value_1d_2 = up_raw_diff[select_1d_2, :]
    value_2d_1 = up_raw_diff[select_2d_1, :]
    value_2d_2 = np.zeros_like(value_cur_1)  # 避免索引问题

    # 起立躺下次数变化趋势
    feature_up_05 = np.sum(value_cur_1 != 0)
    feature_up_06 = np.sum(value_cur_1 != 0) / (np.sum(value_1d_1 != 0) + 0.1)
    feature_up_07 = np.sum(value_cur_1 != 0) / (np.sum(value_2d_1 != 0) + 0.1)
    feature_up_08 = np.sum(value_1d_2 != 0) / (np.sum(value_2d_2 != 0) + 0.1)

    # 起立特征计算
    feature_up_09 = count_continuous_ones(value_cur_1.flatten().tolist())
    feature_up_10 = count_continuous_ones(value_1d_1.flatten().tolist())
    feature_up_11 = feature_up_09 >= 30 and feature_up_10 == 0
    feature_up_12 = feature_up_09 / (feature_up_10 + 0.1)
    feature_up_13 = feature_up_09 - feature_up_10
    feature_up_14 = ((feature_up_10 == 0 and feature_up_12 >= 30) or
                    (feature_up_10 == 1 and feature_up_12 >= 40) or
                    (feature_up_10 == 2 and feature_up_12 >= 20) or
                    (feature_up_10 == 3 and feature_up_12 >= 15) or
                    (feature_up_10 == 4 and feature_up_12 >= 10) or
                    (feature_up_10 == 5 and feature_up_12 >= 8) or
                    (feature_up_10 == 6 and feature_up_12 >= 6) or
                    (10 >= feature_up_10 >= 7 and feature_up_12 >= 4) or
                    (feature_up_10 > 10 and feature_up_12 >= 2.5) or
                    (feature_up_10 > 25 and feature_up_12 >= 2) or
                    (feature_up_10 > 35 and feature_up_12 >= 1.5) or
                    (feature_up_10 > 50 and feature_up_13 >= 25))

    feature_up_15 = count_continuous_ones(np.zeros_like(value_1d_1).flatten().tolist())
    feature_up_16 = feature_up_09 >= feature_up_10 >= feature_up_15 and feature_up_09 != 0

    return np.array([
        feature_up_01, feature_up_02, feature_up_03,
        feature_up_04, feature_up_05, feature_up_06,
        feature_up_07, feature_up_08,
        feature_up_09,
        feature_up_10, feature_up_11, feature_up_12,
        feature_up_13, feature_up_14, feature_up_15,
        feature_up_16
    ])


def _compute_sleep_features(up_raw, hour_ind_1d, cur_hour_ind, hour_ind_2d) -> np.ndarray:
    """计算睡眠特征"""

    def get_sleep(sleep):
        result = [0, 0, 0, 0]
        i = 0
        while i < len(sleep):
            time = 0
            while i < len(sleep) and sleep[i] != 0:
                i += 1
            while i < len(sleep) and sleep[i] != 1:
                i += 1
                time += 1
            if time >= 120:
                result[3] += time
            elif time >= 60:
                result[2] += time
            elif time >= 20:
                result[1] += time
            elif time > 0:
                result[0] += time
            else:
                pass
        result.append(len(sleep) - sum(result))  # 清醒时间
        result = [s / len(sleep) for s in result]  # 各睡眠状态时间分布
        return result

    # 计算睡眠特征
    today_sleep = up_raw[hour_ind_1d + 20: cur_hour_ind + 7].flatten().tolist()
    yesterday_sleep = up_raw[hour_ind_2d + 20: hour_ind_1d + 7].flatten().tolist()
    today_sleep_con = get_sleep(today_sleep)
    yesterday_sleep_con = get_sleep(yesterday_sleep)

    feature_sleep_01 = today_sleep_con[0] - yesterday_sleep_con[0]
    feature_sleep_02 = today_sleep_con[1] - yesterday_sleep_con[1]
    feature_sleep_03 = today_sleep_con[2] - yesterday_sleep_con[2]
    feature_sleep_04 = today_sleep_con[3] - yesterday_sleep_con[3]
    feature_sleep_05 = kl_divergence(today_sleep_con, yesterday_sleep_con)
    feature_sleep_06 = today_sleep_con[0]
    feature_sleep_07 = today_sleep_con[1]
    feature_sleep_08 = today_sleep_con[2]
    feature_sleep_09 = today_sleep_con[3]

    return np.array([
        feature_sleep_01, feature_sleep_02, feature_sleep_03,
        feature_sleep_04, feature_sleep_05, feature_sleep_06,
        feature_sleep_07, feature_sleep_08, feature_sleep_09
    ])


def _compute_78_features(act_raw, br_raw, up_raw, cur_hour_ind, hour_ind_1d, shift_date, raw_data, include_mean_features) -> np.ndarray:
    """计算7-8小时特征"""

    select_today = range(cur_hour_ind + 7, cur_hour_ind + 8)
    select_yesterday = range(hour_ind_1d + 7, hour_ind_1d + 8)

    # 计算平均特征
    mean_features = {}
    if include_mean_features:
        mean_features = compute_meanfeatures(raw_data, shift_date)
        if not mean_features:
            mean_features = {
                'mean_feature_up01': 0, 'mean_feature_up02': 0, 'mean_feature_up03': [],
                'mean_feature_act01': 0, 'mean_feature_act02': 0, 'mean_feature_act03': 0,
                'mean_feature_act04': []
            }

    # 7-8小时起卧特征
    up_value_today = up_raw[select_today, :]
    up_value_yesterday = up_raw[select_yesterday, :]
    feature78_up_01 = np.nansum(up_value_today)
    feature78_up_02 = np.nansum(up_value_today) - np.nansum(up_value_yesterday)
    feature78_up_03 = np.nansum(up_value_today) - mean_features.get('mean_feature_up01', 0)
    feature78_up_04 = count_continuous_ones(up_value_today.flatten().tolist())
    feature78_up_05 = feature78_up_04 - count_continuous_ones(up_value_yesterday.flatten().tolist())
    feature78_up_06 = feature78_up_04 - mean_features.get('mean_feature_up02', 0)

    from dtaidistance import dtw
    feature78_up_07 = compute_windows(up_value_today.flatten(), window=10, step=10, type='up')
    if len(feature78_up_07) > 0 and len(compute_windows(up_value_yesterday.flatten(), window=10, step=10, type='up')) > 0:
        feature78_up_08 = dtw.distance(
            feature78_up_07,
            compute_windows(up_value_yesterday.flatten(), window=10, step=10, type='up')
        )
    else:
        feature78_up_08 = 0

    if len(feature78_up_07) > 0 and len(mean_features.get('mean_feature_up03', [])) > 0:
        feature78_up_09 = dtw.distance(feature78_up_07, mean_features['mean_feature_up03'])
    else:
        feature78_up_09 = 0

    # 7-8小时活动特征
    act_value_today = act_raw[select_today, :]
    act_value_yesterday = act_raw[select_yesterday, :]
    feature78_act_01 = np.nansum(act_value_today)
    feature78_act_02 = np.nansum(act_value_today) - np.nansum(act_value_yesterday)
    feature78_act_03 = np.nansum(act_value_today) - mean_features.get('mean_feature_act01', 0)
    feature78_act_04 = np.sum(act_value_today > 4)
    feature78_act_05 = np.sum(act_value_today > 4) - np.sum(act_value_yesterday > 4)
    feature78_act_06 = np.sum(act_value_today > 4) - mean_features.get('mean_feature_act02', 0)
    feature78_act_07 = np.sum(act_value_today > 15)
    feature78_act_08 = np.sum(act_value_today > 15) - np.sum(act_value_yesterday > 15)
    feature78_act_09 = np.sum(act_value_today > 15) - mean_features.get('mean_feature_act03', 0)

    feature78_act_10 = compute_windows(act_value_today.flatten(), window=10, step=10, type='act')
    if len(feature78_act_10) > 0 and len(compute_windows(act_value_yesterday.flatten(), window=10, step=10, type='act')) > 0:
        feature78_act_11 = dtw.distance(
            feature78_act_10,
            compute_windows(act_value_yesterday.flatten(), window=10, step=10, type='act')
        )
    else:
        feature78_act_11 = 0

    if len(feature78_act_10) > 0 and len(mean_features.get('mean_feature_act04', [])) > 0:
        feature78_act_12 = dtw.distance(feature78_act_10, mean_features['mean_feature_act04'])
    else:
        feature78_act_12 = 0

    return np.array([
        feature78_act_01, feature78_act_02, feature78_act_03,
        feature78_act_04, feature78_act_05, feature78_act_06,
        feature78_act_07, feature78_act_08, feature78_act_09,
        feature78_act_11, feature78_act_12,
        feature78_up_01, feature78_up_02, feature78_up_03,
        feature78_up_04, feature78_up_05, feature78_up_06,
        feature78_up_08, feature78_up_09
    ])


def _get_feature_columns(include_78_features: bool) -> List[str]:
    """获取特征列名"""

    feature_columns = [
        # 活动特征 (18个)
        'act_01', 'act_02', 'act_03', 'act_04', 'act_05', 'act_06',
        'act_07', 'act_08', 'act_09', 'act_10', 'act_11', 'act_12',
        'act_13', 'act_14', 'act_15', 'act_16', 'act_17', 'act_18',

        # 呼吸特征 (20个)
        'br_01', 'br_02', 'br_03', 'br_04', 'br_05', 'br_06',
        'br_07', 'br_08', 'br_09', 'br_10', 'br_11', 'br_12',
        'br_13', 'br_14', 'br_15', 'br_16', 'br_17', 'br_18',
        'br_19', 'br_20',

        # 起卧特征 (16个)
        'up_01', 'up_02', 'up_03', 'up_04', 'up_05', 'up_06',
        'up_07', 'up_08', 'up_09', 'up_10', 'up_11', 'up_12',
        'up_13', 'up_14', 'up_15', 'up_16',

        # 睡眠特征 (9个)
        'sleep_01', 'sleep_02', 'sleep_03', 'sleep_04', 'sleep_05',
        'sleep_06', 'sleep_07', 'sleep_08', 'sleep_09'
    ]

    if include_78_features:
        # 7-8小时特征 (31个)
        feature_columns.extend([
            'act78_01', 'act78_02', 'act78_03', 'act78_04', 'act78_05', 'act78_06',
            'act78_07', 'act78_08', 'act78_09', 'act78_11', 'act78_12',
            'up78_01', 'up78_02', 'up78_03', 'up78_04', 'up78_05', 'up78_06',
            'up78_08', 'up78_09'
        ])

    return feature_columns