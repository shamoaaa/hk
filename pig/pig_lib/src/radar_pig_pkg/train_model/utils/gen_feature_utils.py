"""
通用特征生成工具模块
提供了灵活的特征生成函数，可以统一处理不同类型的特征生成需求
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Union, List, Tuple, Dict, Optional
import logging
from utils.common_utils import replace_negative_br_with_nan

# 设置日志记录
logger = logging.getLogger(__name__)

def generate_features_with_metadata(
    dataset: List[Tuple],
    exp_list: List[str],
    # ... 其他参数保持不变 ...
    time_window_days: int = 3,
    start_hour: int = 0,
    end_hour: int = 7,
    evening_start_hour: int = 20,
    evening_end_hour: int = 24,
    include_78_features: bool = False,
    shift_date: int = 0,
    interval_type: str = '-2_0',
) -> pd.DataFrame:
    """
    生成猪只活动特征，并保留 experiment, pig_id, date 信息，用于对齐标签
    """
    all_features = []
    
    # 对 hr呼吸率 = -1 的值 改为 NAN
    dataset = replace_negative_br_with_nan(dataset)

    for item in dataset:
        experiment = item[0]
        if experiment not in exp_list:
            continue

        pig_id = item[1]
        dates = item[2]
        raw_data = item[3]

        dates_cnt = len(dates)
        hour_step = 2 * 60

        # 数据预处理
        act_raw = raw_data['normalize_act'].values.reshape(-1, hour_step)
        # br_raw = raw_data['br'].values.reshape(-1, hour_step)
        br_raw = raw_data['new_br'].values.reshape(-1, hour_step)
        up_raw = raw_data['is_up'].values.reshape(-1, hour_step)

        # 开始计算特征
        for i in range(time_window_days - 1, dates_cnt):
            # 获取当前对应的日期
            # 注意：dates 是字符串列表，dates[i] 就是当前特征对应的日期
            current_date = dates[i]
            
            cur_hour_ind = i * 24
            hour_ind_1d = (i - 1) * 24
            hour_ind_2d = (i - 2) * 24

            # 时间窗口选择 (保持原代码不变)
            select_cur_1 = range(cur_hour_ind, (cur_hour_ind + end_hour))
            select_cur_2 = range((cur_hour_ind + evening_start_hour), (cur_hour_ind + evening_end_hour))
            select_1d_1 = range(hour_ind_1d, (hour_ind_1d + end_hour))
            select_1d_2 = range((hour_ind_1d + evening_start_hour), (hour_ind_1d + evening_end_hour))
            select_2d_1 = range(hour_ind_2d, (hour_ind_2d + end_hour))
            select_2d_2 = range((hour_ind_2d + evening_start_hour), (hour_ind_2d + evening_end_hour))

            try:
                features = _compute_feature_set(
                    act_raw, br_raw, up_raw,
                    select_cur_1, select_cur_2, select_1d_1, select_1d_2, select_2d_1, select_2d_2,
                    include_78_features, include_mean_features=include_78_features,
                    cur_hour_ind=cur_hour_ind, hour_ind_1d=hour_ind_1d, hour_ind_2d=hour_ind_2d,
                    shift_date=shift_date, raw_data=raw_data, hour_step=hour_step
                )

                # 关键修改：如果包含 NaN，直接跳过 (不添加到列表)，这保证了特征和元数据的一致性
                if np.isnan(features).sum() != 0:
                    continue

                # 将元数据加入到特征列表中
                # 假设 _compute_feature_set 返回的是 list 或 array
                # 我们将其转为 list 并在前面加上元数据
                feature_row = list(features)
                feature_row.insert(0, current_date)    # 在第0列插入 date
                feature_row.insert(0, pig_id)          # 在第0列插入 pig_id
                feature_row.insert(0, experiment)      # 在第0列插入 experiment
                
                all_features.append(feature_row)

            except Exception as e:
                continue

    # 定义列名：元数据列 + 特征列
    metadata_columns = ['experiment', 'pig_id', 'date']
    feature_columns = _get_feature_columns(include_78_features)
    final_columns = metadata_columns + feature_columns
    
    f_df = pd.DataFrame(all_features, columns=final_columns)
    
    return f_df


def generate_features_only(
    dataset: List[Tuple],
    exp_list: List[str],
    time_window_days: int = 3,
    start_hour: int = 0,
    end_hour: int = 7,
    evening_start_hour: int = 20,
    evening_end_hour: int = 24,
    include_78_features: bool = False,
    shift_date: int = 0,
    interval_type: str = '-2_0',
) -> pd.DataFrame:
    """
    生成猪只活动特征的函数

    参数:
        dataset: 数据集，格式为[(实验名称, 猪id, 实验日期列表, 猪特征的dataframe形式),()]
        exp_list: 需要计算特征的实验名称列表
        time_window_days: 时间窗口天数，默认为3天
        start_hour: 开始小时，默认为0点
        end_hour: 结束小时，默认为7点
        evening_start_hour: 晚间开始小时，默认为20点
        evening_end_hour: 晚间结束小时，默认为24点
        include_78_features: 是否包含7-8小时的特征，默认为False
        shift_date: 用于计算平均特征的偏移日期
        interval_type: 区间类型，例如[-2,0]

    返回:
        pd.DataFrame: 生成的特征DataFrame
    """
    all_features = []
    
    #对 hr呼吸率 = -1 的值 改为 NAN
    dataset = replace_negative_br_with_nan(dataset)

    for item in dataset:
        experiment = item[0]
        if experiment not in exp_list:
            continue

        pig_id = item[1]
        dates = item[2]
        raw_data = item[3]

        dates_cnt = len(dates)
        hour_step = 2 * 60

        # 数据预处理
        act_raw = raw_data['normalize_act'].values.reshape(-1, hour_step)
        br_raw = raw_data['br'].values.reshape(-1, hour_step)
        up_raw = raw_data['is_up'].values.reshape(-1, hour_step)

        # 开始计算特征
        for i in range(time_window_days - 1, dates_cnt):
            cur_hour_ind = i * 24
            hour_ind_1d = (i - 1) * 24
            hour_ind_2d = (i - 2) * 24

            # 时间窗口选择
            select_cur_1 = range(cur_hour_ind, (cur_hour_ind + end_hour))  # 当前天的开始到结束
            select_cur_2 = range((cur_hour_ind + evening_start_hour), (cur_hour_ind + evening_end_hour))  # 当前天的晚间
            select_1d_1 = range(hour_ind_1d, (hour_ind_1d + end_hour))  # 前一天的开始到结束
            select_1d_2 = range((hour_ind_1d + evening_start_hour), (hour_ind_1d + evening_end_hour))  # 前一天的晚间
            select_2d_1 = range(hour_ind_2d, (hour_ind_2d + end_hour))  # 前两天的开始到结束
            select_2d_2 = range((hour_ind_2d + evening_start_hour), (hour_ind_2d + evening_end_hour))  # 前两天的晚间

            try:
                # 计算特征
                features = _compute_feature_set(
                    act_raw, br_raw, up_raw,
                    select_cur_1, select_cur_2, select_1d_1, select_1d_2, select_2d_1, select_2d_2,
                    include_78_features, include_mean_features=include_78_features,
                    cur_hour_ind=cur_hour_ind, hour_ind_1d=hour_ind_1d, hour_ind_2d=hour_ind_2d,
                    shift_date=shift_date, raw_data=raw_data, hour_step=hour_step
                )

                if np.isnan(features).sum() != 0:
                    continue

                all_features.append(features)

            except Exception as e:
                continue

    # 创建特征DataFrame
    feature_columns = _get_feature_columns(include_78_features)
    f_df = pd.DataFrame(all_features, columns=feature_columns)
    
    return f_df


def generate_features_unified(
    dataset: List[Tuple],
    label_path: str,
    exp_list: List[str],
    type: str = 'train',
    label_type: str = 'breeding',
    time_window_days: int = 3,
    start_hour: int = 0,
    end_hour: int = 7,
    evening_start_hour: int = 20,
    evening_end_hour: int = 24,
    breeding_offset_days: int = 2,
    include_78_features: bool = False,
    shift_date: int = 0,
    enable_logging: bool = True,
    interval_type: str = '-2_0',
    is_completedata: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    统一的特征生成函数，支持多种特征生成配置

    参数:
        dataset: 数据集，格式为[(实验名称, 猪id, 实验日期列表, 猪特征的dataframe形式),()]
        label_path: 实验真实标签路径
        exp_list: 需要计算特征的实验名称列表
        type: 'train' 或 'test', 'test'时会剔除掉发情未配的猪
        label_type: 标签类型，可选:
            - 'oestrus': 基于发情标签生成特征 (使用pig_label[1])
            - 'breeding': 基于配种标签生成特征 (使用pig_label[2])
            - 'breeding_offset': 基于配种标签生成特征，并减去偏移天数作为发情时间
        time_window_days: 时间窗口天数，默认为3天
        start_hour: 开始小时，默认为0点
        end_hour: 结束小时，默认为7点
        evening_start_hour: 晚间开始小时，默认为20点
        evening_end_hour: 晚间结束小时，默认为24点
        breeding_offset_days: 配种标签的偏移天数，用于计算发情时间
        include_78_features: 是否包含7-8小时的特征，默认为False
        shift_date: 用于计算平均特征的偏移日期
        enable_logging: 是否启用日志记录
        interval_type: 区间类型，例如[-2,0]
        is_completedata: 是否补全数据

    返回:
        Tuple[pd.DataFrame, pd.DataFrame]: (特征DataFrame, 标签DataFrame)
    """

    if enable_logging:
        logger.info(f"开始生成特征 - 类型: {label_type}, 时间窗口: {time_window_days}天")

    # 根据feature_type确定label_column
    feature_type_config = {
        'oestrus': {'label_column': 1, 'breeding_offset_days': 0},
        'breeding': {'label_column': 2, 'breeding_offset_days': 0},
        'breeding_offset': {'label_column': 2, 'breeding_offset_days': breeding_offset_days}
    }

    if label_type not in feature_type_config:
        raise ValueError(f"不支持的feature_type: {label_type}")

    config = feature_type_config[label_type]
    label_column = config['label_column']
    
    #对 hr呼吸率 = -1 的值 改为 NAN
    dataset = replace_negative_br_with_nan(dataset)

    all_features = []
    all_labels = []
    nan_sample_count = 0  # 用于统计含 NaN 的样本数
    for item in dataset:
        experiment = item[0]
        if experiment not in exp_list:
            continue

        pig_id = item[1]
        dates = item[2]
        raw_data = item[3]

        if enable_logging:
            logger.debug(f"处理实验: {experiment}, 猪ID: {pig_id}")

        label_file = os.path.join(label_path, f'{experiment}.csv')
        if not os.path.exists(label_file):
            print(f'----\tNo label file found for {experiment}')
            continue
        label_data = pd.read_csv(label_file, index_col=None, header=0, dtype=str)

        dates_cnt = len(dates)
        hour_step = 2 * 60

        # 数据预处理
        act_raw = raw_data['normalize_act'].values.reshape(-1, hour_step)
        # br_raw = raw_data['br'].values.reshape(-1, hour_step)
        br_raw = raw_data['new_br'].values.reshape(-1, hour_step)
        up_raw = raw_data['is_up'].values.reshape(-1, hour_step)

        radar_id = pig_id.split('_')[1]

        select_ind = label_data.iloc[:, 0] == radar_id
        if (not select_ind.any()):
            print(f'----\tNo label found for {radar_id} in {experiment}')
            continue
        else:
            pig_label = label_data[select_ind].iloc[0].tolist()
            
        print("pig_label:", pig_label)

        # 标签处理
        if len(pig_label[label_column]) < 8:
            # 淘汰标签为-1的猪
            if pig_label[label_column].strip() == '-1':
                if enable_logging:
                    logger.debug(f'{pig_id}淘汰')
                # print(f'{pig_id}淘汰')
                continue
            elif pig_label[label_column].strip() == '0':
                label_date = datetime(1970, 1, 1, 0, 0, 0)
            else:
                raise ValueError(f"label error {pig_label[label_column]}")
        else:
            label_date = datetime.strptime(pig_label[label_column][0:8], '%Y%m%d')

        # 处理配种标签偏移
        if label_type == 'breeding_offset':
            label_date = label_date - timedelta(days=breeding_offset_days)
            
        print("label_date:", label_date)
            
        
        # 新增：处理标签在前两天的情况
        dates_compare = datetime.strptime(str(dates[1]).strip(), '%Y%m%d')
        if len(dates) >= 2 and label_date <= dates_compare and label_date > datetime(1970, 1, 1, 0, 0, 0):
            print(f'{pig_id} 标签日期 {label_date} 未在 dates[1] ({dates[1]}) 之后，跳过')
            continue

        # 测试时剔除发情未配
        if type == 'test':
            if len(pig_label[1]) >= 8 and pig_label[2].strip() == '0':
                print(f'{pig_id}发情未配种，测试时剔除')
                continue
        

        # 开始计算特征
        for i in range(time_window_days - 1, dates_cnt):
            cur_hour_ind = i * 24
            hour_ind_1d = (i - 1) * 24
            hour_ind_2d = (i - 2) * 24

            # 时间窗口选择
            select_cur_1 = range(cur_hour_ind, (cur_hour_ind + end_hour))  # 当前天的开始到结束
            select_cur_2 = range((cur_hour_ind + evening_start_hour), (cur_hour_ind + evening_end_hour))  # 当前天的晚间
            select_1d_1 = range(hour_ind_1d, (hour_ind_1d + end_hour))  # 前一天的开始到结束
            select_1d_2 = range((hour_ind_1d + evening_start_hour), (hour_ind_1d + evening_end_hour))  # 前一天的晚间
            select_2d_1 = range(hour_ind_2d, (hour_ind_2d + end_hour))  # 前两天的开始到结束
            select_2d_2 = range((hour_ind_2d + evening_start_hour), (hour_ind_2d + evening_end_hour))  # 前两天的晚间

            try:
                # 计算特征
                features = _compute_feature_set(
                    act_raw, br_raw, up_raw,
                    select_cur_1, select_cur_2, select_1d_1, select_1d_2, select_2d_1, select_2d_2,
                    include_78_features, include_mean_features=include_78_features,
                    cur_hour_ind=cur_hour_ind, hour_ind_1d=hour_ind_1d, hour_ind_2d=hour_ind_2d,
                    shift_date=shift_date, raw_data=raw_data, hour_step=hour_step
                )

                if np.isnan(features).sum() != 0 and type == 'train':
                    # print('特征中存在nan值')
                    nan_sample_count += 1
                    continue


                all_features.append(features)

                # 生成标签
                label_info = _generate_label_info(
                    interval_type,
                    experiment, pig_id, dates[i],
                    datetime.strptime(dates[i], '%Y%m%d'), label_date
                )
                all_labels.append(label_info)

            except Exception as e:
                if enable_logging:
                    logger.error(f"计算特征失败 - 实验: {experiment}, 猪ID: {pig_id}, 错误: {str(e)}")
                continue

    # 创建特征DataFrame
    feature_columns = _get_feature_columns(include_78_features)
    f_df = pd.DataFrame(all_features, columns=feature_columns)

    # 创建标签DataFrame
    l_df = pd.DataFrame(all_labels, columns=[
        'experiment', 'pig_id', 'date', 'label', 'eva_label', 'oneday_label', 'towday_label'
    ])

    
    print(f"特征生成完成 - 样本数量: {len(f_df)}, 特征维度: {len(feature_columns)}")
    print(f"总共跳过了 {nan_sample_count} 个含 NaN 的样本")
        
    # 判断是否补全数据
    if is_completedata:
        f_df, l_df = complete_data(f_df,l_df,
                                   dataset,
                                   label_path,
                                   exp_list,
                                   interval_type)

    return f_df, l_df


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
    # value_2d_2 = np.zeros_like(value_cur_1)  # 避免索引问题
    value_2d_2 = up_raw_diff[select_2d_2, :]

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
    # feature_up_15 = count_continuous_ones(value_2d_1.flatten().tolist())
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


def _generate_label_info(interval_type,experiment, pig_id, date_str, cur_date, label_date) -> List:
    """生成标签信息"""
    
    date_delta = cur_date - label_date
    
    if interval_type == '-2_0':
        if date_delta == timedelta(days=-2):
            cur_label = 0
            eva_label = 1   # 提前两天报警
            towday_label = 1
            oneday_label = 0
        elif (date_delta == timedelta(days=-1)):
            cur_label = 0
            eva_label = 1   # 提前一天报警
            towday_label = 0
            oneday_label = 1
        elif (date_delta == timedelta(days=0)):
            cur_label = 1
            eva_label = 1
            towday_label = 0
            oneday_label = 0
        else:
            cur_label = 0
            eva_label = 0
            towday_label = 0
            oneday_label = 0

        return [experiment, pig_id, date_str, cur_label, eva_label, oneday_label, towday_label]
    
    if interval_type == '-1_0':
        if date_delta == timedelta(days=-2):
            cur_label = 0
            eva_label = 0   # 提前两天报警
            towday_label = 1
            oneday_label = 0
        elif (date_delta == timedelta(days=-1)):
            cur_label = 0
            eva_label = 1   # 提前一天报警
            towday_label = 0
            oneday_label = 1
        elif (date_delta == timedelta(days=0)):
            cur_label = 1
            eva_label = 1
            towday_label = 0
            oneday_label = 0
        else:
            cur_label = 0
            eva_label = 0
            towday_label = 0
            oneday_label = 0

        return [experiment, pig_id, date_str, cur_label, eva_label, oneday_label, towday_label]
    
    
    if interval_type == '-1_1':
        if date_delta == timedelta(days=-2):
            cur_label = 0
            eva_label = 0  
            towday_label = 1
            oneday_label = 0
        elif (date_delta == timedelta(days=-1)):
            cur_label = 0
            eva_label = 1   # 提前一天报警
            towday_label = 0
            oneday_label = 1
        elif (date_delta == timedelta(days=0)):
            cur_label = 1
            eva_label = 1
            towday_label = 0
            oneday_label = 0
        elif (date_delta == timedelta(days=1)):
            cur_label = 0
            eva_label = 1
            towday_label = 0
            oneday_label = 0
        else:
            cur_label = 0
            eva_label = 0
            towday_label = 0
            oneday_label = 0

        return [experiment, pig_id, date_str, cur_label, eva_label, oneday_label, towday_label]
    
    


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

def compute_meanfeatures(raw_data,shift_date=0):
    # 计算平均特征
    hour_step = 2 * 60
    act_raw = raw_data['normalize_act'].values.reshape(-1, hour_step)
    up_raw = raw_data['is_up'].values.reshape(-1, hour_step)
    mean_feature_up01 = [] # 前三天的起立时间
    mean_feature_up02 = [] # 前三天的起立次数
    mean_feature_up03 = [] # 前三天的滑窗起立结果
    mean_feature_getup = [] # 前三天的起床时间
    mean_feature_act01 = [] # 前三天的平均活动量
    mean_feature_act02 = [] # 前三天的平均高活动次数
    mean_feature_act03 = [] # 前三天的剧烈活动次数
    mean_feature_act04 = [] # 前三天的滑窗活动量结果

    # 跟踪前三天的特征
    use_day = 0
    for i in range(shift_date,shift_date+2):
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
    if use_day==0:
        return {}
    if use_day==2:
        mean_feature_up01 = np.mean(mean_feature_up01)
        mean_feature_up02 = np.mean(mean_feature_up02)
        mean_feature_up03 = np.mean(mean_feature_up03, axis=0)
        mean_feature_act01 = np.mean(mean_feature_act01)
        mean_feature_act02 = np.mean(mean_feature_act02)
        mean_feature_act03 = np.mean(mean_feature_act03)
        mean_feature_act04 = np.mean(mean_feature_act04, axis=0)
    else:
        mean_feature_up01 = mean_feature_up01[0]
        mean_feature_up02 = mean_feature_up02[0]
        mean_feature_up03 = mean_feature_up03[0]
        mean_feature_act01 = mean_feature_act01[0]
        mean_feature_act02 = mean_feature_act02[0]
        mean_feature_act03 = mean_feature_act03[0]
        mean_feature_act04 = mean_feature_act04[0]
    return {'mean_feature_up01': mean_feature_up01,
            'mean_feature_up02': mean_feature_up02,
            'mean_feature_up03': mean_feature_up03,
            'mean_feature_act01': mean_feature_act01,
            'mean_feature_act02': mean_feature_act02,
            'mean_feature_act03': mean_feature_act03,
            'mean_feature_act04': mean_feature_act04
            }


# 保持向后兼容性的包装函数
def generage_oestrus_features(dataset, label_path, exp_list, type='train'):
    """基于发情标签生成特征 - 向后兼容"""
    return generate_features_unified(
        dataset, label_path, exp_list, type,
        feature_type='oestrus', label_column=1
    )


def generage_breeding2_features(dataset, label_path, exp_list, type='train'):
    """基于配种标签生成特征并减去2天 - 向后兼容"""
    return generate_features_unified(
        dataset, label_path, exp_list, type,
        feature_type='breeding_offset', label_column=2, breeding_offset_days=2
    )


def generage_features(dataset, label_path, exp_list, type='train'):
    """基于配种标签生成特征 - 向后兼容"""
    return generate_features_unified(
        dataset, label_path, exp_list, type,
        feature_type='breeding', label_column=2
    )
    


def get_pig_target_days(dataset, label_path, exp_list, label_type='breeding', breeding_offset_days=2):
    """
    获取每只猪label对应的target_day日期，保存成字典

    参数:
    dataset: 数据集，格式为[(实验名称, 猪id, 实验日期列表, 猪特征的dataframe形式),()]
    label_path: 实验真实标签路径
    exp_list: 需要处理的实验名称列表
    label_type: 特征类型，可选:
        - 'oestrus': 基于发情标签生成特征 (使用pig_label[1])
        - 'breeding': 基于配种标签生成特征 (使用pig_label[2])
        - 'breeding_offset': 基于配种标签生成特征，并减去偏移天数作为发情时间
    breeding_offset_days: 配种标签的偏移天数，用于计算发情时间（仅breeding_offset类型有效）

    返回:
    dict: 字典格式为 {pig_id: target_date_str}，其中target_date_str是YYYYMMDD格式的字符串
    """

    target_days_dict = {}

    # 根据feature_type确定label_column
    feature_type_config = {
        'oestrus': {'label_column': 1, 'breeding_offset_days': 0},
        'breeding': {'label_column': 2, 'breeding_offset_days': 0},
        'breeding_offset': {'label_column': 2, 'breeding_offset_days': breeding_offset_days}
    }

    if label_type not in feature_type_config:
        raise ValueError(f"不支持的feature_type: {label_type}")

    config = feature_type_config[label_type]
    label_column = config['label_column']

    for item in dataset:
        experiment = item[0]
        if experiment not in exp_list:
            continue

        pig_id = item[1]
        dates = item[2]
        raw_data = item[3]

        # 读取标签文件
        label_file = os.path.join(label_path, f'{experiment}.csv')
        if not os.path.exists(label_file):
            print(f'----\tNo label file found for {experiment}')
            continue

        label_data = pd.read_csv(label_file, index_col=None, header=0, dtype=str)

        # 提取猪的ID部分（去除实验前缀）
        radar_id = pig_id.split('_')[1]

        # 查找该猪的标签
        select_ind = label_data.iloc[:, 0] == radar_id
        if (not select_ind.any()):
            print(f'----\tNo label found for {radar_id} in {experiment}')
            continue
        else:
            pig_label = label_data[select_ind].iloc[0].tolist()

        # 处理标签日期
        if len(pig_label[label_column]) < 8:
            # 淘汰标签为-1的猪
            if pig_label[label_column].strip() == '-1':
                print(f'{pig_id}标签为-1，跳过')
                continue
            elif pig_label[label_column].strip() == '0':
                print(f'{pig_id}标签为0，无配种时间')
                continue
            else:
                print(f'{pig_id}标签格式错误: {pig_label[label_column]}')
                continue
        else:
            # 解析标签日期
            try:
                label_date = datetime.strptime(pig_label[label_column][0:8], '%Y%m%d')
            except Exception as e:
                print(f'{pig_id}日期解析失败: {pig_label[label_column][0:8]}, 错误: {str(e)}')
                continue

        # 处理配种标签偏移（仅breeding_offset类型）
        if label_type == 'breeding_offset':
            target_date = label_date - timedelta(days=breeding_offset_days)
        else:
            target_date = label_date

        # 转换为字符串格式
        target_date_str = target_date.strftime('%Y%m%d')

        # 添加到字典
        target_days_dict[pig_id] = target_date_str

        # print(f'{pig_id}: 配种时间={label_date.strftime("%Y%m%d")}, target_day={target_date_str}')

    print(f"\n总共处理了 {len(target_days_dict)} 只猪的target_day")
    return target_days_dict


def complete_data(features_df, labels_df, dataset, label_path, exp_list, interval_type='-1_1'):
    """
    补全猪只数据
    
    新逻辑：
    1. 对于每头猪，找到该猪label==1的那一天breeding_day
    2. 找到该猪已有数据最早的一天earliest_day
    3. 如果breeding_day - earliest_day - 5 < 0，就要往前补全数据
    4. 补全天数 = 5 - (breeding_day - earliest_day).days
    5. feature_df用该猪earliest_day对应的特征增加高斯噪声补全
    6. labels_df中的eva_label根据interval_type确定：
       - interval_type=='-1_1': breeding_day的正负1区间内为1，其他为0
       - interval_type=='-2_0': breeding_day的前两天内为1，其他为0
    7. oneday_label表示breeding_day前一天为1，否则为0
    8. towday_label表示breeding_day前两天为1，否则为0
    
    Parameters:
    features_df: 特征数据框
    labels_df: 标签数据框
    dataset: 原始数据集，用于获取猪只的最早数据日期
    label_path: 标签文件路径
    exp_list: 实验列表
    interval_type: 区间类型，'-1_1'或'-2_0'
    """
    
    # 首先获取每只猪的target_day
    print("获取每只猪的target_day...")
    breeding_day_dict = get_pig_target_days(dataset, label_path, exp_list, label_type='breeding')
    
    np.random.seed(42)  # 固定种子
    
    # 确保所有日期格式统一为'YYYYMMDD'
    def normalize_date(date_val):
        """标准化日期格式为'YYYYMMDD'"""
        date_str = str(date_val)
        # 如果是 'YYYY-MM-DD HH:MM:SS' 格式
        if '-' in date_str and ':' in date_str:
            return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S').strftime('%Y%m%d')
        # 如果是 'YYYY-MM-DD' 格式
        elif '-' in date_str:
            return pd.to_datetime(date_str, format='%Y-%m-%d').strftime('%Y%m%d')
        # 如果已经是 'YYYYMMDD' 格式
        else:
            return date_str
    
    # 复制数据避免修改原数据
    features_df_copy = features_df.copy()
    labels_df_copy = labels_df.copy()
    
    # 合并原数据和新数据
    # 为原features_df添加pig_id和date列（如果还没有的话）
    if 'pig_id' not in features_df_copy.columns:
        # 假设features_df和labels_df的行是对应的
        features_df_copy['pig_id'] = labels_df_copy['pig_id'].values
    if 'date' not in features_df_copy.columns:
        features_df_copy['date'] = labels_df_copy['date'].values
    
    # 确保原features_df的日期格式也是'YYYYMMDD'
    features_df_copy['date'] = features_df_copy['date'].apply(normalize_date)
    # 处理labels_df的日期格式
    labels_df_copy['date'] = labels_df_copy['date'].apply(normalize_date)
    
    # 按猪只分组处理
    pig_groups = labels_df_copy.groupby('pig_id')
    
    # 存储需要添加的新数据
    new_features_rows = []
    new_labels_rows = []
    
    # 用于统计补全情况
    completion_stats = {
        'total_pigs': 0,
        'pigs_to_complete': 0,
        'days_completed': 0,
        'pigs_completed': 0
    }
    
    for pig_id, pig_data in pig_groups:
        completion_stats['total_pigs'] += 1
        
        # 从breeding_day_dict中获取该猪的breeding_day
        if pig_id in breeding_day_dict:
            breeding_day_str = breeding_day_dict[pig_id]
            breeding_day = pd.to_datetime(breeding_day_str, format='%Y%m%d')
        else:
            print(f"Warning: No breeding_day found for pig_id {pig_id}. Skipping.")
            continue
            
        # 找到该猪已有数据最早的一天
        earliest_day_str = pig_data['date'].min()
        earliest_day = pd.to_datetime(earliest_day_str, format='%Y%m%d')
        
        # 计算需要补全的天数
        days_before_breeding = (breeding_day - earliest_day).days
        days_to_complete = 5 - days_before_breeding
        
        if days_to_complete > 0:
            completion_stats['pigs_to_complete'] += 1
            print(f"Pig {pig_id}: earliest_day={earliest_day.strftime('%Y%m%d')}, breeding_day={breeding_day.strftime('%Y%m%d')}, days_to_complete={days_to_complete}")
            
            # 找到该猪在earliest_day对应的特征数据
            # 首先找到earliest_day对应的索引
            earliest_date_mask = pig_data['date'] == earliest_day_str
            if not earliest_date_mask.any():
                print(f"Warning: No data found for earliest_day {earliest_day_str} for pig {pig_id}. Skipping.")
                continue
                
            earliest_idx = pig_data[earliest_date_mask].index[0]
            earliest_features = features_df_copy.loc[earliest_idx]
            
            # 获取数值列
            numeric_cols = features_df_copy.select_dtypes(include=[np.number]).columns
            
            # 补全缺失的天数
            for i in range(1, days_to_complete + 1):
                # 计算补全的日期
                complete_date = earliest_day - timedelta(days=i)
                complete_date_str = complete_date.strftime('%Y%m%d')
                
                # 补全features_df：使用最早一天的特征增加高斯噪声
                new_features = earliest_features.copy()
                
                # 为数值列添加高斯噪声
                if len(numeric_cols) > 0:
                    # 计算相对噪声（使用特征标准差的1%）
                    feature_std = features_df_copy[features_df_copy['pig_id'] == pig_id][numeric_cols].std()
                    noise_scale = 0.01  # 1%的噪声
                    noise = np.random.normal(0, noise_scale * feature_std, len(numeric_cols))
                    new_features[numeric_cols] = new_features[numeric_cols] + noise
                
                # 创建新的特征行
                new_features_dict = new_features.to_dict()
                new_features_dict['pig_id'] = pig_id
                new_features_dict['date'] = complete_date_str
                new_features_rows.append(new_features_dict)
                
                # 补全labels_df
                new_label_row = {
                    'experiment': pig_data['experiment'].iloc[0],
                    'pig_id': pig_id,
                    'date': complete_date_str,
                    'label': 0,  # 补全的数据label为0
                }
                
                # 计算与breeding_day的差值
                days_diff = (breeding_day - complete_date).days
                
                # 根据interval_type计算eva_label
                if interval_type == '-1_1':
                    new_label_row['eva_label'] = 1 if -1 <= days_diff <= 1 else 0
                elif interval_type == '-2_0':
                    new_label_row['eva_label'] = 1 if -2 <= days_diff <= 0 else 0
                else:
                    new_label_row['eva_label'] = 0
                
                # 计算oneday_label和towday_label
                new_label_row['oneday_label'] = 1 if days_diff == 1 else 0
                new_label_row['towday_label'] = 1 if days_diff == 2 else 0
                
                new_labels_rows.append(new_label_row)
            
            completion_stats['days_completed'] += days_to_complete
            completion_stats['pigs_completed'] += 1
            print(f"  Completed {days_to_complete} days for pig {pig_id}")
    
    print(f"\n补全统计:")
    print(f"  总猪只数: {completion_stats['total_pigs']}")
    print(f"  需要补全的猪只数: {completion_stats['pigs_to_complete']}")
    print(f"  实际补全的猪只数: {completion_stats['pigs_completed']}")
    print(f"  总补全天数: {completion_stats['days_completed']}")
    
    # 如果有新数据需要添加
    if new_features_rows and new_labels_rows:
        # 创建新的DataFrame
        new_features_df = pd.DataFrame(new_features_rows)
        new_labels_df = pd.DataFrame(new_labels_rows)
        
        
        # 合并数据
        final_features_df = pd.concat([features_df_copy, new_features_df], ignore_index=True)
        final_labels_df = pd.concat([labels_df_copy, new_labels_df], ignore_index=True)
        
        # 按猪只ID和日期排序
        final_labels_df = final_labels_df.sort_values(['pig_id', 'date']).reset_index(drop=True)
        final_features_df = final_features_df.sort_values(['pig_id', 'date']).reset_index(drop=True)
        
        print(f"\n补全后数据量:")
        print(f"  特征数据: {len(features_df_copy)} -> {len(final_features_df)} (+{len(final_features_df) - len(features_df_copy)})")
        print(f"  标签数据: {len(labels_df_copy)} -> {len(final_labels_df)} (+{len(final_labels_df) - len(labels_df_copy)})")
        
        # 特阵删除"pig_id"和"date"
        model_feature_names = features_df.columns.drop(['pig_id', 'date'], errors='ignore').tolist()
        final_features_df = final_features_df[model_feature_names]
        
        return final_features_df, final_labels_df
    else:
        print("\n无需补全数据")
        return features_df_copy, labels_df_copy
    
    

def delete_data(features_df, labels_df, dataset, label_path, exp_list):
    """
    删除不满足条件的猪只数据
    
    删除规则：
    如果 breeding_day - earliest_day - 5 < 0，则删除该猪所有数据
    
    Parameters:
    features_df: 特征数据框
    labels_df: 标签数据框
    dataset: 原始数据集，用于获取猪只的最早数据日期
    label_path: 标签文件路径
    exp_list: 实验列表
    
    Returns:
    features_df_clean: 删除后的特征数据框
    labels_df_clean: 删除后的标签数据框
    """
    
    # 获取每只猪的breeding_day
    print("获取每只猪的breeding_day...")
    breeding_day_dict = get_pig_target_days(dataset, label_path, exp_list, label_type='breeding')
    
    # 复制数据避免修改原数据
    features_df_copy = features_df.copy()
    labels_df_copy = labels_df.copy()
    
    # 确保所有日期格式统一为'YYYYMMDD'
    def normalize_date(date_val):
        """标准化日期格式为'YYYYMMDD'"""
        date_str = str(date_val)
        # 如果是 'YYYY-MM-DD HH:MM:SS' 格式
        if '-' in date_str and ':' in date_str:
            return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S').strftime('%Y%m%d')
        # 如果是 'YYYY-MM-DD' 格式
        elif '-' in date_str:
            return pd.to_datetime(date_str, format='%Y-%m-%d').strftime('%Y%m%d')
        # 如果已经是 'YYYYMMDD' 格式
        else:
            return date_str
    
    # 处理labels_df的日期格式
    labels_df_copy['date'] = labels_df_copy['date'].apply(normalize_date)
    
    # 为features_df添加pig_id列（如果还没有的话）
    if 'pig_id' not in features_df_copy.columns:
        features_df_copy['pig_id'] = labels_df_copy['pig_id'].values
    
    # 按猪只分组处理
    pig_groups = labels_df_copy.groupby('pig_id')
    
    # 存储需要删除的猪只ID
    delete_pig_ids = []
    
    # 用于统计
    total_pigs = 0
    pigs_deleted = 0
    
    print("\n检查每只猪的数据完整性...")
    
    for pig_id, pig_data in pig_groups:
        total_pigs += 1
        
        # 从breeding_day_dict中获取该猪的breeding_day
        if pig_id in breeding_day_dict:
            breeding_day_str = breeding_day_dict[pig_id]
            breeding_day = pd.to_datetime(breeding_day_str, format='%Y%m%d')
        else:
            print(f"Warning: No breeding_day found for pig_id {pig_id}. Skipping.")
            continue
            
        # 找到该猪已有数据最早的一天
        earliest_day_str = pig_data['date'].min()
        earliest_day = pd.to_datetime(earliest_day_str, format='%Y%m%d')
        
        # 计算 breeding_day - earliest_day - 5
        days_diff = (breeding_day - earliest_day).days - 5
        
        # 如果 < 0，则删除该猪数据
        if days_diff < 0:
            delete_pig_ids.append(pig_id)
            pigs_deleted += 1
            print(f"Pig {pig_id}: breeding_day={breeding_day.strftime('%Y%m%d')}, "
                  f"earliest_day={earliest_day.strftime('%Y%m%d')}, "
                  f"diff={days_diff} < 0, 删除该猪数据")
    
    # 删除不满足条件的猪只数据
    if delete_pig_ids:
        print(f"\n删除的猪只: {delete_pig_ids}")
        labels_df_clean = labels_df_copy[~labels_df_copy['pig_id'].isin(delete_pig_ids)]
        features_df_clean = features_df_copy[~features_df_copy['pig_id'].isin(delete_pig_ids)]
    else:
        print("\n没有需要删除的猪只数据")
        labels_df_clean = labels_df_copy
        features_df_clean = features_df_copy
    
    # 删除pig_id列（如果原始features_df中没有）
    if 'pig_id' not in features_df.columns:
        features_df_clean = features_df_clean.drop('pig_id', axis=1, errors='ignore')
    
    print(f"\n删除统计:")
    print(f"  总猪只数: {total_pigs}")
    print(f"  删除猪只数: {pigs_deleted}")
    print(f"  剩余猪只数: {total_pigs - pigs_deleted}")
    print(f"  特征数据量: {len(features_df_copy)} -> {len(features_df_clean)}")
    print(f"  标签数据量: {len(labels_df_copy)} -> {len(labels_df_clean)}")
    
    return features_df_clean, labels_df_clean