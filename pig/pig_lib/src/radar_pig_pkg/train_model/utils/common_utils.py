import pickle
from typing import List, Tuple, Union
import multiprocessing as mp
from functools import partial
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

def count_negative_br(new_dataset):
    """
    统计new_dataset中所有br=-1的总数量
    
    参数:
        new_dataset: 二维列表，其中每个元素的[3]是一个DataFrame
    
    返回:
        int: br=-1的总数量
    """
    total_count = 0
    for item in new_dataset:
        df = item[3]  # 获取DataFrame
        # 统计当前DataFrame中br=-1的数量
        count = (df['br'] == -1).sum()
        total_count += count
    
    return total_count

def replace_negative_br_with_nan(new_dataset):
    """
    将new_dataset中所有br=-1的值替换为NaN
    
    参数:
        new_dataset: 二维列表，其中每个元素的[3]是一个DataFrame
    
    返回:
        修改后的new_dataset（原地修改，无需返回）
    """
    for item in new_dataset:
        df = item[3]  # 获取DataFrame
        # 将br=-1的值替换为NaN
        df.loc[df['br'] == -1, 'br'] = np.nan
    
    return new_dataset  # 返回修改后的数据（虽然已原地修改）


def convert_features_to_numeric(features_df):
    """
    将features_df中的object类型列转换为数值类型
    
    参数:
        features_df: 特征DataFrame
    
    返回:
        转换后的DataFrame
    """
    # 创建副本以避免修改原始数据
    df = features_df.copy()
    
    # 遍历所有列
    for col in df.columns:
        # 如果列的数据类型是object，尝试转换为数值
        if df[col].dtype == 'object':
            try:
                # 尝试转换为float
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                # 如果转换失败，可能是字符串类型，可以尝试其他处理方式
                print(f"警告: 列 {col} 无法转换为数值类型")
    
    return df


def find_pig_data(dataset, pig_id):
    """
    在dataset中查找指定pig_id的数据
    
    参数:
        dataset: 二维列表
        pig_id: 要查找的pig_id
    
    返回:
        找到的实验数据（整个元素），如果没找到返回None
    """
    for item in dataset:
        if item[1] == pig_id:  # 假设实验名在索引0的位置
            return item
    return -1

def find_first_one(nums):
    """
    找出列表中第一个1的下标
    Args:
        nums: 输入的数字列表
    Returns:
        第一个1的下标，如果不存在则返回-1
    """
    for i in range(len(nums)):
        if nums[i] == 1:
            return i
    return -1

def find_last_one(nums):
    """
    找出列表中最后一个1的下标
    Args:
        nums: 输入的数字列表
    Returns:
        最后一个1的下标，如果不存在则返回-1
    """
    for i in range(len(nums)-1, -1, -1):
        if nums[i] == 1:
            return i
    return -1


def calculate_pig_sign(select_pred, select_label):
    """
    计算单个猪的标志值
    Args:
        select_pred: 预测值列表
        select_eva_label: 真实标签列表
    Returns:
        sign: 标志值
    """
    sign = "未预测"
    pred_day = find_first_one(select_pred)
    true_day = find_last_one(select_label)
    if(true_day == -1 and pred_day == -1):
        return "未配种/发情 且 未预测"
    if(true_day == -1 and pred_day != -1):
        return "未配种/发情 但 预测"
    if(true_day != -1 and pred_day != -1):
        sign = str(pred_day - true_day)
    return sign



def add_missing_data(item, date_delta, noise_std=0):
    """
    根据date_delta添加缺失的数据
    
    Parameters:
    item: list - [exp_name, file_name, date_list, df]
    date_delta: datetime.timedelta - 时间差
    noise_std: float - 高斯噪声标准差
    
    Returns:
    list - 添加后的item
    """
    
    np.random.seed(42)  # 固定种子
    
    exp_name, file_name, date_list, df = item
    
    missing_days = date_delta.days
    rows_per_day = 2880
    
    # 解析第一个有效日期用于添加
    first_valid_date = None
    for date_str in date_list:
        if date_str.strip() not in ['0', '-1'] and isinstance(date_str, str):
            first_valid_date = date_str
            break
    
    if first_valid_date is None:
        print("警告: 未找到有效的日期用于添加")
        return item
    
    # 转换第一个有效日期为datetime对象
    try:
        if len(first_valid_date) == 8:  # YYYYMMDD
            first_date = datetime.strptime(first_valid_date, '%Y%m%d')
        else:
            # 处理带小时的日期格式
            first_date = datetime.strptime(first_valid_date[:8], '%Y%m%d')
    except:
        print(f"警告: 无法解析日期 {first_valid_date}")
        return item
    
    if missing_days < 0:
        num_missing_days = abs(missing_days)
        total_missing_rows = num_missing_days * rows_per_day
        
        # 取第一天的数据（0~2879行）
        first_day_data = df.iloc[:rows_per_day].copy()
        
        # 生成需要添加的日期列表
        missing_dates = []
        add_data_list = []
        
        for i in range(num_missing_days):
            # 计算需要添加的日期（从第一个有效日期往前推）
            add_date = first_date - timedelta(days=num_missing_days-i)
            add_date_str = add_date.strftime('%Y%m%d')
            missing_dates.append(add_date_str)
            
            # 复制第一天的数据
            day_copy = first_day_data.copy()
            
            # 对数值列添加高斯噪声（除了ideal_channel和alive列）
            for col in day_copy.select_dtypes(include=[np.number]).columns:
                # ideal_channel和alive列直接复制，不添加噪声
                if col not in ['ideal_channel', 'alive']:
                    noise = np.random.normal(0, noise_std, size=day_copy.shape[0])
                    day_copy[col] = day_copy[col] + noise
            
            add_data_list.append(day_copy)
        
        # 拼接所有添加的数据
        if add_data_list:
            add_df = pd.concat(add_data_list, ignore_index=True)
            
            # 将添加数据放在原始数据前面
            df = pd.concat([add_df, df], ignore_index=True)
            
            # 更新日期列表（将添加的日期加到前面）
            date_list = missing_dates + date_list
            
            print(f"已添加 {num_missing_days} 天数据，共 {total_missing_rows} 行")
            print(f"添加日期: {missing_dates}")
        else:
            print("警告: 未生成添加数据")
    else:
        print("无需添加数据")
    
    # 返回更新后的item（列表格式）
    return [exp_name, file_name, date_list, df]


def compare_added_data(new_item, date_delta, rows_per_day=2880):
    """
    显示补全数据的前10条（所有补的天数都显示）
    
    Parameters:
    new_item: list - 添加数据后的item [exp_name, file_name, date_list, df]
    date_delta: datetime.timedelta - 时间差
    rows_per_day: int - 每天的数据行数，默认2880
    
    Returns:
    None - 直接打印显示结果
    """
    exp_name, file_name, date_list, df = new_item
    
    missing_days = date_delta.days
    
    if missing_days >= 0:
        print("无需显示，没有添加数据")
        return
    
    num_missing_days = abs(missing_days)
    
    print(f"=== 补全数据展示 ===")
    print(f"补全天数: {num_missing_days}")
    print(f"补全猪只: {file_name}")
    print(f"补全日期: {date_list[:num_missing_days]}")
    print(f"原始第一天日期: {date_list[num_missing_days]}")
    
    # 显示每一天补全数据的前10条
    for i in range(num_missing_days):
        print(f"\n--- 第{i+1}天补全数据 (日期: {date_list[i]}) ---")
        start_idx = i * rows_per_day
        end_idx = start_idx + 10  # 只取前10条
        day_data = df.iloc[start_idx:end_idx].copy()
        print(day_data)
    
    # 显示原始第一天数据的前10条
    print(f"\n--- 原始第一天数据 (日期: {date_list[num_missing_days]}) ---")
    start_idx = num_missing_days * rows_per_day
    end_idx = start_idx + 10
    original_day_data = df.iloc[start_idx:end_idx].copy()
    print(original_day_data)
    
    
def add_data(dataset, label_path):
    new_dataset = []
    for item in dataset:
        experiment = item[0]  
        pig_id = item[1]  
        dates = item[2]   
        raw_data = item[3]  

        label_file = os.path.join(label_path, f'{experiment}.csv')
        if not os.path.exists(label_file):
            print(f'----\tNo label file found for {experiment}')
            continue
        label_data = pd.read_csv(label_file, index_col=None, header=0, dtype=str)
        
        # 选出不为0和-1的数据中的最早时间（使用strip()处理空白字符）
        valid_times = label_data.iloc[:,2][
            (label_data.iloc[:,2].notna()) &  # 排除NaN值
            (label_data.iloc[:,2].str.strip() != '0') & 
            (label_data.iloc[:,2].str.strip() != '-1') &
            (label_data.iloc[:,2].str.strip() != '')
        ]
        
        if len(valid_times) > 0:
            # 获取最小值并确保是字符串
            earliest_valid_time_str = str(valid_times.min())
            # 使用strip()清理空白字符后再解析
            earliest_breeding_time = datetime.strptime(earliest_valid_time_str.strip()[0:8],'%Y%m%d')
            print(f"最早时间: {earliest_breeding_time}")
        else:
            print("没有找到有效的非零时间数据")
            continue
            
        # 选出当前最早数据时间
        first_date_str = dates[0]
        if isinstance(first_date_str, str):
            first_date_str = first_date_str.strip()
            earliest_time = datetime.strptime(first_date_str[0:8],'%Y%m%d')
        else:
            print("当前数据日期格式错误")
            continue
        
        # 计算差值
        delta = earliest_breeding_time - timedelta(days=5) - earliest_time
        
        # 添加新数据
        new_item = add_missing_data(item, delta)
        
        # 补全展示
        compare_added_data(new_item, delta)

        new_dataset.append(new_item)

    return new_dataset



def _process_single_item(item, noise_scale: float = 0.1):
    """处理单个item - 移到模块级别以便pickle"""
    np.random.seed(42)
    if len(item) >= 4 and isinstance(item[3], pd.DataFrame):
        df = item[3]

        # 预计算统计信息
        col_std = df.std(skipna=True)
        col_mean = df.mean(skipna=True)
        col_min = df.min(skipna=True)  # 新增：每列的最小值
        col_std[col_std == 0] = 1e-8

        df_filled = df.copy()
        n_rows = len(df)
        nan_mask = df.isna()

        # 为每列处理
        for col_idx, col_name in enumerate(df.columns):
            if nan_mask[col_name].sum() == 0:
                continue

            nan_indices = np.where(nan_mask[col_name])[0]

            # 为每个NaN位置寻找填充值 - 从后往前遍历
            for nan_idx in reversed(nan_indices):  # 修改：使用reversed从后往前遍历
                if nan_idx < 2400:
                    # 2400行之前：使用当前逻辑
                    fill_value = None
                    for day_offset in range(1, 8):
                        target_idx = nan_idx + day_offset * 2880
                        if target_idx >= n_rows:
                            fill_value = col_mean[col_name]
                            if pd.isna(fill_value):
                                fill_value = 0
                            break
                        if not pd.isna(df.iloc[target_idx, col_idx]):
                            fill_value = df.iloc[target_idx, col_idx]
                            break
                    else:
                        fill_value = col_mean[col_name]
                        if pd.isna(fill_value):
                            fill_value = 0

                    # 添加噪声
                    noise = np.random.normal(0, col_std[col_name] * noise_scale)
                    imputed_value = fill_value + noise

                    # 如果填补后是负数，就用该特征的最小值填补
                    if imputed_value < 0:
                        imputed_value = col_min[col_name]
                        if pd.isna(imputed_value):
                            imputed_value = 0

                    df_filled.iloc[nan_idx, col_idx] = imputed_value
                else:
                    # 2400行之后：使用插值或邻近均值
                    # 尝试向前和向后查找非NaN值
                    forward_value = None
                    backward_value = None
                    
                    # 向前查找（最近的先前非NaN值）
                    for i in range(nan_idx - 1, max(-1, nan_idx - 10), -1):
                        if not pd.isna(df.iloc[i, col_idx]):
                            forward_value = df.iloc[i, col_idx]
                            break
                    
                    # 向后查找（最近的后续非NaN值）
                    for i in range(nan_idx + 1, min(n_rows, nan_idx + 10)):
                        if not pd.isna(df.iloc[i, col_idx]):
                            backward_value = df.iloc[i, col_idx]
                            break
                    
                    # 根据找到的值选择填充策略
                    if forward_value is not None and backward_value is not None:
                        # 前后都有值：使用线性插值
                        imputed_value = (forward_value + backward_value) / 2
                    elif forward_value is not None:
                        # 只有前值：使用前值
                        imputed_value = forward_value
                    elif backward_value is not None:
                        # 只有后值：使用后值
                        imputed_value = backward_value
                    else:
                        # 前后都找不到：使用列均值
                        imputed_value = col_mean[col_name]
                        if pd.isna(imputed_value):
                            imputed_value = 0
                    
                    df_filled.iloc[nan_idx, col_idx] = imputed_value

        # 返回处理后的item
        new_item = list(item)
        new_item[3] = df_filled
        return tuple(new_item) if isinstance(item, tuple) else new_item
    else:
        return item

def fill_nan_parallel(dataset: Union[List, Tuple],
                     noise_scale: float = 0.02,
                     n_processes: int = None,
                     output_path: str = None) -> Union[List, Tuple]:
    """
    并行版本的NaN填补函数

    使用多进程并行处理多个DataFrame

    Parameters:
    -----------
    dataset : Union[List, Tuple]
        包含多个DataFrame的列表或元组（通常包含至少4个元素）
    noise_scale : float, default 0.1
        添加噪声的标准差倍数
    n_processes : int, optional
        并行进程数，默认是CPU核心数和数据项数中的较小值
    output_path : str, optional
        保存处理后数据的pkl文件路径

    Returns:
    --------
    Union[List, Tuple]
        处理后的dataset
    """

    # 检查输入类型
    is_tuple = isinstance(dataset, tuple)
    
    dataset = replace_negative_br_with_nan(dataset)

    # 确定进程数
    if n_processes is None:
        n_processes = min(mp.cpu_count(), len(dataset))

    # 如果只有一个进程或只有一个item，直接处理
    if n_processes <= 1 or len(dataset) <= 1:
        new_dataset = [_process_single_item(item, noise_scale) for item in dataset]
    else:
        # 使用多进程 - 使用partial固定noise_scale参数
        process_func = partial(_process_single_item, noise_scale=noise_scale)

        with mp.Pool(processes=n_processes) as pool:
            new_dataset = pool.map(process_func, dataset)

    # 保存为pkl文件
    if output_path is not None:
        with open(output_path, 'wb') as f:
            pickle.dump(new_dataset, f)
        print(f"Dataset saved to {output_path}")

    return tuple(new_dataset) if is_tuple else new_dataset