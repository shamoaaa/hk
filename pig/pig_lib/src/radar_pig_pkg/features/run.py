import os
import logging
import traceback

import numpy as np
import pandas as pd

from .core.pig_oestrus_features import PigOestrusFeatureExtractor
from ..utils.common_utils import get_merged_data_hdf5


def feature_extract(file, out_path, pig_prop):

    try:

        filename, _ = os.path.splitext(os.path.basename(file))
        out_file = os.path.join(out_path, f"{filename}-features.csv")
        
        if not os.path.exists(out_file):
        
            # 创建PigOestrusAnalyzer实例
            analyzer = PigOestrusFeatureExtractor()

            # 获取hdf5数据
            merged_data = get_merged_data_hdf5(file)  # 72000 * 23
            l = merged_data.shape[0]
            min_num = int(l / 20 / 60)

            # 创建存储空间
            act_dict = {}
            act_dict['amp'] = np.zeros(2*min_num)
            act_dict['normal'] = np.zeros(2*min_num)
            br_dict = {}
            br_dict['main_freq'] = np.zeros(2*min_num)
            br_dict['new_main_freq'] = np.zeros(2*min_num)
            br_dict['ideal_channel'] = np.zeros(2*min_num)
            hr_dict = {}
            hr_dict['main_freq'] = np.zeros(2*min_num)
            is_up_list = np.zeros(2*min_num)
            alive_list = np.zeros(2*min_num)

            # 以60秒为窗口、30秒为步长，提取特征
            for min_index in range(2 * min_num - 1):
                # 截取1分钟数据
                start_index = 600 * min_index
                end_index = min(l, start_index + 1200)
                frame_data = merged_data[start_index:end_index, :]  # 1200 * 23

                if pig_prop==2:
                    analyzer.init_param(frame_data, 6)
                else:
                    analyzer.init_param(frame_data, 4)

                alive_result_30_sec, is_up, act, br, hr = analyzer.process()

                act_dict['amp'][min_index] = act['act_mean']
                act_dict['normal'][min_index] = act['act_normal']
                br_dict['main_freq'][min_index] = br['br_freq']
                br_dict['new_main_freq'][min_index] = br['new_br_freq']
                br_dict['ideal_channel'][min_index] = br['ideal_channel']
                hr_dict['main_freq'][min_index] = hr['hr_freq']
                is_up_list[min_index] = is_up
                alive_list[min_index] = alive_result_30_sec

            act_dict['amp'][-1] = act_dict['amp'][-2]
            act_dict['normal'][-1] = act_dict['normal'][-2]
            br_dict['main_freq'][-1] = br_dict['main_freq'][-2]
            br_dict['new_main_freq'][-1] = br_dict['new_main_freq'][-2]
            br_dict['ideal_channel'][-1] = br_dict['ideal_channel'][-2]
            hr_dict['main_freq'][-1] = hr_dict['main_freq'][-2]
            is_up_list[-1] = is_up_list[-2]
            alive_list[-1] = alive_list[-2]

            array_2d = np.column_stack((act_dict['amp'], act_dict['normal'], br_dict['main_freq'], br_dict['new_main_freq'], br_dict['ideal_channel'], hr_dict['main_freq'], is_up_list, alive_list))
            df = pd.DataFrame(array_2d, columns=['act', 'normalize_act', 'br', 'new_br', 'ideal_channel', 'hr', 'is_up', 'alive'])
            
            #增加混淆特征
            df = df.pipe(add_confusion_features)
            
            df.to_csv(out_file, index=False, na_rep='-1')
    
    except Exception as e:
        logging.error(f'{file} occur exception!')
        logging.error(str(e))
        logging.error(traceback.format_exc())
        
        
        
def add_confusion_features(current_df):
    # 1. 安全检查
    if current_df is None or current_df.empty:
        return current_df

    # 2. 定义 10 个不同的权重向量
    # 每一行代表一个新特征(feature1 到 feature10)的权重组合
    weights = [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], # feature1 的权重
        [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], # feature2 的权重 (倒序)
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], # feature3 的权重 (平均)
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], # feature4 的权重 (间隔)
        [0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4], # feature5 的权重 (梯形)
        [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1], # feature6 的权重
        [0.3, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1], # feature7 的权重
        [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85], # feature8 的权重
        [0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5], # feature9 的权重 (包含负权重，增加混淆度)
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08] # feature10 的权重
    ]
    
    w_matrix = np.array(weights)
    data_values = current_df.values.astype(float)
    new_features_matrix = np.dot(data_values, w_matrix.T)
    
    # 4. 将计算结果添加到 DataFrame 中
    for i in range(10):
        current_df[f'feature{i+1}'] = new_features_matrix[:, i]
        
    return current_df