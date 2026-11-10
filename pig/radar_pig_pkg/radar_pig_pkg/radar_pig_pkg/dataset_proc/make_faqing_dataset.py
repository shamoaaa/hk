import os
import random
import shutil
import logging
import traceback

from ..utils.common_utils import generate_time_strings


__all__ = ['make_faqing_dataset']

def move_split_dataset(exp_dir, train_dir, 
                       radar_ids, cannot_use_radar_ids, 
                       place, exp_id, start_time_str, end_time_str):
    
    time_list = generate_time_strings(start_time_str, end_time_str)

    # 获取所有的雷达编号
    can_use_radar_ids = []
    for id in radar_ids:
        if id not in cannot_use_radar_ids:
            can_use_radar_ids.append(id)

    # 遍历目录
    date_list = os.listdir(exp_dir)
    for date in date_list:
        one_day_data_path = os.path.join(exp_dir, date)

        data_list = os.listdir(one_day_data_path)
        for data in data_list:
            try:
                radar_id = data.split('-')[0]
                data_time = data.split('-')[1][:-5]

                # 训练集
                if radar_id not in cannot_use_radar_ids and data_time in time_list:

                    source_dir = os.path.join(one_day_data_path, data)
                    dest_dir = os.path.join(train_dir, f'{place}_{exp_id}', f'{place}_{radar_id}_{exp_id}', date)
                    if not os.path.exists(dest_dir):
                        os.makedirs(dest_dir)

                    shutil.move(source_dir, dest_dir)
            except:
                logging.error(f'{one_day_data_path} occur exception!')
                logging.error(traceback.format_exc())