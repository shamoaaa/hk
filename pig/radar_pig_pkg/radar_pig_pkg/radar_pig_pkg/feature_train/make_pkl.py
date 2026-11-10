import os
import logging
from datetime import datetime

import pickle
import numpy as np
import pandas as pd



import warnings
warnings.filterwarnings("ignore")


__all__ = ['make_feature_pkl']

def _sort_check_date(date_list):
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


def make_feature_pkl(feature_dir, out_pkl):

    all_data = {}
    new_dataset = []

    for experiment in os.listdir(feature_dir):
        if not os.path.isdir(os.path.join(feature_dir, experiment)):
            continue
        logging.info(f'Processing {experiment}...')

        exp_path = os.path.join(feature_dir, experiment)
        for pig_id in os.listdir(exp_path):
            if not os.path.isdir(os.path.join(exp_path, pig_id)):
                continue
            logging.info(f'\tPig_id: {pig_id}...')

            pid_id_path = os.path.join(exp_path, pig_id)
            date_list = os.listdir(pid_id_path)

            # 检查日期列表是否有序，且连续
            date_check_flag, sorted_date_list = _sort_check_date(date_list)
            if (date_check_flag is False):
                # raise ValueError(f'Date list {date_list} is not sorted correctly.')
                logging.error('\n'+ '*'*20)
                logging.error(f'Date list {date_list} is not sorted correctly.')
                logging.error('\n'+ '*'*20)
                continue

            pig_data = []
            dates = []
            for date in sorted_date_list:

                date_path = os.path.join(pid_id_path, date)
                if not os.path.isdir(date_path):
                    continue

                # 获取当前文件名规则
                file_pattern = os.listdir(date_path)
                if (len(file_pattern) == 0):
                    logging.warning(f'\t\tNo files found in {date_path}. Skipping...')
                    continue

                file_pattern = file_pattern[0]
                cut_len = len("-features.csv") + 2
                file_pattern = file_pattern[:-cut_len]

                # 读取一天得特征数据，没有得补充nan
                data_list = []
                for index in range(24):
                    cur_file = f'{file_pattern}{index:02d}-features.csv'
                    cur_file_path = os.path.join(date_path, cur_file)
                    if not os.path.exists(cur_file_path):
                        logging.warning(f'\t\t\tFile {cur_file} does not exist. use Nan replace')
                        cur_data = pd.DataFrame(np.nan, index=range(120), columns=['act', 'br', 'hr', 'is_up', 'ideal_channel', 'normalize_act', 'alive'])
                    else:
                        # print(f'\t\t\tProcessing {cur_file}...')
                        cur_data = pd.read_csv(cur_file_path, dtype='float32')
                    data_list.append(cur_data)

                data_day = pd.concat(data_list, ignore_index=True, axis=0)
                # data_day = pd.concat([data_day, cur_data], ignore_index=True, axis=0).infer_objects()
                # data_day.shape
                pig_data.append(data_day)
                dates.append(date)
            if pig_data == []:
                continue
            pig_datas = pd.concat(pig_data, axis=0)
            new_dataset.append((experiment, pig_id, dates, pig_datas))

    all_data.keys()

    with open(out_pkl, 'wb') as f:  # 注意 'wb' 模式
        pickle.dump(new_dataset, f)

    # fill_nan_parallel(new_dataset, 0.02, n_processes, os.path.join(out_pkl_dir, oestrus_pkl_name))