import os
import logging
import traceback
import concurrent.futures

import oss2

from ..utils.common_utils import sn2id
from ..utils.oss_utils import get_bucket, download_file


__all__ = ['download_new_radar_files']

def _download_from_oss_mt(prefix, bucket, 
                         local_directory, start_time, end_time, 
                         max_retries):

    for obj in oss2.ObjectIterator(bucket, prefix=prefix):
        if obj.key.endswith('/'):
            continue

        relative_path = obj.key[len(prefix):]
        file_datetime = relative_path[0:8]

        if start_time <= file_datetime <= end_time:
            file_date = relative_path[0:8]
            save_path = os.path.join(local_directory, file_date)
            os.makedirs(save_path, exist_ok=True)

            local_path = os.path.join(save_path, os.path.basename(obj.key))
            
            args = (obj.key, local_path, bucket, max_retries)
            download_file(args)


def _multi_prefix_download(prefixes, bucket, 
                          local_directory, start_date_str, end_date_str, 
                          max_retries, max_workers):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_list = []

        for prefix in prefixes:
            # 为每个prefix创建独立目录
            prefix_name = prefix.strip('/').split('/')[-1]
            save_dir = os.path.join(local_directory, prefix_name)
            os.makedirs(save_dir, exist_ok=True)

            future_list.append(executor.submit(
                _download_from_oss_mt, 
                prefix, bucket, save_dir, start_date_str, end_date_str, 
                max_retries
            ))
        concurrent.futures.wait(future_list)


def download_new_radar_files(start_date_str, end_date_str, 
                             radar_ids, local_directory, 
                             oss_info, max_retries, max_workers):
    
    try:
        # 本地保存目录 
        os.makedirs(local_directory, exist_ok=True)

        # 生成oss文件夹地址
        radar_ids = list(set(sn2id(radar_ids)))
        pre = "pigoestrus/rawdata/"
        prefixes = []

        for radar_id in radar_ids:
            prefixes.append(pre + radar_id + '/')

        # 创建认证对象
        bucket = get_bucket(oss_info)

        _multi_prefix_download(prefixes, bucket, 
                            local_directory, start_date_str, end_date_str, 
                            max_retries, max_workers)
    
    except:
        logging.error(traceback.format_exc())