import os
import logging
import traceback
from multiprocessing import Pool

from ..utils.oss_utils import get_bucket, download_file


__all__ = ['download_old_radar_files']

def _get_files_to_download(local_directory, oss_directory, bucket, max_retries):
    files_to_download = []

    try:
        next_marker = ''
        while True:
            result = bucket.list_objects(oss_directory, marker=next_marker)

            for obj in result.object_list:
                oss_file_path = obj.key
                relative_path = os.path.relpath(oss_file_path, oss_directory)

                # 如果本地有文件，则会跳过
                local_file = os.path.join(local_directory, relative_path)
                if os.path.exists(local_file):
                    continue

                files_to_download.append((oss_file_path, local_file, bucket, max_retries))
                
            next_marker = result.next_marker
            if not result.is_truncated:
                break
    except Exception as e:
        logging.error(f'Error listing objects in bucket: {str(e)}')
    return files_to_download


def download_old_radar_files(local_directory, oss_directory, 
                             oss_info, max_retries, 
                             num_processes):
    try:
        # 本地保存目录 
        os.makedirs(local_directory, exist_ok=True)

        # 创建认证对象
        bucket = get_bucket(oss_info)

        files_to_download = _get_files_to_download(local_directory, oss_directory, bucket, max_retries)
        with Pool(processes=num_processes) as pool:
            pool.map(download_file, files_to_download)
            
    except:
        logging.error(traceback.print_exc())
