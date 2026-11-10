import os
import logging
import traceback
from collections import defaultdict

from ..utils.common_utils import sn2id, generate_date_strings
from ..utils.oss_utils import get_bucket


def upload_file(local_file, oss_file_path, bucket):
    try:
        # 上传文件
        with open(local_file, 'rb') as file_obj:
            bucket.put_object(oss_file_path, file_obj)
        logging.info(f'Uploaded: {local_file} to {oss_file_path}')

        return True
    except Exception as e:
        # logging.error(f'Failed to upload {local_file}: {str(e)}')
        return False


def upload_directory(local_directory, oss_directory, date_list, radar_ids, bucket, max_retries):
    retry_count = defaultdict(int)
    files_to_upload = []
    fail_num = 0
    success_num = 0
    all_num = 0

    # 获取需要上传的文件
    for date in date_list:
        directory = os.path.join(local_directory, date)

        for root, _, files in os.walk(directory):
            for file_name in files:
                dev_id = str.split(file_name, '-')[0]
                if dev_id in radar_ids:

                    local_file = os.path.join(root, file_name)
                    relative_path = os.path.relpath(local_file, local_directory)
                    oss_file_path = os.path.join(oss_directory, relative_path).replace("\\", "/")
                    files_to_upload.append((local_file, oss_file_path))
                    all_num += 1

    # 开始上传文件
    while files_to_upload:
        local_file, oss_file_path = files_to_upload.pop(0)
        success = upload_file(local_file, oss_file_path, bucket)

        if success:
            success_num += 1
        else:
            if retry_count[local_file] < max_retries:
                retry_count[local_file] += 1
                files_to_upload.append((local_file, oss_file_path))
                logging.warning(f'Retrying {local_file}. Attempt {retry_count[local_file]} of {max_retries}')
            else:
                fail_num += 1
                logging.error(f'Exceeded maximum retries for {local_file}, giving up.')

    logging.info(f'Need upload {all_num} files, success: {success_num}, failed: {fail_num}. Please retry!')


def upload_old_radar(start_date_str, end_date_str, 
                     radar_ids, local_directory, oss_directory, 
                     oss_info, max_retries):
    try:

        date_list = generate_date_strings(start_date_str, end_date_str)
        radar_ids = sn2id(radar_ids)
        bucket = get_bucket(oss_info)

        # 开始上传
        upload_directory(local_directory, oss_directory, date_list, radar_ids, bucket, max_retries)
    
    except:
        print(traceback.format_exc())