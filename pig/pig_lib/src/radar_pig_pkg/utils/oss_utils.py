import os
import time
import logging

import oss2

def get_bucket(oss_info):
    access_key_id = oss_info['access_key_id']
    access_key_secret = oss_info['access_key_secret']
    endpoint = oss_info['endpoint']
    bucket_name = oss_info['bucket_name']

    # 创建认证对象
    auth = oss2.Auth(access_key_id, access_key_secret)
    # 创建Bucket对象
    bucket = oss2.Bucket(auth, endpoint, bucket_name)

    return bucket


def download_file(args):
    oss_file_path, local_file, bucket, max_retries = args

    for attempt in range(max_retries):
        try:
            if not os.path.exists(local_file):
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                bucket.get_object_to_file(oss_file_path, local_file)
                logging.info(f'Downloaded: {oss_file_path} to {local_file}')
            else:
                logging.info(f'Skipped: {local_file} already exists')
            break
        except Exception as e:
            logging.warning(f'Failed to download {oss_file_path} on attempt {attempt + 1}: {str(e)}')
            time.sleep(2 ** attempt)

            if attempt == max_retries - 1:
                logging.error(f'Giving up after {max_retries} attempts for file: {oss_file_path}')