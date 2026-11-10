import os
import logging

from radar_pig_pkg.dataset_proc.download_old_radar import download_old_radar_files


# 配置日志文件
log_filename = '/data2/chenming40/log/download_old_radar_log.txt'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 配置阿里云OSS信息
oss_info = {
    'access_key_id': 'lfb7wasdizzk5aggkqov', 
    'access_key_secret': 'xeqr6i9c82p701eiydk3b1ovlk58h79ti83pnaxy', 
    'endpoint': 'https://wens.com.cn', 
    'bucket_name': 'oss-zzcq'
}

# 本地目录
local_directory = r'/data2/wens_data/dataset/un_split/changtang_20260109'
if not os.path.exists(local_directory):
    os.makedirs(local_directory)
# 云端目录
oss_directory = 'radar_data/faqing/changtang/20260109/'

# 下载最大重试次数
max_retries = 3

# 配置多进程数量
num_processes = 16

download_old_radar_files(local_directory, oss_directory, 
                         oss_info, max_retries, 
                         num_processes)