import os
import logging

from radar_pig_pkg.dataset_proc.download_new_radar import download_new_radar_files


# 配置日志文件
log_filename = '/data2/chenming40/log/download_new_radar_log.txt'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 配置阿里云OSS信息
oss_info = {
    'access_key_id': 'lfb7wasdizzk5aggkqov', 
    'access_key_secret': 'xeqr6i9c82p701eiydk3b1ovlk58h79ti83pnaxy', 
    'endpoint': 'https://wens.com.cn', 
    'bucket_name': 'oss-zzcq'
}

# 本地目录
local_directory = r'/data2/wens_data/dataset/un_split/changtang_20260103'
if not os.path.exists(local_directory):
    os.makedirs(local_directory)
# 云端目录
oss_directory = 'pigoestrus/rawdata/'

# 待上传的雷达编号，字符串数组
radar_ids = ['CH2962726','CH2962263']
# 上传数据的起始时间和结束时间，区间前闭后闭，%Y%m%d格式
start_date_str = '20260104'
end_date_str = '20260104'

# 上传最大重试次数
max_retries = 3

# 配置多线程数量
max_workers = 16

download_new_radar_files(start_date_str, end_date_str, 
                         radar_ids, local_directory, 
                         oss_info, max_retries, max_workers)