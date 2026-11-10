import logging

# 配置日志文件
log_filename = '/root/upload_old_radar_log.txt'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 配置阿里云OSS信息
oss_info = {
    'access_key_id': 'lfb7wasdizzk5aggkqov', 
    'access_key_secret': 'xeqr6i9c82p701eiydk3b1ovlk58h79ti83pnaxy', 
    'endpoint': 'https://wens.com.cn', 
    'bucket_name': 'oss-zzcq'
}

# 要上传的本地文件夹路径
local_directory = '/data/gather/tmp/radar_data/hdf5'
# OSS中目标文件夹名
oss_directory = 'radar_data/faqing/changtang/20251130/'
# 上传最大重试次数
max_retries = 3

# 待上传的雷达编号，字符串数组
radar_ids = ['30540','31299','31083']
# 上传数据的起始时间和结束时间，区间前闭后闭，%Y%m%d格式
start_date_str = '20250101'
end_date_str = '20250103'

upload_old_radar(start_date_str, end_date_str, 
                 radar_ids, local_directory, 
                 oss_directory, oss_info, max_retries)