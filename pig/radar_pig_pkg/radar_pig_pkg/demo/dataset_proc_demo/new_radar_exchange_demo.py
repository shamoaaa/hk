import logging


from radar_pig_pkg.dataset_proc.new_radar_exchange import new_radar_raw2hdf

# 配置日志文件
log_filename = '/data2/chenming40/log/new_radar_exchange_log.txt'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 原始数据目录，及从云端下载下来的本地目录
local_directory = r'/data2/wens_data/dataset/un_split/changtang_20260103'
# hdf数据目录
hdf_data_folder = r'/data2/wens_data/dataset/un_split/changtang_20260104'

new_radar_raw2hdf(local_directory, hdf_data_folder)