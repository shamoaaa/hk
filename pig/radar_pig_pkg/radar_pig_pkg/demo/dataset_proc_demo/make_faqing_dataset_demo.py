import logging


from radar_pig_pkg.dataset_proc.make_faqing_dataset import move_split_dataset

# 配置日志文件
log_filename = '/data2/chenming40/log/make_faqing_dataset_log.txt'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 剔除的雷达编号
cannot_use_radar_ids = ['CH2962263']

# 经产猪的雷达编号
jingchan_radar_ids = ['CH2962726']

# 后备猪的雷达编号
houbei_radar_ids = ['CH2962263']

# 实验数据的地址
exp_dir = r'/data2/wens_data/dataset/un_split/changtang_20260104'

# 经产猪数据集的地址
jingchan_dir = r'/data2/wens_data/dataset/jingchan'

# 后备猪数据集的地址
houbei_dir = r'/data2/wens_data/dataset/houbei'

place = 'changtang'
exp_id = '20250505'
start_time_str = '2026010400'
end_time_str = '2026010408'

# 划分
move_split_dataset(exp_dir, jingchan_dir, 
                   jingchan_radar_ids, cannot_use_radar_ids,  
                   place, exp_id, start_time_str, end_time_str)

move_split_dataset(exp_dir, houbei_dir,  
                   houbei_radar_ids, cannot_use_radar_ids,  
                   place, exp_id, start_time_str, end_time_str)