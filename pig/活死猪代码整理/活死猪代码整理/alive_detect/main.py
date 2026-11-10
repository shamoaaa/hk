import os
import logging
import traceback
import multiprocessing
import warnings
import numpy as np

# Limit threads for various libraries
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")

import common_value
from utils import get_merged_data_hdf5
from pig_oestrus_features import PigOestrusFeatureExtractor

# --- 日志配置部分 ---
log_format = '%(asctime)s - [PID:%(process)d] - %(levelname)s - %(message)s'

# 1. 配置主日志 (记录所有数据，包括数组)
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler('alive_detect.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 2. 配置错误日志 (仅记录不匹配的数据)
error_logger = logging.getLogger('error_logger')
error_logger.setLevel(logging.INFO)
# 防止错误日志传播到根日志导致在控制台重复打印
error_logger.propagate = False 
error_handler = logging.FileHandler('alive_detect_errors.log')
error_handler.setFormatter(logging.Formatter(log_format))
error_logger.addHandler(error_handler)
# ---------------------

def run(file, label):
    try:
        analyzer = PigOestrusFeatureExtractor(4, 20)

        # 获取hdf5数据
        merged_data = get_merged_data_hdf5(file)  # 72000 * 23

        # 获取活动水平、呼吸、心率
        alive = analyzer.process(merged_data)

        # 提取需要记录的特定值，防止数组越界
        if len(alive) > 118:
            alive_118_val = int(alive[118])
        else:
            alive_118_val = -1 # 越界时给个默认值

        # --- 主日志：记录所有数据 ---
        logger.info(f"File: {file}")
        logger.info(f"Label: {label}")
        logger.info(f"Alive Array: {alive}")
        logger.info(f"Alive[118]: {alive_118_val}")

        # 更新混淆矩阵
        common_value.set_confusion_matrix(label, alive_118_val)

        common_value.add_finish_num()
        progress = (common_value.get_finish_num() / common_value.get_total_num()) * 100
        # 进度条依然使用 print 输出到标准错误或标准输出，避免被日志格式干扰
        print(f'\rOverall progress: {round(progress, 3)}%', end='', flush=True)

        # --- 错误日志：仅记录不匹配的数据 ---
        if label != alive_118_val:
            error_logger.info(f"Mismatch File: {file}")
            error_logger.info(f"  -> Label: {label}, Alive[118]: {alive_118_val}")
            error_logger.info(f"  -> Alive Array: {alive}")
            error_logger.info("-" * 50) # 分隔线方便阅读
        
    except Exception as e:
        # 捕获异常并记录到主日志
        logger.error(f"Error processing file {file}: {e}")
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    folder = r'D:\project\pig\mypig\活死猪代码整理\活死猪代码整理\alive_detect\data'
    folder_json = {
        'dead_pig': 2,
        # 'dead_pig_mistake': 1,
        'have_pig': 1,
        'no_pig': 0,
        # '20250908': 1,
    }

    para = []
    for key in folder_json.keys():

        dataset_path = os.path.join(folder, key)
        if os.path.exists(dataset_path):
            file_list = [os.path.join(dataset_path, name) for name in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, name))]
        else:
            continue

        for file in file_list:
            para.append((file, folder_json[key]))


    total_num = len(para)
    print(total_num)
    common_value.set_total_num(total_num)

    cpu_count = 18
    p = multiprocessing.Pool(cpu_count)
    p.starmap(run, para)

    p.close()
    p.join()

    common_value.get_confusion_matrix()
