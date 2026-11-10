import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import logging

# 配置日志文件
log_filename = r'D:\project\pig\mypig\radar_pig_pkg\workspace\log1.txt'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


from ..radar_pig_pkg.feature_train.train_model import train_model

if __name__ == '__main__':
    
    dataset_dir = r'D:\project\pig\mypig\radar_pig_pkg\dataset'
    workspace_dir = r'D:\project\pig\mypig\radar_pig_pkg\workspace'
    cpu_count = 2
    # train_exp_list = ['changtang_20250505', 'changtang_20250523', 'changtang_20250610', 'changtang_20250627', 'changtang_20250717', 'changtang_20250803', 'changtang_20250822']
    # test_exp_list = ['changtang_20250908', 'changtang_20250926', 'changtang_20251031']
    train_exp_list = ['changtang_20250908']
    test_exp_list = ['changtang_20250908']
    label_dir = r"D:\project\pig\mypig\Oestrus\label_backup"
    train_model(dataset_dir, workspace_dir, cpu_count, train_exp_list, test_exp_list, label_dir)