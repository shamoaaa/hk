import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import shutil
import logging
import multiprocessing

import warnings
warnings.filterwarnings("ignore") # 忽略所有警告


from radar_pig_pkg.features.run import feature_extract


def main():
    # 配置日志文件
    log_filename = r'D:\project\pig\mypig\pig_lib\out\test.log'
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    dataset_dir = r'D:\project\pig\mypig\pig_lib\data'  # 数据集路径
    feature_dir = r'D:\project\pig\mypig\pig_lib\out'    # 原始特征存储路径
    cpu_count = 1

    # 是否需要全部重跑
    need_rerun = False
    if need_rerun:
        shutil.rmtree(feature_dir)
    os.makedirs(feature_dir, exist_ok=True)

    # 全部跑，还是跑指定的实验批次
    select_all_flag = True
    select_exp_name = []

    exp_list = []
    if select_all_flag:
        exp_list = os.listdir(dataset_dir)
    else:
        exp_list = select_exp_name

    # 生成参数
    para = []

    for one_exp in os.listdir(dataset_dir):
        one_exp_dir = os.path.join(dataset_dir, one_exp)
        one_radar_list = os.listdir(one_exp_dir)

        pig_prop = 2 if 'houbei' in dataset_dir else 1

        for one_radar in one_radar_list:
            one_day_dir = os.path.join(one_exp_dir, one_radar)
            date_list = os.listdir(one_day_dir)

            for date in date_list:
                hdf_dir = os.path.join(one_day_dir, date)
                hdf_list = os.listdir(hdf_dir)

                # 创建结果输出目录
                out_path = os.path.join(feature_dir, one_exp, one_radar, date)
                os.makedirs(out_path, exist_ok=True)

                for hdf in hdf_list:
                    hdf_path = os.path.join(hdf_dir, hdf)

                    para.append((hdf_path, out_path, pig_prop))


    total_num = len(para)
    print(total_num)

    p = multiprocessing.Pool(cpu_count)
    p.starmap(feature_extract, para)

    p.close()
    p.join()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()