import os
import shutil
import multiprocessing

from ..features.run import feature_extract


def feature_extraction(dataset_dir, feature_dir, cpu_count, exp_list):

    os.makedirs(feature_dir, exist_ok=True)
    shutil.rmtree(feature_dir)
    os.makedirs(feature_dir, exist_ok=True)

    # 生成参数
    para = []
    for one_exp in exp_list:
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

    p = multiprocessing.Pool(cpu_count)
    p.starmap(feature_extract, para)

    p.close()
    p.join()