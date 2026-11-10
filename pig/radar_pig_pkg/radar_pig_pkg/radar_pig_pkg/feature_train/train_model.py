import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import pickle
import shutil
import logging
import traceback

import warnings
warnings.filterwarnings("ignore")

from ..train.utils.common_utils import fill_nan_parallel
from ..train.breeding import train_breeding, test_breeding
from ..train.oestrus import train_oestrus, test_oestrus
from .feature_extraction import feature_extraction
from .make_pkl import make_feature_pkl


def train_model(dataset_dir, workspace_dir, cpu_count, train_exp_list, test_exp_list, label_dir):

    # ---------- 进行特征提取 ----------
    try:
        logging.info('Start to extract features...')
        feature_dir = os.path.join(workspace_dir, 'feature')
        exp_list = list(set(train_exp_list+test_exp_list))
        feature_extraction(dataset_dir, feature_dir, cpu_count, exp_list)
        logging.info('Finish extract features!')
    except:
        logging.error(traceback.format_exc())
        if os.path.exists(feature_dir):
            shutil.rmtree(feature_dir)
        return

    # ---------- 制作配种pkl ----------
    try:
        logging.info('Start to make peizhong pkl...')
        peizhong_pkl_name = 'peizhong.pkl'
        peizhong_pkl_path = os.path.join(workspace_dir, peizhong_pkl_name)
        make_feature_pkl(feature_dir, peizhong_pkl_path)
        logging.info('Finish make peizhong pkl!')
    except:
        logging.error(traceback.format_exc())
        if os.path.exists(feature_dir):
            shutil.rmtree(feature_dir)
        if os.path.exists(peizhong_pkl_path):
            os.remove(peizhong_pkl_path)
        return

    # peizhong_pkl_path = r'/data1/chenming40/pkl/train_feature_20251127.pkl'
    # # peizhong_pkl_path = r'/data2/chenming40/workspace/faqing.pkl'
    # faqing_pkl_path = r'/data2/chenming40/workspace/faqing.pkl'
    # ---------- 制作发情pkl ----------
    try:
        logging.info('Start to make faqing pkl...')
        faqing_pkl_name = 'faqing.pkl'
        faqing_pkl_path = os.path.join(workspace_dir, faqing_pkl_name)
        with open(peizhong_pkl_path, 'rb') as f:  # 注意 'rb' 模式
            peizhong_pkl = pickle.load(f)
        fill_nan_parallel(peizhong_pkl, 0.02, cpu_count, faqing_pkl_path)
        logging.info('Finish make faqing pkl!')
    except:
        logging.error(traceback.format_exc())
        if os.path.exists(feature_dir):
            shutil.rmtree(feature_dir)
        if os.path.exists(faqing_pkl_path):
            os.remove(faqing_pkl_path)
        return
    
    # # ========== 删除csv数据 ==========
    # if os.path.exists(feature_dir):
    #     shutil.rmtree(feature_dir)

    try:
        # ========== 配种模型 ==========
        breeding_model_path = os.path.join(workspace_dir, 'breeding.model')
        # 训练配种模型
        train_breeding(peizhong_pkl_path, label_dir, train_exp_list, breeding_model_path)
        # 测试配种模型
        test_breeding(peizhong_pkl_path, label_dir, test_exp_list, breeding_model_path)

        # ========== 查情模型 ==========
        oestrus_model_path = os.path.join(workspace_dir, 'oestrus.model')
        # 训练查情模型
        train_oestrus(faqing_pkl_path, label_dir, train_exp_list, oestrus_model_path)
        # 测试查情模型
        test_oestrus(faqing_pkl_path, label_dir, test_exp_list, oestrus_model_path)
    except:
        logging.error(traceback.format_exc())
        if os.path.exists(peizhong_pkl_path):
            os.remove(faqing_pkl_path)
        if os.path.exists(faqing_pkl_path):
            os.remove(faqing_pkl_path)
        if os.path.exists(breeding_model_path):
            os.remove(breeding_model_path)
        if os.path.exists(oestrus_model_path):
            os.remove(oestrus_model_path)
        logging.error(traceback.format_exc())
        return

    # # 删除数据
    # if os.path.exists(peizhong_pkl_path):
    #     os.remove(faqing_pkl_path)
    # if os.path.exists(faqing_pkl_path):
    #     os.remove(faqing_pkl_path)
