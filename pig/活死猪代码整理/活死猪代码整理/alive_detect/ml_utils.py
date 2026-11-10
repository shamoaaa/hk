import os

# Limit threads for various libraries
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import traceback
import multiprocessing
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

import common_value
from utils import get_merged_data_hdf5
from pig_oestrus_features import PigOestrusFeatureExtractor

def get_onepig_feature(hdf5_file, analyzer):
    '''
    Args:
    hdf5_file: 单个猪的一小时hdf5源数据文件路径
    analyzer: 猪特征处理对象
    Return:
    字典形式的单猪特征，和逻辑判断输出标签
    feature: [[amp_max, amp_min, amp_mean, max_channel, ideal_channel, br_amp, up_down],...]
    label: [] 0-无猪 1-活猪 2-死猪
    '''

    merged_data = get_merged_data_hdf5(hdf5_file)
    feature, label = analyzer.process_ml(merged_data)
    return feature, label

def process(hdf5_file, type):
    '''
    针对死猪不确定的机器学习使用，对死猪标签利用逻辑判断结果重新整理，
    Args:
    hdf5_file: 猪的雷达源数据hdf5文件
    type: 0-无猪，1-有猪，2-无猪, -1-不确定用于， 当为无猪和有猪时将标签全部置为0和1，死猪额外判断修正
    '''
    analyzer = PigOestrusFeatureExtractor(4,20)
    feature, label = get_onepig_feature(hdf5_file, analyzer)
    flag = 0 # 表示标签是否可信
    # 标签修正
    if type==0:
        label = [0 for _ in range(len(label))]
        flag = 1
    if type==1:
        label = [1 for _ in range(len(label))]
        flag = 1
    if type==2:
        # 死猪特殊判断
        no_num, have_num, dead_num = 0, 0, 0
        for lb in label:
            if lb==0:
                no_num += 1
            elif lb==1:
                have_num += 1
            else:
                dead_num += 1
        if have_num > 0.85*len(label):
            label = [1 for _ in range(len(label))]
            flag = 1
        elif no_num > 0.85*len(label):
            label = [0 for _ in range(len(label))]
            flag = 1
        elif dead_num > 0.85*len(label):
            label = [2 for _ in range(len(label))]
            flag = 1
        else:
            flag = 0 # 不确定标签 用作无标签数据集
        
    return feature, label, flag


def getfeature(file_path):
    '''
    Args:
    file_path: 包含活猪、无猪、死猪数据的文件夹，全部读入后处理成有猪无猪数据集，活猪死猪数据集
    Return:
    dataset
    label
    '''
    pig_dir = []
    feature = [] # 窗口特征
    label = [] # 窗口标签
    nolabel_feature = [] # 无标签窗口特征
    logic_label = [] # 无标签窗口逻辑判断结果
    for dir in os.listdir(file_path):
        pig_dir.append(dir)  # 获取不同猪的文件名 no_pig, dead_pig, have_pig
    for dir in pig_dir:
        if dir=='no_pig':
            type=0
        if dir=='have_pig':
            type=1
        if dir=='dead_pig':
            type=2
        for hdf5_file in os.listdir(os.path.join(file_path, dir)):
            hdf5_file = os.path.join(file_path, dir, hdf5_file)
            f, lb, flag = process(hdf5_file, type)  
            if flag==1:
                feature += f
                label += lb
            else:
                nolabel_feature += f
                logic_label += lb
    # 已经获得每个窗口对应的特征，以及其标签，后续整理出有猪无猪数据集和活猪死猪数据集用于训练，无标签留作半监督学习使用
    have_no_dataset = []
    have_no_label = []
    alive_dead_dataset = []
    alive_dead_label = []
    for i in range(len(label)):
        if label[i]==0:
            have_no_dataset.append(feature[i])
            have_no_label.append(0)
        elif label[i]==1:
            have_no_dataset.append(feature[i])
            have_no_label.append(1)
            alive_dead_dataset.append(feature[i])
            alive_dead_label.append(1)
        else:
            have_no_dataset.append(feature[i])
            have_no_label.append(1)
            alive_dead_dataset.append(feature[i])
            alive_dead_label.append(0)
    
    return have_no_dataset, have_no_label, alive_dead_dataset, alive_dead_label, nolabel_feature, logic_label
