from .train_test_core import *


def train_oestrus(feature_file_path, label_path, train_exp_list, model_save_path, interval_type='-1_1'):
    """
    查情训练XGBoost模型

    参数:
        feature_file_path (str): 训练特征文件路径
        label_path (str): 标签文件路径
        train_exp_list (list): 训练实验列表
        model_save_path (str): 模型保存路径
        interval_type (str): 时间间隔类型，默认'-1_1'

    返回:
        model: 训练好的XGBoost模型
        dict: 训练过程信息
    """
    return train_impl(
        feature_file_path=feature_file_path,
        label_path=label_path,
        train_exp_list=train_exp_list,
        model_save_path=model_save_path,
        label_type='breeding_offset',
        interval_type=interval_type
    )


def test_oestrus(feature_file_path, label_path, test_exp_list, model_path, interval_type='-1_1'):
    """
    查情测试XGBoost模型

    参数:
        feature_file_path (str): 测试特征文件路径
        label_path (str): 标签文件路径
        test_exp_list (list): 测试实验列表
        model_path (str): 模型文件路径
        interval_type (str): 时间间隔类型，默认'-1_1'

    返回:
        dict: 测试结果
    """
    return test_impl(
        feature_file_path=feature_file_path,
        label_path=label_path,
        test_exp_list=test_exp_list,
        model_path=model_path,
        result_save_path=None,
        label_type='breeding_offset',
        interval_type=interval_type
    )

