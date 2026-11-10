import os
import json
import logging
import traceback

import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split

from .utils.gen_feature_utils import generate_features_unified
from .utils.evaluate_utils_refactored import EvaluationConfig, ExperimentEvaluator


def train_impl(feature_file_path, label_path, train_exp_list, model_save_path, label_type, interval_type):
    """
    训练XGBoost模型内部实现

    参数:
        feature_file_path (str): 训练特征文件路径
        label_path (str): 标签文件路径
        train_exp_list (list): 训练实验列表
        model_save_path (str): 模型保存路径
        label_type (str): 标签类型 'breeding' 或 'check'
        interval_type (str): 时间间隔类型

    返回:
        model: 训练好的XGBoost模型
        dict: 训练过程信息
    """
    # logging.info(f"开始训练模型 - 类型: {label_type}, 时间间隔: {interval_type}")
    logging.info(f"开始训练模型 - 类型: {label_type}")
    logging.info(f"特征文件路径: {feature_file_path}")
    logging.info(f"标签路径: {label_path}")
    logging.info(f"训练实验列表: {train_exp_list}")
    logging.info(f"模型保存路径: {model_save_path}")

    # 创建模型保存目录
    model_dir = os.path.dirname(model_save_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        logging.info(f"创建模型保存目录: {model_dir}")

    # 加载训练数据
    logging.info("加载训练数据...")
    with open(feature_file_path, 'rb') as f:
        dataset = pickle.load(f)

    # logging.info(f"训练集数量: {len(dataset)}")

    # 生成特征
    logging.info("生成训练特征...")
    features_df, labels_df = generate_features_unified(
        dataset, label_path, train_exp_list,
        type='train', label_type=label_type, interval_type=interval_type
    )

    logging.info(f"特征生成完成 - 样本数量: {len(features_df)}, 特征维度: {features_df.shape[1]}")

    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, labels_df['eva_label'], test_size=0.2, random_state=42
    )

    # 创建DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # 设置模型参数
    params = {
        'nthread': 4,
        'n_jobs': 4,
        'eval_metric': 'logloss',
        'eta': 0.015,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
    }

    logging.info("开始训练XGBoost模型...")

    # 训练模型
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        verbose_eval=10,
        early_stopping_rounds=10
    )

    # 记录训练信息
    train_info = {
        'label_type': label_type,
        'interval_type': interval_type,
        'best_score': model.best_score,
        'best_iteration': model.best_iteration,
        'feature_dim': features_df.shape[1],
        'train_samples': len(X_train),
        'val_samples': len(X_test)
    }

    logging.info(f"模型训练完成")
    logging.info(f"最佳得分: {train_info['best_score']}")
    logging.info(f"最佳迭代次数: {train_info['best_iteration']}")

    # 保存模型
    logging.info(f"\n保存模型到: {model_save_path}")
    model.save_model(model_save_path)
    logging.info(f"模型已成功保存")

    # # 保存训练信息
    # info_save_path = model_save_path.replace('.model', '_info.json')
    # with open(info_save_path, 'w', encoding='utf-8') as f:
    #     json.dump(train_info, f, indent=2, ensure_ascii=False)
    # logging.info(f"训练信息已保存到: {info_save_path}")

    return model, train_info


def test_impl(feature_file_path, label_path, test_exp_list, model_path, result_save_path, label_type, interval_type):
    """
    测试XGBoost模型内部实现

    参数:
        feature_file_path (str): 测试特征文件路径
        label_path (str): 标签文件路径
        test_exp_list (list): 测试实验列表
        model_path (str): 模型文件路径
        result_save_path (str): 结果保存地址
        label_type (str): 标签类型 'breeding' 或 'breeding_offset'
        interval_type (str): 时间间隔类型

    返回:
        dict: 测试结果
    """
    # logging.info(f"开始测试模型 - 类型: {label_type}, 时间间隔: {interval_type}")
    logging.info(f"开始测试模型 - 类型: {label_type}")
    logging.info(f"测试特征文件路径: {feature_file_path}")
    logging.info(f"标签路径: {label_path}")
    logging.info(f"测试实验列表: {test_exp_list}")
    logging.info(f"模型路径: {model_path}")
    # logging.info(f"结果保存路径: {result_save_path}")

    # 创建结果保存目录
    # os.makedirs(result_save_path, exist_ok=True)

    # 加载测试数据
    logging.info("加载测试数据...")
    with open(feature_file_path, 'rb') as f:
        dataset = pickle.load(f)

    # logging.info(f"测试集数量: {len(dataset)}")

    # 加载模型
    logging.info("加载模型...")
    model = xgb.Booster()
    model.load_model(model_path)

    # 导入评估工具
    try:
        
        # 配置评估参数
        config = EvaluationConfig(
            label_type=label_type,
            interval_type=interval_type,
            save_results=True,
            # output_dir=result_save_path,
            output_dir=None,
        )

        # 创建评估器
        evaluator = ExperimentEvaluator(model, config, label_path)

        # 逐个实验评估
        logging.info("开始逐个实验评估...")
        result_list = []
        for exp in test_exp_list:
            logging.info(f"评估实验: {exp}")
            single_test_exp = [exp]
            result = evaluator.evaluate(dataset, single_test_exp)
            result_list.append(result)

        # 汇总结果
        logging.info("\n测试结果汇总:")
        logging.info("         准确率              精确率               召回率")
        for i, result in enumerate(result_list):
            logging.info(f"{test_exp_list[i]}: {result[1:4]}")

        # 返回结果
        test_results = {
            'label_type': label_type,
            'interval_type': interval_type,
            'experiments': test_exp_list,
            'results': result_list,
            'summary': {
                'accuracy': [r[1] for r in result_list],
                'precision': [r[2] for r in result_list],
                'recall': [r[3] for r in result_list]
            }
        }

        return test_results

    except:
        logging.error(traceback.format_exc())
        return None
