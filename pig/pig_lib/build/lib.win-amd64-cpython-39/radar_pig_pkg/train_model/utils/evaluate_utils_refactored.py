"""
重构后的evaluate_utils.py - 降低耦合性的版本
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# 导入必要的工具函数
from utils.common_utils import calculate_pig_sign, convert_features_to_numeric


@dataclass
class EvaluationConfig:
    """评估配置类，集中管理所有配置参数"""
    label_type: str = 'breeding'
    breeding_offset_days: int = 2
    interval_type: str = '-2_0'
    threshold: float = 0.52
    save_results: bool = False
    output_dir: str = None
    is_completedata: bool = False
    importance_type: str = 'weight'
    top_n_features: int = 20


class DataPreparator:
    """数据准备类，负责特征生成和预处理"""

    def __init__(self, label_path: str):
        self.label_path = label_path

    def prepare_features(self, dataset, test_exp_list, config: EvaluationConfig):
        """准备评估特征"""
        from utils.gen_feature_utils import generate_features_unified

        # features_df, labels_df = generate_features_unified(
        #     dataset,
        #     self.label_path,
        #     test_exp_list,
        #     type='test',
        #     label_type=config.label_type,
        #     interval_type=config.interval_type,
        #     breeding_offset_days=config.breeding_offset_days,
        #     is_completedata=config.is_completedata
        # )
        
        from utils.improved_feature_utils import generate_improved_features_unified

        features_df, labels_df = generate_improved_features_unified(
            dataset,
            self.label_path,
            test_exp_list,
            type='test',
            label_type=config.label_type,
            interval_type=config.interval_type,
            breeding_offset_days=config.breeding_offset_days,
            is_completedata=config.is_completedata,
            include_correlation_features=False,
            include_statistical_features=False,
            include_temporal_features=False
        )


        features_df = convert_features_to_numeric(features_df)
        return features_df, labels_df


class ModelPredictor:
    """模型预测类，负责模型预测相关逻辑"""

    def __init__(self, model, config: EvaluationConfig):
        self.model = model
        self.config = config

    def predict(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """进行预测并返回概率和类别"""
        
        dtest = xgb.DMatrix(features_df)
        y_pred_proba = self.model.predict(dtest)
        y_pred = np.array([1 if x > self.config.threshold else 0 for x in y_pred_proba], dtype=int)
        
        nan_rows_mask = features_df.isna().any(axis=1)
        y_pred[nan_rows_mask] = 0
        y_pred_proba[nan_rows_mask] = 0.0
        
        return y_pred, y_pred_proba


class MetricsCalculator:
    """指标计算类，负责各种评估指标的计算"""

    @staticmethod
    def calculate_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Union[float, np.ndarray, List[np.ndarray]]]:
        """计算基本评估指标"""
        cm = confusion_matrix(y_true, y_pred)
        TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

        # 创建对应TN, FP, FN, TP的mask
        TN_mask = (y_true == 0) & (y_pred == 0)
        FP_mask = (y_true == 0) & (y_pred == 1)
        FN_mask = (y_true == 1) & (y_pred == 0)
        TP_mask = (y_true == 1) & (y_pred == 1)

        # 手动构建交换后的矩阵
        cm_swapped = np.array([[TP, FP],
                            [FN, TN]])
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_pred_proba) ,
            'cm' : cm_swapped,
            'masks': [TP_mask, FP_mask, FN_mask, TN_mask]
        }

    @staticmethod
    def calculate_pig_level_metrics(labels_df: pd.DataFrame, y_pred: np.ndarray, y_pred_prob: np.ndarray) -> Dict:
        """计算猪级别的指标"""
        test_pigs = labels_df['pig_id'].unique()
        pig_results = []
        sign_count = {}

        # 初始化混淆矩阵计数
        c00, c01, c10, c11 = 0, 0, 0, 0

        # 初始化混淆矩阵对应的猪ID列表
        # 矩阵格式: [[c11, c01], [c10, c00]] 即 [[TP, FP], [FN, TN]]
        tp_pigs, fp_pigs, fn_pigs, tn_pigs = [], [], [], []

        for pig_id in test_pigs:
            select_mask = labels_df['pig_id'] == pig_id
            select_df = labels_df.loc[select_mask]
            select_label = select_df['label'].values
            select_eva_label = select_df['eva_label'].values
            select_pred = y_pred[select_mask]
            select_pred_prob = y_pred_prob[select_mask]

            # 计算sign值
            sign = calculate_pig_sign(select_pred, select_label)
            sign_count[sign] = sign_count.get(sign, 0) + 1

            # 记录猪的预测结果
            pig_results.append({
                'pig_id': pig_id,
                'label': select_label.tolist(),
                'eva_label': select_eva_label.tolist(),
                'pred': select_pred.tolist(),
                'pred_prob': [round(x, 2) for x in select_pred_prob.tolist()]
            })
            
            
            #新增，只考虑第一次预测
            indices = np.where(select_pred == 1)[0]
            if len(indices) > 0:
                select_pred[:] = 0
                select_pred[indices[0]] = 1

            # 更新混淆矩阵并记录猪ID
            if sum(select_label) > 0:
                # 真实为正例
                if sum(select_pred & select_eva_label) == 0:
                    # 预测为负例，False Negative
                    c10 += 1
                    fn_pigs.append(pig_id)
                else:
                    # 预测为正例，True Positive
                    c11 += 1
                    tp_pigs.append(pig_id)
            else:
                # 真实为负例
                if sum(select_pred) > 0:
                    # 预测为正例，False Positive
                    c01 += 1
                    fp_pigs.append(pig_id)
                else:
                    # 预测为负例，True Negative
                    c00 += 1
                    tn_pigs.append(pig_id)

        # 计算最终指标
        recall = c11 / (c11 + c10) if (c11 + c10) > 0 else 0
        precision = c11 / (c11 + c01) if (c11 + c01) > 0 else 0
        accuracy = (c11 + c00) / (c11 + c01 + c00 + c10) if (c11 + c01 + c00 + c10) > 0 else 0

        confusion_matrix = [[c11, c01], [c10, c00]]

        # 按照混淆矩阵的顺序返回猪ID列表
        confusion_matrix_pigs = [tp_pigs, fp_pigs, fn_pigs, tn_pigs]

        return {
            'pig_results': pig_results,
            'sign_count': sign_count,
            'confusion_matrix': confusion_matrix,
            'confusion_matrix_pigs': confusion_matrix_pigs,
            'recall': recall,
            'precision': precision,
            'accuracy': accuracy
        }


class ResultSaver(ABC):
    """结果保存抽象基类"""

    @abstractmethod
    def save(self, data, path: str):
        """保存数据的抽象方法"""
        pass


class CSVResultSaver(ResultSaver):
    """CSV结果保存器"""

    def save(self, data: pd.DataFrame, path: str):
        """保存为CSV格式"""
        data.to_csv(path, index=False, encoding='utf-8-sig')


class PNGResultSaver(ResultSaver):
    """PNG图片保存器"""

    def save(self, fig, path: str):
        """保存为PNG格式"""
        fig.savefig(path, dpi=300, bbox_inches='tight')


class ExperimentDataProcessor:
    """实验数据处理器，负责处理和整理实验数据，方便后续画图"""

    def __init__(self, label_path: str):
        self.label_path = label_path

    def process_experiment_data(self, dataset, labels_df, y_pred,
                               label_type='breeding', breeding_offset_days=2):
        """
        处理实验数据，整理预测结果和真实标签

        参数:
        dataset: 原始数据集
        labels_df: 标签DataFrame
        y_pred: 预测结果
        label_type: 标签类型
        breeding_offset_days: 配种偏移天数

        返回:
        dict: 包含处理后的实验数据
        """
        # 获取基本信息
        test_pigs = labels_df['pig_id'].unique()
        experiment_list = labels_df['experiment'].unique()
        exp_dict = {exp: i for i, exp in enumerate(experiment_list)}
        exp_date = {}
        
        for item in dataset:
            exp = item[0]
            if exp not in experiment_list:
                continue
            if exp not in exp_date:
                exp_date[exp] = item[2]
        
        all_data = [{} for _ in range(len(experiment_list))]
        dates = []
        
        for expnum in experiment_list:
            label_file = os.path.join(self.label_path, expnum + '.csv')
            label_data = pd.read_csv(label_file, index_col=None, header=0, dtype=str)
            exp_data = labels_df[labels_df['experiment'] == expnum]
            dates.append(range(len(exp_date[expnum])))
            pig_ids = exp_data['pig_id'].unique()
            
            for pig_id in pig_ids:
                exp = exp_dict[expnum]
                id = pig_id.split('_')[1]
                if pig_id not in all_data[exp]:
                    all_data[exp][pig_id] = {'y_pred': [], 'y_true': []}
                
                select_index = exp_data.index[exp_data['pig_id'] == pig_id]
                pig_pred = [0 for _ in range(len(exp_date[expnum]))] 
                pred_date = exp_data['date'][select_index].tolist()
                pred_index = [exp_date[expnum].index(date) for date in pred_date if date in exp_date[expnum]]
                
                for i, index in enumerate(pred_index):
                    pig_pred[index] = int(y_pred[select_index][i])
                all_data[exp][pig_id]['y_pred'] = pig_pred
                
                pig_label = [0 for _ in range(len(exp_date[expnum]))]
                select_index = label_data.iloc[:, 0] == id
                pl = label_data[select_index].iloc[0].tolist()
                
                
                if label_type == 'breeding':
                    if len(pl[2]) >= 8:
                        ld = pl[2][:8]
                        ind = exp_date[expnum].index(ld)
                        pig_label[ind] = 1
                elif label_type == 'breeding_offset':
                    if len(pl[2]) >= 8:
                        ld = pl[2][:8]
                        ind = exp_date[expnum].index(ld) - breeding_offset_days
                        pig_label[ind] = 1
                elif label_type == 'oestrus':
                    if len(pl[1]) >= 8:
                        ld = pl[1][:8]
                        ind = exp_date[expnum].index(ld)
                        pig_label[ind] = 1
                all_data[exp][pig_id]['y_true'] = pig_label

        return {
            'experiment_list': experiment_list,
            'all_data': all_data,
            'dates': dates
        }


class ExperimentEvaluator:
    """实验评估主类 - 组合各个组件"""

    def __init__(self, model, config: EvaluationConfig = None, label_path: str = None):
        self.config = config or EvaluationConfig()
        self.predictor = ModelPredictor(model, self.config)
        self.metrics_calculator = MetricsCalculator()
        self.label_path = label_path

        # 修复label_path的传递
        if label_path:
            self.data_processor = ExperimentDataProcessor(label_path)
        else:
            # 如果没有提供label_path，使用config中的
            label_path = getattr(self.config, 'label_path', None)
            if label_path:
                self.data_processor = ExperimentDataProcessor(label_path)
            else:
                self.data_processor = ExperimentDataProcessor('')

        # 初始化结果保存器
        self.savers = {
            'csv': CSVResultSaver(),
            'png': PNGResultSaver()
        }

        # 初始化可视化器
        self.visualizer = Visualizer()
        self.feature_visualizer = FeatureImportanceVisualizer()

    def evaluate(self, dataset, test_exp_list) -> List:
        """执行完整的评估流程"""
        # 1. 准备数据
        preparator = DataPreparator(self.label_path)
        features_df, labels_df = preparator.prepare_features(dataset, test_exp_list, self.config)

        # 2. 进行预测
        y_pred, y_pred_proba = self.predictor.predict(features_df)

        # 3. 计算指标
        basic_metrics = self.metrics_calculator.calculate_basic_metrics(
            labels_df['eva_label'], y_pred, y_pred_proba
        )

        pig_level_metrics = self.metrics_calculator.calculate_pig_level_metrics(
            labels_df, y_pred, y_pred_proba
        )
        
        features_TP = features_df[basic_metrics['masks'][0]]
        features_FP = features_df[basic_metrics['masks'][1]]
        features_FN = features_df[basic_metrics['masks'][2]]
        features_TN = features_df[basic_metrics['masks'][3]]
        
        features_cm = [features_TP, features_FP, features_FN, features_TN]

        # 4. 创建评估DataFrame
        eval_df = self._create_evaluation_dataframe(basic_metrics)

        # 5. 保存结果（如果需要）
        if self.config.save_results and self.config.output_dir:
            self._save_results(
                dataset, labels_df, y_pred,
                self.config.output_dir, self.config.label_type,
                self.config.breeding_offset_days
            )

        # 6. 返回结果
        return [
            pig_level_metrics['confusion_matrix'],
            pig_level_metrics['accuracy'],
            pig_level_metrics['precision'],
            pig_level_metrics['recall'],
            pig_level_metrics['sign_count'],
            pig_level_metrics['confusion_matrix_pigs'],
            pd.DataFrame(pig_level_metrics['pig_results']),
            features_cm,
            eval_df,
        ]

    def _save_results(self, dataset, labels_df, y_pred, output_dir,
                     label_type, breeding_offset_days):
        """
        保存实验的详细结果，包括CSV文件和可视化图表

        参数:
        dataset: 原始数据集
        labels_df: 标签DataFrame
        y_pred: 预测结果
        output_dir: 输出目录
        label_type: 标签类型
        breeding_offset_days: 配种偏移天数
        """
        try:
            # 1. 处理实验数据
            processed_data = self.data_processor.process_experiment_data(
                dataset, labels_df, y_pred, label_type, breeding_offset_days
            )

            # 2. 创建输出目录
            os.makedirs(output_dir, exist_ok=True)

            # 3. 保存每个实验的结果
            for i, expnum in enumerate(processed_data['experiment_list']):
                if i >= len(processed_data['all_data']):
                    continue

                exp_data = processed_data['all_data'][i]
                exp_dates = processed_data['dates'][i]

                # 准备数据用于保存
                result_data = []
                index = []
                real_values = []
                pred_values = []
                radar_ids = []

                for pig_id, data in exp_data.items():
                    # 提取雷达ID
                    radar_id = pig_id.split('_')[0] + '_' + pig_id.split('_')[1]
                    radar_ids.append(radar_id)

                    # 添加预测和真实标签数据
                    index.append(f'{radar_id}_pred')
                    index.append(f'{radar_id}_true')

                    pred = data['y_pred']
                    lab = data['y_true']

                    result_data.append(pred)
                    result_data.append(lab)
                    real_values.append(lab)
                    pred_values.append(pred)

                # 4. 保存CSV文件
                df = pd.DataFrame(result_data, index=index)
                csv_path = os.path.join(output_dir, f'{expnum}_result.csv')
                df.to_csv(csv_path, encoding='utf-8-sig')
                print(f"已保存: {csv_path}")

                # 5. 保存PNG图表
                png_path = os.path.join(output_dir, f'{expnum}_result.png')
                self.visualizer.plot_prediction_results(
                    real_values, pred_values, exp_dates,
                    radar_ids, expnum, png_path
                )
                print(f"已保存: {png_path}")

        except Exception as e:
            print(f"保存结果时发生错误: {str(e)}")
            raise

    def _create_evaluation_dataframe(self, metrics: Dict) -> pd.DataFrame:
        """创建评估结果DataFrame"""


        return pd.DataFrame([{
            "准确率": f"{metrics['accuracy']:.4f}",
            "精确率": f"{metrics['precision']:.4f}",
            "召回率": f"{metrics['recall']:.4f}",
            "F1分数": f"{metrics['f1']:.4f}",
            "AUC分数": f"{metrics['auc']:.4f}",
            "混淆矩阵": f"{metrics['cm']}"
        }])

    def save_feature_importance(self, model, features_df, output_dir,
                              top_n=20, importance_type='weight'):
        """
        保存特征重要性图表

        参数:
        model: 训练好的模型
        features_df: 特征DataFrame
        output_dir: 输出目录
        top_n: 显示前N个特征
        importance_type: 重要性类型
        """
        try:
            os.makedirs(output_dir, exist_ok=True)

            # 使用内置重要性图
            fig1, ax1 = self.feature_visualizer.plot_importance(
                model, features_df, top_n,
                os.path.join(output_dir, 'feature_importance_builtin.png'),
                importance_type
            )

            # 使用自定义重要性图
            fig2, ax2 = self.feature_visualizer.plot_custom_importance(
                model, features_df, top_n,
                os.path.join(output_dir, 'feature_importance_custom.png'),
                importance_type
            )

            # 保存特征重要性数据
            importance_df = self.get_feature_importance_df(model, importance_type)
            importance_df.to_csv(
                os.path.join(output_dir, 'feature_importance.csv'),
                index=False, encoding='utf-8-sig'
            )

            print(f"特征重要性图表已保存到: {output_dir}")

        except Exception as e:
            print(f"保存特征重要性时发生错误: {str(e)}")
            raise

    def get_feature_importance_df(self, model, importance_type='weight'):
        """
        获取特征重要性的DataFrame

        参数:
        model: 训练好的模型
        importance_type: 重要性类型

        返回:
        DataFrame: 特征重要性数据
        """
        importance = model.get_score(importance_type=importance_type)
        importance_df = pd.DataFrame(
            list(importance.items()),
            columns=['feature', 'importance']
        )
        importance_df = importance_df.sort_values(
            'importance', ascending=False
        ).reset_index(drop=True)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        return importance_df


class Visualizer:
    """可视化类，负责所有绘图功能"""

    def __init__(self, style='default'):
        self.style = style
        self._setup_style()

    def _setup_style(self):
        """设置绘图样式"""
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

    def plot_prediction_results(self, real_values, pred_values, dates,
                              group_labels, title, save_path=None):
        """绘制预测结果对比图"""
        def get_color(r, p):
            if r == 0 and p == 0:
                return 'white'
            elif r == 1 and p == 0:
                return 'blue'
            elif r == 0 and p == 1:
                return 'red'
            elif r == 1 and p == 1:
                return 'purple'
            else:
                return 'grey'

        n_groups = len(group_labels)
        block_num = len(dates)

        cell_width = 0.3
        cell_height = 0.1
        fig_width = block_num * cell_width + 2
        fig_height = n_groups * cell_height + 2

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        for row in range(n_groups):
            real = real_values[row]
            pred = pred_values[row]
            for col in range(len(real)):
                color = get_color(real[col], pred[col])
                rect = plt.Rectangle(
                    (col * cell_width, (n_groups - row - 1) * cell_height),
                    cell_width, cell_height,
                    facecolor=color, edgecolor='black', linewidth=0.5
                )
                ax.add_patch(rect)

        # 设置坐标轴
        ax.set_xlabel("Date")
        ax.set_ylabel("Sample")
        ax.set_xticks([(i + 0.5) * cell_width for i in range(block_num)])
        ax.set_xticklabels(range(len(dates)), rotation=0)
        ax.set_yticks([(n_groups - i - 0.5) * cell_height for i in range(n_groups)])
        ax.set_yticklabels(group_labels)
        ax.set_xlim(0, block_num * cell_width)
        ax.set_ylim(0, n_groups * cell_height)
        ax.set_aspect('auto')
        plt.axis('on')
        plt.title(f'{title} Prediction Results')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, ax


class FeatureImportanceVisualizer:
    """特征重要性可视化类"""

    def __init__(self, style='default'):
        self.style = style
        self._setup_style()

    def _setup_style(self):
        """设置绘图样式"""
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

    def plot_importance(self, model, top_n=20, save_path=None,
                       importance_type='weight', figsize=(12, 8)):
        """绘制特征重要性图"""
        from xgboost import plot_importance

        fig, ax = plt.subplots(figsize=figsize)
        plot_importance(model, ax=ax, max_num_features=top_n,
                       importance_type=importance_type, show_values=False)
        ax.set_title(f'XGBoost特征重要性 ({importance_type}) - Top {top_n}',
                    fontsize=14, pad=20)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, ax
    
    def plot_multiple_importance_types(self, model, save_dir=None, top_n=20, figsize=(15, 10)):
        """
        绘制多种重要性类型的对比图

        参数:
            model: 训练好的XGBoost模型
            save_dir: str - 保存目录（可选）
            top_n: int - 显示前N个最重要的特征
            figsize: tuple - 图形大小

        返回:
            fig - matplotlib图形对象
        """
        # 定义不同的重要性类型
        importance_types = ['weight', 'gain', 'cover']
        titles = ['权重', '增益', '覆盖率']

        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        for idx, (imp_type, title) in enumerate(zip(importance_types, titles)):
            # 获取特征重要性
            importance = model.get_score(importance_type=imp_type)

            # 转换为DataFrame并排序
            importance_df = pd.DataFrame(list(importance.items()), columns=['feature', 'importance'])
            importance_df = importance_df.sort_values('importance', ascending= False).head(top_n)

            # 绘制条形图
            bars = axes[idx].barh(range(len(importance_df)), importance_df['importance'],
                                color='lightcoral', edgecolor='black', alpha=0.7)

            # 设置标签
            axes[idx].set_yticks(range(len(importance_df)))
            axes[idx].set_yticklabels(importance_df['feature'], fontsize=8)
            axes[idx].set_xlabel(f'{title}分数', fontsize=10)
            axes[idx].set_title(f'特征{title}重要性', fontsize=12)

            # 添加数值标签
            for i, bar in enumerate(bars):
                width = bar.get_width()
                axes[idx].text(width + max(importance_df['importance']) * 0.01,
                            bar.get_y() + bar.get_height()/2,
                            f'{width:.1f}',
                            ha='left', va='center', fontsize=8)

            # 反转y轴
            axes[idx].invert_yaxis()

            # 添加网格线
            axes[idx].grid(True, axis='x', alpha=0.3)

        # 设置总标题
        fig.suptitle('XGBoost特征重要性对比', fontsize=16, y=0.98)

        # 调整布局
        plt.tight_layout()

        # 保存图片
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'feature_importance_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特征重要性对比图已保存到: {save_path}")

        return fig
    

def show_result_dataframe(test_exp_list, result_list):
    # 1. 定义列名顺序
    columns = [
        '实验名', 
        '准确率', 
        '精确率', 
        '召回率', 
        '-3', '-2', '-1', '0', '1', '2', '未预测', "未配种/发情 且 未预测", "未配种/发情 但 预测"
    ]
    
    # 2. 准备存储数据的列表
    data_rows = []
    
    # 3. 遍历数据进行提取
    for i in range(len(test_exp_list)):
        # 获取当前实验的数据
        item = result_list[i]
        exp_name = test_exp_list[i]
        
        # 提取准确率、精确率、召回率 (索引 1, 2, 3)
        metrics = item[1:4]
        
        # 提取分类统计字典 (索引 4)
        # 使用 .get() 方法防止某个标签缺失时报错，默认补 0
        count_dict = item[4]
        
        # 按照指定顺序提取字典中的值
        # 顺序：'-3', '-2', '-1', '0', '1', '2', '未预测'
        counts = [
            count_dict.get('-3', 0),
            count_dict.get('-2', 0),
            count_dict.get('-1', 0),
            count_dict.get('0', 0),
            count_dict.get('1', 0),
            count_dict.get('2', 0),
            count_dict.get('未预测', 0),
            count_dict.get('未配种/发情 且 未预测', 0),
            count_dict.get('未配种/发情 但 预测', 0)
        ]
        
        # 将实验名、指标和统计数合并为一行数据
        row = [exp_name] + metrics + counts
        data_rows.append(row)
        
    # 4. 创建 DataFrame
    df = pd.DataFrame(data_rows, columns=columns)
    
    return df

# 为了保持向后兼容性，保留原有的函数接口
def evaluate_experiment(dataset, label_path, test_exp_list, model,
                       save_results=False, output_dir=None, label_type='breeding',
                       breeding_offset_days=2, interval_type='-2_0',
                       is_completedata=False):
    """
    评估实验的性能指标（保持向后兼容的接口）

    注意：此函数是重构后的简化版本，推荐使用ExperimentEvaluator类
    """
    config = EvaluationConfig(
        label_type=label_type,
        breeding_offset_days=breeding_offset_days,
        interval_type=interval_type,
        save_results=save_results,
        output_dir=output_dir,
        is_completedata=is_completedata
    )

    evaluator = ExperimentEvaluator(model, config, label_path)
    return evaluator.evaluate(dataset, test_exp_list)