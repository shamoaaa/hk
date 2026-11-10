# 猪分类模型模块使用说明

## 文件说明

`pig_classifier.py` - 猪分类模型训练与测试一体化模块

## 功能概述

该模块用于猪的分类识别，支持三种状态：
- **0**: 无猪
- **1**: 活猪
- **2**: 死猪

## 数据格式

### 输入数据格式（pkl 文件）

训练数据和测试数据应为 pandas DataFrame，包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `max_channel` | list | 雷达检测到最大通道（120 个时间点） |
| `ideal_channel` | list | 理想通道（120 个时间点） |
| `br_amp` | list | 呼吸振幅（120 个时间点） |
| `br_cv` | list | 呼吸波动率（120 个时间点） |
| `up_down` | list | 是否起卧（120 个时间点） |
| `label` | int | 标签：0-无猪，1-活猪，2-死猪 |
| `file_path` | str | 数据文件路径 |

## 函数说明

### 1. `train_model(train_data_path, model_save_path, segment_size=30)`

训练猪分类模型。

**参数：**
- `train_data_path` (str): 训练数据 pkl 文件路径
- `model_save_path` (str): 模型保存路径
- `segment_size` (int): 片段大小，默认 30（将 120 分钟数据分为 4 个片段）

**返回值：**
- `model`: 训练好的 XGBoost 模型
- `feature_cols`: 特征列名列表

**示例：**
```python
from pig_classifier import train_model

model, feature_cols = train_model(
    train_data_path='pig_features_data_train.pkl',
    model_save_path='pig_classifier_model.pkl'
)
```

### 2. `test_model(test_data_path, model_path, segment_size=30, use_fusion=True)`

测试猪分类模型，打印混淆矩阵和分类报告。

**参数：**
- `test_data_path` (str): 测试数据 pkl 文件路径
- `model_path` (str): 模型文件路径
- `segment_size` (int): 片段大小，需与训练时一致
- `use_fusion` (bool): 是否使用融合算法，默认 True

**返回值：**
- `dict`: 包含以下键：
  - `accuracy_direct`: 直接预测准确率
  - `accuracy_fused`: 融合后准确率
  - `cm_direct`: 直接预测混淆矩阵
  - `cm_fused`: 融合后混淆矩阵
  - `y_pred_fused`: 融合后预测结果
  - `y_test_fused`: 融合后真实标签
  - `confidence_scores`: 置信度分数

**示例：**
```python
from pig_classifier import test_model

results = test_model(
    test_data_path='pig_features_data_test.pkl',
    model_path='pig_classifier_model.pkl',
    use_fusion=True
)

print(f"直接预测准确率：{results['accuracy_direct']:.4f}")
print(f"融合后准确率：{results['accuracy_fused']:.4f}")
```

## 特征提取说明

模块将每个 1 小时的样本分割成多个片段（默认 4 个，每段 15 分钟），从每个片段中提取以下特征：

### 呼吸振幅特征
- `br_amp_mean`: 平均呼吸振幅
- `breathing_stop_ratio`: 呼吸停止检测（振幅<0.2 的比例）
- `breathing_high_ratio`: 高呼吸检测（振幅>1.0 的比例）

### 呼吸波动率特征
- `br_cv_mean`: 平均波动率
- `br_cv_high_ratio`: 高波动率占比（>1.0）
- `br_cv_mid_ratio`: 中波动率占比（0.5-1.0）
- `br_cv_low_ratio`: 低波动率占比（<0.5）

### 通道相关特征
- `channel_match_ratio`: 通道匹配率
- `max_channel_changes`: 通道变化次数
- `max_channel_no`: 无猪通道占比（21-23）
- `max_channel_dead`: 死猪通道占比（12-14）
- `max_channel_13`: 通道 13 占比
- `ideal_channel_13`: 理想通道 13 占比

### 起卧行为特征
- `up_down_ratio`: 起卧时间占比
- `up_down_transitions`: 状态转换次数
- `up_down_avg_duration`: 平均起卧持续时间

### 组合特征
- `activity_score`: 活动分数（起卧比×通道变化）
- `breathing_stability`: 呼吸稳定性

### 基于规则的分类比例
- `rate_alive`: 活猪规则匹配比例
- `rate_dead`: 死猪规则匹配比例
- `rate_empty`: 无猪规则匹配比例

## 融合算法说明

测试时，模块会对每个样本的多个片段预测结果进行融合：

1. **投票法**: 4 个片段投票，多数票决定最终结果（要求票数>40%）
2. **概率平均法**: 计算每个类别的平均概率，选择概率最高的类别
3. **融合策略**: 投票法失效时使用概率平均法结果

## 完整使用示例

```python
from pig_classifier import train_model, test_model

# 1. 训练模型
print("开始训练模型...")
model, feature_cols = train_model(
    train_data_path='pig_features_data_train.pkl',
    model_save_path='pig_classifier_model.pkl',
    segment_size=30
)

# 2. 测试模型
print("\n开始测试模型...")
results = test_model(
    test_data_path='pig_features_data_test.pkl',
    model_path='pig_classifier_model.pkl',
    segment_size=30,
    use_fusion=True
)

# 3. 输出结果
print(f"\n最终结果：")
print(f"直接预测准确率：{results['accuracy_direct']:.4f}")
print(f"融合后准确率：{results['accuracy_fused']:.4f}")
```

## 直接运行

该模块支持直接运行：

```bash
python pig_classifier.py
```

将自动使用 `pig_features_data_train.pkl` 和 `pig_features_data_test.pkl` 进行训练和测试。

## 依赖库

- pandas
- numpy
- xgboost
- scikit-learn