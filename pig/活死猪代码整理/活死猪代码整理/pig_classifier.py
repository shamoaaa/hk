"""
猪分类模型训练与测试模块
功能：封装训练和测试代码，用于猪的分类（无猪、活猪、死猪）

标签说明：
- 0: 无猪
- 1: 活猪
- 2: 死猪

原始特征：
- max_channel: 雷达检测到最大通道
- ideal_channel: 理想通道
- br_amp: 呼吸振幅（1小时内提取的列表）
- br_cv: 呼吸的波动率（1小时内提取的列表）
- up_down: 是否起卧（1小时内提取的列表）
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import defaultdict


# ==================== 特征提取函数 ====================

def extract_features_for_segment(br_amp_segment, br_cv_segment, max_ch_segment, ideal_ch_segment, up_down_segment):
    """从单个时间片段中提取特征"""
    features = {}

    # ============ 1. 呼吸振幅 (br_amp) 特征 ============
    features['br_amp_mean'] = np.mean(br_amp_segment)
    features['breathing_stop_ratio'] = np.mean(br_amp_segment < 0.2)
    features['breathing_high_ratio'] = np.mean(br_amp_segment > 1.0)

    # ============ 2. 呼吸波动率 (br_cv) 特征 ============
    features['br_cv_mean'] = np.mean(br_cv_segment)
    features['br_cv_high_ratio'] = np.mean(br_cv_segment > 1.0)
    features['br_cv_mid_ratio'] = np.mean((0.5 < br_cv_segment) & (br_cv_segment < 1))
    features['br_cv_low_ratio'] = np.mean(br_cv_segment < 0.5)

    # ============ 3. 通道相关特征 ============
    channel_match = (max_ch_segment == ideal_ch_segment).astype(float)
    features['channel_match_ratio'] = np.mean(channel_match)
    max_ch_diff = np.diff(max_ch_segment)
    features['max_channel_changes'] = np.sum(max_ch_diff != 0)
    features['max_channel_no'] = np.mean((max_ch_segment >= 21) & (max_ch_segment <= 23))
    features['max_channel_dead'] = np.mean((max_ch_segment >= 12) & (max_ch_segment <= 14))
    features['max_channel_13'] = np.mean(max_ch_segment == 13)
    features['ideal_channel_13'] = np.mean(ideal_ch_segment == 13)

    # ============ 4. 起卧行为特征 ============
    features['up_down_ratio'] = np.mean(up_down_segment)
    up_down_diff = np.abs(np.diff(up_down_segment))
    features['up_down_transitions'] = np.sum(up_down_diff > 0.5)

    # 连续起卧时长
    consecutive_lengths = []
    current_length = 0
    for val in up_down_segment:
        if val > 0.5:
            current_length += 1
        else:
            if current_length > 0:
                consecutive_lengths.append(current_length)
            current_length = 0
    if current_length > 0:
        consecutive_lengths.append(current_length)
    features['up_down_avg_duration'] = np.mean(consecutive_lengths) if consecutive_lengths else 0

    # ============ 5. 组合特征 ============
    features['activity_score'] = features['up_down_ratio'] * features['max_channel_changes']
    features['breathing_stability'] = features['br_amp_mean'] * (1 - features['br_cv_high_ratio'])

    # ============ 6. 基于规则分类的比例特征 ============
    rule_rates = calc_rule_rates_for_segment(max_ch_segment, br_amp_segment, up_down_segment)
    features['rate_alive'] = rule_rates['rate_alive']
    features['rate_dead'] = rule_rates['rate_dead']
    features['rate_empty'] = rule_rates['rate_empty']

    return features


def calc_rule_rates_for_segment(mc_arr, amp_arr, up_arr):
    """基于规则的分类比例计算（针对单个片段）"""
    n = len(mc_arr)
    if n == 0:
        return {'rate_alive': 0, 'rate_dead': 0, 'rate_empty': 0}

    # 活猪：(mc <= 20 and amp >= 0.5) or is_up
    mask_alive = ((mc_arr <= 20) & (amp_arr >= 0.5)) | (up_arr == 1)
    # 死猪：13 <= mc <= 15 and amp < 0.5
    mask_dead = (mc_arr >= 13) & (mc_arr <= 15) & (amp_arr < 0.5)
    # 无猪：其余情况
    mask_empty = ~(mask_alive | mask_dead)

    return {
        'rate_alive': np.sum(mask_alive) / n,
        'rate_dead': np.sum(mask_dead) / n,
        'rate_empty': np.sum(mask_empty) / n
    }


def extract_features_with_segments(row, segment_size=30):
    """将样本分割成多个片段并提取每个片段的特征"""
    max_ch = np.array(row['max_channel'])
    ideal_ch = np.array(row['ideal_channel'])
    br_amp = np.array(row['br_amp'])
    br_cv = np.array(row['br_cv'])
    up_down = np.array(row['up_down'])

    total_points = len(br_amp)
    num_segments = total_points // segment_size

    segments_features = []

    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size

        br_amp_seg = br_amp[start_idx:end_idx]
        br_cv_seg = br_cv[start_idx:end_idx]
        max_ch_seg = max_ch[start_idx:end_idx]
        ideal_ch_seg = ideal_ch[start_idx:end_idx]
        up_down_seg = up_down[start_idx:end_idx]

        features = extract_features_for_segment(
            br_amp_seg, br_cv_seg, max_ch_seg, ideal_ch_seg, up_down_seg
        )
        features['segment_id'] = i
        features['label'] = row['label']
        features['file_path'] = row['file_path']
        segments_features.append(features)

    return segments_features


# ==================== 融合预测函数 ====================

def fuse_predictions(file_paths, segment_preds, segment_probas, true_labels):
    """
    融合每个样本的多个片段预测结果

    融合算法：
    1. 投票法：4个片段投票，多数票决定最终结果（要求票数大于总数的40%）
    2. 概率平均：计算每个类别的平均概率，选择概率最高的类别
    3. 加权融合：投票法失效时使用概率平均法结果
    """
    unique_files = np.unique(file_paths)
    fused_preds = []
    fused_proba_max = []
    true_labels_fused = []

    for file_path in unique_files:
        mask = file_paths == file_path
        file_preds = segment_preds[mask]
        file_probas = segment_probas[mask]
        file_true = true_labels[mask][0]

        # 方法 1: 投票法（修改版）- 要求大于总数的0.4且最大
        votes = np.bincount(file_preds, minlength=3)
        total_votes = len(file_preds)
        threshold = total_votes * 0.4

        max_votes = np.max(votes)
        max_vote_class = np.argmax(votes)

        if max_votes > threshold:
            vote_pred = max_vote_class
        else:
            vote_pred = None

        # 方法 2: 概率平均法
        avg_proba = np.mean(file_probas, axis=0)
        proba_pred = np.argmax(avg_proba)

        # 方法 3: 融合投票法和概率平均法
        if vote_pred is not None:
            weighted_pred = vote_pred
        else:
            weighted_pred = proba_pred

        final_pred = weighted_pred

        fused_preds.append(final_pred)
        fused_proba_max.append(np.max(avg_proba))
        true_labels_fused.append(file_true)

    return np.array(fused_preds), np.array(true_labels_fused), np.array(fused_proba_max)


# ==================== 训练函数 ====================

def train_model(train_data_path, model_save_path, segment_size=30):
    """
    训练猪分类模型

    参数:
        train_data_path: str, 训练数据pkl文件路径
        model_save_path: str, 模型保存路径
        segment_size: int, 片段大小（默认30）

    返回:
        model: 训练好的XGBoost模型
        feature_cols: list, 特征列名
    """
    print(f"加载训练数据: {train_data_path}")
    df_train = pd.read_pickle(train_data_path)
    print(f"训练集形状：{df_train.shape}")
    print(f"训练集标签分布：无猪 -0: {(df_train['label']==0).sum()}, 活猪 -1: {(df_train['label']==1).sum()}, 死猪 -2: {(df_train['label']==2).sum()}")

    # 提取特征
    print(f"\n正在提取训练集特征（片段大小：{segment_size}）...")
    train_features = []
    for idx, row in df_train.iterrows():
        segment_features = extract_features_with_segments(row, segment_size=segment_size)
        train_features.extend(segment_features)
        if (idx + 1) % 10000 == 0:
            print(f"已处理 {idx + 1}/{len(df_train)} 训练样本")

    train_features_df = pd.DataFrame(train_features)
    print(f"训练集特征形状（片段级）：{train_features_df.shape}")

    # 准备训练数据
    feature_cols = [c for c in train_features_df.columns if c not in ['label', 'file_path', 'segment_id']]
    X_train = train_features_df[feature_cols].values
    y_train = train_features_df['label'].values

    print(f"\n训练集片段数：{len(X_train)}")
    print(f"训练集标签分布：0={sum(y_train==0)}, 1={sum(y_train==1)}, 2={sum(y_train==2)}")

    # 训练模型
    print("\n正在训练 XGBoost 模型...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss',
        subsample=0.6,
        colsample_bytree=0.9
    )

    model.fit(X_train, y_train)
    print("模型训练完成！")

    # 保存模型
    model_data = {
        'model': model,
        'feature_cols': feature_cols,
        'segment_size': segment_size
    }
    with open(model_save_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n模型已保存至: {model_save_path}")

    return model, feature_cols


# ==================== 测试函数 ====================

def test_model(test_data_path, model_path, segment_size=30, use_fusion=False):
    """
    测试猪分类模型

    参数:
        test_data_path: str, 测试数据pkl文件路径
        model_path: str, 模型文件路径
        segment_size: int, 片段大小（默认30，需与训练时一致）
        use_fusion: bool, 是否使用融合算法（默认False）

    返回:
        dict: 包含准确率、混淆矩阵等结果
    """
    # 加载模型
    print(f"加载模型: {model_path}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    feature_cols = model_data['feature_cols']
    print(f"模型加载成功，特征列数: {len(feature_cols)}")

    # 加载测试数据
    print(f"\n加载测试数据: {test_data_path}")
    df_test = pd.read_pickle(test_data_path)
    print(f"测试集形状：{df_test.shape}")
    print(f"测试集标签分布：无猪 -0: {(df_test['label']==0).sum()}, 活猪 -1: {(df_test['label']==1).sum()}, 死猪 -2: {(df_test['label']==2).sum()}")

    # 提取特征
    print(f"\n正在提取测试集特征（片段大小：{segment_size}）...")
    test_features = []
    for idx, row in df_test.iterrows():
        segment_features = extract_features_with_segments(row, segment_size=segment_size)
        test_features.extend(segment_features)
        if (idx + 1) % 5000 == 0:
            print(f"已处理 {idx + 1}/{len(df_test)} 测试样本")

    test_features_df = pd.DataFrame(test_features)
    print(f"测试集特征形状（片段级）：{test_features_df.shape}")

    # 准备测试数据
    X_test = test_features_df[feature_cols].values
    y_test = test_features_df['label'].values
    file_paths_test = test_features_df['file_path'].values
    segment_ids_test = test_features_df['segment_id'].values

    print(f"\n测试集片段数：{len(X_test)}")
    print(f"测试集标签分布：0={sum(y_test==0)}, 1={sum(y_test==1)}, 2={sum(y_test==2)}")

    # 预测
    print("\n正在进行预测...")
    y_pred_segments = model.predict(X_test)
    y_proba_segments = model.predict_proba(X_test)

    # 直接结果（不融合）
    print("\n" + "=" * 60)
    print("=== 直接结果混淆矩阵（不融合，每个分片就是一个样本）===")
    print("=" * 60)

    accuracy_direct = accuracy_score(y_test, y_pred_segments)
    print(f"\n直接预测测试集准确率：{accuracy_direct:.4f}")

    cm_direct = confusion_matrix(y_test, y_pred_segments)
    print(f"\n=== 混淆矩阵 (直接结果，不融合) ===")
    print("预测 →")
    print("真实 ↓    无猪 (0)  活猪 (1)  死猪 (2)")
    print(f"无猪 (0)    {cm_direct[0][0]:^5}   {cm_direct[0][1]:^5}   {cm_direct[0][2]:^5}")
    print(f"活猪 (1)    {cm_direct[1][0]:^5}   {cm_direct[1][1]:^5}   {cm_direct[1][2]:^5}")
    print(f"死猪 (2)    {cm_direct[2][0]:^5}   {cm_direct[2][1]:^5}   {cm_direct[2][2]:^5}")

    # print("\n=== 分类报告 (直接结果，不融合) ===")
    # print(classification_report(y_test, y_pred_segments, target_names=['无猪 (0)', '活猪 (1)', '死猪 (2)']))

    # 融合结果
    if use_fusion:
        print("\n" + "=" * 60)
        print("=== 融合后结果 ===")
        print("=" * 60)

        y_pred_fused, y_test_fused, confidence_scores = fuse_predictions(
            file_paths_test, y_pred_segments, y_proba_segments, y_test
        )

        accuracy_fused = accuracy_score(y_test_fused, y_pred_fused)
        print(f"\n融合后测试集准确率：{accuracy_fused:.4f}")

        cm_fused = confusion_matrix(y_test_fused, y_pred_fused)
        print(f"\n=== 混淆矩阵 (融合后) ===")
        print("预测 →")
        print("真实 ↓    无猪 (0)  活猪 (1)  死猪 (2)")
        print(f"无猪 (0)    {cm_fused[0][0]:^5}   {cm_fused[0][1]:^5}   {cm_fused[0][2]:^5}")
        print(f"活猪 (1)    {cm_fused[1][0]:^5}   {cm_fused[1][1]:^5}   {cm_fused[1][2]:^5}")
        print(f"死猪 (2)    {cm_fused[2][0]:^5}   {cm_fused[2][1]:^5}   {cm_fused[2][2]:^5}")

        # print("\n=== 分类报告 (融合后) ===")
        # print(classification_report(y_test_fused, y_pred_fused, target_names=['无猪 (0)', '活猪 (1)', '死猪 (2)']))

        return {
            'accuracy_direct': accuracy_direct,
            'accuracy_fused': accuracy_fused,
            'cm_direct': cm_direct,
            'cm_fused': cm_fused,
            'y_pred_fused': y_pred_fused,
            'y_test_fused': y_test_fused,
            'confidence_scores': confidence_scores
        }
    else:
        return {
            'accuracy_direct': accuracy_direct,
            'cm_direct': cm_direct,
            'y_pred_segments': y_pred_segments,
            'y_test': y_test
        }


# ==================== 主程序 ====================

if __name__ == "__main__":
    # 示例用法
    train_data_path = 'pig_features_data_train.pkl'
    test_data_path = 'pig_features_data_cs.pkl'
    model_save_path = 'pig_classifier_model.pkl'

    # # 训练模型
    # print("=" * 60)
    # print("开始训练模型")
    # print("=" * 60)
    # model, feature_cols = train_model(train_data_path, model_save_path)
    # print(f"\n特征列数: {len(feature_cols)}")
    # print(f"特征列: {feature_cols}")

    # 测试模型
    print("\n" + "=" * 60)
    print("开始测试模型")
    print("=" * 60)
    results = test_model(test_data_path, model_save_path, use_fusion=False)
    print(f"\n最终结果：")
    print(f"直接预测准确率: {results['accuracy_direct']:.4f}")
    # print(f"融合后准确率: {results['accuracy_fused']:.4f}")