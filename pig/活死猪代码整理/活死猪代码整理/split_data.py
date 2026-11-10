import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

# 加载原始数据
print("正在加载数据...")
df = pd.read_pickle('pig_features_data_merged.pkl')
print(f"原始数据形状：{df.shape}")
print(f"标签分布：无猪 -0: {(df['label']==0).sum()}, 活猪 -1: {(df['label']==1).sum()}, 死猪 -2: {(df['label']==2).sum()}")

# 分层抽样分割训练集和测试集（8:2）
print("\n正在进行分层抽样分割...")
train_df, test_df = train_test_split(
    df,
    test_size=0.2,  # 20% 作为测试集
    random_state=42,
    stratify=df['label']  # 按标签分层抽样
)

print(f"训练集样本数：{len(train_df)}")
print(f"测试集样本数：{len(test_df)}")
print(f"训练集标签分布：0={(train_df['label']==0).sum()}, 1={(train_df['label']==1).sum()}, 2={(train_df['label']==2).sum()}")
print(f"测试集标签分布：0={(test_df['label']==0).sum()}, 1={(test_df['label']==1).sum()}, 2={(test_df['label']==2).sum()}")

# 保存训练集和测试集
print("\n正在保存数据...")
train_df.to_pickle('pig_features_data_train.pkl')
test_df.to_pickle('pig_features_data_test.pkl')

# 保存测试集 file_paths
file_paths_test = test_df['file_path'].tolist()
with open('test_file.pkl', 'wb') as f:
    pickle.dump(file_paths_test, f)

print("\n数据分割完成！")
print("  - pig_features_data_train.pkl: 训练集")
print("  - pig_features_data_test.pkl: 测试集")
print("  - test_file.pkl: 测试集文件路径列表")
