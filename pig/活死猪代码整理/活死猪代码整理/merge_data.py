import pandas as pd
import pickle

# 读取3个特征dataframe
df_filtered = pd.read_pickle("pig_features_data_all_filtered.pkl")
df_label1 = pd.read_pickle("pig_features_data_all_label1.pkl")
df_houbei = pd.read_pickle("pig_features_data_all_train_houbei.pkl")

print(f"filtered: {df_filtered.shape}, label1: {df_label1.shape}, houbei: {df_houbei.shape}")

# 读取需要保留的文件路径列表
with open("hdf5_file_paths.pkl", "rb") as f:
    valid_paths = pickle.load(f)
if not isinstance(valid_paths, set):
    valid_paths = set(valid_paths)

# 第一步：合并 houbei 和 label1，按file_path去重，再根据路径列表过滤
df_step1 = pd.concat([df_houbei, df_label1], ignore_index=True).drop_duplicates(subset=["file_path"], keep="first")
df_step1 = df_step1[df_step1["file_path"].isin(valid_paths)].reset_index(drop=True)
print(f"houbei + label1 合并去重并过滤后: {df_step1.shape}")

# 第二步：再合并 filtered，按file_path去重
df_final = pd.concat([df_step1, df_filtered], ignore_index=True).drop_duplicates(subset=["file_path"], keep="first").reset_index(drop=True)
print(f"最终合并后: {df_final.shape}")

# 保存
df_final.to_pickle("pig_features_data_merged.pkl")
print("已保存至 pig_features_data_merged.pkl")
