
import pandas as pd

import numpy as np

from dtaidistance import dtw
import math

def kl_divergence(p, q):
    epsilon = 1e-10
    p = np.asarray(p, dtype=np.float32) + epsilon
    q = np.asarray(q, dtype=np.float32) + epsilon
    return np.sum(p * np.log(p / q))

def count_continuous_ones(data):
    # data 是一个由 0 和 1 组成的列表
    count = 0
    in_segment = False
    for bit in data:
        if bit == 1:
            if not in_segment:
                count += 1
                in_segment = True
        else:
            in_segment = False
    
    return count

def generate_breeding_features_v1(dataset):
    '''
    dataset:dataframe的数组->((br, normalize_act, is_up), (br, normalize_act, is_up),...)
    对于每头猪需要'br','is_up','normalize_act'三个特征，需要三天的数据：
    第一天 0~7 20~24  第二天 0~7 20~24 第三天 0~7
    实际输入以三天都为0~24进行输入, 非利用时间段数据可使用0进行补偿
    '''
    all_features = []
    pig_pred_mask = [] # 对模型预测了的天打上标记
    pig_dates = {} # 方便后续收集猪的按天结果

    for i, item in enumerate(dataset):
        # 解包数据
        pig_id = item[0]
        raw_data = item[1].copy()

        # 自动计算天数 (假设数据是连续的，每天 24 小时)
        hour_step = 2 * 60  # 2分钟一个点
        total_points = len(raw_data['normalize_act'].values)
        dates_cnt = math.ceil(total_points / (24 * hour_step))

        # 记录该猪的天数
        pig_dates[pig_id] = dates_cnt

        act_raw = raw_data['normalize_act'].values.reshape(-1, hour_step)
        br_raw = raw_data['br'].values.reshape(-1, hour_step)
        up_raw = raw_data['is_up'].values.reshape(-1, hour_step)


        # 从第三天开始计算特征
        for i in range(dates_cnt):
            if i<2:
                pig_pred_mask.append(-1)
                continue
            cur_hour_ind = i * 24
            hour_ind_1d = (i - 1) * 24
            hour_ind_2d = (i - 2) * 24

            select_cur_1 = range(cur_hour_ind, (cur_hour_ind + 7)) # 当前天的0点到7点
            select_cur_2 = range((cur_hour_ind + 20), (cur_hour_ind + 24)) # 当前天的20点到24点
            select_1d_1 = range(hour_ind_1d, (hour_ind_1d + 7))  # 前一天的0点到7点
            select_1d_2 = range((hour_ind_1d + 20), (hour_ind_1d + 24)) # 前一天的20点到24点
            select_2d_1 = range(hour_ind_2d, (hour_ind_2d + 7))  # 前两天的0点到7点
            select_2d_2 = range((hour_ind_2d + 20), (hour_ind_2d + 24))  # 前两天的20点到24点

            value_cur_1 = act_raw[select_cur_1, :]
            value_cur_2 = act_raw[select_cur_2, :]
            value_1d_1 = act_raw[select_1d_1, :]
            value_1d_2 = act_raw[select_1d_2, :]
            value_2d_1 = act_raw[select_2d_1, :]
            value_2d_2 = act_raw[select_2d_2, :]

            feature_act_01 = np.nansum(value_cur_1) - np.nansum(value_1d_1)
            feature_act_02 = np.nansum(value_cur_1) - np.nansum(value_2d_1)
            feature_act_03 = np.nansum(value_1d_2) - np.nansum(value_2d_2)

            feature_act_04 = np.sum(value_cur_1 > 4) - np.sum(value_1d_1 > 4)
            feature_act_05 = np.sum(value_cur_1 > 4) - np.sum(value_2d_1 > 4)
            feature_act_06 = np.sum(value_1d_2 > 4) - np.sum(value_2d_2 > 4)

            feature_act_07 = np.sum(value_cur_1 > 15) - np.sum(value_1d_1 > 15)
            feature_act_08 = np.sum(value_cur_1 > 15) - np.sum(value_2d_1 > 15)
            feature_act_09 = np.sum(value_1d_2 > 15) - np.sum(value_2d_2 > 15)

            feature_act_10 = np.nansum(value_cur_1) / (np.nansum(value_1d_1) + 0.1)
            feature_act_11 = np.nansum(value_cur_1) / (np.nansum(value_2d_1) + 0.1)
            feature_act_12 = np.nansum(value_1d_2) / (np.nansum(value_2d_2) + 0.1)

            feature_act_13 = np.sum(value_cur_1 > 4) / (np.sum(value_1d_1 > 4) + 0.1)
            feature_act_14 = np.sum(value_cur_1 > 4) / (np.sum(value_2d_1 > 4) + 0.1)
            feature_act_15 = np.sum(value_1d_2 > 4) / (np.sum(value_2d_2 > 4) + 0.1)

            feature_act_16 = np.sum(value_cur_1 > 15) / (np.sum(value_1d_1 > 15) + 0.1)
            feature_act_17 = np.sum(value_cur_1 > 15) / (np.sum(value_2d_1 > 15) + 0.1)
            feature_act_18 = np.sum(value_1d_2 > 15) / (np.sum(value_2d_2 > 15) + 0.1)

            # 呼吸特征
            value_cur_1 = br_raw[select_cur_1, :]
            value_cur_2 = br_raw[select_cur_2, :]
            value_1d_1 = br_raw[select_1d_1, :]
            value_1d_2 = br_raw[select_1d_2, :]
            value_2d_1 = br_raw[select_2d_1, :]
            value_2d_2 = br_raw[select_2d_2, :]

            feature_br_01 = np.nanmedian(value_cur_1)
            feature_br_02 = np.nanmedian(value_1d_2)
            feature_br_03 = np.nanmedian(value_1d_1)
            feature_br_04 = np.nanmedian(value_2d_2)
            feature_br_05 = np.nanmedian(value_2d_1)
            
            feature_br_06 = feature_br_01 - feature_br_03
            feature_br_07 = feature_br_01 - feature_br_05
            feature_br_08 = feature_br_03 - feature_br_05
            feature_br_09 = feature_br_02 - feature_br_04
            
            feature_br_10 = feature_br_01 / (feature_br_03 + 0.01)
            feature_br_11 = feature_br_01 / (feature_br_05 + 0.01)
            feature_br_12 = feature_br_03 / (feature_br_05 + 0.01)
            feature_br_13 = feature_br_02 / (feature_br_04 + 0.01)
            
            feature_br_14 = (feature_br_01 + feature_br_02) / (feature_br_03 + feature_br_04 + 0.01)
            feature_br_15 = np.nanpercentile(value_cur_1, 75) / (np.nanpercentile(value_cur_1, 25) + 0.01)
            feature_br_16 = np.nanpercentile(value_1d_1, 75) / (np.nanpercentile(value_1d_1, 25) + 0.01)
            feature_br_17 = np.nanpercentile(value_2d_1, 75) / (np.nanpercentile(value_2d_1, 25) + 0.01)
            feature_br_18 = np.nanpercentile(value_1d_2, 75) / (np.nanpercentile(value_1d_2, 25) + 0.01)
            
            feature_br_19 = np.sum(value_cur_1 > 40)
            feature_br_20 = np.sum(value_1d_1 > 40)

            # 起卧特征
            value_cur_1 = up_raw[select_cur_1, :]
            value_cur_2 = up_raw[select_cur_2, :]
            value_1d_1 = up_raw[select_1d_1, :]
            value_1d_2 = up_raw[select_1d_2, :]
            value_2d_1 = up_raw[select_2d_1, :]
            value_2d_2 = up_raw[select_2d_2, :]

            feature_up_01 = np.nansum(value_cur_1)
            feature_up_02 = np.nansum(value_cur_1) - np.nansum(value_1d_1)
            feature_up_03 = np.nansum(value_cur_1) - np.nansum(value_2d_1)
            feature_up_04 = np.nansum(value_1d_2) - np.nansum(value_2d_2)

            up_raw_diff = np.diff(up_raw)
            value_cur_1 = up_raw_diff[select_cur_1, :]
            value_cur_2 = up_raw_diff[select_cur_2, :]
            value_1d_1 = up_raw_diff[select_1d_1, :]
            value_1d_2 = up_raw_diff[select_1d_2, :]
            value_2d_1 = up_raw_diff[select_2d_1, :]
            value_2d_2 = np.zeros_like(value_cur_1)  # 避免索引问题

            feature_up_05 = np.sum(value_cur_1 != 0)
            feature_up_06 = np.sum(value_cur_1 != 0) / (np.sum(value_1d_1 != 0) + 0.1)
            feature_up_07 = np.sum(value_cur_1 != 0) / (np.sum(value_2d_1 != 0) + 0.1)
            feature_up_08 = np.sum(value_1d_2 != 0) / (np.sum(value_2d_2 != 0) + 0.1)

            feature_up_09 = count_continuous_ones(value_cur_1.flatten().tolist())
            feature_up_10 = count_continuous_ones(value_1d_1.flatten().tolist())
            feature_up_11 = feature_up_09 >= 30 and feature_up_10 == 0
            feature_up_12 = feature_up_09 / (feature_up_10 + 0.1)
            feature_up_13 = feature_up_09 - feature_up_10
            feature_up_14 = ((feature_up_10==0 and feature_up_12>=30) or 
                                (feature_up_10==1 and feature_up_12>=40) or 
                                (feature_up_10==2 and feature_up_12>=20) or 
                                (feature_up_10==3 and feature_up_12>=15) or 
                                (feature_up_10==4 and feature_up_12>=10) or 
                                (feature_up_10==5 and feature_up_12>=8) or 
                                (feature_up_10==6 and feature_up_12>=6) or 
                                (10>=feature_up_10>=7 and feature_up_12>=4) or 
                                (feature_up_10>10 and feature_up_12>=2.5) or 
                                (feature_up_10>25 and feature_up_12>=2) or 
                                (feature_up_10>35 and feature_up_12>=1.5) or 
                                (feature_up_10>50 and feature_up_13>=25))
            feature_up_15 = count_continuous_ones(np.zeros_like(value_1d_1).flatten().tolist())
            feature_up_16 = feature_up_09>=feature_up_10>=feature_up_15 and feature_up_09!=0

            # 睡眠特征
            # 统计睡眠分布,0~10min,10~30min 30~60min >60min 清醒 浅睡眠 中度睡眠 深度睡眠
            def get_sleep(sleep):
                result = [0,0,0,0]
                i=0
                while i<len(sleep):
                    time = 0
                    while(i<len(sleep) and sleep[i]!=0):
                        i+=1
                    while(i<len(sleep) and sleep[i]!=1):
                        i+=1
                        time+=1
                    if(time>=120):
                        result[3] += time
                    elif(time>=60):
                        result[2] += time
                    elif(time>=20):
                        result[1] += time
                    elif(time>0):
                        result[0] += time
                    else:
                        pass
                result.append(len(sleep)-sum(result)) # 清醒时间
                result = [s/len(sleep) for s in result]
                return result
            
            today_sleep = up_raw[hour_ind_1d+20 : cur_hour_ind+7].flatten().tolist()
            yesterday_sleep = up_raw[hour_ind_2d+20 : hour_ind_1d+7].flatten().tolist()
            today_sleep_con = get_sleep(today_sleep)
            yesterday_sleep_con = get_sleep(yesterday_sleep)

            feature_sleep_01 = today_sleep_con[0]-yesterday_sleep_con[0]
            feature_sleep_02 = today_sleep_con[1]-yesterday_sleep_con[1]
            feature_sleep_03 = today_sleep_con[2]-yesterday_sleep_con[2]
            feature_sleep_04 = today_sleep_con[3]-yesterday_sleep_con[3]
            # feature_sleep_05 = np.corrcoef(today_sleep_con, yesterday_sleep_con)[0, 1] # 计算两天睡眠分布的相关性
            feature_sleep_05 = kl_divergence(today_sleep_con, yesterday_sleep_con) # 计算两天睡眠分布的KL散度
            feature_sleep_06 = today_sleep_con[0]
            feature_sleep_07 = today_sleep_con[1]
            feature_sleep_08 = today_sleep_con[2]
            feature_sleep_09 = today_sleep_con[3]
            
            features = np.array([
                feature_act_01, feature_act_02, feature_act_03,
                        feature_act_04, feature_act_05, feature_act_06,
                        feature_act_07, feature_act_08, feature_act_09,
                        feature_act_10, feature_act_11, feature_act_12,
                        feature_act_13, feature_act_14, feature_act_15,
                        feature_act_16, feature_act_17, feature_act_18,
                        feature_br_01, feature_br_02, feature_br_03,
                        feature_br_04, feature_br_05, feature_br_06,
                        feature_br_07, feature_br_08, feature_br_09,
                        feature_br_10, feature_br_11, feature_br_12,
                        feature_br_13, feature_br_14, feature_br_15,
                        feature_br_16, feature_br_17, feature_br_18,
                        feature_br_19, feature_br_20,
                        feature_up_01, feature_up_02, feature_up_03,
                        feature_up_04, feature_up_05, feature_up_06,
                        feature_up_07, feature_up_08, 
                        feature_up_09,
                        feature_up_10, feature_up_11, feature_up_12,
                        feature_up_13, feature_up_14, feature_up_15,
                        feature_up_16,
                        feature_sleep_01,feature_sleep_02, feature_sleep_03, feature_sleep_04,feature_sleep_05,feature_sleep_06, feature_sleep_07, feature_sleep_08, feature_sleep_09,
                        ])

            if np.isnan(features).sum()!=0:
                # 特征中存在nan值，当呼吸率在时间段内都为nan的情况下发生
                pig_pred_mask.append(-1)
                continue
            all_features.append(features)
            pig_pred_mask.append(1)

    f_df = pd.DataFrame(all_features, columns=[
        'act_01', 'act_02', 'act_03',
        'act_04', 'act_05', 'act_06',
        'act_07', 'act_08', 'act_09',
        'act_10', 'act_11', 'act_12',
        'act_13', 'act_14', 'act_15',
        'act_16', 'act_17', 'act_18',
        'br_01', 'br_02', 'br_03',
        'br_04', 'br_05', 'br_06',
        'br_07', 'br_08', 'br_09',
        'br_10', 'br_11', 'br_12',
        'br_13', 'br_14', 'br_15',
        'br_16', 'br_17', 'br_18',
        'br_19', 'br_20',
        'up_01', 'up_02', 'up_03',
        'up_04', 'up_05', 'up_06',
        'up_07', 'up_08',
        'up_09',
        'up_10', 'up_11', 'up_12',
        'up_13', 'up_14', 'up_15',
        'up_16',
        'sleep_01','sleep_02', 'sleep_03', 'sleep_04','sleep_05','sleep_06', 'sleep_07', 'sleep_08', 'sleep_09'
    ])
    return f_df, pig_pred_mask, pig_dates

def compute_windows(raw_data, window, step, type):
    # 计算滑窗特征
    window *= 2
    step *= 2
    start = 0
    if type=='act':
        result = []
        while start + window <= len(raw_data):
            act_window = raw_data[start:start+window]
            result.append(np.nanmean(act_window))
            start += step
    if type=='up':
        result = []
        while start + window <= len(raw_data):
            up_window = raw_data[start:start+window]
            result.append(np.nansum(up_window))
            start += step
    return result

def compute_meanfeatures(data):
    # 计算最前面两天可用的7~8， 如果整个周期内7~8特征都不存在则返回空，当前样本应舍弃不好预测
    pig_meanfeature = {}
    for item in data:
        pig_id = item[0]
        raw_data = item[1]
        hour_step = 2 * 60
        date = math.ceil(len(raw_data['normalize_act'].values)/2880)
        act_raw = raw_data['normalize_act'].values.reshape(-1, hour_step)
        up_raw = raw_data['is_up'].values.reshape(-1, hour_step)
        
        mean_feature_up01 = [] # 起立时间
        mean_feature_up02 = [] # 起立次数
        mean_feature_up03 = [] # 滑窗起立结果
        mean_feature_act01 = [] # 平均活动量
        mean_feature_act02 = [] # 平均高活动次数
        mean_feature_act03 = [] # 剧烈活动次数
        mean_feature_act04 = [] # 滑窗活动量结果

        # 跟踪前两天的特征
        use_day = 0
        for i in range(date):
            cur_hour_ind = i * 24
            select_today = range(cur_hour_ind+7, cur_hour_ind+8)
            up_value_today = up_raw[select_today, :]
            act_value_today = act_raw[select_today, :]
            if np.isnan(up_value_today).sum()==120:
                continue
            use_day += 1
            mean_feature_up01.append(np.nansum(up_value_today))
            mean_feature_up02.append(count_continuous_ones(up_value_today.flatten().tolist()))
            mean_feature_up03.append(compute_windows(up_value_today.flatten(), window=10, step=10, type='up'))
            mean_feature_act01.append(np.nansum(act_value_today))
            mean_feature_act02.append(np.sum(act_value_today > 4))
            mean_feature_act03.append(np.sum(act_value_today > 15))
            mean_feature_act04.append(compute_windows(act_value_today.flatten(), window=10, step=10, type='act'))
            if use_day==2:
                break
        if use_day < 2:
            pig_meanfeature[pig_id] = {}
        if use_day==2:
            mean_feature_up01 = np.mean(mean_feature_up01)
            mean_feature_up02 = np.mean(mean_feature_up02)
            mean_feature_up03 = np.mean(mean_feature_up03, axis=0)
            mean_feature_act01 = np.mean(mean_feature_act01)
            mean_feature_act02 = np.mean(mean_feature_act02)
            mean_feature_act03 = np.mean(mean_feature_act03)
            mean_feature_act04 = np.mean(mean_feature_act04, axis=0)

        pig_meanfeature[pig_id] =  {'mean_feature_up01': mean_feature_up01,
                'mean_feature_up02': mean_feature_up02,
                'mean_feature_up03': mean_feature_up03,
                'mean_feature_act01': mean_feature_act01,
                'mean_feature_act02': mean_feature_act02,
                'mean_feature_act03': mean_feature_act03,
                'mean_feature_act04': mean_feature_act04
                }
    return pig_meanfeature

def generate_test_features_v2(dataset, pig_meanfeature):
    '''
    dataset:列表或元组的dataframe->((pig_id1, (br, normalize_act, is_up)), (pigid2, (br, normalize_act, is_up)),...)
    对于每头猪需要'br','is_up','normalize_act'三个特征，可将多天0~24点数据拼在一起，模型会以三天为长度输出预测结果
    这里涉及到7~8特征的使用，模型会计算在实验前期2天的7~8特征作为这头猪平常状态下特征，因此输入特征包含从实验开始到待预测的当天
    '''
    all_features = []
    pig_pred_mask = [] # 对模型预测了的天打上标记
    pig_dates = {} # 方便后续收集猪的按天结果
    for item in dataset:
        pig_id = item[0]
        raw_data = item[1]

        # 自动计算天数 (假设数据是连续的，每天 24 小时)
        hour_step = 2 * 60  # 2分钟一个点
        total_points = len(raw_data['normalize_act'].values)
        dates_cnt = math.ceil(total_points / (24 * hour_step))

        pig_dates[pig_id] = dates_cnt

        act_raw = raw_data['normalize_act'].values.reshape(-1, hour_step)
        br_raw = raw_data['br'].values.reshape(-1, hour_step)
        up_raw = raw_data['is_up'].values.reshape(-1, hour_step)
        mean_feature = pig_meanfeature[pig_id]
        if len(mean_feature)==0:
            # 无可用7~8特征，模型拒绝预测
            for _ in range(dates_cnt):
                pig_pred_mask.append(-2)
            continue
        mean_feature_up01 = mean_feature['mean_feature_up01']
        mean_feature_up02 = mean_feature['mean_feature_up02']
        mean_feature_up03 = mean_feature['mean_feature_up03']
        mean_feature_act01 = mean_feature['mean_feature_act01']
        mean_feature_act02 = mean_feature['mean_feature_act02']
        mean_feature_act03 = mean_feature['mean_feature_act03']
        mean_feature_act04 = mean_feature['mean_feature_act04']
        # 从第三天开始计算特征
        for i in range(dates_cnt):
            if(i<2):
                pig_pred_mask.append(-1)
                continue
            cur_hour_ind = i * 24
            hour_ind_1d = (i - 1) * 24
            hour_ind_2d = (i - 2) * 24
            #7~8特征
            select_today = range(cur_hour_ind+7, cur_hour_ind+8) # 当前天的7点到8点
            select_yesterday = range(hour_ind_1d+7, hour_ind_1d+8) # 前一天的7点到8点

            # 起卧特征
            up_value_today = up_raw[select_today, :]
            up_value_yesterday = up_raw[select_yesterday, :]
            feature78_up_01 = np.nansum(up_value_today)
            feature78_up_02 = np.nansum(up_value_today) - np.nansum(up_value_yesterday)
            feature78_up_03 = np.nansum(up_value_today) - mean_feature_up01
            feature78_up_04 = count_continuous_ones(up_value_today.flatten().tolist())
            feature78_up_05 = feature78_up_04 - count_continuous_ones(up_value_yesterday.flatten().tolist())
            feature78_up_06 = feature78_up_04 - mean_feature_up02
            feature78_up_07 = compute_windows(up_value_today.flatten(), window=10, step=10, type='up')
            feature78_up_08 = dtw.distance(feature78_up_07, compute_windows(up_value_yesterday.flatten(), window=10, step=10, type='up'))
            feature78_up_09 = dtw.distance(feature78_up_07, mean_feature_up03)

            # 活动特征
            act_value_today = act_raw[select_today, :]
            act_value_yesterday = act_raw[select_yesterday, :]
            feature78_act_01 = np.nansum(act_value_today)
            feature78_act_02 = np.nansum(act_value_today) - np.nansum(act_value_yesterday)
            feature78_act_03 = np.nansum(act_value_today) - mean_feature_act01
            feature78_act_04 = np.sum(act_value_today > 4)
            feature78_act_05 = np.sum(act_value_today > 4) - np.sum(act_value_yesterday > 4)
            feature78_act_06 = np.sum(act_value_today > 4) - mean_feature_act02
            feature78_act_07 = np.sum(act_value_today > 15)
            feature78_act_08 = np.sum(act_value_today > 15) - np.sum(act_value_yesterday > 15)
            feature78_act_09 = np.sum(act_value_today > 15) - mean_feature_act03
            feature78_act_10 = compute_windows(act_value_today.flatten(), window=10, step=10, type='act')
            feature78_act_11 = dtw.distance(feature78_act_10, compute_windows(act_value_yesterday.flatten(), window=10, step=10, type='act'))
            feature78_act_12 = dtw.distance(feature78_act_10, mean_feature_act04)



            select_cur_1 = range(cur_hour_ind, (cur_hour_ind + 7)) # 当前天的0点到7点
            select_cur_2 = range((cur_hour_ind + 20), (cur_hour_ind + 24)) # 当前天的20点到24点
            select_1d_1 = range(hour_ind_1d, (hour_ind_1d + 7))  # 前一天的0点到7点
            select_1d_2 = range((hour_ind_1d + 20), (hour_ind_1d + 24)) # 前一天的20点到24点
            select_2d_1 = range(hour_ind_2d, (hour_ind_2d + 7))  # 前两天的0点到7点
            select_2d_2 = range((hour_ind_2d + 20), (hour_ind_2d + 24))  # 前两天的20点到24点

            value_cur_1 = act_raw[select_cur_1, :]
            value_cur_2 = act_raw[select_cur_2, :]
            value_1d_1 = act_raw[select_1d_1, :]
            value_1d_2 = act_raw[select_1d_2, :]
            value_2d_1 = act_raw[select_2d_1, :]
            value_2d_2 = act_raw[select_2d_2, :]

            feature_act_01 = np.nansum(value_cur_1) - np.nansum(value_1d_1)
            feature_act_02 = np.nansum(value_cur_1) - np.nansum(value_2d_1)
            feature_act_03 = np.nansum(value_1d_2) - np.nansum(value_2d_2)

            feature_act_04 = np.sum(value_cur_1 > 4) - np.sum(value_1d_1 > 4)
            feature_act_05 = np.sum(value_cur_1 > 4) - np.sum(value_2d_1 > 4)
            feature_act_06 = np.sum(value_1d_2 > 4) - np.sum(value_2d_2 > 4)

            feature_act_07 = np.sum(value_cur_1 > 15) - np.sum(value_1d_1 > 15)
            feature_act_08 = np.sum(value_cur_1 > 15) - np.sum(value_2d_1 > 15)
            feature_act_09 = np.sum(value_1d_2 > 15) - np.sum(value_2d_2 > 15)

            feature_act_10 = np.nansum(value_cur_1) / (np.nansum(value_1d_1) + 0.1)
            feature_act_11 = np.nansum(value_cur_1) / (np.nansum(value_2d_1) + 0.1)
            feature_act_12 = np.nansum(value_1d_2) / (np.nansum(value_2d_2) + 0.1)

            feature_act_13 = np.sum(value_cur_1 > 4) / (np.sum(value_1d_1 > 4) + 0.1)
            feature_act_14 = np.sum(value_cur_1 > 4) / (np.sum(value_2d_1 > 4) + 0.1)
            feature_act_15 = np.sum(value_1d_2 > 4) / (np.sum(value_2d_2 > 4) + 0.1)

            feature_act_16 = np.sum(value_cur_1 > 15) / (np.sum(value_1d_1 > 15) + 0.1)
            feature_act_17 = np.sum(value_cur_1 > 15) / (np.sum(value_2d_1 > 15) + 0.1)
            feature_act_18 = np.sum(value_1d_2 > 15) / (np.sum(value_2d_2 > 15) + 0.1)

            # 呼吸特征
            value_cur_1 = br_raw[select_cur_1, :]
            value_cur_2 = br_raw[select_cur_2, :]
            value_1d_1 = br_raw[select_1d_1, :]
            value_1d_2 = br_raw[select_1d_2, :]
            value_2d_1 = br_raw[select_2d_1, :]
            value_2d_2 = br_raw[select_2d_2, :]

            feature_br_01 = np.nanmedian(value_cur_1)
            feature_br_02 = np.nanmedian(value_1d_2)
            feature_br_03 = np.nanmedian(value_1d_1)
            feature_br_04 = np.nanmedian(value_2d_2)
            feature_br_05 = np.nanmedian(value_2d_1)
            
            feature_br_06 = feature_br_01 - feature_br_03
            feature_br_07 = feature_br_01 - feature_br_05
            feature_br_08 = feature_br_03 - feature_br_05
            feature_br_09 = feature_br_02 - feature_br_04
            
            feature_br_10 = feature_br_01 / (feature_br_03 + 0.01)
            feature_br_11 = feature_br_01 / (feature_br_05 + 0.01)
            feature_br_12 = feature_br_03 / (feature_br_05 + 0.01)
            feature_br_13 = feature_br_02 / (feature_br_04 + 0.01)
            
            feature_br_14 = (feature_br_01 + feature_br_02) / (feature_br_03 + feature_br_04 + 0.01)
            feature_br_15 = np.nanpercentile(value_cur_1, 75) / (np.nanpercentile(value_cur_1, 25) + 0.01)
            feature_br_16 = np.nanpercentile(value_1d_1, 75) / (np.nanpercentile(value_1d_1, 25) + 0.01)
            feature_br_17 = np.nanpercentile(value_2d_1, 75) / (np.nanpercentile(value_2d_1, 25) + 0.01)
            feature_br_18 = np.nanpercentile(value_1d_2, 75) / (np.nanpercentile(value_1d_2, 25) + 0.01)
            
            feature_br_19 = np.sum(value_cur_1 > 40)
            feature_br_20 = np.sum(value_1d_1 > 40)

            # 起卧特征
            value_cur_1 = up_raw[select_cur_1, :]
            value_cur_2 = up_raw[select_cur_2, :]
            value_1d_1 = up_raw[select_1d_1, :]
            value_1d_2 = up_raw[select_1d_2, :]
            value_2d_1 = up_raw[select_2d_1, :]
            value_2d_2 = up_raw[select_2d_2, :]

            feature_up_01 = np.nansum(value_cur_1)
            feature_up_02 = np.nansum(value_cur_1) - np.nansum(value_1d_1)
            feature_up_03 = np.nansum(value_cur_1) - np.nansum(value_2d_1)
            feature_up_04 = np.nansum(value_1d_2) - np.nansum(value_2d_2)

            up_raw_diff = np.diff(up_raw)
            value_cur_1 = up_raw_diff[select_cur_1, :]
            value_cur_2 = up_raw_diff[select_cur_2, :]
            value_1d_1 = up_raw_diff[select_1d_1, :]
            value_1d_2 = up_raw_diff[select_1d_2, :]
            value_2d_1 = up_raw_diff[select_2d_1, :]
            value_2d_2 = up_raw_diff[select_2d_2, :]

            feature_up_05 = np.sum(value_cur_1 != 0)
            feature_up_06 = np.sum(value_cur_1 != 0) / (np.sum(value_1d_1 != 0) + 0.1)
            feature_up_07 = np.sum(value_cur_1 != 0) / (np.sum(value_2d_1 != 0) + 0.1)
            feature_up_08 = np.sum(value_1d_2 != 0) / (np.sum(value_2d_2 != 0) + 0.1)

            feature_up_09 = count_continuous_ones(value_cur_1.flatten().tolist())
            feature_up_10 = count_continuous_ones(value_1d_1.flatten().tolist())
            feature_up_11 = feature_up_09 >= 30 and feature_up_10 == 0
            feature_up_12 = feature_up_09 / (feature_up_10 + 0.1)
            feature_up_13 = feature_up_09 - feature_up_10
            feature_up_14 = ((feature_up_10==0 and feature_up_12>=30) or 
                                (feature_up_10==1 and feature_up_12>=40) or 
                                (feature_up_10==2 and feature_up_12>=20) or 
                                (feature_up_10==3 and feature_up_12>=15) or 
                                (feature_up_10==4 and feature_up_12>=10) or 
                                (feature_up_10==5 and feature_up_12>=8) or 
                                (feature_up_10==6 and feature_up_12>=6) or 
                                (10>=feature_up_10>=7 and feature_up_12>=4) or 
                                (feature_up_10>10 and feature_up_12>=2.5) or 
                                (feature_up_10>25 and feature_up_12>=2) or 
                                (feature_up_10>35 and feature_up_12>=1.5) or 
                                (feature_up_10>50 and feature_up_13>=25))
            feature_up_15 = count_continuous_ones(value_2d_1.flatten().tolist())
            feature_up_16 = feature_up_09>=feature_up_10>=feature_up_15 and feature_up_09!=0

            # 睡眠特征
            # 统计睡眠分布,0~10min,10~30min 30~60min >60min 清醒 浅睡眠 中度睡眠 深度睡眠
            def get_sleep(sleep):
                result = [0,0,0,0]
                i=0
                while i<len(sleep):
                    time = 0
                    while(i<len(sleep) and sleep[i]!=0):
                        i+=1
                    while(i<len(sleep) and sleep[i]!=1):
                        i+=1
                        time+=1
                    if(time>=120):
                        result[3] += time
                    elif(time>=60):
                        result[2] += time
                    elif(time>=20):
                        result[1] += time
                    elif(time>0):
                        result[0] += time
                    else:
                        pass
                result.append(len(sleep)-sum(result)) # 清醒时间
                result = [s/len(sleep) for s in result]
                return result
            
            today_sleep = up_raw[hour_ind_1d+20 : cur_hour_ind+7].flatten().tolist()
            yesterday_sleep = up_raw[hour_ind_2d+20 : hour_ind_1d+7].flatten().tolist()
            today_sleep_con = get_sleep(today_sleep)
            yesterday_sleep_con = get_sleep(yesterday_sleep)


            feature_sleep_01 = today_sleep_con[0]-yesterday_sleep_con[0]
            feature_sleep_02 = today_sleep_con[1]-yesterday_sleep_con[1]
            feature_sleep_03 = today_sleep_con[2]-yesterday_sleep_con[2]
            feature_sleep_04 = today_sleep_con[3]-yesterday_sleep_con[3]
            feature_sleep_05 = kl_divergence(today_sleep_con, yesterday_sleep_con) # 计算两天睡眠分布的KL散度
            feature_sleep_06 = today_sleep_con[0]
            feature_sleep_07 = today_sleep_con[1]
            feature_sleep_08 = today_sleep_con[2]
            feature_sleep_09 = today_sleep_con[3]

            
            features = np.array([
                feature_act_01, feature_act_02, feature_act_03,
                        feature_act_04, feature_act_05, feature_act_06,
                        feature_act_07, feature_act_08, feature_act_09,
                        feature_act_10, feature_act_11, feature_act_12,
                        feature_act_13, feature_act_14, feature_act_15,
                        feature_act_16, feature_act_17, feature_act_18,
                        feature_br_01, feature_br_02, feature_br_03,
                        feature_br_04, feature_br_05, feature_br_06,
                        feature_br_07, feature_br_08, feature_br_09,
                        feature_br_10, feature_br_11, feature_br_12,
                        feature_br_13, feature_br_14, feature_br_15,
                        feature_br_16, feature_br_17, feature_br_18,
                        feature_br_19, feature_br_20,
                        feature_up_01, feature_up_02, feature_up_03,
                        feature_up_04, feature_up_05, feature_up_06,
                        feature_up_07, feature_up_08, 
                        feature_up_09,
                        feature_up_10, feature_up_11, feature_up_12,
                        feature_up_13, feature_up_14, feature_up_15,
                        feature_up_16,
                        feature_sleep_01,feature_sleep_02, feature_sleep_03, feature_sleep_04,feature_sleep_05,feature_sleep_06, feature_sleep_07, feature_sleep_08, feature_sleep_09,
                        feature78_act_01,feature78_act_02, feature78_act_03,
                        feature78_act_04, feature78_act_05, feature78_act_06,
                        feature78_act_07, feature78_act_08, feature78_act_09,
                        feature78_act_11, feature78_act_12,
                        feature78_up_01, feature78_up_02, feature78_up_03,
                        feature78_up_04, feature78_up_05, feature78_up_06,
                        feature78_up_08, feature78_up_09,
                        ])

            if np.isnan(features).sum()!=0:
                # 特征中存在nan值，由呼吸全是nan值引起，模型拒绝预测
                pig_pred_mask.append(-1)
                continue
            all_features.append(features)
            pig_pred_mask.append(1)

    f_df = pd.DataFrame(all_features, columns=[
        'act_01', 'act_02', 'act_03',
        'act_04', 'act_05', 'act_06',
        'act_07', 'act_08', 'act_09',
        'act_10', 'act_11', 'act_12',
        'act_13', 'act_14', 'act_15',
        'act_16', 'act_17', 'act_18',
        'br_01', 'br_02', 'br_03',
        'br_04', 'br_05', 'br_06',
        'br_07', 'br_08', 'br_09',
        'br_10', 'br_11', 'br_12',
        'br_13', 'br_14', 'br_15',
        'br_16', 'br_17', 'br_18',
        'br_19', 'br_20',
        'up_01', 'up_02', 'up_03',
        'up_04', 'up_05', 'up_06',
        'up_07', 'up_08',
        'up_09',
        'up_10', 'up_11', 'up_12',
        'up_13', 'up_14', 'up_15',
        'up_16',
        'sleep_01','sleep_02', 'sleep_03', 'sleep_04','sleep_05','sleep_06', 'sleep_07', 'sleep_08', 'sleep_09',
        'act78_01','act78_02', 'act78_03',
        'act78_04', 'act78_05', 'act78_06',
        'act78_07', 'act78_08', 'act78_09',
        'act78_11', 'act78_12',
        'up78_01', 'up78_02', 'up78_03',
        'up78_04', 'up78_05', 'up78_06',
        'up78_08', 'up78_09',
    ])


    return f_df, pig_pred_mask, pig_dates



# ========== V2 特征提取（youhua 3天版 90维 rich2_3dx） ==========

# --- V2 常量 ---
_V2_PPD = 2880          # points per day
_V2_H = 120             # points per hour
_V2_EPS = 1e-9
_V2_WARMUP = 2          # 只需 2 天暖机
_V2_CENTROID_MIN_PROB = 0.01
_V2_SERIES_NAMES = ['act_full', 'act_morn', 'act_night', 'br_med', 'hr_med', 'br_high',
                    'up_dur', 'up_trans', 'up_night']
_V2_SERIES_INTRADAY = ['act_conc', 'act_peakhr', 'act_entropy', 'act_hourmax',
                       'up_maxbout', 'up_acthrs', 'hr_max', 'hr_range', 'br_range']

# 导出给 model.py 的 from .breeding_utils import PPD 使用
PPD = _V2_PPD


def _v2_winsum(arr, day, h_start, h_end):
    s = day * _V2_PPD + h_start * _V2_H
    e = day * _V2_PPD + h_end * _V2_H
    seg = arr[s:e]
    return float(np.nansum(seg)) if seg.size else np.nan


def _v2_winmed(arr, day, h_start, h_end):
    s = day * _V2_PPD + h_start * _V2_H
    e = day * _V2_PPD + h_end * _V2_H
    seg = arr[s:e]
    return float(np.nanmedian(seg)) if seg.size and not np.all(np.isnan(seg)) else np.nan


def _v2_longest_run(x):
    max_run = cur = 0
    for v in x:
        if v == 1:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 0
    return max_run


def _v2_daily_series(sig, n_days):
    """
    逐日聚合序列。生产对齐：最后一天（当天）只用 0-8h，历史天用全天 0-24h。
    生产实际可用数据：前天/昨天 0-24h 完整，今天只有 0-8h。
    """
    act, br, hr, up = sig['normalize_act'], sig['new_br'], sig['hr'], sig['is_up']
    last_day = n_days - 1  # 最后一天 = 当天
    TODAY_H = 8            # 当天可用时段结束 (0-8h)

    def _h_end(d):
        """第 d 天的有效小时数终点：最后一天8h，其他天24h。"""
        return TODAY_H if d == last_day else 24

    S = {}
    # 活动量（各天用对应窗口）
    if act is not None:
        S['act_full'] = np.array([_v2_winsum(act, d, 0, _h_end(d)) for d in range(n_days)])
        S['act_morn'] = np.array([_v2_winsum(act, d, 0, min(7, _h_end(d))) for d in range(n_days)])
        S['act_night'] = np.array([_v2_winsum(act, d, 19, 24) if d != last_day else np.nan for d in range(n_days)])
    else:
        S['act_full'] = S['act_morn'] = S['act_night'] = np.full(n_days, np.nan)
    # 呼吸中位数
    S['br_med'] = np.array([_v2_winmed(br, d, 0, _h_end(d)) for d in range(n_days)]) if br is not None else np.full(n_days, np.nan)
    # 心率中位数
    S['hr_med'] = np.array([_v2_winmed(hr, d, 0, _h_end(d)) for d in range(n_days)]) if hr is not None else np.full(n_days, np.nan)
    # 呼吸高占比
    if br is not None:
        bh = []
        for d in range(n_days):
            he = _h_end(d)
            seg = br[d*_V2_PPD:d*_V2_PPD + he*_V2_H]
            bh.append(float(np.nanmean(seg > 40)) if seg.size and not np.all(np.isnan(seg)) else np.nan)
        S['br_high'] = np.array(bh)
    else:
        S['br_high'] = np.full(n_days, np.nan)
    # 起立
    if up is not None:
        dur, trans, nt = [], [], []
        for d in range(n_days):
            he = _h_end(d)
            seg = up[d*_V2_PPD:d*_V2_PPD + he*_V2_H]
            dur.append(float(np.nansum(seg)) if seg.size else np.nan)
            s2 = np.nan_to_num(seg)
            trans.append(float(np.sum((s2[1:]==1)&(s2[:-1]==0))) if seg.size else np.nan)
            # 夜间起立：当天没有(0-8h无19-24h段)
            nt.append(_v2_winsum(up, d, 19, 24) if d != last_day else np.nan)
        S['up_dur'] = np.array(dur)
        S['up_trans'] = np.array(trans)
        S['up_night'] = np.array(nt)
    else:
        S['up_dur'] = S['up_trans'] = S['up_night'] = np.full(n_days, np.nan)
    return S


def _v2_daily_series_intraday(sig, n_days):
    """
    日内形态序列。生产对齐：最后一天只用 0-8h。
    """
    act, br, hr, up = sig['normalize_act'], sig['new_br'], sig['hr'], sig['is_up']
    last_day = n_days - 1
    TODAY_H = 8
    S = {k: np.full(n_days, np.nan) for k in _V2_SERIES_INTRADAY}
    for d in range(n_days):
        s0 = d * _V2_PPD
        h_end = TODAY_H if d == last_day else 24
        n_pts = h_end * _V2_H  # 当天可用点数

        # 活动日内形态
        if act is not None and s0 + n_pts <= len(act):
            seg = np.nan_to_num(act[s0:s0+n_pts])
            hrs = seg[:h_end*_V2_H].reshape(h_end, _V2_H)
            hsum = hrs.sum(axis=1)
            tot = hsum.sum()
            if tot > _V2_EPS:
                p = hsum / tot
                S['act_conc'][d] = float(hsum.max() / tot)
                S['act_peakhr'][d] = float(np.argmax(hsum))
                S['act_entropy'][d] = float(-np.sum(p[p>0]*np.log(p[p>0])))
                S['act_hourmax'][d] = float(hsum.max())
        # 起立日内形态
        if up is not None and s0 + n_pts <= len(up):
            day_up = np.nan_to_num(up[s0:s0+n_pts])
            S['up_maxbout'][d] = float(_v2_longest_run(day_up))
            uh = day_up[:h_end*_V2_H].reshape(h_end, _V2_H)
            S['up_acthrs'][d] = float(np.sum(uh.sum(axis=1)>0))
        # 心率
        if hr is not None and s0 + n_pts <= len(hr):
            seg = hr[s0:s0+n_pts]
            if not np.all(np.isnan(seg)):
                S['hr_max'][d] = float(np.nanpercentile(seg, 95))
                S['hr_range'][d] = float(np.nanpercentile(seg, 95) - np.nanpercentile(seg, 5))
        # 呼吸范围
        if br is not None and s0 + n_pts <= len(br):
            seg = br[s0:s0+n_pts]
            if not np.all(np.isnan(seg)):
                S['br_range'][d] = float(np.nanpercentile(seg, 95) - np.nanpercentile(seg, 5))
    return S


def _v2_build_feature_rows(S, n_days):
    """3天版特征：每个序列产出 _lvl/_d1/_r1/_d2/_vsmax2 共5个，18序列×5=90维。"""
    series_names = list(_V2_SERIES_NAMES) + list(_V2_SERIES_INTRADAY)
    days = list(range(_V2_WARMUP, n_days))
    rows, names = [], None
    for d in days:
        feat, nm = [], []
        for name in series_names:
            x = S[name]
            lvl = x[d]
            d1 = x[d] - x[d-1] if d-1 >= 0 else np.nan
            r1 = x[d] / (x[d-1] + _V2_EPS) if d-1 >= 0 else np.nan
            d2 = x[d] - x[d-2] if d-2 >= 0 else np.nan
            if d-2 >= 0:
                vsmax2 = x[d] - max(x[d-1], x[d-2])
            elif d-1 >= 0:
                vsmax2 = x[d] - x[d-1]
            else:
                vsmax2 = np.nan
            feat += [lvl, d1, r1, d2, vsmax2]
            nm += [f'{name}_lvl', f'{name}_d1', f'{name}_r1', f'{name}_d2', f'{name}_vsmax2']
        rows.append(feat)
        if names is None:
            names = nm
    return np.array(rows, dtype=float), names


def _v2_centroid_select(probs):
    """重心选日：取概率加权重心对应的天。"""
    above = [(i, probs[i]) for i in range(len(probs)) if probs[i] >= _V2_CENTROID_MIN_PROB]
    if above:
        wsum = sum(i*p for i, p in above)
        wtot = sum(p for _, p in above)
        ci = int(round(wsum / wtot))
        ci = max(0, min(ci, len(probs)-1))
        return ci, float(probs[ci])
    ci = int(np.argmax(probs))
    return ci, float(probs[ci])


def generate_breeding_features_v2(dataset):
    '''
    V2 特征提取：90维 rich2_3dx（只需3天数据）。
    接口与 generate_breeding_features_v1 完全一致：
        输入: dataset = [('pig_id', feature_df), ...]
        输出: (extract_feature_df, pig_pred_mask, pig_dates)
    '''
    all_features = []
    pig_pred_mask = []
    pig_dates = {}

    for item in dataset:
        pig_id = item[0]
        raw_data = item[1].copy()

        act = raw_data['normalize_act'].values.astype(float) if 'normalize_act' in raw_data.columns else None
        br = raw_data['br'].values.astype(float) if 'br' in raw_data.columns else None
        up = raw_data['is_up'].values.astype(float) if 'is_up' in raw_data.columns else None
        hr = raw_data['hr'].values.astype(float) if 'hr' in raw_data.columns else None

        if act is None or len(act) == 0:
            pig_dates[pig_id] = 0
            continue

        n_days = len(act) // _V2_PPD
        pig_dates[pig_id] = n_days

        for d in range(min(_V2_WARMUP, n_days)):
            pig_pred_mask.append(-1)

        if n_days < 3:
            continue

        sig = {'normalize_act': act, 'new_br': br, 'hr': hr, 'is_up': up}
        S = _v2_daily_series(sig, n_days)
        S.update(_v2_daily_series_intraday(sig, n_days))
        X, names = _v2_build_feature_rows(S, n_days)

        for row in X:
            if np.isnan(row).all():
                pig_pred_mask.append(-1)
            else:
                all_features.append(row)
                pig_pred_mask.append(1)

    if len(all_features) == 0:
        f_df = pd.DataFrame()
    else:
        f_df = pd.DataFrame(all_features, columns=names)

    return f_df, pig_pred_mask, pig_dates
