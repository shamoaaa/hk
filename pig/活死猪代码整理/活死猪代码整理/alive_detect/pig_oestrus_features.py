# -- coding: utf-8 -*-
# import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import sosfiltfilt
from utils import get_amp, butter_bandpass, calculate_envelope


class PigOestrusFeatureExtractor:

    def __init__(self, act_channel=4, sample_rate=20):
        self.act_channel = act_channel
        self.sample_rate = sample_rate

        # 创建滤波器
        self.generate_filter()

    def init_space(self, min_num):

        self.alive_status_list = []
        self.alive = np.zeros(2*min_num)


    def generate_filter(self):
        # 构建起卧滤波器
        self.up_down_sos = butter_bandpass(0.1, 6, self.sample_rate, order=8)

    
    def process(self, merged_data):
        self.l = merged_data.shape[0]
        self.min_num = int(self.l /20 / 60)

        # 初始化空间
        self.init_space(self.min_num)

        # 以60秒为窗口、30秒为步长，提取特征
        for min_index in range(2 * self.min_num - 1):
            
            # 截取1分钟数据
            start_index = 600 * min_index
            end_index = min(self.l, start_index + 1200)
            frame_data = merged_data[start_index:end_index, :]  # 1200 * 23

            self.get_feature(frame_data, min_index)

        self.alive[-1] = self.alive[-2]

        return self.alive
    
    def get_feature(self, frame_data, min_index):
        # 获取幅值
        m, n = frame_data.shape
        amp = np.zeros([m, n])
        weight_data = None
        for i_frame in range(m):
            amp[i_frame, :], weight_data = get_amp(frame_data[i_frame, :], i_frame, weight_data)

        # 获取最佳通道
        ideal_channel = np.argmax(np.mean(np.abs(amp), axis=0))
        max_channel = np.argmax(np.mean(np.abs(frame_data), axis=0))

        # 获取起卧
        all_phase = np.unwrap(np.angle(frame_data[:, 0:5]))
        if min_index == 0:
            up_down = self.get_up_down(all_phase[0:600, :], amp[0:600, 0:5])

        up_down = self.get_up_down(all_phase[600:1200, :], amp[600:1200, 0:5])

        # 活体检测
        br_sos = butter_bandpass(0.1, 3, 20, order=8)
        filter_phase = sosfiltfilt(br_sos, np.unwrap(np.angle(frame_data[:, max_channel])))
        envelope_low, envelope_high = calculate_envelope(filter_phase, 20)
        br_env = np.abs(envelope_high - envelope_low)
        br_amp = np.mean(br_env)
        
        self.get_alive_status(amp[:, ideal_channel], max_channel, ideal_channel, br_amp, up_down)
        result = self.alive_detect()
        if result != -2:
            self.alive_status_list = []

        self.alive[min_index] = result

    def get_up_down(self, phase, amp):
        """
        计算起卧

        :return: True表示起，False表示卧
        """
        
        # 对相位进行滤波
        # 判断前5个通道的相位是否存在大于2
        is_over = False
        for i in range(phase.shape[1]):
            phase[:, i] = sosfiltfilt(self.up_down_sos,  phase[:, i])

            count = np.sum(phase[:, i] > 2)
            if count > 5:
                is_over = True

        if is_over:
            if max(amp[:, 1])>=1e4 and np.var(phase[:, 1], ddof=0)>=0.1 and max(amp[:, 2])>=1e4 and np.var(phase[:, 2], ddof=0)>=0.1:
                return True
            elif max(amp[:, 3])>=5e3 and np.var(phase[:, 3], ddof=0)>=0.5 and max(amp[:, 4])>=5e3 and np.var(phase[:, 4], ddof=0)>=0.5:
                return True
            else:
                return False
            
        return False


    def get_alive_status(self, amp, max_channel, ideal_channel, br_amp, is_up):
        if np.sum(amp) < 1200:    # 丢数据为什么用sum相消幅值的总和判断，而不是用幅值的长度判断？
            self.alive_status_list.append(-1)    # 丢数据
            return

        # 活猪
        # 优化：并且幅值要大，避免全通道都没有照到物体的情况
        if (max_channel <= 20 and br_amp >= 0.5) or is_up:
            self.alive_status_list.append(1)
            return

        # 无猪or死猪
        elif 13 <= max_channel <= 15 and br_amp < 0.5:
            self.alive_status_list.append(2)    # 死猪
            return
        elif max_channel < 13:  # 这里可以优化一下，如果猪起来了，但是没有检测到
            self.alive_status_list.append(-1)    # 异常
            return
        else:
            self.alive_status_list.append(0)    # 无猪
            return


    def alive_detect(self):
        if len(self.alive_status_list) < 119:
            return -2
        
        no_num = 0
        have_num = 0
        dead_num = 0
        nan_num = 0
        for x in self.alive_status_list:
            if x == -1:
                nan_num += 1
            elif x == 0:
                no_num += 1
            elif x == 1:
                have_num += 1
            elif x == 2:
                dead_num += 1
        
        if have_num >= 100:
            return 1
        elif no_num >= 100:
            return 0
        elif dead_num >= 100:
            return 2
        else:
            return -1
        
        # if len(self.alive_status_list) < 59:
        #     return -2
        
        # no_num = 0
        # have_num = 0
        # dead_num = 0
        # nan_num = 0
        # for x in self.alive_status_list:
        #     if x == -1:
        #         nan_num += 1
        #     elif x == 0:
        #         no_num += 1
        #     elif x == 1:
        #         have_num += 1
        #     elif x == 2:
        #         dead_num += 1
        
        # if have_num >= 50:
        #     return 1
        # elif no_num >= 50:
        #     return 0
        # elif dead_num >= 50:
        #     return 2
        # else:
        #     return -1

        # if len(self.alive_status_list) < 30:
        #     return -2
        # # print(self.alive_status_list)
        
        # no_num = 0
        # have_num = 0
        # dead_num = 0
        # nan_num = 0
        # for x in self.alive_status_list:
        #     if x == -1:
        #         nan_num += 1
        #     elif x == 0:
        #         no_num += 1
        #     elif x == 1:
        #         have_num += 1
        #     elif x == 2:
        #         dead_num += 1
        
        # if have_num >= 25:
        #     return 1
        # elif no_num >= 25:
        #     return 0
        # elif dead_num >= 25:
        #     return 2
        # else:
        #     return -1

    def process_ml(self, merged_data):
        '''
        返回机器学习模型需要的特征，包含1200个数据点的对消幅值amp、1个最大通道、1个最优通道、1个呼吸幅值、1个起卧状态
        与逻辑判断保持一致，以60s为窗口，30s为步长进行提取，同时利用原有的逻辑判断过滤去-1(异常)的特征
        考虑训练使用的标签问题：有猪(包含死猪)和无猪具有置信度为1的标签，而死猪并不确定猪的具体死亡时间，考虑使用逻辑判断的结果作为标签做半监督学习
        label: -1-异常 0-无猪 1-活猪 2-死猪
        '''
        # TODO: amp数量级较大需要处理，最大通道和最优通道为1~23的整数，考虑÷23进行处理，呼吸幅值量级未知逻辑判断时为>0.5 考虑减去0.5， 起卧状态为0或1无需修改

        self.l = merged_data.shape[0]
        self.min_num = int(self.l/20/60)

        feature = []
        label = []
        # 以60秒为窗口、30秒为步长，提取特征
        for min_index in range(2*self.min_num-1):

            # 截取1分钟数据
            start_index = 600*min_index
            end_index = min(self.l, start_index+1200)
            frame_data = merged_data[start_index:end_index, :]
            amp, max_channel, ideal_channel, br_amp, up_down, lb = self.get_feature_ml(frame_data, min_index)
            if lb==-1:
                continue
            f = []
            amp_max = np.max(amp)/10000
            amp_min = np.min(amp)/10000
            amp_mean = np.mean(amp)/10000
            max_channel = max_channel / 23.0
            ideal_channel = ideal_channel / 23.0
            br_amp = br_amp-0.5

            f.append(amp_max)
            f.append(amp_min)
            f.append(amp_mean)
            f.append(max_channel)
            f.append(ideal_channel)
            f.append(br_amp)
            f.append(up_down)
            feature.append(f)
            label.append(lb)
        
        return feature, label
    
    def get_feature_ml(self, frame_data, min_index):
        # 获取幅值
        m, n = frame_data.shape
        amp = np.zeros([m, n])
        weight_data = None
        for i_frame in range(m):
            amp[i_frame, :], weight_data = get_amp(frame_data[i_frame, :], i_frame, weight_data)

        # 获取最佳通道
        ideal_channel = np.argmax(np.mean(np.abs(amp), axis=0))
        max_channel = np.argmax(np.mean(np.abs(frame_data), axis=0))

        # 获取起卧
        all_phase = np.unwrap(np.angle(frame_data[:, 0:5]))
        if min_index == 0:
            up_down = self.get_up_down(all_phase[0:600, :], amp[0:600, 0:5])

        up_down = self.get_up_down(all_phase[600:1200, :], amp[600:1200, 0:5])

        # 活体检测
        br_sos = butter_bandpass(0.1, 3, 20, order=8)
        filter_phase = sosfiltfilt(br_sos, np.unwrap(np.angle(frame_data[:, max_channel])))
        envelope_low, envelope_high = calculate_envelope(filter_phase, 20)
        br_env = np.abs(envelope_high - envelope_low)
        br_amp = np.mean(br_env)
        
        amp = amp[:, ideal_channel]
        label = -2
        if np.sum(amp) < 1200:    # 丢数据为什么用sum相消幅值的总和判断，而不是用幅值的长度判断？
            label = -1    # 丢数据

        # 活猪
        # 优化：并且幅值要大，避免全通道都没有照到物体的情况
        if (max_channel <= 20 and br_amp >= 0.5) or up_down:
            label = 1

        # 无猪or死猪
        elif 13 <= max_channel <= 15 and br_amp < 0.5:
            label = 2    # 死猪

        elif max_channel < 13:  # 这里可以优化一下，如果猪起来了，但是没有检测到
            label = -1   # 异常

        else:
            label = 0    # 无猪

        return amp, max_channel, ideal_channel, br_amp, up_down, label  # 返回特征 和 逻辑判断参考标签      