import warnings
import numpy as np
from scipy.signal import sosfiltfilt, hilbert, find_peaks

from .utils import calculate_envelope, get_amp, butter_bandpass, butter_bandpass2, dual_bandpass_filter, br_data_filter2, row_normalize
from .find_pulse import calc_breath
from .find_pulse_env_ctrl import calc_breath as calc_breath_env_ctrl

warnings.filterwarnings("ignore")


class PigOestrusFeatureExtractor:

    def __init__(self, sample_rate=20):

        self.sample_rate = sample_rate

        # 构建活动、起卧滤波器
        self.act_sos = butter_bandpass(0.1, 6, self.sample_rate, order=8)

        # 构建心率滤波器
        self.hr_sos = butter_bandpass(0.67, 2, self.sample_rate, order=8)

        self.weight_data = None
        self.frame_data = None
        self.amp = None
        self.phase = None

        self.act_channel = None
        self.ideal_channel = None
        self.max_channel = None


    def init_param(self, frame_data, act_channel):
        self.weight_data = None
        self.frame_data = frame_data
        self.act_channel = act_channel

        m, n = frame_data.shape
        self.m = m
        self.n = n

        # 获取对消幅值
        self.amp = np.zeros([m, n])
        for i_frame in range(m):
            self.amp[i_frame, :] = get_amp(self, frame_data[i_frame, :], i_frame)

        # 获取最佳通道
        self.ideal_channel = np.argmax(np.mean(np.abs(self.amp), axis=0))
        self.max_channel = np.argmax(np.mean(np.abs(self.frame_data), axis=0))

        # 获取相位
        self.phase = np.unwrap(np.angle(frame_data[:, self.ideal_channel]))


    def process(self):
        # 提取活动量
        act_mean, act_env = self.get_act(self.amp[:, self.act_channel])
        act_normal = self.get_norma_act()

        # 提取呼吸
        br_freq, _ = self.get_br(self.phase)

        # 提取新呼吸
        new_br_freq, _ = self.get_br_env_ctrl(act_mean, self.ideal_channel, 0)

        # 提取心跳
        hr_freq, _ = self.get_hr(self.phase, br_freq)

        # 获取起卧
        all_phase = np.unwrap(np.angle(self.frame_data[:, 0:5]))
        is_up = self.get_up(all_phase[0:600, :], self.amp[0:600, 0:5])

        # 活体检测
        alive_result_30_sec = self.get_alive_result(self.amp[0:600, :], self.frame_data[0:600, :], is_up)

        act = {
            'act_mean': np.mean(act_mean[0:600]),
            'act_normal': act_normal
        }

        br = {
            'ideal_channel': self.ideal_channel,
            'br_freq': br_freq,
            'new_br_freq': new_br_freq
        }

        hr = {
            'hr_freq': hr_freq
        }

        return alive_result_30_sec, is_up, act, br, hr


    def feature_extraction(self):

        # 提取活动量
        act_mean, act_env = self.get_act(self.amp[:, self.act_channel])

        # 提取呼吸
        br_freq, _ = self.get_br(self.phase)

        # 提取心跳
        hr_freq, _ = self.get_hr(self.phase, br_freq)

        # 获取起卧
        all_phase = np.unwrap(np.angle(self.frame_data[:, 0:5]))
        is_up = self.get_up(all_phase[0:600, :], self.amp[0:600, 0:5])

        # 活体检测
        alive_result_30_sec = self.get_alive_result(self.amp[0:600, :], self.frame_data[0:600, :], is_up)

        act = {
            'act_mean': np.mean(act_mean[0:600])
        }

        br = {
            'ideal_channel': self.ideal_channel,
            'br_freq': br_freq,
        }

        hr = {
            'hr_freq': hr_freq
        }

        return alive_result_30_sec, is_up, act, br, hr


    def env_ctrl(self):

        # 提取活动量
        act_mean, act_env = self.get_act(self.amp[:, self.act_channel])

        # 提取呼吸
        br_freq, _ = self.get_br_env_ctrl(act_mean, self.ideal_channel, 0)

        # 提取心跳
        hr_freq, _ = self.get_hr(self.phase, br_freq)

        # 获取起卧
        all_phase = np.unwrap(np.angle(self.frame_data[:, 0:5]))
        is_up = self.get_up(all_phase[0:600, :], self.amp[0:600, 0:5])

        # 活体检测
        alive_result_30_sec = self.get_alive_result(self.amp[0:600, :], self.frame_data[0:600, :], is_up)

        act = {
            'act_mean': np.mean(act_mean[0:600])
        }

        br = {
            'br_freq': br_freq,
        }

        hr = {
            'hr_freq': hr_freq
        }

        return alive_result_30_sec, is_up, act, br, hr
        # return alive_result_30_sec, br


    def get_act(self, act_amp):
        """
        计算活动水平

        :return: 活动水平的计算结果 (例如: 活动得分或活动标记)
        """
        act_amp = sosfiltfilt(self.act_sos, act_amp.T)

        # 滑动平均
        window_size = 1 * 20
        window = np.ones(window_size) / window_size
        act_mean_amp = np.convolve(act_amp, window, mode='same')

        # 获取包络线
        _, act_env = calculate_envelope(act_mean_amp, 100)

        return np.abs(act_mean_amp), np.abs(act_env)


    def get_norma_act(self):
        normalize_amp = np.zeros([self.m, self.n])
        for i_frame in range(self.m):
            normalize_amp[i_frame:i_frame+1, :] = row_normalize(np.abs(self.frame_data[i_frame:i_frame+1, :]))
        
        return np.sum(np.square(np.diff(normalize_amp[0:600, self.act_channel])))


    def get_br(self, phase):
        # 滤波
        filter_phase = dual_bandpass_filter(phase, 20, 0.1, 1.25, order=61)

        _, _, _, _, _, main_freq = calc_breath(filter_phase, fs=20)

        try:
            return round(main_freq), filter_phase
        except:
            return -1.0, filter_phase


    def get_br_env_ctrl(self, act_mean, ideal_channel, min_index):
        amp2 = butter_bandpass2(np.abs(self.frame_data), 20, 0.15, 2.5)
        ideal_channel2 = np.argmax(np.mean(np.abs(amp2), axis=0))
        kernel = np.array([0.2, 0.6, 0.2])
        # 选择理想通道附近的四个通道
        if ideal_channel2 == 0:
            channel_start = 0
        elif ideal_channel2 > 19:
            channel_start = 19
        else:
            channel_start = ideal_channel2 - 1

        frame_fusion_data = np.dot(self.frame_data[:, channel_start:(channel_start + 3)], kernel)
        frame_phase = np.unwrap(np.angle(frame_fusion_data))

        # 滤波
        filter_phase = br_data_filter2(frame_phase)

        _, _, _, _, _, main_freq = calc_breath_env_ctrl(filter_phase, act_mean, ideal_channel, min_index, fs=20)

        try:
            return round(main_freq), filter_phase
        except:
            return -1.0, filter_phase


    def get_hr(self, phase, br_freq):
        # 滤波
        filter_phase = sosfiltfilt(self.hr_sos, phase)

        if br_freq == -1:
            return -1, filter_phase

        # 差分、归一化
        diff_phase = np.diff(filter_phase)
        analytic_signal = hilbert(diff_phase)
        hr_enhi = np.abs(analytic_signal)
        diff_phase = diff_phase / hr_enhi

        # 剔除异常值ֵ
        diff_phase[np.isinf(diff_phase)] = 1
        diff_phase[np.isnan(diff_phase)] = 1
        diff_phase[np.abs(diff_phase) < 1e-2] = 1
        diff_phase[np.abs(diff_phase) > 1e2] = 1

        # 获取主频
        locs, _ = find_peaks(-diff_phase, height=0.75, distance=10)
        rr_diff = np.diff(locs)
        main_freq = round(20.0 / np.median(rr_diff) * 60, 0)

        return main_freq, filter_phase


    def get_up(self, phase, amp):
        # 对相位进行滤波
        # 判断前5个通道的相位是否存在大于2
        is_over = False
        for i in range(phase.shape[1]):
            phase[:, i] = sosfiltfilt(self.act_sos, phase[:, i])

            count = np.sum(phase[:, i] > 2)
            if count > 5:
                is_over = True

        if is_over:
            if max(amp[:, 1]) >= 1e4 and np.var(phase[:, 1], ddof=0) >= 0.1 and max(amp[:, 2]) >= 1e4 and np.var(
                    phase[:, 2], ddof=0) >= 0.1:
                return 1
            elif max(amp[:, 3]) >= 5e3 and np.var(phase[:, 3], ddof=0) >= 0.5 and max(amp[:, 4]) >= 5e3 and np.var(
                    phase[:, 4], ddof=0) >= 0.5:
                return 1
            else:
                return 0

        return 0


    def get_alive_result(self, amp, frame_data, is_up):
        ideal_channel = np.argmax(np.mean(np.abs(amp), axis=0))
        # max_channel = np.argmax(np.mean(np.abs(frame_data), axis=0))

        br_sos = butter_bandpass(0.1, 3, 20, order=8)
        filter_phase = sosfiltfilt(br_sos, np.unwrap(np.angle(frame_data[:, ideal_channel])))
        envelope_low, envelope_high = calculate_envelope(filter_phase, 20)
        br_env = np.abs(envelope_high - envelope_low)
        br_amp = np.mean(br_env)

        alive_result_30_sec = self.get_alive_status(amp[:, ideal_channel], ideal_channel, ideal_channel, br_amp, is_up)
        return alive_result_30_sec


    def get_alive_status(self, amp, max_channel, ideal_channel, br_amp, is_up):
        if np.sum(amp) < 1200:
            return -1

        # 活猪
        # 优化：并且幅值要大，避免全通道都没有照到物体的情况
        if (max_channel <= 20 and br_amp >= 0.5) or is_up:
            return 1

        # 无猪or死猪
        elif 13 <= max_channel <= 15 and br_amp < 0.5:
            return 2
        elif max_channel < 13:
            return -1
        else:
            return 0

