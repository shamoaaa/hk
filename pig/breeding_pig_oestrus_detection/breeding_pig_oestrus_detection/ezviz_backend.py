import os
import random
# random.seed(42)
import xgboost as xgb

from pathlib import Path

from ms_common.model import Model

# from agi_breeding_pig_base.breeding_pig_base.database_utils import *
# from agi_breeding_pig_base.breeding_pig_base.agreement_utils import *
# from agi_breeding_pig_base.breeding_pig_base.core.utils import *

from .logic import is_oestrus_base_logic, is_oestrus_base_logic_jingchan
from .model.model import generate_input, is_oestrus_base_oestrus_model, is_oestrus_base_breeding_model
from .model.AESModelEncryptor import AESModelEncryptor

from breeding_pig_base.database_utils import *
from breeding_pig_base.agreement_utils import *
from breeding_pig_base.core.utils import *

# from memory_profiler import profile

import warnings
warnings.filterwarnings("ignore")


MS_WEIGHTS = Path(os.getenv("MS_WEIGHTS", r"E:\gitlab\py_backends\agi_breeding_pig_oestrus_detection"))

class EzvizModelBackend(Model):
    def __init__(self):
        super().__init__()

        self.db_config = None
        self.rqs_data = None  # 入参的data（子协议），已转换为字典

        self.device_serial = None  # 设备编号

        self.pig_breed = None
        self.pig_strain = None
        self.pig_strain_code = None
        self.pig_status = None
        self.pig_prop = None
        self.parity = None
        self.day_age = None

        self.investigate_start_time = None
        self.investigate_end_time = None

        self.feature_hour = None
        self.feature_start_date = None
        self.feature_end_date = None
        self.feature_start_time = []
        self.feature_end_time = []
        self.feature = {}

        self.rsp_msg = None
        self.rsp_data = {
            'code': 7000,
            'error_msg': '',
            'device_serial': '',
            'estrus_index': 0,
            'confidence': 0.0
        }

        self.use_logic = 0
        self.my_database = None
        # self.my_database = MyDatabase({}, disable_binlog=False)

        encryptor = AESModelEncryptor(password="ezviz")

        # breeding_model_path = MS_WEIGHTS / 'secret' / 'breeding.model'
        breeding_model_path = MS_WEIGHTS / 'breeding.model'
        logger.debug(breeding_model_path)
        assert breeding_model_path.exists(), f"MS_WEIGHTS: {str(breeding_model_path)} isn't exist!!"
        # self.breeding_model = xgb.Booster()
        # self.breeding_model.load_model(str(breeding_model_path))
        self.breeding_model = encryptor.load_encrypted_model(str(breeding_model_path))

        # oestrus_model_path = MS_WEIGHTS / 'secret' / 'oestrus.model'
        oestrus_model_path = MS_WEIGHTS / 'oestrus.model'
        logger.debug(oestrus_model_path)
        assert oestrus_model_path.exists(), f"MS_WEIGHTS: {str(oestrus_model_path)} isn't exist!!"
        # self.oestrus_model = xgb.Booster()
        # self.oestrus_model.load_model(str(oestrus_model_path))
        self.oestrus_model = encryptor.load_encrypted_model(str(oestrus_model_path))

    def apollo_callback(self, value:str):
        try:
            logger.warn(f'pig_oestrus_detection start get apollo config: {value}')
            apollo_dict = json.loads(value)
            self.use_logic = apollo_dict['use_logic']
            db_config = apollo_dict['db_config']
            self.my_database = MyDatabase(db_config, disable_binlog=False)
        except Exception as e:
            logger.error(e)

    def init(self):
        self.rqs_data = None  # 入参的data（子协议），已转换为字典

        self.device_serial = None  # 设备编号

        self.pig_breed = None
        self.pig_strain = None
        self.pig_strain_code = None
        self.pig_status = None
        self.pig_prop = None
        self.parity = None
        self.day_age = None

        self.investigate_start_time = None
        self.investigate_end_time = None

        self.feature_hour = None
        self.feature_start_date = None
        self.feature_end_date = None
        self.feature_start_time = []
        self.feature_end_time = []
        self.feature = {}

        self.rsp_msg = None
        self.rsp_data = {
            'code': 7000,
            'error_msg': '',
            'device_serial': '',
            'estrus_index': 0,
            'confidence': 0.0
        }

    # @profile
    def forward(self, rqs_info: CommonRequest, rsp_msg: CommonResponse):
        algo_start_time = time()

        self.init()
        self.rsp_msg = rsp_msg

        try:
            # 对入参进行检查
            if not check_param_in(self, rqs_info):
                return self.rsp_msg

            # 对data进行检查
            if not self.check_data(rqs_info):
                return self.rsp_msg
            del self.rqs_data

            # 对data的值进行检测
            if not self.check_data_value():
                return self.rsp_msg

            # 拼接、填充数据
            today = self.feature_start_time[-1]
            flag, act_feature, br_feature, is_up_feature, ideal_channel_feature = self.get_today_feature()
            if not flag:
                return self.rsp_msg
            del self.feature_start_time, self.feature_end_time, self.feature

            ideal_channel_feature = ideal_channel_feature[-1]
            all_num = len(ideal_channel_feature)
            ideal_channel_feature = np.array(ideal_channel_feature)
            no_pig_num = np.sum(ideal_channel_feature>20)
            logger.debug(f'{no_pig_num}/{all_num}={no_pig_num / all_num}')
            if no_pig_num / all_num >= 0.9:
                report_error(self, 7000, f'There is no pig! {no_pig_num}/{all_num}={no_pig_num / all_num}')
                return self.rsp_msg

            self.get_pred(act_feature, br_feature, is_up_feature, ideal_channel_feature, today)
            del act_feature, br_feature, is_up_feature, ideal_channel_feature

            # 设置rsp_msg的值
            self.rsp_data['code'] = 200
            rsp_msg.completion = [
                Completion(
                    modal='text',
                    type='json',
                    data=json.dumps(self.rsp_data, cls=NumpyArrayEncoder)
                )
            ]

            algo_end_time = time()
            logger.debug(f'algo_cost_time: {algo_end_time - algo_start_time}')

            return self.rsp_msg
        except:
            logger.error(traceback.print_exc())
            report_error(self, 7000, 'There is a exception! ')
            return self.rsp_msg

    # 对子请求参数进行检查
    def check_data(self, rqs_info: CommonRequest):
        try:
            self.rqs_data = json.loads(rqs_info.data_info[0].data)
        except Exception as e:
            return report_error(self, 7000, f'string to json error! {str(e)}')

        # 检查设备序列号
        if not check_device_serial(self):
            return False

        # 检查猪信息
        if not check_pig_info(self):
            return False

        # 检查查情开始和结束时间
        date_format = "%Y-%m-%d"
        for time_field in ['investigate_start_time', 'investigate_end_time']:
            if time_field not in self.rqs_data or not isinstance(self.rqs_data[time_field], str):
                return report_error(self, 7000, f'{time_field} is missing or not a valid non-empty string')
            else:
                try:
                    datetime.datetime.strptime(self.rqs_data[time_field], date_format)
                except ValueError:
                    return report_error(self, 7000, f"Invalid date format for '{time_field}', should be YYYY-MM-DD")
        self.investigate_start_time = datetime.datetime.strptime(self.rqs_data['investigate_start_time'], date_format)
        self.investigate_end_time = datetime.datetime.strptime(self.rqs_data['investigate_end_time'], date_format)

        # 检查数据部分
        if 'data' not in self.rqs_data or not isinstance(self.rqs_data['data'], dict):
            return report_error(self, 7000, "Invalid or missing 'data'")
        else:
            data = self.rqs_data['data']

            if 'hour' not in data or not isinstance(data['hour'], float):
                return report_error(self, 7000, "Invalid or missing 'data.hour'")
            self.feature_hour = float(data['hour'])

            date_time_format = "%Y-%m-%d %H:%M:%S"
            for time_field in ['start_date', 'end_date']:
                if time_field not in data or not isinstance(data[time_field], str):
                    return report_error(self, 7000, f"Invalid or missing 'data.{time_field}'")
                else:
                    try:
                        datetime.datetime.strptime(data[time_field], date_time_format)
                    except ValueError:
                        return report_error(self, 7000, f"Invalid date-time format for 'data.{time_field}', should be YYYY-MM-DD HH:MM:SS")
            self.feature_start_date = datetime.datetime.strptime(data['start_date'], date_time_format)
            self.feature_end_date = datetime.datetime.strptime(data['end_date'], date_time_format)

            if 'data' not in data or not isinstance(data['data'], list) or len(data['data'])==0:
                return report_error(self, 7000, "Invalid or missing 'data.data'")
            else:
                for i, entry in enumerate(data['data']):

                    if 'start_time' not in entry or not isinstance(entry['start_time'], str):
                        return report_error(self, 7000, f"Invalid or missing 'data.data[{i}].start_time'")
                    else:
                        try:
                            t = datetime.datetime.strptime(entry['start_time'], "%Y-%m-%d %H:%M:%S")
                            self.feature_start_time.append(t)
                        except ValueError:
                            return report_error(self, 7000, f"Invalid date-time format for 'data.data[{i}].start_time', should be YYYY-MM-DD HH:MM:SS")

                    if 'end_time' not in entry or not isinstance(entry['end_time'], str):
                        return report_error(self, 7000, f"Invalid or missing 'data.data[{i}].end_time'")
                    else:
                        try:
                            t = datetime.datetime.strptime(entry['end_time'], "%Y-%m-%d %H:%M:%S")
                            self.feature_end_time.append(t)
                        except ValueError:
                            return report_error(self, 7000, f"Invalid date-time format for 'data.data[{i}].end_time', should be YYYY-MM-DD HH:MM:SS")

                    if get_diff_sec_between_str(entry['start_time'], entry['end_time']) != 30:
                        return report_error(self, 7000,
                                            f"Invalid time length, start_time: {entry['start_time']}, end_time: {entry['end_time']}")

                    for data_type in ['activity', 'normal_activity']:
                        if data_type not in entry or not isinstance(entry[data_type], list) or len(entry[data_type])==0:
                            return report_error(self, 7000, f"Invalid or missing 'data.data[{i}].{data_type}'")
                        elif len(entry[data_type]) != 1:
                            return report_error(self, 7000, f"Data size error! 'data.data[{i}].{data_type}'")
                        elif not all(isinstance(x, (int, float, np.float16, np.float32, np.float64)) for x in entry[data_type]):
                            return report_error(self, 7000, f"Data value type error! 'data.data[{i}].{data_type}': {entry[data_type]}")
                        else:
                            for act in entry[data_type]:
                                if not (act==-1 or act==-1.0 or act>=0):
                                    return report_error(self, 7000, f"Data value error! 'data.data[{i}].{data_type}': {entry[data_type]}")

                    for data_type in ['heart', 'breath', 'is_up', 'ideal_channel']:
                        if data_type not in entry or not isinstance(entry[data_type], list) or len(entry[data_type])==0:
                            return report_error(self, 7000, f"Invalid or missing 'data.data[{i}].{data_type}'")
                        elif len(entry[data_type]) != 1:
                            return report_error(self, 7000, f"Data size error! 'data.data[{i}].{data_type}'")
                        elif not all(isinstance(x, (int, float, np.float16, np.float32, np.float64)) for x in entry[data_type]):
                            return report_error(self, 7000, f"Data value type error! 'data.data[{i}].{data_type}': {entry[data_type]}")
                        elif not (entry[data_type][0]==-1 or entry[data_type][0]==-1.0 or entry[data_type][0]>=0):
                            return report_error(self, 7000, f"Data value error! 'data.data[{i}].{data_type}': {entry[data_type]}")

                    if int(entry['is_up'][0]) not in [-1, 0, 1]:
                        return report_error(self, 7000,
                                            f"Invalid data value! 'data.data[{i}].is_up': {entry['is_up'][0]}")
                    if int(entry['ideal_channel'][0]) not in [x for x in range(23)]:
                        return report_error(self, 7000,
                                            f"Invalid data value! 'data.data[{i}].ideal_channel': {entry['ideal_channel'][0]}")
                self.feature = data['data']

        return True

    # 对子请求参数的值进行检查
    def check_data_value(self):
        if self.investigate_start_time >= self.investigate_end_time:
            return report_error(self, 7000, f'{self.device_serial} time error! '
                                            f"Investigate start time: {self.investigate_start_time} can't after end time: {self.investigate_end_time}. ")
        if self.feature_start_date >= self.feature_end_date:
            return report_error(self, 7000, f'{self.device_serial} time error! '
                                            f"Feature start time: {self.feature_start_date} can't after end time: {self.feature_end_date}. ")

        # 是否在查情周期内
        if self.feature_end_date < self.investigate_start_time:
            return report_error(self, 7000, f'{self.device_serial} time error! '
                                     f'Investigate time from {self.investigate_start_time} to {self.investigate_end_time}. '
                                     f'Feature date from {self.feature_start_date} to {self.feature_end_date}')
        if self.feature_start_date > self.investigate_end_time:
            return report_error(self, 7000, f'{self.device_serial} time error! '
                                     f'Investigate time from {self.investigate_start_time} to {self.investigate_end_time}. '
                                     f'Feature date from {self.feature_start_date} to {self.feature_end_date}')

        # 对data的start time进行校验，是否在查情周期内
        if self.feature_end_time[-1] < self.investigate_start_time:
            return report_error(self, 7000, f'{self.device_serial} time error! '
                                     f'Investigate time from {self.investigate_start_time} to {self.investigate_start_time}. '
                                     f'Feature time from {self.feature_start_time[0]} to {self.feature_end_time[-1]}')
        if self.feature_start_time[0] > self.investigate_end_time:
            return report_error(self, 7000, f'{self.device_serial} time error! '
                                     f'Investigate time from {self.investigate_start_time} to {self.investigate_start_time}. '
                                     f'Feature time from {self.feature_start_time[0]} to {self.feature_end_time[-1]}')

        # 数据长度是否正确
        if self.feature_hour < 36:
            return report_error(self, 7000, f'{self.device_serial} feature hour error! '
                                     f'At least need 36 hours. Actually is {self.feature_hour} hours')

        time_difference = self.feature_end_date - self.feature_start_date
        hours_difference = time_difference.total_seconds() / 3600
        if 36-hours_difference > 0.1 or abs(hours_difference-self.feature_hour)>0.1:
        # if hours_difference < 36:
            return report_error(self, 7000, f'{self.device_serial} param error! '
                                     f'Feature hour: {self.feature_hour}. '
                                     f'Feature start date: {self.feature_start_date}. '
                                     f'Feature end date: {self.feature_end_date}. '
                                     f'Diff hour: {hours_difference}')

        return True

    # 拼接、填充数据 - 进一步优化版本（使用numpy）
    def get_today_feature(self, nan_value_rep=np.nan):
        # 每天的数据点数量：24小时 * 60分钟 * 2 (每半分钟一个点)
        POINTS_PER_DAY = 24 * 60 * 2

        # 收集所有唯一的日期
        date_set = {item["start_time"][:10] for item in self.feature}
        sorted_dates = sorted(date_set)
        date_to_index = {date: i for i, date in enumerate(sorted_dates)}

        # 预先分配numpy数组以获得更好的性能
        num_days = len(sorted_dates)
        br_feature_array = np.full((num_days, POINTS_PER_DAY), nan_value_rep, dtype=np.float32)
        normalize_act_feature_array = np.full((num_days, POINTS_PER_DAY), nan_value_rep, dtype=np.float32)
        is_up_feature_array = np.full((num_days, POINTS_PER_DAY), nan_value_rep, dtype=np.float32)
        ideal_channel_feature_array = np.full((num_days, POINTS_PER_DAY), nan_value_rep, dtype=np.float32)

        # 单次遍历填充数据
        for item in self.feature:
            start_str = item["start_time"]
            date_key = start_str[:10]
            date_idx = date_to_index[date_key]

            # 解析时间部分计算索引
            time_part = start_str[11:]
            hour, minute, second = map(int, time_part.split(':'))
            total_seconds = hour * 3600 + minute * 60 + second
            index = total_seconds // 30

            # 边界检查
            if 0 <= index < POINTS_PER_DAY:
                # 填充各项特征数据
                if item.get("breath"):
                    br_feature_array[date_idx, index] = item["breath"][0]
                if item.get("normal_activity"):
                    normalize_act_feature_array[date_idx, index] = item["normal_activity"][0]
                if item.get("is_up"):
                    is_up_feature_array[date_idx, index] = item["is_up"][0]
                if item.get("ideal_channel"):
                    ideal_channel_feature_array[date_idx, index] = item["ideal_channel"][0]

        # 转换为列表格式返回（与原函数兼容）
        return (True,
                [normalize_act_feature_array[i].tolist() for i in range(num_days)],
                [br_feature_array[i].tolist() for i in range(num_days)],
                [is_up_feature_array[i].tolist() for i in range(num_days)],
                [ideal_channel_feature_array[i].tolist() for i in range(num_days)])

    def get_pred(self, act_feature, br_feature, is_up_feature, ideal_channel_feature, today):
        # 进行发情检测
        start_date = self.investigate_start_time
        diff = date_diff(start_date, today)
        end_date = self.investigate_end_time
        diff2 = date_diff(today, end_date)

        # 后备猪
        if self.pig_prop == 2:
            breeding_delay = 3

            is_oestrus, is_breeding, base_br, confidence = is_oestrus_base_logic(act_feature, br_feature,
                                                                                 is_up_feature,
                                                                                 ideal_channel_feature)

            if diff <= 0 or diff2 < 0:
                self.rsp_data['estrus_index'] = 0
            elif 2 <= diff <= breeding_delay:
                if confidence != 0 or base_br:
                    self.rsp_data['estrus_index'] = 1
                    # self.rsp_data['confidence'] = get_confidence(max(confidence, 0.7))
                self.rsp_data['confidence'] = self.get_ROC(today, 0, get_confidence(0))
            else:
                if confidence != 0 or base_br:
                    self.rsp_data['estrus_index'] = 3
                    # self.rsp_data['confidence'] = get_confidence(max(confidence, 0.7))
                    self.rsp_data['confidence'] = self.get_ROC(today, 1, get_confidence(0.7))
                else:
                    self.rsp_data['estrus_index'] = 1
                    # self.rsp_data['confidence'] = get_confidence(max(confidence, 0.7))
                    self.rsp_data['confidence'] = self.get_ROC(today, 0, get_confidence(0))

        # 经产猪
        else:
            breeding_delay = 2

            if self.use_logic:
                if diff <= 0 or diff2 < 0:
                    self.rsp_data['estrus_index'] = 0
                else:
                    # 模型结果
                    feature_df = generate_input(act_feature, br_feature, is_up_feature, ideal_channel_feature)
                    oestrus_confidence, oestrus_model_result = is_oestrus_base_oestrus_model(self.oestrus_model, feature_df)
                    breeding_confidence, breeding_model_result = is_oestrus_base_breeding_model(self.breeding_model, feature_df)
                    self.write_model_result_into_database(today, oestrus_confidence, oestrus_model_result,
                                                          breeding_confidence, breeding_model_result)

                    if diff <= breeding_delay:
                        if oestrus_model_result != 0:
                            self.rsp_data['estrus_index'] = int(oestrus_model_result)
                            # self.rsp_data['confidence'] = max(float(oestrus_confidence), get_confidence(0.7))
                            self.rsp_data['confidence'] = self.get_ROC(today, 0, get_confidence(0.7))
                        else:
                            self.rsp_data['estrus_index'] = 0
                            # self.rsp_data['confidence'] = 0.0
                            self.rsp_data['confidence'] = self.get_ROC(today, 0, get_confidence(0))
                    else:
                        is_oestrus, is_breeding, base_br, confidence = is_oestrus_base_logic(act_feature, br_feature,
                                                                                             is_up_feature,
                                                                                             ideal_channel_feature)

                        if confidence != 0 or base_br:
                            if is_breeding != 0 or base_br:
                                self.rsp_data['estrus_index'] = 3
                                # self.rsp_data['confidence'] = get_confidence(max(confidence, 0.7))
                                self.rsp_data['confidence'] = self.get_ROC(today, 1, get_confidence(0.7))
                            else:
                                self.rsp_data['estrus_index'] = 1
                                # self.rsp_data['confidence'] = get_confidence(max(confidence, 0.7))
                                self.rsp_data['confidence'] = self.get_ROC(today, 0, get_confidence(0))
                        else:
                            self.rsp_data['estrus_index'] = 0
                            self.rsp_data['confidence'] = self.get_ROC(today, 0, get_confidence(0))
            else:
                if diff <= 0 or diff2 < 0:
                    self.rsp_data['estrus_index'] = 0

                elif diff <= breeding_delay:
                    # 模型结果
                    feature_df = generate_input(act_feature, br_feature, is_up_feature, ideal_channel_feature)
                    oestrus_confidence, oestrus_model_result = is_oestrus_base_oestrus_model(self.oestrus_model, feature_df)
                    logger.info(f"{self.device_serial} oestrus result on {today} is {oestrus_confidence} {oestrus_model_result}")

                    if oestrus_model_result != 0:
                        self.rsp_data['estrus_index'] = int(oestrus_model_result)
                        # self.rsp_data['confidence'] = max(float(oestrus_confidence), get_confidence(0.7))
                        self.rsp_data['confidence'] = self.get_ROC(today, 0, get_confidence(0))
                    else:
                        self.rsp_data['estrus_index'] = 0
                        # self.rsp_data['confidence'] = 0.0
                        self.rsp_data['confidence'] = self.get_ROC(today, 0, get_confidence(0))

                elif diff == breeding_delay + 1:
                    # --------------------逻辑结果--------------------
                    is_oestrus, is_breeding, base_br, confidence = is_oestrus_base_logic_jingchan(act_feature,
                                                                                                  br_feature,
                                                                                                  is_up_feature,
                                                                                                  ideal_channel_feature)

                    # --------------------模型结果--------------------
                    feature_df = generate_input(act_feature, br_feature, is_up_feature, ideal_channel_feature)
                    breeding_confidence, breeding_model_result = is_oestrus_base_breeding_model(self.breeding_model, feature_df)
                    logger.info(f"{self.device_serial} breeding result on {today} is {breeding_confidence} {breeding_model_result}")

                    # 删除数据库
                    del_query = ("DELETE FROM result_base_model "
                                    "WHERE device_serial=%s AND date<=%s")
                    value = (self.device_serial, today.strftime('%Y-%m-%d'))
                    self.my_database.database_change(del_query, value)

                    if breeding_confidence >= 0.7:
                        self.rsp_data['estrus_index'] = 3
                        # self.rsp_data['confidence'] = max(float(breeding_confidence), get_confidence(0.9))
                    elif confidence >= 0.4 or is_breeding > 0:
                        # 写入数据库
                        insert_query = ("INSERT INTO result_base_model "
                                        "(device_serial, date, confidence) "
                                        "VALUES (%s, %s, %s)")
                        value = (self.device_serial, (today + datetime.timedelta(days=1)).strftime('%Y-%m-%d'),
                                    float(breeding_confidence))
                        self.my_database.database_change(insert_query, value)

                    if self.rsp_data['estrus_index'] == 3:
                        self.rsp_data['confidence'] = self.get_ROC(today, 1, max(float(breeding_confidence), get_confidence(0.9)))
                    else:
                        self.rsp_data['confidence'] = self.get_ROC(today, 0, get_confidence(0.3))

                else:
                    # --------------------逻辑结果--------------------
                    is_oestrus, is_breeding, base_br, confidence = is_oestrus_base_logic_jingchan(act_feature,
                                                                                                  br_feature,
                                                                                                  is_up_feature,
                                                                                                  ideal_channel_feature)
                    if is_breeding > 0:
                        self.rsp_data['estrus_index'] = 3
                        # self.rsp_data['confidence'] = max(float(confidence), get_confidence(0.7))

                    # --------------------模型结果--------------------
                    feature_df = generate_input(act_feature, br_feature, is_up_feature, ideal_channel_feature)
                    breeding_confidence, breeding_model_result = is_oestrus_base_breeding_model(self.breeding_model, feature_df, 0.5)
                    logger.info(f"{self.device_serial} breeding result on {today} is {breeding_confidence} {breeding_model_result}")

                    if int(breeding_model_result)==1:
                        self.rsp_data['estrus_index'] = 3
                        # self.rsp_data['confidence'] = max(float(breeding_confidence), get_confidence(0.7))

                    # 读取数据库
                    select_query = (f"SELECT confidence FROM result_base_model "
                                    f"WHERE device_serial=%s AND date=%s")
                    value = (self.device_serial, today.strftime('%Y-%m-%d'))
                    flag, result = self.my_database.database_select(select_query, value)
                    if flag and result is not None and len(result) != 0:
                        self.rsp_data['estrus_index'] = 3
                        # self.rsp_data['confidence'] = max(float(breeding_confidence), float(result[0][0]), get_confidence(0.7))
                    # 删除数据库
                    del_query = ("DELETE FROM result_base_model "
                                 "WHERE device_serial=%s AND date<=%s")
                    value = (self.device_serial, today.strftime('%Y-%m-%d'))
                    self.my_database.database_change(del_query, value)

                    if self.rsp_data['estrus_index'] == 3:
                        self.rsp_data['confidence'] = self.get_ROC(today, 1, max(float(breeding_confidence), get_confidence(0.7)))
                    else:
                        self.rsp_data['confidence'] = self.get_ROC(today, 0, breeding_confidence)

        if self.rsp_data['confidence'] < 0.5 and self.rsp_data['estrus_index'] == 3:
            self.rsp_data['estrus_index'] = 0
        if self.rsp_data['confidence'] >= 0.5 and self.rsp_data['estrus_index'] != 3:
            self.rsp_data['confidence'] = get_confidence(0.3)

        if self.rsp_data['confidence'] >= 0.5 and self.rsp_data['estrus_index'] != 3:
            logger.error(f"{self.device_serial} confidence is {self.rsp_data['confidence']} but estrus_index is {self.rsp_data['estrus_index']}")
        if self.rsp_data['confidence'] < 0.5 and self.rsp_data['estrus_index'] == 3:
            logger.error(f"{self.device_serial} confidence is {self.rsp_data['confidence']} but estrus_index is {self.rsp_data['estrus_index']}")

    def write_model_result_into_database(self, date, oestrus_confidence, oestrus_model_result, breeding_confidence, breeding_model_result):
        insert_query = (f"INSERT INTO model_result "
                        f"(device_serial, date, oestrus_confidence, oestrus_model_result, breeding_confidence, breeding_model_result) "
                        f"VALUES (%s, %s, %s, %s, %s, %s)")
        value = (self.device_serial, date.strftime("%Y-%m-%d"), float(oestrus_confidence), int(oestrus_model_result), float(breeding_confidence), int(breeding_model_result))

        try:
            if not self.my_database.database_change(insert_query, value):
                msg = (f"{self.device_serial} insert model result: {date} {oestrus_confidence} {oestrus_model_result} "
                       f"{breeding_confidence} {breeding_model_result} failed!")
                logger.info(msg)
            else:
                msg = (f"{self.device_serial} insert model result: {date} {oestrus_confidence} {oestrus_model_result} "
                       f"{breeding_confidence} {breeding_model_result} success!")
                logger.info(msg)
        except:
            msg = f"{self.device_serial} insert model result: {date} {oestrus_model_result} {breeding_model_result} failed!"
            logger.error(msg)

    def get_ROC(self, today, oestrus_index, oestrus_confidence):

        # 获取今天是查情第几天
        day_diff = date_diff(self.investigate_start_time, today)
        day_len = date_diff(self.investigate_start_time, self.investigate_end_time) - 1

        # 查数据库
        select_query = (f"SELECT device_serial, investigate_start_date, cur_day, x, y_list, y_smooth_list, is_peak_detected, peak_position FROM confidence_msg "
                        f"WHERE device_serial=%s")
        flag, result = self.my_database.database_select(select_query, self.device_serial)

        # 如果没有这条记录
        if (not flag) or result is None or len(result) == 0:
            x = 0
            y_list = [float(oestrus_confidence)]
            y_smooth_list = []
            y_smooth_list, is_peak_detected, peak_position = pred_prob_smooth(x, y_list, y_smooth_list, day_len, False, -1, oestrus_index)

            # 写入数据库
            insert_query = ("INSERT INTO confidence_msg "
                            "(device_serial, investigate_start_date, cur_day, x, y_list, y_smooth_list, is_peak_detected, peak_position) "
                            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)")
            value = (self.device_serial, self.investigate_start_time, today, int(x), json.dumps(y_list), json.dumps(y_smooth_list), int(is_peak_detected), int(peak_position))
            self.my_database.database_change(insert_query, value)

        # 如果有这条记录
        else:
            (device_serial, investigate_start_date, cur_day, x, y_list, y_smooth_list, is_peak_detected, peak_position) = result[0]

            # 如果查情时间不一致，说明重置了
            if investigate_start_date != self.investigate_start_time.date():
                x = 0
                y_list = [float(oestrus_confidence)]
                y_smooth_list = []
                y_smooth_list, is_peak_detected, peak_position = pred_prob_smooth(x, y_list, y_smooth_list, day_len,
                                                                                  False, -1, oestrus_index)
                # 更新数据库
                update_query = ("UPDATE confidence_msg "
                                "SET investigate_start_date=%s, cur_day=%s, x=%s, y_list=%s, y_smooth_list=%s, is_peak_detected=%s, peak_position=%s "
                                "WHERE device_serial=%s")
                value = (self.investigate_start_time, today, int(x), json.dumps(y_list), json.dumps(y_smooth_list),
                         int(is_peak_detected), int(peak_position), self.device_serial)
                self.my_database.database_change(update_query, value)
            elif today.date() <= cur_day:
                y_smooth_list = json.loads(y_smooth_list)
            else:
                x = x + 1
                y_list = json.loads(y_list)
                y_list.append(float(oestrus_confidence))
                y_smooth_list = json.loads(y_smooth_list)
                y_smooth_list, is_peak_detected, peak_position = pred_prob_smooth(x, y_list, y_smooth_list, day_len,
                                                                                  is_peak_detected, peak_position, oestrus_index)
                # 更新数据库
                update_query = ("UPDATE confidence_msg "
                                "SET investigate_start_date=%s, cur_day=%s, x=%s, y_list=%s, y_smooth_list=%s, is_peak_detected=%s, peak_position=%s "
                                "WHERE device_serial=%s")
                value = (self.investigate_start_time, today, int(x), json.dumps(y_list), json.dumps(y_smooth_list),
                         int(is_peak_detected), int(peak_position), self.device_serial)
                self.my_database.database_change(update_query, value)

        result = y_smooth_list[-1]
        return round(result, 4)


def pred_prob_smooth(x, y_list, y_smooth_list, total_steps, is_peak_detected, peak_position, is_breeding):

    if x == 0:
        if y_list[-1] > 0.9 or is_breeding == 1:
            y_smooth_list.append(float(y_list[-1]))
        else:
            # 第一个点：起始低
            y_smooth_list.append(float(y_list[-1] * 0.5))
    # --- 峰值前阶段：强制上升 ---
    elif not is_peak_detected:
        window_data = y_list

        # 检测是否应该进入峰值
        if x >= 3 and len(y_smooth_list) >= 3:
            # 检查最近3点是否呈现下降趋势（可能过早出现峰值）
            recent_3 = y_smooth_list[-3:]
            is_descending = recent_3[0] > recent_3[1] > recent_3[2]

            if is_descending and x < total_steps // 2:
                # 过早下降，强制继续上升
                current_avg = np.mean(window_data)
                expected_rise = 0.3 + 0.5 * (x / (total_steps // 2))

                if current_avg < expected_rise:
                    # 使用上升权重
                    weights = np.linspace(0.1, 0.9, len(window_data))
                    weights = weights / np.sum(weights)
                    base_smooth = np.sum(weights * window_data)
                    base_smooth = max(base_smooth, y_smooth_list[-1] * 1.05)  # 强制上升
                else:
                    base_smooth = np.mean(window_data)
            else:
                # 正常平滑
                weights = np.linspace(0.2, 0.8, len(window_data))
                weights = weights / np.sum(weights)
                base_smooth = np.sum(weights * window_data)

        else:
            # 前期简单平滑
            weights = np.linspace(0.1, 0.9, len(window_data))
            weights = weights / np.sum(weights)
            base_smooth = np.sum(weights * window_data)

        # 峰值检测条件
        if x >= total_steps // 3 and len(y_smooth_list) >= 2:
            # 检查是否出现转折点（峰值候选）
            if y_smooth_list[-1] > base_smooth and y_smooth_list[-1] > 0.5:
                is_peak_detected = True
                peak_position = x - 1

        if is_breeding == 1 and base_smooth < 0.5:
            y_smooth_list.append(float(random.uniform(0.6, 0.8)))
        else:
            y_smooth_list.append(float(min(base_smooth, 0.9)))
    # --- 峰值后阶段：强制下降 ---
    else:
        window_data = y_smooth_list[max(0, x - 4):x + 1]

        # 计算从峰值位置开始的相对进度
        progress_after_peak = (x - peak_position) / (total_steps - peak_position)

        # 强制下降权重：越远离峰值，近期权重越低
        decline_intensity = min(1.0, progress_after_peak * 2)  # 加速下降
        recent_weight = max(0.05, 0.3 * (1 - decline_intensity))

        weights = [0.3, 0.25, 0.2, 0.15, recent_weight][-len(window_data):]
        weights = [w / sum(weights) for w in weights]
        base_smooth = np.sum(np.array(weights) * window_data)

        # --- 强制单一峰值机制 ---
        if len(y_smooth_list) > 0:
            # 禁止出现第二个峰值：当前值不能超过历史最高值
            historical_max = max(y_smooth_list)
            if base_smooth > historical_max:
                base_smooth = historical_max * 0.95  # 强制低于历史峰值

            # 强制下降趋势
            if base_smooth > y_smooth_list[-1]:
                # 如果出现上升，强制转为下降
                expected_decline = 0.03 + 0.07 * progress_after_peak
                base_smooth = y_smooth_list[-1] * (1 - expected_decline)

        # 末尾特别处理
        if x >= total_steps - 2:
            if x == total_steps - 2:
                base_smooth = min(base_smooth, 0.25)
            elif x == total_steps - 1:
                base_smooth = min(base_smooth, 0.15)
                if base_smooth > 0.2:
                    base_smooth = 0.1

        y_smooth_list.append(float(max(base_smooth, 0.05)))

    return y_smooth_list, is_peak_detected, peak_position


def get_confidence(base_confidence=0.0):
    if base_confidence == 0:
        return round(random.uniform(0, 0.1), 4)
    elif base_confidence == 0.3:
        return round(random.uniform(0.1, 0.3), 4)
    elif base_confidence == 0.7:
        return round(random.uniform(0.65, 0.75), 4)
    elif base_confidence == 0.8:
        return round(random.uniform(0.65, 0.9), 4)
    elif base_confidence == 0.9:
        return round(random.uniform(0.9, 0.95), 4)
    elif base_confidence == 1:
        return round(random.uniform(0.95, 1), 4)


def count_value_in_2d_list(two_d_list, value_to_count1, value_to_count2):
    count = 0
    for row in two_d_list:
        count += row.count(value_to_count1)
        count += row.count(value_to_count2)
    return count


def get_diff_sec_between_str(start_time, end_time):
    format = "%Y-%m-%d %H:%M:%S"
    return (datetime.datetime.strptime(end_time, format) - datetime.datetime.strptime(start_time, format)).total_seconds()


def date_diff(label_date, pred_date):
    # 计算两日期之间的差值
    delta = pred_date.date() - label_date.date()

    # 获取天数差
    day_difference = delta.days

    return day_difference
