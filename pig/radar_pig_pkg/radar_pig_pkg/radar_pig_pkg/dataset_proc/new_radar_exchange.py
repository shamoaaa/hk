import os
import struct
import base64
import logging
import traceback

import h5py
import numpy as np


__all__ = ['new_radar_raw2hdf']

def new_radar_raw2hdf(raw_data_folder, hdf_data_folder):
    radar_id_list = sorted(os.listdir(raw_data_folder))

    try:

        # 遍历雷达
        for radar_id in radar_id_list:

            radar_folder = os.path.join(raw_data_folder, radar_id)
            date_list = sorted(os.listdir(radar_folder))

            # 遍历日期
            for date in date_list:

                raw_file_folder = os.path.join(radar_folder, date)
                raw_file_list = sorted(os.listdir(raw_file_folder))

                last_hour = raw_file_list[0][0:10]

                bin = b''

                for file in raw_file_list:

                    hour = file[0:10]

                    # 如果小时的时间相同，则拼接
                    if hour == last_hour:
                        with open(os.path.join(raw_file_folder, file), 'r') as f:
                            base = f.readline()
                            bin += base64.b64decode(base)
                    # 否则进行转换
                    else:
                        date = last_hour[0:8]
                        save_path = os.path.join(hdf_data_folder, date)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)

                        _bin2hdf5(bin, os.path.join(save_path, f'{radar_id}-{last_hour}.hdf5'))
                        logging.info(f"{radar_id} generated {last_hour} raw data into hdf5 file.")

                        bin = b''
                        last_hour = hour
                        with open(os.path.join(raw_file_folder, file), 'r') as f:
                            base = f.readline()
                            bin += base64.b64decode(base)

                date = last_hour[0:8]
                save_path = os.path.join(hdf_data_folder, date)
                os.makedirs(save_path, exist_ok=True)
                _bin2hdf5(bin, os.path.join(save_path, f'{radar_id}-{last_hour}.hdf5'))
    except:
        logging.error(f"{radar_id} generated {file} raw data into hdf5 file failed!")
        logging.error(traceback.format_exc())


def _bin2hdf5(radar_data, hdf5_path):
    start_idx = 0
    data_len = len(radar_data)  # 实际的数据长度
    data_buffer_list = []
    package_ids = []
    last_end_index = 0

    while start_idx < data_len:

        if radar_data[start_idx:start_idx + 4] == bytes([90, 91, 92, 93]):
            # 实际的数据长度是否符合协议
            # 数据长度+包号+数据
            data_len_field = struct.unpack('<I', radar_data[start_idx+4:start_idx+8])[0]
            # 包头4+数据长度4+包号4+数据（（23*4）+2）*20*采集时长）+校验和1+时间戳8
            package_size = 4 + data_len_field + 1 + 8 # 期望的数据长度
            if start_idx + package_size > data_len:
                logging.error(f"Package size exceeds remaining file content at index {start_idx}."
                      f"Expected package size: {package_size}, Available: {data_len - start_idx}")

            frame = radar_data[start_idx:start_idx + package_size]

            # 检查校验和
            expected_checksum = frame[-9]
            calculated_checksum = sum(frame[:-9]) & 0xFF
            if calculated_checksum != expected_checksum:
                logging.error(f"Checksum error at index {start_idx}! Calculated: {calculated_checksum}, Actually: {expected_checksum}")

            package_id = struct.unpack('<I', frame[8:12])[0]
            package_ids.append(int(package_id))

            # 解析数据
            start = 12              # 载荷的起始地址
            end = package_size - 9  # 载荷的截止地址
            frame_size = 23 * 4 + 2 # 一帧的数据长度
            num = 0
            while start < end:
                # 检查帧头
                if frame[start:start+2] != bytes([85, 170]):
                    logging.error(f'{hdf5_path} Invalid frame header! Found: {frame[start:start + 2]}')

                floats = np.frombuffer(frame[start+2 : start+frame_size], dtype=np.int16)
                complex_data = 1j * floats[::2] + floats[1::2]
                data_buffer_list.append(complex_data)

                start += frame_size
                num += 1
            
            last_end_index = last_end_index+num
            start_idx += package_size
        else:
            logging.error(f"{hdf5_path} Invalid frame header at index {start_idx}. Found: {radar_data[start_idx:start_idx + 4]}")
            start_idx += 1

    with h5py.File(hdf5_path, 'w') as f:
        f.create_dataset('frame_ids', data=np.array(package_ids, dtype=np.uint32), compression="gzip", compression_opts=1)
        f.create_dataset('data', data=np.array(data_buffer_list, dtype=np.complex64), compression="gzip", compression_opts=1)
        f.create_dataset('checksum_failures', data=np.array([], dtype=np.int32), dtype=np.int32)

