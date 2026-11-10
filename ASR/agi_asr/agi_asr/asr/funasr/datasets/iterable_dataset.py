"""Iterable dataset module."""
import copy
from io import StringIO
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Iterator
from typing import Tuple
from typing import Union
from typing import List

import kaldiio
import numpy as np
import torch
import torchaudio
from torch.utils.data.dataset import IterableDataset


SUPPORT_AUDIO_TYPE_SETS = ['flac', 'mp3', 'ogg', 'opus', 'wav', 'pcm']

def load_kaldi(input):
    retval = kaldiio.load_mat(input)
    if isinstance(retval, tuple):
        assert len(retval) == 2, len(retval)
        if isinstance(retval[0], int) and isinstance(retval[1], np.ndarray):
            # sound scp case
            rate, array = retval
        elif isinstance(retval[1], int) and isinstance(retval[0], np.ndarray):
            # Extended ark format case
            array, rate = retval
        else:
            raise RuntimeError(f"Unexpected type: {type(retval[0])}, {type(retval[1])}")

        # Multichannel wave fie
        # array: (NSample, Channel) or (Nsample)

    else:
        # Normal ark case
        assert isinstance(retval, np.ndarray), type(retval)
        array = retval
    return array


def load_bytes(input):
    middle_data = np.frombuffer(input, dtype=np.int16)
    middle_data = np.asarray(middle_data)
    if middle_data.dtype.kind not in 'iu':
        raise TypeError("'middle_data' must be an array of integers")
    dtype = np.dtype('float32')
    # if dtype.kind != 'f':
    #     raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(middle_data.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    array = np.frombuffer((middle_data.astype(dtype) - offset) / abs_max, dtype=np.float32)
    return array

def load_pcm(input):
    with open(input,"rb") as f:
        bytes = f.read()
    return load_bytes(bytes)

DATA_TYPES = {
    "sound": lambda x: torchaudio.load(x)[0].numpy(),
    "pcm": load_pcm,
    "kaldi_ark": load_kaldi,
    "bytes": load_bytes,
    "waveform": lambda x: x,
    "npy": np.load,
    "text_int": lambda x: np.loadtxt(
        StringIO(x), ndmin=1, dtype=np.long, delimiter=" "
    ),
    "csv_int": lambda x: np.loadtxt(StringIO(x), ndmin=1, dtype=np.long, delimiter=","),
    "text_float": lambda x: np.loadtxt(
        StringIO(x), ndmin=1, dtype=np.float32, delimiter=" "
    ),
    "csv_float": lambda x: np.loadtxt(
        StringIO(x), ndmin=1, dtype=np.float32, delimiter=","
    ),
    "text": lambda x: x,
}


class IterableESPnetDataset(IterableDataset):
    """Pytorch Dataset class for ESPNet.

    Examples:
        >>> dataset = IterableESPnetDataset([('wav.scp', 'input', 'sound'),
        ...                                  ('token_int', 'output', 'text_int')],
        ...                                )
        >>> for uid, data in dataset:
        ...     data
        {'input': per_utt_array, 'output': per_utt_array}
    """

    def __init__(
            self,
            path_name_type_list: Collection[Tuple[any, str, str]],
            preprocess: Callable[
                [str, Dict[str, np.ndarray]], Dict[str, np.ndarray]
            ] = None,
            float_dtype: str = "float32",
            fs: dict = None,
            mc: bool = False,
            int_dtype: str = "long",
            key_file: str = None,
    ):
        # assert check_argument_types()
        if len(path_name_type_list) == 0:
            raise ValueError(
                '1 or more elements are required for "path_name_type_list"'
            )

        # path_name_type_list = copy.deepcopy(path_name_type_list)

        self.preprocess = preprocess

        # self.float_dtype = float_dtype
        # self.int_dtype = int_dtype
        # self.key_file = key_file
        self.fs = fs
        # self.mc = mc

        # self.debug_info = {}
        self.path_name_type_list = []

        if not isinstance(path_name_type_list[0], (Tuple, List)):
            path = path_name_type_list[0]
            name = path_name_type_list[1]
            _type = path_name_type_list[2]
            _uid = path_name_type_list[3] if len(path_name_type_list) > 3 else 'utt_id'
            # self.debug_info[name] = path, _type

            self.path_name_type_list.append((path, name, _type, _uid))
           
        else:
            if len(path_name_type_list[0]) == 3:
                for path, name, _type in path_name_type_list:
                    _uid = 'utt_id'
                    # self.debug_info[name] = path, _type
                    self.path_name_type_list.append((path, name, _type, _uid))
            else:
                for path, name, _type, _uid in path_name_type_list:
                    # self.debug_info[name] = path, _type
                    self.path_name_type_list.append((path, name, _type, _uid))

        self.apply_utt2category = False

    # def has_name(self, name) -> bool:
    #     return name in self.debug_info

    # def names(self) -> Tuple[str, ...]:
    #     return tuple(self.debug_info)

    # def __repr__(self):
    #     _mes = self.__class__.__name__
    #     _mes += "("
    #     for name, (path, _type) in self.debug_info.items():
    #         _mes += f'\n  {name}: {{"path": "{path}", "type": "{_type}"}}'
    #     _mes += f"\n  preprocess: {self.preprocess})"
    #     return _mes

    def __iter__(self) -> Iterator[Tuple[Union[str, int], Dict[str, np.ndarray]]]:
        
        if self.path_name_type_list[0][2] == "bytes" or self.path_name_type_list[0][2] == "waveform":
            linenum = len(self.path_name_type_list)
            
            for i in range(linenum):
                data = {}
                value = self.path_name_type_list[i][0]
                uid = self.path_name_type_list[i][3]
                name = self.path_name_type_list[i][1]
                _type = self.path_name_type_list[i][2]
                func = DATA_TYPES[_type]
                array = func(value)
                if self.fs is not None and (name == "speech" or name == "ref_speech"):
                    audio_fs = self.fs["audio_fs"]
                    model_fs = self.fs["model_fs"]
                    if audio_fs is not None and model_fs is not None:
                        array = torch.from_numpy(array)
                        array = array.unsqueeze(0)
                        array = torchaudio.transforms.Resample(orig_freq=audio_fs,
                                                       new_freq=model_fs)(array)
                        array = array.squeeze(0).numpy()

                data[name] = array

                if self.preprocess is not None:
                    data = self.preprocess(uid, data)

                # for name in data:
                #     value = data[name]
                #     if not isinstance(value, np.ndarray):
                #         raise RuntimeError(
                #             f'All values must be converted to np.ndarray object '
                #             f'by preprocessing, but "{name}" is still {type(value)}.')
                #     # Cast to desired type
                #     if value.dtype.kind == 'f':
                #         value = value.astype(self.float_dtype)
                #     elif value.dtype.kind == 'i':
                #         value = value.astype(self.int_dtype)
                #     else:
                #         raise NotImplementedError(
                #             f'Not supported dtype: {value.dtype}')
                #     data[name] = value

                yield uid, data


