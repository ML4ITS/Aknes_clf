"""
Crop (with a fixed window size) -> Resize to be square-sized.
During inference, a model runs over a stack of cropped-resized spectrograms with some `t` interval.

Possible augmentations:
- ColorJitter
- GaussianBlur
"""
import numpy as np
import torch
from utils.types_ import *


def random_pick_timesteps_nc(norm_ts: float, shift_len: int, crop_size: int, input_len: int) -> (int, int):
    min_ = shift_len + crop_size // 2 + 1
    max_ = input_len - shift_len - crop_size // 2 - 1
    ts = norm_ts * (max_ - min_) + min_
    ts = int(round(ts))  # reference time step

    # select a neighboring sample
    # shift the timestep according to the normal dist.
    sigma = shift_len / 2
    while True:
        ts_l = np.random.normal(ts, sigma, size=1)[0]
        ts_l = np.round(ts_l).astype(int)
        if (ts - shift_len <= ts_l) and (ts_l <= ts + shift_len):
            break

    # select far-away sample
    min_ = crop_size // 2 + 1
    max_ = input_len - crop_size // 2 - 1
    while True:
        norm_ts_k = np.random.uniform(0., 1.)
        ts_k = norm_ts_k * (max_ - min_) + min_
        ts_k = np.round(ts_k).astype(int)
        if (ts_k < ts - shift_len) or (ts + shift_len < ts_k):
            break
    return ts, ts_l, ts_k


def scale_up_ts_nc(norm_ts: float, norm_ts_l: float, norm_ts_k: float, input_len: int):
    min_ = 0.
    max_ = input_len

    # scale-up reference time step
    ts = int(np.floor(norm_ts * (max_ - min_) + min_))

    # scale-up neighboring time step
    ts_l = int(np.floor(norm_ts_l * (max_ - min_) + min_))

    # scale-up non-neighboring time step
    ts_k = int(np.floor(norm_ts_k * (max_ - min_) + min_))

    return ts, ts_l, ts_k


class NeighboringCrop(object):
    def __init__(self,
                 crop_window_size_rate: float = 0.1,
                 shift_rate: float = 0.25,):
        """
        :param crop_window_size_rate: sub-size of the original spectrogram by `this rate` which is the crop_size.
        :param shift_rate: sub-size of crop_size by `this rate`; A randomly-selected point in time is randomly shifted by U(-`shift_rate` * crop_size, `shift_rate` * crop_size); (0 <= param <= 1)
        """
        self.crop_window_size_rate = crop_window_size_rate
        self.shift_rate = shift_rate

    def __call__(self, input) -> List[Tensor]:
        """
        :param Sxx1: (n_sensors, H, W)
        :param Sxx2: (n_sensors, H, W)
        :return:
        """
        Sxx1, Sxx2, norm_ts_nc = input

        input_len = Sxx1.shape[2]
        crop_size = np.floor(self.crop_window_size_rate * input_len).astype(int)

        norm_ts, norm_ts_l, norm_ts_k = norm_ts_nc['ts'], norm_ts_nc['ts_l'], norm_ts_nc['ts_k']
        ts, ts_l, ts_k = scale_up_ts_nc(norm_ts, norm_ts_l, norm_ts_k, input_len)

        # crop
        Sxx_view = Sxx1[:, :, ts - crop_size // 2:ts + crop_size // 2]
        Sxx_view_l = Sxx2[:, :, ts_l - crop_size // 2:ts_l + crop_size // 2]
        Sxx_view_k = Sxx2[:, :, ts_k - crop_size // 2:ts_k + crop_size // 2]

        return [Sxx_view, Sxx_view_l, Sxx_view_k]


class RandomCrop(object):
    def __init__(self,
                 crop_window_size_rate: float = 0.1,):
        """
        :param crop_window_size_rate: sub-size of the original spectrogram by `this rate` which is the crop_size.
        """
        self.crop_window_size_rate = crop_window_size_rate

    def _random_pick_timesteps_rc(self, input_len: int) -> Tuple[int, int]:
        # sample reference time step
        global_norm_min = 0. + self.crop_window_size_rate
        global_norm_max = 1. - self.crop_window_size_rate
        norm_ts1, norm_ts2 = np.random.uniform(global_norm_min, global_norm_max, size=2)

        # scale up
        min_ = 0.
        max_ = input_len
        ts1 = int(np.floor(norm_ts1 * (max_ - min_) + min_))
        ts2 = int(np.floor(norm_ts2 * (max_ - min_) + min_))
        return ts1, ts2

    def __call__(self, input) -> List[Tensor]:
        """
        :param Sxx1: (n_sensors, H, W)
        :param Sxx2: (n_sensors, H, W)
        :return:
        """
        Sxx1, Sxx2, norm_ts_nc = input

        spectrogram_len = Sxx1.shape[2]
        crop_size = np.floor(self.crop_window_size_rate * spectrogram_len).astype(int)

        # crop
        ts1, ts2 = self._random_pick_timesteps_rc(spectrogram_len)
        Sxx_view1 = Sxx1[:, :, ts1 - crop_size // 2:ts1 + crop_size // 2]
        Sxx_view2 = Sxx2[:, :, ts2 - crop_size // 2:ts2 + crop_size // 2]
        Sxx_view_k = torch.zeros(Sxx_view2.shape)  # exists to match the format with `NeighboringCrop`

        return [Sxx_view1, Sxx_view2, Sxx_view_k]


class ToTensor(object):
    def __call__(self, input: Tensor):
        for i in range(len(input)):
            if isinstance(input[i], np.ndarray):
                input[i] = torch.from_numpy(input[i]).float()
        return input


# index dictionary (must be defined)
augmentations = {'NeighboringCrop': NeighboringCrop,
                 'RandomCrop': RandomCrop,
                 'ToTensor': ToTensor,
                 }
