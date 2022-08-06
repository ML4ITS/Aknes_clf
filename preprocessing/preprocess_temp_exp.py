import os
import logging
import warnings

from torchvision.transforms import Compose
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import obspy
from scipy.signal import spectrogram
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import cv2

from utils import root_dir
from original_implementation.preprocess import signaltonoise


def sample_norm_timesteps_nc(crop_window_size_rate: float, shift_rate: float):
    norm_shift_len = shift_rate * crop_window_size_rate
    global_norm_min = 0. + norm_shift_len + crop_window_size_rate
    global_norm_max = 1. - norm_shift_len - crop_window_size_rate

    # sample reference time step
    norm_ts = np.random.uniform(global_norm_min, global_norm_max)

    # sample neighboring time step
    norm_sigma = norm_shift_len / 2
    while True:
        norm_ts_l = np.random.normal(norm_ts, norm_sigma)
        if (norm_ts - norm_shift_len < norm_ts_l) and (norm_ts_l < norm_ts + norm_shift_len):
            break

    # sample non-neighboring time step
    while True:
        norm_ts_k = np.random.uniform(global_norm_min, global_norm_max)
        if (norm_ts_k < norm_ts - norm_shift_len - crop_window_size_rate) or (norm_ts + norm_shift_len + crop_window_size_rate < norm_ts_k):
            break

    return norm_ts, norm_ts_l, norm_ts_k


class NORSARDataset(Dataset):
    def __init__(self,
                 transform: Compose = None,
                 sampling_freq: int = 1000,
                 nperseg: int = 80,  # 64; 80
                 log_scale: bool = True,
                 minmax_scale: bool = True,
                 resize: bool = True,
                 resize_size: int = 128,
                 return_single_spectrogram: bool = False,
                 is_ssl: bool = False,
                 train_data_ratio: float = 0.8,
                 test_data_ratio: float = 0.2,
                 train_test_split_rand_seed: int = 0,
                 kind: str = 'train',
                 **kwargs):
        """
        :param transform:
        :param sampling_freq:
        :param nperseg:
        :param log_scale:
        :param minmax_scale:
        :param resize:
        :param resize_size:
        :param return_single_spectrogram:
        :param is_ssl:
        :param train_data_ratio:
        :param test_data_ratio:
        :param train_test_split_rand_seed:
        :param kind:
        :param kwargs:
        """
        super(NORSARDataset, self).__init__()

        self.transform = transform
        self.sampling_freq = sampling_freq
        self.nperseg = nperseg
        self.log_scale = log_scale
        self.minmax_scale = minmax_scale
        self.resize = resize
        self.resize_size = resize_size
        self.return_single_spectrogram = return_single_spectrogram
        self.is_ssl = is_ssl
        self.kind = kind

        # load the csv file for indexing the .dat files
        data_dir_fname = root_dir.joinpath('dataset', 'data_dir.csv')
        self.data_dir_csv = pd.read_csv(data_dir_fname)  # (n_samples, 2)
        self.data_dir_csv['fname'] = self.data_dir_csv['fname'].apply(lambda s: s.replace('\\', os.sep))

        # if it's not self-supervised learning, we exclude samples with the class `Unlabeled`
        if not self.is_ssl:
            self.data_dir_csv = self.data_dir_csv[self.data_dir_csv['event'] != 'Unlabeled']

        # label_encoder
        self.label_encoder = LabelEncoder()
        labels = self.label_encoder.fit_transform(self.data_dir_csv['event'].values)
        logging.info('You can use `dataset.label_encoder` to decode the numeric label.')

        # train-test split
        sample_indices = np.arange(self.data_dir_csv.shape[0])
        train_sample_indices, test_sample_indices, train_labels, test_labels = train_test_split(sample_indices,
                                                                                                labels,
                                                                                                train_size=train_data_ratio,
                                                                                                test_size=test_data_ratio,
                                                                                                stratify=labels,
                                                                                                random_state=train_test_split_rand_seed)
        if self.kind == 'train':
            self.data_dir_csv = self.data_dir_csv.iloc[train_sample_indices, :]
        elif self.kind == 'test':
            self.data_dir_csv = self.data_dir_csv.iloc[test_sample_indices, :]
        else:
            raise ValueError
        self.data_dir_csv.reset_index(inplace=True)

        self._len = self.data_dir_csv.shape[0]

    def __getitem__(self, idx):
        fname = root_dir.joinpath(self.data_dir_csv['fname'][idx])
        fname = r'{}'.format(fname)  # to prevent some reading error
        label = self.data_dir_csv['event'][idx]

        # encode label
        label = self.label_encoder.transform([label])[0]

        # Read the file --> returns a Stream object containing 8*3 Trace objects --> the stream object is converted into np.array
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st = obspy.read(fname)  # st: stream
        st_size = st.__len__()
        if self.return_single_spectrogram:
            rand_idx1, rand_idx2 = np.random.randint(0, st_size), np.random.randint(0, st_size)
            st1, st2 = st[rand_idx1].data[np.newaxis, :], st[rand_idx2].data[np.newaxis, :]  # [1 x 16000]
            st = np.concatenate((st1, st2), axis=0)  # [2 x 16000]
        else:
            ts_len = st[0].data.shape[0]
            st_ = np.zeros((st_size, ts_len))  # [24 x 16000]; 16s x 1000Hz
            for i, tr in enumerate(st):
                st_[i, :] = tr.data  # (16000,)
            st = st_

        snr = signaltonoise(st, axis=-1)

        # convert .dat into a spectrogram
        f, t, Sxx = spectrogram(st, fs=self.sampling_freq, nperseg=self.nperseg)  # Sxx: spectrogram; (n_sensors, H, W)
        Sxx = np.abs(Sxx)

        # scale
        n_sensors = Sxx.shape[0]
        if self.log_scale:
            epsilon = 1.  # to prevent noise from being exaggerated by log.
            Sxx = np.log10(Sxx + epsilon)

        if self.minmax_scale:
            flatSxx = Sxx.reshape(n_sensors, -1)  # (n_sensors, HW)
            q = np.nanquantile(flatSxx, [0., 1.], axis=1).T  # (n_sensors, 2)
            min_ = q[:, 0][:, None, None]  # (n_sensors, 1, 1)
            max_ = q[:, 1][:, None, None]  # (n_sensors, 1, 1)
            Sxx = (Sxx - min_) / (max_ - min_)  # (n_sensors, H, W)

        # resize
        if self.resize:
            increase_ratio = (self.resize_size / Sxx.shape[1])
            H, W = int(Sxx.shape[1] * increase_ratio), int(Sxx.shape[2] * increase_ratio)
            Sxx_resized = np.zeros((n_sensors, H, W))  # (n_sensors, H, W)
            for i in range(n_sensors):
                Sxx_resized[i, :, :] = cv2.resize(Sxx[i], (W, H))
            Sxx = Sxx_resized  # (n_sensors, H, W)

        # augmentation
        if self.return_single_spectrogram:
            Sxx_view1, Sxx_view2 = Sxx[[0], :, :], Sxx[[1], :, :]  # (H, W)
            first_transform = str(self.transform.transforms[0])
            if first_transform in ['NeighboringCrop', 'RandomCrop']:
                crop_window_size_rate = self.transform.transforms[0].crop_window_size_rate
                if first_transform == 'NeighboringCrop':
                    shift_rate = self.transform.transforms[0].shift_rate
                    norm_ts, norm_ts_l, norm_ts_k = sample_norm_timesteps_nc(crop_window_size_rate, shift_rate)
                    norm_ts_nc = {'ts': norm_ts, 'ts_l': norm_ts_l, 'ts_k': norm_ts_k}
                elif first_transform == 'RandomCrop':
                    norm_ts_nc = {'ts': 0, 'ts_l': 0, 'ts_k': 0}
                else:
                    raise ValueError
                Sxx_view1, Sxx_view_l, Sxx_view_k = self.transform([Sxx_view1, Sxx_view2, norm_ts_nc])
                return (Sxx_view1, Sxx_view_l, Sxx_view_k), label, snr  # Sxx: (1, H, W)
            # elif first_transform == 'ToTensor':
            else:
                norm_ts_nc = {'ts': 0, 'ts_l': 0, 'ts_k': 0}
                Sxx_view1, Sxx_view_l, norm_ts_nc = self.transform([Sxx_view1, Sxx_view2, norm_ts_nc])
                return Sxx_view1, label, snr
            # else:
            #     raise ValueError
        else:
            return Sxx, label, snr  # Sxx: (24, H, W)

    def __len__(self):
        return self._len


if __name__ == '__main__':
    import os
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from preprocessing.augmentations import *

    os.chdir(root_dir)

    # transforms = Compose([NeighboringCrop(), ToTensor()])
    # transforms = Compose([RandomCrop(), ToTensor()])
    transforms = Compose([ToTensor()])
    data_loader = DataLoader(NORSARDataset(transform=transforms,
                                           return_single_spectrogram=False,  # True / False
                                           is_ssl=False,
                                           kind='train'),
                             batch_size=64,
                             shuffle=False,
                             drop_last=True,
                             num_workers=0)

    batch_idx = 0

    if data_loader.dataset.return_single_spectrogram:
        for (Sxx_view, Sxx_view_l, Sxx_view_k), label, snr in data_loader:
            break

        print('Sxx_view.shape:', Sxx_view.shape)
        print('Sxx_view_l.shape:', Sxx_view_l.shape)
        print('Sxx_view_k.shape:', Sxx_view_k.shape)
        print('label.shape:', label.shape)
        print('snr:', snr)

        # visualize
        channels = Sxx_view.shape[1]
        for c in range(channels):
            n_plots = 3
            fig, (ax1, ax2, ax3) = plt.subplots(n_plots, 1, figsize=(8, 2*n_plots))

            im1 = ax1.imshow(Sxx_view[batch_idx, 0, :, :], aspect='auto')
            im2 = ax2.imshow(Sxx_view_l[batch_idx, 0, :, :], aspect='auto')
            im3 = ax3.imshow(Sxx_view_k[batch_idx, 0, :, :], aspect='auto')

            fig.colorbar(im3)

            ax1.invert_yaxis()
            ax2.invert_yaxis()
            ax3.invert_yaxis()
            plt.show()

    else:
        for Sxx_view, label, snr in data_loader:
            break

        print('Sxx_view.shape:', Sxx_view.shape)
        print('snr.shape:', snr.shape)

        sensor_idx = 0
        n_imfs = Sxx_view.shape[2]

        n_plots = 2
        fig, (ax1, ax2) = plt.subplots(n_plots, 1, figsize=(8, 2*n_plots))
        im1 = ax1.imshow(Sxx_view[batch_idx, sensor_idx, :, :], aspect='auto')  # real
        ax1.invert_yaxis()
        ax2.invert_yaxis()
        ax1.set_title(f'label: {label[batch_idx]}')
        fig.colorbar(im1)
        plt.show()