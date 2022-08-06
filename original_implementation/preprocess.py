import os
import logging
import warnings

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import obspy
from scipy.signal import spectrogram
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import cv2

from utils import root_dir


def signaltonoise(a, axis=0, ddof=0):
    """
    (source) https://github.com/scipy/scipy/blob/v0.14.0/scipy/stats/stats.py#L1864
    The signal-to-noise ratio of the input data.
    Returns the signal-to-noise ratio of `a`, here defined as the mean
    divided by the standard deviation.
    Parameters
    ----------
    a : array_like
        An array_like object containing the sample data.
    axis : int or None, optional
        If axis is equal to None, the array is first ravel'd. If axis is an
        integer, this is the axis over which to operate. Default is 0.
    ddof : int, optional
        Degrees of freedom correction for standard deviation. Default is 0.
    Returns
    -------
    s2n : ndarray
        The mean to standard deviation ratio(s) along `axis`, or 0 where the
        standard deviation is 0.
    """
    a = np.asanyarray(a)
    # m = a.mean(axis)
    m = 1.  # the mean of the geophone sensor is always zero. We put 1 as a unit value.
    sd = a.std(axis=axis, ddof=ddof)
    # return np.where(sd == 0, 0, m/sd)
    return np.where(sd == 0, 0, m / sd)


class AknesOriginal(Dataset):
    def __init__(self,
                 sampling_freq: int = 1000,
                 nperseg: int = 80,
                 log_scale: bool = True,
                 minmax_scale: bool = True,
                 resize: bool = True,
                 resize_size: int = 128,
                 train_data_ratio: float = 0.8,
                 test_data_ratio: float = 0.2,
                 train_test_split_rand_seed: int = 0,
                 kind: str = 'train',
                 **kwargs):
        super().__init__()

        self.sampling_freq = sampling_freq
        self.nperseg = nperseg
        self.log_scale = log_scale
        self.minmax_scale = minmax_scale
        self.resize = resize
        self.resize_size = resize_size

        # load the csv file for indexing the .dat files
        data_dir_fname = root_dir.joinpath('dataset', 'data_dir.csv')
        self.data_dir_csv = pd.read_csv(data_dir_fname)  # (n_samples, 2)
        self.data_dir_csv = self.data_dir_csv[self.data_dir_csv['event'] != 'Unlabeled']  # remove unlabeled class samples
        self.data_dir_csv['fname'] = self.data_dir_csv['fname'].apply(lambda s: s.replace('\\', os.sep))

        # fit-transform label encoder
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
        if kind == 'train':
            self.data_dir_csv = self.data_dir_csv.iloc[train_sample_indices, :]
        elif kind == 'test':
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
        # and create a sample (n_ts_channels, ts_len); ts: timeseries
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st = obspy.read(fname)  # st: stream
        st_size = st.__len__()
        ts_len = st[0].data.shape[0]
        st_ = np.zeros((st_size, ts_len))  # [24 x 16000]; 16s x 1000Hz
        aux_info = {'norm_stn_ratio': np.array([])}  # stn_ratio: signal-to-noise ratio
        for i, tr in enumerate(st):
            st_[i, :] = tr.data  # (16000,)
            aux_info['norm_stn_ratio'] = np.append(aux_info['norm_stn_ratio'], np.array([signaltonoise(tr.data)]))
        st = st_  # [24 x 16000]
        aux_info['norm_stn_ratio'] = aux_info['norm_stn_ratio'] / np.sum(aux_info['norm_stn_ratio'].reshape(-1))  # (24, )
        aux_info['norm_stn_ratio'] = aux_info['norm_stn_ratio'].reshape(-1, 1)  # [24 x 1]

        # weighted stacking
        st = st * aux_info['norm_stn_ratio']  # (24, 16000)
        st = np.sum(st, axis=0, keepdims=True)  # (1, 16000)

        # convert .dat into a spectrogram
        f, t, Sxx = spectrogram(st, fs=self.sampling_freq, nperseg=self.nperseg,)  # Sxx: spectrogram
        Sxx = np.abs(Sxx)

        # scale
        n_sensors = Sxx.shape[0]
        if self.log_scale:
            epsilon = 1.  # 1e-2  # to prevent noise from being exaggerated by log.
            Sxx = np.log10(Sxx + epsilon)

        if self.minmax_scale:
            flatSxx = Sxx.reshape(n_sensors, -1)  # (24, HW)
            min_ = np.min(flatSxx, axis=1)[:, np.newaxis, np.newaxis]  # (24, 1, 1)
            max_ = np.max(flatSxx, axis=1)[:, np.newaxis, np.newaxis]  # (24, 1, 1)
            Sxx = (Sxx - min_) / (max_ - min_)  # [24 x H x W]

        # resize such that the smaller side (i.e., frequency axis) is adjusted to (e.g., 224).
        if self.resize:
            increase_ratio = (self.resize_size / Sxx.shape[1])
            H, W = int(Sxx.shape[1] * increase_ratio), int(Sxx.shape[2] * increase_ratio)
            Sxx_resized = np.zeros((n_sensors, H, W))  # [24(1) x H x W]
            for i in range(n_sensors):
                Sxx_resized[i, :, :] = cv2.resize(Sxx[i], (W, H))
            Sxx = Sxx_resized  # [24(1) x H x W]

        Sxx = torch.from_numpy(Sxx).float()
        label = torch.tensor(label).long()

        torch.cuda.empty_cache()

        # return st, Sxx, label
        return Sxx, label

    def __len__(self):
        return self._len


def build_datapipeline(config: dict):
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']

    # build datasets
    train_dataset = AknesOriginal(kind='train', **config['dataset'])
    test_dataset = AknesOriginal(kind='test', **config['dataset'])

    # build data_loaders
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=num_workers)
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=num_workers)

    return train_data_loader, test_data_loader


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    os.chdir(root_dir)

    # DataLoader
    data_loader = DataLoader(AknesOriginal(),
                             batch_size=32,
                             shuffle=True,
                             drop_last=True,
                             num_workers=0)
    label_encoder = data_loader.dataset.label_encoder

    # fetch a mini-batch
    for st, Sxx, label in data_loader:
        break
    label_str = label_encoder.inverse_transform(label)

    print('Sxx.shape:', Sxx.shape)
    print('label.shape:', label.shape)
    print('label.shape:', label.shape)

    # visualize
    target_class = 'Regional' #'Slope_Multi' #'Rockfall'
    senosr_idx = 0
    for idx in range(Sxx.shape[0]):
        if label_str[idx] == target_class:
            fig, axes = plt.subplots(2, 1, figsize=(8, 3))

            ax1 = axes[0]
            im1 = ax1.imshow(Sxx[idx, senosr_idx, :, :], aspect='auto')
            ax1.invert_yaxis()
            fig.colorbar(im1, orientation="horizontal")
            ax1.set_title(f'label_str: {label_str[idx]}')

            ax2 = axes[1]
            ax2.plot(st[idx, 0, :])
            plt.tight_layout()
    plt.show()
