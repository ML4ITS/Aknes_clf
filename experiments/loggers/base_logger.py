from abc import ABC, abstractmethod
import os

import torch

from utils.get_root_dir import get_root_dir


class BaseLogger(object):
    def __init__(self):
        pass

    @abstractmethod
    def log(self, *args, **kwargs):
        pass

    def save_model(self, current_epoch: int, params: dict, models: dict, dir: str = 'checkpoints'):
        """
        Saves a model checkpoint.
        :param current_epoch: `self.current_epoch` in `pl.LightningModule`
        :param models
        :return:
        """
        current_epoch = current_epoch + 1  # make `epoch` starts from 1 instead of 0
        if current_epoch % params['model_save_ep_period'] == 0:
            if not os.path.isdir(get_root_dir().joinpath('checkpoints')):
                os.mkdir(get_root_dir().joinpath('checkpoints'))

            for k, model in models.items():
                torch.save({
                    'epoch': current_epoch,
                    'model_state_dict': model.state_dict(),
                }, get_root_dir().joinpath(dir, f'{k}-ep_{current_epoch}.ckpt'))

        print(f'# pretrained-model is saved in {dir}.')
