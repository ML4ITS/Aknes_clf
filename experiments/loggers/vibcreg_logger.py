import torch
import wandb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

from .base_logger import BaseLogger
from utils.types_ import *


class VIbCRegTNCLogger(BaseLogger):
    def __init__(self):
        super(VIbCRegTNCLogger, self).__init__()

    def log(self,
            kind: str,
            outs: Tensor,
            current_epoch: int,
            params: dict,
            y: Tensor = None,
            label: Tensor = None,
            label_encoder: LabelEncoder = None) -> None:
        """
        Logs on wandb
        :param kind: `train` or `validate`
        :param outs: `outs` from `training_epoch_end` in the LightningModule
        :param current_epoch: `self.current_epoch` in the LightningModule
        :param params: `params` in the LightningModule
        :param y: self.y in the LightningModule
        :param label: self.label in the LightningModule
        """

        if kind not in ['train', 'validate']:
            raise ValueError('Type correct `kind`.')

        mean_outs = {k: 0. for k in outs[0].keys()}
        for k in mean_outs.keys():
            for i in range(len(outs)):
                mean_outs[k] += outs[i][k]
            mean_outs[k] /= len(outs)

        # log numerical status
        log_items = {'epoch': current_epoch}
        for k in mean_outs.keys():
            log_items[f'{kind}/{k}'] = mean_outs[k]
        wandb.log(log_items)

        # log kNN accuracy
        if (kind == 'train') and (current_epoch % params['knn_acc_record_ep_period'] == 0):
            y, label = y.numpy(), label.numpy()
            kNN_clf = KNeighborsClassifier(n_neighbors=5, n_jobs=4)
            y_, label_ = [], []
            for i in range(label.shape[0]):
                if label_encoder.inverse_transform([label[i]])[0] != 'Unlabeled':
                    label_.append(label[i])
                    y_.append(y[i, :])
            y, label = np.array(y_), np.array(label_)

            # on `y` - euclidean distance
            kNN_clf.fit(y, label)
            pred_labels = kNN_clf.predict(y)
            kNN_acc = accuracy_score(label, pred_labels)
            wandb.log({'epoch': current_epoch, 'kNN_acc-y': kNN_acc})
