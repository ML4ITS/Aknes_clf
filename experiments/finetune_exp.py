import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import root_dir
from utils.types_ import *


class Classifier(nn.Module):
    def __init__(self,
                 in_size: int,
                 n_classes: int = 8):
        super().__init__()
        self.model = nn.Linear(in_size, n_classes)
        # self.model = nn.Sequential(nn.Dropout(0.5),
        #                            nn.Linear(in_size, n_classes))

    def forward(self, input: Tensor) -> Tensor:
        return self.model(input)


class GiniImpurity(object):
    def __init__(self, momentum: float = 0.1):
        self.momentum = momentum

        # running_(..) gets updated by exponential moving average
        self.running_mean = 0.5  # initialized
        self.running_std = 0.1  # initialized

    def compute_gini(self, softmax_output: Tensor) -> Tensor:
        """
        :param softmax_output: probabilistic output after the softmax; (B, n_classes)
        """
        impurities = 1 - torch.sum(softmax_output ** 2, dim=1)  # (B, )
        return impurities

    def update_stats(self, impurities: Tensor) -> None:
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * torch.mean(impurities)
        self.running_std = (1 - self.momentum) * self.running_std + self.momentum * torch.std(impurities)


def compute_balanced_class_weight(label_encoder: LabelEncoder) -> Tensor:
    """
    Following (https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)
    NB! Noise class is ignored.
    """
    workdir = root_dir.joinpath('dataset', 'AknesDataFiles', 'DataSamples')
    dirnames = [item for item in os.listdir(workdir)
                if os.path.isdir(workdir.joinpath(item)) and (item != 'Unlabeled')]

    sample_count = {}
    for dirname in dirnames:
        if dirname == 'Noise':
            continue
        label_num_idx = label_encoder.transform([dirname])[0]
        sample_count[label_num_idx] = len(os.listdir(workdir.joinpath(dirname)))
    sample_count = dict(sorted(sample_count.items()))

    n_classes = len(sample_count.keys())
    sample_count_values = np.array(list(sample_count.values()))
    n_samples = np.sum(sample_count_values)
    class_weight = n_samples / (n_classes * sample_count_values)
    class_weight = np.concatenate(([0.], class_weight))  # 0. for the Noise class
    return torch.from_numpy(class_weight).float()


class FinetuneExperiment(pl.LightningModule):
    def __init__(self,
                 encoder: nn.Module,
                 config_ft: dict,
                 n_train_samples: int = None,
                 label_encoder: LabelEncoder = None):
        super().__init__()
        self.encoder = encoder
        self.config_ft = config_ft
        self.params = config_ft['exp_params']
        batch_size = config_ft['dataset']['batch_size']
        max_epochs = config_ft['trainer_params']['max_epochs']
        self.T_max = max_epochs * np.ceil(n_train_samples / batch_size)  # Maximum number of iterations
        self.label_encoder = label_encoder

        self.classifier = Classifier(in_size=self.params['out_size_enc'], n_classes=self.params['n_classes'])
        class_weight = compute_balanced_class_weight(label_encoder) if self.params['use_class_weight'] else None
        print('class_weight:', class_weight)
        self.criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=self.params['label_smoothing'])

        self.gini = GiniImpurity()

        if self.params['freeze_encoders']:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, Sxx: Tensor) -> Tensor:
        return self.encoder(Sxx)

    def training_step(self, batch, batch_idx):
        self.encoder.eval() if self.params['freeze_encoders'] or self.params['freeze_bn_stat_train'] else self.encoder.train()
        self.classifier.train()

        # Sxx, label = batch  # Sxx: (B, n_sensors, H, W)
        Sxx, label = batch
        Sxx = Sxx.float()
        label = label.long()

        y = self.forward(Sxx)  # (B, D)
        out = self.classifier(y)  # (B, 8)
        loss = self.criterion(out, label)

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        return {'loss': loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.encoder.eval()
        self.classifier.eval()

        Sxx, label = batch  # Sxx: (B, n_sensors, H, W)
        Sxx = Sxx.float()
        label = label.long().cpu()

        # stack representations along the sensor-axis
        n_sensors = Sxx.shape[1]
        ys = None
        for sensor_idx in range(n_sensors):
            Sxx_single_sensor = Sxx[:, [sensor_idx], :, :]  # (B, H, W)
            y = self.forward(Sxx_single_sensor).detach()  # (B, D)

            if ys is None:
                ys = torch.zeros(y.shape[0], n_sensors, y.shape[-1]).float().to(Sxx.device)  # [B x 24 x D]
            ys[:, sensor_idx, :] = y

        # classifier
        avg_prob_pred = torch.zeros(ys.shape[0], self.params['n_classes']).float()  # (B, 8)
        for sensor_idx in range(n_sensors):
            out = self.classifier(ys[:, sensor_idx, :]).detach()  # (B, 8)
            # out = nn.functional.softmax(out, dim=1)
            out = nn.functional.log_softmax(out, dim=1).cpu()

            # impurities = self.gini.compute_gini(out)[:, None]  # (B, 1)
            # out = out * impurities
            # print('impurities:', impurities)

            avg_prob_pred += out
        avg_prob_pred /= n_sensors

        # gini
        # impurities = self.gini.compute_gini(avg_prob_pred)  # (B, )
        # self.gini.update_stats(impurities)
        # noise_label_indices = impurities > (self.gini.running_mean + 1 * self.gini.running_std)
        # noise_num_label = self.label_encoder.transform(['Noise'])[0]
        # if torch.sum(noise_label_indices) != 0:
        #     noise_event_prob = torch.zeros(torch.sum(noise_label_indices), len(self.label_encoder.classes_)).to(avg_prob_pred.device)
        #     noise_event_prob[:, noise_num_label] = 1.
        #     avg_prob_pred[noise_label_indices, :] = noise_event_prob
        # print(torch.cat((label.reshape(-1, 1), impurities.reshape(-1, 1), noise_label_indices.reshape(-1, 1)), dim=1))

        # compute predictions
        pred_label = torch.argmax(avg_prob_pred, dim=-1)  # (B,)
        pred_result = (label == pred_label)  # (B,)

        # loss
        avg_prob_pred = np.exp(avg_prob_pred)  # to make it be like an output after `softmax(out)` instead of `log_softmax(out)`
        loss = -torch.log(avg_prob_pred[range(len(label)), label]).mean()

        return {'pred_result': pred_result,
                'label': label,
                'pred_label': pred_label,
                'loss': loss}

    def configure_optimizers(self):
        opt = torch.optim.Adam([{'params': self.encoder.parameters(), 'lr': self.params['LR']['encoder']},
                                {'params': self.classifier.parameters(), 'lr': self.params['LR']['clf']}],
                               weight_decay=self.params['weight_decay'])
        return {"optimizer": opt, "lr_scheduler": CosineAnnealingLR(opt, self.T_max)}

    def training_epoch_end(self, outs) -> None:
        mean_outs = {'loss': 0.}
        for out in outs:
            mean_outs['loss'] += out['loss'].item()
        for k in mean_outs.keys():
            mean_outs[k] /= len(outs)

        # log
        log_items = {'epoch': self.current_epoch,
                     'train/loss': mean_outs['loss']}
        for k, v in log_items.items():
            self.log(k, v)

    def validation_epoch_end(self, outs) -> None:
        pred_results = np.array([])
        labels = np.array([])
        pred_labels = np.array([])
        mean_outs = {'loss': 0.}
        for out in outs:
            pred_results = np.append(pred_results, out['pred_result'].numpy())
            labels = np.append(labels, out['label'].numpy())
            pred_labels = np.append(pred_labels, out['pred_label'].numpy())
            mean_outs['loss'] += out['loss'].item()
        pred_acc = np.mean(pred_results)
        for k in mean_outs.keys():
            mean_outs[k] /= len(outs)

        # log
        log_items = {'epoch': self.current_epoch,
                     'validate/pred_acc': pred_acc,
                     'validate/loss': mean_outs['loss']}
        for k, v in log_items.items():
            self.log(k, v)

        # log confusion matrix
        try:
            cmat = confusion_matrix(labels, pred_labels)
            class_names = list(self.label_encoder.classes_)
            df_cmat = pd.DataFrame(cmat, index=class_names, columns=class_names).astype(int)
            heatmap = sns.heatmap(df_cmat, annot=True, fmt='d')
            heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
            heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.title(f'epoch: {self.current_epoch+1}')
            plt.tight_layout()
            wandb.log({'val_confusion_mat': wandb.Image(plt)})
            plt.close()
        except ValueError:
            # this can happen during fine-tuning with a small dataset
            pass
