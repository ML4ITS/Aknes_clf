from argparse import ArgumentParser
import pytorch_lightning as pl
import numpy as np
import wandb
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from experiments.finetune_exp import compute_balanced_class_weight
from utils.types_ import *


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--configs', type=str, help="Path to the dataset config.",
                        default='config.yaml')
    return parser.parse_args()


class Experiment(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 config: dict,
                 n_train_samples: int = None,
                 label_encoder: LabelEncoder = None,
                 ):
        super().__init__()
        self.model = model
        self.config = config
        batch_size = config['dataset']['batch_size']
        max_epochs = config['trainer_params']['max_epochs']
        self.T_max = max_epochs * np.ceil(n_train_samples / batch_size)  # Maximum number of iterations
        self.label_encoder = label_encoder

        class_weight = compute_balanced_class_weight(label_encoder) if self.config['exp_params']['use_class_weight'] else None
        print('# class_weight:', class_weight)
        self.criterion = nn.CrossEntropyLoss(weight=class_weight)

    def forward(self, input: Tensor) -> Tensor:
        return self.model(input)  # [B, 8]

    def training_step(self, batch, batch_idx) -> dict:
        self.model.train()

        Sxx, label = batch  # Sxx: [B x 1 x H x W]

        # propagate
        out = self.forward(Sxx)  # [B x 8]

        # loss
        loss = self.criterion(out, label)

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        return {'loss': loss}

    def validation_step(self, batch, batch_idx) -> dict:
        self.model.eval()

        Sxx, label = batch  # Sxx: [B x 1 x H x W]

        # propagate
        out = self.forward(Sxx)  # [B x 8]

        # loss
        loss = self.criterion(out, label)

        # compute predictions
        pred_label = torch.argmax(nn.functional.softmax(out.detach(), dim=1), dim=1)
        pred_result = (label == pred_label)  # (B,)

        return {'loss': loss.detach(),
                'pred_result': pred_result.cpu(),
                'label': label.cpu(),
                'pred_label': pred_label.cpu(),
                }

    def configure_optimizers(self) -> dict:
        opt = torch.optim.Adam([{'params': self.model.parameters(), 'lr': self.config['exp_params']['LR']}
                                ], weight_decay=self.config['exp_params']['weight_decay'])
        return {"optimizer": opt, "lr_scheduler": CosineAnnealingLR(opt, self.T_max)}

    def training_epoch_end(self, outs) -> None:
        mean_outs = {k: 0. for k in outs[0].keys()}
        for k in mean_outs.keys():
            for i in range(len(outs)):
                mean_outs[k] += outs[i][k]
            mean_outs[k] /= len(outs)

        # log
        log_items = {'epoch': self.current_epoch,
                     'train/loss': mean_outs['loss']}
        wandb.log(log_items)

    def validation_epoch_end(self, outs) -> None:
        mean_outs = {'loss': 0.}
        pred_results = np.array([])
        labels = np.array([])
        pred_labels = np.array([])
        for out in outs:
            mean_outs['loss'] += out['loss'].item()
            pred_results = np.append(pred_results, out['pred_result'].numpy())
            labels = np.append(labels, out['label'].numpy())
            pred_labels = np.append(pred_labels, out['pred_label'].numpy())
        for k in mean_outs.keys():
            mean_outs[k] /= len(outs)

        pred_acc = np.mean(pred_results)

        # log
        log_items = {'epoch': self.current_epoch,
                     'valid/loss': mean_outs['loss'],
                     'valid/pred_acc': pred_acc}
        wandb.log(log_items)

        # log confusion matrix
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
        log_items = {'epoch': self.current_epoch,
                     'val_confusion_mat': wandb.Image(plt)}
        wandb.log(log_items)
        plt.close()


if __name__ == '__main__':
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import LearningRateMonitor

    from utils.load_yaml import load_yaml_param_settings
    from preprocess import build_datapipeline
    from backbones import backbones_with_clf

    # load configs
    args = load_args()
    configs = load_yaml_param_settings(args.configs)

    # data pipeline
    train_data_loader, test_data_loader = build_datapipeline(configs)

    # build model (encoder & classifier)
    model = backbones_with_clf[configs['backbone_with_clf_type']]()

    # pl-experiment & pl-trainer
    experiment = Experiment(model,
                            configs,
                            n_train_samples=len(train_data_loader.dataset),
                            label_encoder=train_data_loader.dataset.label_encoder)
    wandb_logger = WandbLogger(project='aknes-original',
                               name=None,
                               config=configs)
    trainer = pl.Trainer(**configs['trainer_params'],
                         logger=wandb_logger,
                         checkpoint_callback=False,
                         callbacks=[LearningRateMonitor(logging_interval='epoch')])
    trainer.fit(experiment, train_dataloaders=train_data_loader, val_dataloaders=test_data_loader)
    wandb.finish()
