import tempfile
from pathlib import Path
import os

import pytorch_lightning as pl
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder

from models.vibcreg import VIbCReg
from models.tnc import Discriminator as DiscriminatorTNC
from utils.types_ import *


class VIbCRegTNCExperiment(pl.LightningModule):
    def __init__(self,
                 config_ssl: dict,
                 exp_logger=None,
                 n_train_samples: int = None,
                 label_encoder: LabelEncoder = None):
        super().__init__()

        self.config_ssl = config_ssl
        self.params = config_ssl['exp_params']
        self.exp_logger = exp_logger
        batch_size = config_ssl['dataset']['batch_size']
        max_epochs = config_ssl['trainer_params']['max_epochs']
        self.T_max = max_epochs * np.ceil(n_train_samples / batch_size)  # Maximum number of iterations
        self.label_encoder = label_encoder

        self.vibcreg = VIbCReg(**config_ssl['model_params'])
        self.D_tnc = DiscriminatorTNC(input_size=config_ssl['model_params']['out_size_enc'])

        self.train_y = None
        self.train_label = None

        self.temp_dirpath = Path(tempfile.mkdtemp()).parent.joinpath('checkpoints')
        if not os.path.isdir(self.temp_dirpath):
            os.mkdir(self.temp_dirpath)

    def forward(self, *args):
        return self.vibcreg(*args)

    def training_step(self, batch, batch_idx):
        self.vibcreg.train()
        self.D_tnc.train()

        (Sxx_view, Sxx_view_l, Sxx_view_k), label = batch  # Sxx: (1, H, W)

        # forward
        y, z = self.forward(Sxx_view)  # y: reference repr
        y_l, z_l = self.forward(Sxx_view_l)  # y_l: neighboring repr
        y_k = self.vibcreg.encoder(Sxx_view_k)  # y_k: non-neighboring repr

        # loss: vibcreg
        loss_hist = {}
        vibcreg_loss = self.vibcreg.loss_function(z, z_l, self.params, loss_hist)
        loss_hist['loss'] = vibcreg_loss

        # loss: TNC
        tnc_loss = self.D_tnc.loss_function(y, y_l, y_k,
                                            self.params['use_diag_loss'],
                                            loss_hist)
        loss_hist['loss'] += self.params['rho'] * tnc_loss

        # get some data for tracking training status
        if batch_idx in [0, 1, 2, 3]:
            if batch_idx == 0:
                self.train_y = y.detach().cpu()  # `z` == `mu` [B x latent_dim]
                self.train_label = label.view(-1).cpu()  # [B]
            else:
                self.train_y = torch.cat((self.train_y, y.detach().cpu()), dim=0)
                self.train_label = torch.cat((self.train_label, label.view(-1).cpu()), dim=0)

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        for k in loss_hist.keys():
            if k != 'loss':
                try:
                    loss_hist[k] = loss_hist[k].detach()
                except AttributeError:
                    pass

        return loss_hist

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.vibcreg.eval()
        self.D_tnc.eval()

        (Sxx_view, Sxx_view_l, Sxx_view_k), label = batch  # Sxx: (1, H, W)

        # forward
        y, z = self.forward(Sxx_view)  # y: reference repr
        y_l, z_l = self.forward(Sxx_view_l)  # y_l: neighboring repr
        y_k = self.vibcreg.encoder(Sxx_view_k)  # y_k: non-neighboring repr

        # loss: vibcreg
        loss_hist = {}
        vibcreg_loss = self.vibcreg.loss_function(z, z_l, self.params, loss_hist)
        loss_hist['loss'] = vibcreg_loss

        # loss: TNC
        tnc_loss = self.D_tnc.loss_function(y, y_l, y_k,
                                            self.params['use_diag_loss'],
                                            loss_hist)
        loss_hist['loss'] += self.params['rho'] * tnc_loss

        for k in loss_hist.keys():
            if k != 'loss':
                try:
                    loss_hist[k] = loss_hist[k].detach()
                except AttributeError:
                    pass

        return loss_hist

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{'params': self.vibcreg.parameters()},
                                      {'params': self.D_tnc.parameters()}],
                                     lr=self.params['LR'],
                                     weight_decay=self.params['weight_decay'])
        return {"optimizer": optimizer,
                "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.T_max)}

    def training_epoch_end(self, outs) -> None:
        self.exp_logger.log('train', outs, self.current_epoch, self.params, self.train_y, self.train_label, self.label_encoder)
        save_models = {}
        save_models['encoder'] = self.vibcreg.encoder
        try:
            self.exp_logger.save_model(self.current_epoch, self.params, save_models)
        except PermissionError:
            self.exp_logger.save_model(self.current_epoch, self.params, save_models, self.temp_dirpath)

    def validation_epoch_end(self, outs) -> None:
        self.exp_logger.log('validate', outs, self.current_epoch, self.params)
