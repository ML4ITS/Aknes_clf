from argparse import ArgumentParser

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from utils import root_dir, load_yaml_param_settings, build_data_pipeline, build_encoder, load_pretrained_encoder
from experiments.finetune_exp import FinetuneExperiment


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config_ft', type=str, help="Path to the finetune config.",
                        default=root_dir.joinpath('configs', 'finetune.yaml'))
    return parser.parse_args()


if __name__ == '__main__':
    # load configs
    args = load_args()
    config_ft = load_yaml_param_settings(args.config_ft)

    # data pipeline
    train_data_loader, test_data_loader = build_data_pipeline(config_ft)

    # build encoder and load the pre-trained encoder
    encoder = build_encoder(config_ft)
    load_pretrained_encoder(config_ft, encoder)

    # fit
    experiment = FinetuneExperiment(encoder,
                                    config_ft,
                                    n_train_samples=train_data_loader.dataset.__len__(),
                                    label_encoder=train_data_loader.dataset.label_encoder)
    wandb_logger = WandbLogger(project='aknes-FT',
                               name=None,
                               config=config_ft)
    trainer = pl.Trainer(**config_ft['trainer_params'],
                         logger=wandb_logger,
                         checkpoint_callback=False,
                         callbacks=[LearningRateMonitor(logging_interval='epoch')],
                         gradient_clip_val=config_ft['exp_params']['gradient_clip_val'],
                         #strategy='ddp',
                         )
    trainer.fit(experiment, train_dataloaders=train_data_loader, val_dataloaders=test_data_loader)
    wandb.finish()
