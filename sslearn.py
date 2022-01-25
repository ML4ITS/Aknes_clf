"""
sslearn: self-supervised learning
"""
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb

from utils import root_dir, load_yaml_param_settings, build_data_pipeline, check_if_dataset_exists, download_dataset
from experiments.loggers import exp_loggers
from experiments import experiments


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config_ssl', type=str, help="Path to the dataset config.",
                        default=root_dir.joinpath('configs', 'ssl.yaml'))
    return parser.parse_args()


if __name__ == '__main__':
    # check if the dataset exists. If not, download one.
    if not check_if_dataset_exists():
        download_dataset()

    # load configs
    args = load_args()
    config_ssl = load_yaml_param_settings(args.config_ssl)

    # data pipeline
    train_data_loader, test_data_loader = build_data_pipeline(config_ssl)

    # pl-experiment & pl-trainer
    exp_logger_ = exp_loggers[config_ssl['model_params']['name']]()
    experiment = experiments[config_ssl['model_params']['name']](config_ssl,
                                                                 exp_logger_,
                                                                 n_train_samples=train_data_loader.dataset.__len__(),
                                                                 label_encoder=train_data_loader.dataset.label_encoder)
    wandb_logger = WandbLogger(project='aknes-SSL',
                               name=f'{config_ssl["model_params"]["name"]}',
                               config=config_ssl)
    trainer = pl.Trainer(**config_ssl['trainer_params'],
                         logger=wandb_logger,
                         checkpoint_callback=False,
                         callbacks=[LearningRateMonitor(logging_interval='epoch')],
                         gradient_clip_val=config_ssl['exp_params']['gradient_clip_val'])
    print(f"======= Training {config_ssl['model_params']['name']}=======")
    trainer.fit(experiment, train_dataloaders=train_data_loader, val_dataloaders=test_data_loader)
    wandb.finish()
