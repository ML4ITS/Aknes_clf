from argparse import ArgumentParser
import logging
from utils.get_root_dir import get_root_dir


def load_pretrained_encoder(config_finetune, encoder, kind: str):
    """
    Load a pretrained encoder
    :param config_finetune:
    :param kind: 'encoder' or 'encoder_ts'
    :return:
    """
    if kind not in ['encoder', 'encoder_ts']:
        raise ValueError
    if config_finetune['ckpt_fname'][kind].lower() != 'none':
        checkpoint = torch.load(get_root_dir().joinpath(config_finetune['ckpt_fname'][kind]), map_location='cpu')
        encoder.load_state_dict(checkpoint['model_state_dict'])
    else:
        config_finetune['exp_params']['LR_enc'][kind] = config_finetune['exp_params']['LR_clf']
        logging.info('LR_clf is set to LR_enc since the training starts from scratch.')


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config_dataset', type=str, help="Path to the dataset config.",
                        default=get_root_dir().joinpath("configs/dataset/norsar_ft.yaml"))
    parser.add_argument('--config_model', type=str, help="Path to the model config",
                        default=get_root_dir().joinpath('configs/models/vibcreg.yaml'))
    parser.add_argument('--config_finetune', type=str, help="Path to the finetune config",
                        default=get_root_dir().joinpath('configs/evaluation/finetune.yaml'))
    return parser.parse_args()


if __name__ == '__main__':
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import LearningRateMonitor
    import torch
    import wandb

    from utils.load_yaml import load_yaml_param_settings
    from utils.build_data_pipeline import build_data_pipeline
    from models import models
    from experiments.finetune_exp import FinetuneExperiment
    # from experiments.finetune_exp2 import FinetuneExperiment
    # from experiments.finetune_exp3 import FinetuneExperiment

    # load configs
    args = load_args()
    config_dataset = load_yaml_param_settings(args.config_dataset)
    config_model = load_yaml_param_settings(args.config_model)
    config_finetune = load_yaml_param_settings(args.config_finetune)

    # data pipeline
    train_data_loader, test_data_loader = build_data_pipeline(config_dataset)

    # build model and load the pre-trained encoder
    encoder = models[config_model['model_params']['name']](**config_model['model_params']).encoder
    encoder_ts = models[config_model['model_params']['name']](**config_model['model_params']).encoder_ts
    load_pretrained_encoder(config_finetune, encoder, kind='encoder')
    load_pretrained_encoder(config_finetune, encoder_ts, kind='encoder_ts')

    # fit
    experiment = FinetuneExperiment(encoder,
                                    config_finetune['exp_params'],
                                    n_train_samples=train_data_loader.dataset.__len__(),
                                    batch_size=config_dataset['batch_size'],
                                    max_epochs=config_finetune['trainer_params']['max_epochs'],
                                    label_encoder=train_data_loader.dataset.label_encoder)
    wandb_logger = WandbLogger(project='norsar-FT',
                               name=None,
                               config={**config_dataset, **config_finetune})
    trainer = pl.Trainer(**config_finetune['trainer_params'],
                         logger=wandb_logger,
                         checkpoint_callback=False,
                         callbacks=[LearningRateMonitor(logging_interval='epoch')],
                         gradient_clip_val=config_finetune['exp_params']['gradient_clip_val'],
                         #strategy='ddp',
                         )
    trainer.fit(experiment, train_dataloaders=train_data_loader, val_dataloaders=test_data_loader)
    wandb.finish()

