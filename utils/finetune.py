import logging
import torch
import torch.nn as nn
import backbones
from .get_root_dir import root_dir


def load_pretrained_encoder(config_ft: dict, encoder: nn.Module) -> None:
    if config_ft['load_encoder']['ckpt_fname'].lower() == 'none':
        config_ft['exp_params']['LR']['encoder'] = config_ft['exp_params']['LR']['clf']
        logging.info('LR_clf is set to LR_enc since the training starts from scratch.')
    else:
        checkpoint = torch.load(root_dir.joinpath(config_ft['load_encoder']['ckpt_fname']), map_location='cpu')
        encoder.load_state_dict(checkpoint['model_state_dict'])
        print(f'Pre-trained encoder is successfully loaded: {config_ft["load_encoder"]["ckpt_fname"]}')


def build_encoder(config_ft: dict) -> nn.Module:
    encoder = backbones.backbones[config_ft['load_encoder']['backbone_type']]()
    return encoder
