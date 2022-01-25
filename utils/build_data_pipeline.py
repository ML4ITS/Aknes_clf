from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from preprocessing.augmentations import augmentations
from preprocessing.preprocess import NORSARDataset
from .types_ import *


def build_data_pipeline(config_ssl: dict) -> Tuple[DataLoader, DataLoader]:
    batch_size = config_ssl['dataset']['batch_size']
    num_workers = config_ssl['dataset']['num_workers']

    # build transforms
    augs = []
    for aug_name in config_ssl['augmentations']:
        augs.append(augmentations[aug_name]())
    transforms = Compose(augs)

    # build datasets
    train_dataset = NORSARDataset(transform=transforms,
                                  return_single_spectrogram=config_ssl['dataset']['return_single_spectrogram_train'],
                                  kind='train',
                                  **config_ssl['dataset'])
    test_dataset = NORSARDataset(transform=transforms,
                                 return_single_spectrogram=config_ssl['dataset']['return_single_spectrogram_test'],
                                 kind='test',
                                 **config_ssl['dataset'])

    # build data_loaders
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=num_workers,
                                   pin_memory=True if num_workers > 0 else False)
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=num_workers,
                                  pin_memory=True if num_workers > 0 else False)

    return train_data_loader, test_data_loader
