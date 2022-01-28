from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from preprocessing.augmentations import augmentations
from preprocessing.preprocess import NORSARDataset
from .types_ import *


def build_data_pipeline(config: dict) -> Tuple[DataLoader, DataLoader]:
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']

    # build transforms
    augs = []
    for aug_name in config['augmentations']:
        augs.append(augmentations[aug_name]())
    transforms = Compose(augs)

    # build datasets
    train_dataset = NORSARDataset(transform=transforms,
                                  return_single_spectrogram=config['dataset']['return_single_spectrogram_train'],
                                  kind='train',
                                  **config['dataset'])
    test_dataset = NORSARDataset(transform=transforms,
                                 return_single_spectrogram=config['dataset']['return_single_spectrogram_test'],
                                 kind='test',
                                 **config['dataset'])

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
