import os
import gdown
from zipfile import ZipFile
from utils.get_root_dir import root_dir


def download_dataset():
    url = "https://drive.google.com/u/0/uc?id=1UZ_wAc1Wvv6NfyaxgV5vaps9KT1rvZz3&export=download"
    output = root_dir.joinpath('dataset', 'AknesDataFiles', 'aknes_DataSamples.zip')
    gdown.download(url, str(output))

    with ZipFile(str(output), 'r') as zipObj:
        zipObj.extractall(root_dir.joinpath('dataset', 'AknesDataFiles'))
    os.unlink(str(output))


def check_if_dataset_exists():
    dirname = root_dir.joinpath('dataset', 'AknesDataFiles')
    fnames = os.listdir(dirname)

    if 'DataSamples' not in fnames:
        return False
    else:
        return True


if __name__ == '__main__':
    if not check_if_dataset_exists():
        download_dataset()
