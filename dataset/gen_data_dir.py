"""
Generates a csv file that states:
1) file names (.dat files)
2) event

The generated csv file is used in the `Dataset`.
"""
import tempfile
import shutil
import os
import pandas as pd
from pathlib import Path


def gen_dat_dir():
    df = pd.DataFrame(columns=['fname', 'event'])

    for root, dirs, files in os.walk(os.path.join('.', 'AknesDataFiles', 'DataSamples')):
        for file in files:
            if '.dat' in file:
                fname = Path(os.path.join('dataset', root, file))
                event = str(fname.parent).split(os.sep)[3]

                df = df.append({'fname': fname,
                                'event': event},
                               ignore_index=True)

    df.to_csv(os.path.join('.', 'data_dir.csv'), index=False)


if __name__ == '__main__':
    gen_dat_dir()
