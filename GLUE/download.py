# 下载GLUE的数据集

import os
import requests
from tqdm import tqdm
from zipfile import ZipFile


DATASET_FOLDER = r'GLUE/datasets/'

GLUE = {
    'CoLA':'https://dl.fbaipublicfiles.com/glue/data/CoLA.zip',
    'SST-2':'https://dl.fbaipublicfiles.com/glue/data/SST-2.zip',
    'STS-B':'https://dl.fbaipublicfiles.com/glue/data/STS-B.zip',
    'QQP':'https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip',
    'MNLI':'https://dl.fbaipublicfiles.com/glue/data/MNLI.zip',
    'QNLI':'https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip',
    'RTE':'https://dl.fbaipublicfiles.com/glue/data/RTE.zip',
    'WNLI':'https://dl.fbaipublicfiles.com/glue/data/WNLI.zip',
}


def download():
    if not os.path.exists(DATASET_FOLDER):
        os.makedirs(DATASET_FOLDER)
    for name, url in tqdm(GLUE.items()):
        if os.path.exists(DATASET_FOLDER + name):
            continue
        print('downloading {}'.format(name))
        r = requests.get(url)
        with open(DATASET_FOLDER + name + '.zip', 'wb') as f:
            f.write(r.content)
        with ZipFile(DATASET_FOLDER + name + '.zip', 'r') as zip_ref:
            zip_ref.extractall(DATASET_FOLDER)
        os.remove(DATASET_FOLDER + name + '.zip')


if __name__ == '__main__':
    download()
