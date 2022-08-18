# 下载SuperGLUE的数据集

import os
import requests
from tqdm import tqdm
from zipfile import ZipFile


DATASET_FOLDER = r'SuperGLUE/datasets/'

SuperGLUE = {
    'AX-b':'https://dl.fbaipublicfiles.com/glue/superglue/data/v2/AX-b.zip',
    'CB':'https://dl.fbaipublicfiles.com/glue/superglue/data/v2/CB.zip',
    'COPA':'https://dl.fbaipublicfiles.com/glue/superglue/data/v2/COPA.zip',
    'MultiRC':'https://dl.fbaipublicfiles.com/glue/superglue/data/v2/MultiRC.zip',
    'RTE':'https://dl.fbaipublicfiles.com/glue/superglue/data/v2/RTE.zip',
    'WiC':'https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WiC.zip',
    'WSC':'https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WSC.zip',
    'BoolQ':'https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip',
    'ReCoRD':'https://dl.fbaipublicfiles.com/glue/superglue/data/v2/ReCoRD.zip',
    'AX-g':'https://dl.fbaipublicfiles.com/glue/superglue/data/v2/AX-g.zip',
}


def download():
    if not os.path.exists(DATASET_FOLDER):
        os.makedirs(DATASET_FOLDER)
    for name, url in tqdm(SuperGLUE.items()):
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
