# 下载CLUE的数据集

import os
import requests
from tqdm import tqdm
from zipfile import ZipFile


DATASET_FOLDER = r'CLUE/datasets/'

CLUE = {
    'afqmc':'https://storage.googleapis.com/cluebenchmark/tasks/afqmc_public.zip',
    'c3':'https://storage.googleapis.com/cluebenchmark/tasks/c3_public.zip',
    'chid':'https://storage.googleapis.com/cluebenchmark/tasks/chid_public.zip',
    'cluener':'https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip',
    'cmnli':'https://storage.googleapis.com/cluebenchmark/tasks/cmnli_public.zip',
    'cmrc2018':'https://storage.googleapis.com/cluebenchmark/tasks/cmrc2018_public.zip',
    'csl':'https://storage.googleapis.com/cluebenchmark/tasks/csl_public.zip',
    'iflytek':'https://storage.googleapis.com/cluebenchmark/tasks/iflytek_public.zip',
    'ocnli':'https://storage.googleapis.com/cluebenchmark/tasks/ocnli_public.zip',
    'tnews':'https://storage.googleapis.com/cluebenchmark/tasks/tnews_public.zip',
    'wsc':'https://storage.googleapis.com/cluebenchmark/tasks/cluewsc2020_public.zip',
}


def download():
    if not os.path.exists(DATASET_FOLDER):
        os.makedirs(DATASET_FOLDER)
    for name, url in tqdm(CLUE.items()):
        if os.path.exists(DATASET_FOLDER + name):
            continue
        print('downloading {}'.format(name))
        r = requests.get(url)
        with open(DATASET_FOLDER + name + '.zip', 'wb') as f:
            f.write(r.content)
        with ZipFile(DATASET_FOLDER + name + '.zip', 'r') as zip_ref:
            zip_ref.extractall(DATASET_FOLDER + name)
        os.remove(DATASET_FOLDER + name + '.zip')


if __name__ == '__main__':
    download()
