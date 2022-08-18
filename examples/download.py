# 下载examples的数据集

import os
import requests
from tqdm import tqdm
from zipfile import ZipFile


DATASET_FOLDER = r'examples/datasets/'

EXAMPLES = {

}


def download():
    if not os.path.exists(DATASET_FOLDER):
        os.makedirs(DATASET_FOLDER)
    for name, url in tqdm(EXAMPLES.items()):
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
