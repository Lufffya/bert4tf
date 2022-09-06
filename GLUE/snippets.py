# GLUE 评测
# 模型配置文件

import numpy as np
import os, sys, csv
from tqdm import tqdm
from download import download
sys.path.append('../bert4tf')
from bert4tf2.backend import set_gelu
from bert4tf2.backend import tf, keras, K
from bert4tf2.tokenizer import Tokenizer
from bert4tf2.bert import build_bert_model
from bert4tf2.snippets import sequence_padding
from bert4tf2.snippets import DataGenerator


download()

# 通用参数
data_path = r'GLUE/datasets/'

# 模型路径
config_path = r'models/uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = r'models/uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = r'models/uncased_L-12_H-768_A-12/vocab.txt'
