#! -*- coding: utf-8 -*-
# GLUE 评测
# 模型配置文件

import numpy as np
import os, sys, csv
from tqdm import tqdm
sys.path.append('../bert4tf')
from bert4tf.backend import set_gelu
from bert4tf.backend import tf, keras, K
from bert4tf.tokenizer import Tokenizer
from bert4tf.bert import build_bert_model
from bert4tf.snippets import sequence_padding
from bert4tf.snippets import DataGenerator
from bert4tf.layers import Dropout, Lambda, Dense
from bert4tf.optimizers import Adam


# 模型路径
config_path = r'/home/zxc/uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = r'/home/zxc//uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = r'/home/zxc/uncased_L-12_H-768_A-12/vocab.txt'

# 通用参数
data_path = r'GLUE/datasets/'
