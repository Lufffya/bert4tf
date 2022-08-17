# 模型配置文件

import os, sys, json
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.append('../bert4tf')
from bert4tf.backend import tf, keras, K
from bert4tf.tokenizer import Tokenizer
from bert4tf.bert import build_bert_model


config_path_zh = r'models/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path_zh = r'models/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path_zh = r'models/chinese_L-12_H-768_A-12/vocab.txt'

config_path_en = r'models/uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path_en = r'models/uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path_en = r'models/uncased_L-12_H-768_A-12/vocab.txt'

config_path_wwm_zh = r'models/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path_wwm_zh = r'modelschinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path_wwm_zh = r'models/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'
