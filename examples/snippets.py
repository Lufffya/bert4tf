# 模型配置文件

import re, os, sys, json
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.append('../bert4tf')
from bert4tf2.backend import tf, keras, K
from bert4tf2.tokenizer import Tokenizer
from bert4tf2.bert import build_bert_model


data_path = r'examples/datasets/'

# BERT中文Base
config_path_zh = r'models/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path_zh = r'models/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path_zh = r'models/chinese_L-12_H-768_A-12/vocab.txt'

# BERT英文Base
config_path_en = r'models/uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path_en = r'models/uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path_en = r'models/uncased_L-12_H-768_A-12/vocab.txt'

# Sim-BERT中文Base
config_path_sim_zh = r'models/chinese_simbert_L-12_H-768_A-12/bert_config.json'
checkpoint_path_sim_zh = r'models/chinese_simbert_L-12_H-768_A-12/bert_model.ckpt'
dict_path_sim_zh = r'models/chinese_simbert_L-12_H-768_A-12/vocab.txt'

# RoBERTa中文全词掩码
config_path_wwm_zh = r'models/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path_wwm_zh = r'modelschinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path_wwm_zh = r'models/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'
