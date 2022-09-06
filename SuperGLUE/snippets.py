# SuperGLUE评测
# 模型配置文件

import numpy as np
import os, sys, csv, json
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
data_path = r'SuperGLUE/datasets/'
learning_rate = 4e-5
pooling = 'first'

# 权重目录
# if not os.path.exists('weights'):
#     os.mkdir('weights')

# 输出目录
# if not os.path.exists('results'):
#     os.mkdir('results')

# 模型路径
config_path = r'models/uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = r'models/uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = r'models/uncased_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 预训练模型
base = build_bert_model(config_path, checkpoint_path, application='unilm', return_keras_model=False)

# 模型参数
last_layer = 'Transformer-%s-FeedForward-Norm' % (base.num_hidden_layers - 1)

if pooling == 'first':
    pooling_layer = keras.layers.Lambda(lambda x: x[:, 0])
elif pooling == 'avg':
    pooling_layer = keras.layers.GlobalAveragePooling1D()
elif pooling == 'max':
    pooling_layer = keras.layers.GlobalMaxPooling1D()
