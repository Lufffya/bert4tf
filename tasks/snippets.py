import os, sys
sys.path.append('../bert4tf')
from bert4tf.backend import tf, keras, K
from bert4tf.tokenizer import Tokenizer
from bert4tf.bert import build_bert_model


config_path = r'/home/zxc/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = r'/home/zxc//chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = r'/home/zxc/chinese_L-12_H-768_A-12/vocab.txt'
