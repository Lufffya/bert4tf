#! -*- coding: utf-8 -*-
# bert 基础测试: 结合MLM的Gibbs采样

import sys
import numpy as np
from tqdm import tqdm
sys.path.append('../bert4tf')
from bert4tf.bert import build_bert_model
from bert4tf.tokenizer import Tokenizer
from bert4tf.snippets import to_array


config_path = '/home/zxc/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/zxc//chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/zxc/chinese_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
model = build_bert_model(config_path=config_path, checkpoint_path=checkpoint_path, with_mlm=True)  # 建立模型, 加载权重

sentences = []
# init_sent = u'科学技术是第一生产力。'  # 给定句子或者None
init_sent = None  # 给定句子或者None
minlen, maxlen = 8, 32
steps = 1000
converged_steps = 100
vocab_size = tokenizer._vocab_size

if init_sent is None:
    length = np.random.randint(minlen, maxlen + 1)
    tokens = ['[CLS]'] + ['[MASK]'] * length + ['[SEP]']
    token_ids = tokenizer.tokens_to_ids(tokens)
    segment_ids = [0] * len(token_ids)
else:
    token_ids, segment_ids = tokenizer.encode(init_sent)
    length = len(token_ids) - 2

for _ in tqdm(range(steps), desc='Sampling'):
    # Gibbs采样流程: 随机mask掉一个token, 然后通过MLM模型重新采样这个token
    i = np.random.choice(length) + 1
    token_ids[i] = tokenizer._token_mask_id
    probas = model.predict(to_array([token_ids], [segment_ids]))[0, i]
    token = np.random.choice(vocab_size, p=probas)
    token_ids[i] = token
    sentences.append(tokenizer.decode(token_ids))

print(u'部分随机采样结果:')
for _ in range(100):
    print(np.random.choice(sentences[converged_steps:]))