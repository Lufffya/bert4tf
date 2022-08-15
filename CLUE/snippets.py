# CLUE 评测 - 模型配置文件

import os, sys
from download import download
sys.path.append('../bert4tf')
from bert4tf.backend import tf, keras, K
from bert4tf.tokenizer import Tokenizer
from bert4tf.bert import build_bert_model
from bert4tf.optimizers import Adam, AdaFactor
from bert4tf.optimizers import extend_with_gradient_accumulation


download()

# 通用参数
data_path = r'CLUE/datasets/'
learning_rate = 5e-4
pooling = 'first'

# 权重目录
# if not os.path.exists(r'CLUE/weights'):
#     os.mkdir(r'CLUE/weights')

# 输出目录
# if not os.path.exists(r'CLUE/results'):
#     os.mkdir(r'CLUE/results')

# 模型路径
config_path = r'models/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = r'models/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = r'models/chinese_L-12_H-768_A-12/vocab.txt'

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

# 优化器
AdaFactorG = extend_with_gradient_accumulation(AdaFactor, name='AdaFactorG')

optimizer = AdaFactor(learning_rate=learning_rate, beta1=0.9, min_dim_size_to_factor=10**6)

optimizer2 = AdaFactorG(
    learning_rate=learning_rate,
    beta1=0.9,
    min_dim_size_to_factor=10**6,
    grad_accum_steps=2
)

optimizer4 = AdaFactorG(
    learning_rate=learning_rate,
    beta1=0.9,
    min_dim_size_to_factor=10**6,
    grad_accum_steps=4
)
