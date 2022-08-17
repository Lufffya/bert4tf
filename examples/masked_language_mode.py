#! -*- coding: utf-8 -*-
# bert 基础测试: MLM

from snippets import *
from bert4tf.snippets import to_array


# 建立分词器
tokenizer = Tokenizer(dict_path_zh, do_lower_case=True)

# 建立模型, 加载权重
model = build_bert_model(config_path=config_path_zh, checkpoint_path=checkpoint_path_zh, with_mlm=True)

token_ids, segment_ids = tokenizer.encode(u'科学技术是第一生产力')

# mask掉“技术”
token_ids[3] = token_ids[4] = tokenizer._token_mask_id
token_ids, segment_ids = to_array([token_ids], [segment_ids])

# 用mlm模型预测被mask掉的部分
probas = model.predict([token_ids, segment_ids])[0]
print(tokenizer.decode(probas[3:5].argmax(axis=1)))  # 结果正是“技术”
