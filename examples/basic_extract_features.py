# bert 基础测试: 检查word embeddings
# 如输入相同, 则表示bert权重加载正确

from snippets import *
from bert4tf.snippets import to_array


def bert_tf_embedding():
    tokenizer = Tokenizer(dict_path_zh, do_lower_case=True)  # 建立分词器
    model = build_bert_model(config_path_zh, checkpoint_path_zh)  # 建立模型, 加载权重

    # 编码测试
    token_ids, segment_ids = tokenizer.encode(u'语言模型')
    token_ids, segment_ids = to_array([token_ids], [segment_ids])

    print('\n ===== predicting =====\n')
    print(model.predict([token_ids, segment_ids]))
    #   """
    # 输出:
    # [[[-0.63251007  0.2030236   0.07936534 ...  0.49122632 -0.20493352
    #     0.2575253 ]
    #   [-0.7588351   0.09651865  1.0718756  ... -0.6109694   0.04312154
    #     0.03881441]
    #   [ 0.5477043  -0.792117    0.44435206 ...  0.42449304  0.41105673
    #     0.08222899]
    #   [-0.2924238   0.6052722   0.49968526 ...  0.8604137  -0.6533166
    #     0.5369075 ]
    #   [-0.7473459   0.49431565  0.7185162  ...  0.3848612  -0.74090636
    #     0.39056838]
    #   [-0.8741375  -0.21650358  1.338839   ...  0.5816864  -0.4373226
    #     0.56181806]]]
    # """

    print('\n ===== reloading and predicting =====\n')
    model.save('bert_test_model')
    del model
    model = keras.models.load_model('bert_test_model')
    print(model.predict([token_ids, segment_ids]))


def tensorflow_hub_bert_embedding():
    import tensorflow_hub as hub
    import tensorflow_text as text  # Imports TF ops for preprocessing.

    BERT_MODEL = 'https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/4'
    PREPROCESS_MODEL = 'https://tfhub.dev/tensorflow/bert_zh_preprocess/3'

    sentences = [u'语言模型']

    preprocess = hub.load(PREPROCESS_MODEL)
    bert = hub.load(BERT_MODEL)
    inputs = preprocess(sentences)
    outputs = bert(inputs)

    print("Sentences:")
    print(sentences)

    print("\nBERT inputs:")
    print(inputs)

    print("\nPooled embeddings:")
    print(outputs["pooled_output"])

    print("\nPer token embeddings:")
    print(outputs["sequence_output"])


if __name__ == '__main__':
    bert_tf_embedding()
    # tensorflow_hub_bert_embedding()
    print('=== complete ===')
