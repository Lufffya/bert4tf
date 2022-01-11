#! -*- coding:utf-8 -*-
# SST-2: The Stanford Sentiment Treebank(斯坦福情感树库)
# describe: 情感分析, 判断句子所表述的情感(积极或消极)
# metric: accuracy

from snippets import *


num_classes = 2
maxlen = 128
batch_size = 32


def load_data(filename):
    """加载数据
    单条格式: (文本, 标签id)
    """
    D = []
    i = 1
    with open(filename, encoding='utf-8') as f:
        for l in f:
            if i == 1:  # 跳过数据第一行
                i = 2
            else:
                text, label = l.strip().split('\t')
                D.append((text, int(label)))
    return D


def load_data_test(filename):
    """加载test数据
    单条格式: (文本, 标签id)
    """
    D = []
    i = 1
    with open(filename, encoding='utf-8') as f:
        for l in f:
            if i == 1:  # 跳过数据第一行
                i = 2
            else:
                _, text = l.strip().split('\t')
                D.append((text, 0))
    return D


# 加载数据集
train_data = load_data(data_path + r'SST-2/train.tsv')
valid_data = load_data(data_path + r'SST-2/dev.tsv')

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)

# 加载预训练模型
bert = build_bert_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_pool=True,
    return_keras_model=False
)

output = Dropout(rate=0.1)(bert.model.output)
output = Dense(units=2, activation='softmax',kernel_initializer=bert.initializer)(output)
model = keras.models.Model(bert.model.input, output)
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(2e-5), metrics=['accuracy'])
model.summary()


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = self.evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # model.save_weights('best_model_SST-2.weights')
        print(u'val_acc: %.5f, best_val_acc: %.5f\n' % (val_acc, self.best_val_acc))

    @staticmethod
    def evaluate(data):
        total, right = 0., 0.
        for x_true, y_true in data:
            y_pred = model.predict(x_true).argmax(axis=1)
            y_true = y_true[:, 0]
            total += len(y_true)
            right += (y_true == y_pred).sum()
        return right / total


def test_predict(in_file, out_file):
    """输出测试结果到文件
    结果文件可以提交到 https://gluebenchmark.com 评测
    """
    test_data = load_data_test(in_file)
    test_generator = data_generator(test_data, batch_size)

    results = []
    for x_true, _ in tqdm(test_generator, ncols=0):
        y_pred = model.predict(x_true).argmax(axis=1)
        results.extend(y_pred)

    with open(out_file, 'w', encoding='utf-8') as f:
        csv_writer = csv.writer(f, delimiter='\t')
        csv_writer.writerow(["index", "prediction"])
        # 写入tsv文件内容
        for i, pred in enumerate(results):
            csv_writer.writerow([i, pred])
        # 关闭文件
    f.close()


if __name__ == '__main__':
    evaluator = Evaluator()

    model.fit(train_generator.forfit(), steps_per_epoch=len(train_generator), epochs=10, callbacks=[evaluator])

    # model.load_weights('best_model_SST-2.weights')
    # 预测测试集, 输出到结果文件
    # test_predict(in_file = './datasets/SST-2/test.tsv', out_file = './results/SST-2.tsv')

else:
    pass
    # model.load_weights('best_model_SST-2.weights')
