# SuperGLUE评测
# RTE文本相似度
# 思路：premise和hypothesis拼接后取[CLS]然后接Dense+Softmax分类

from snippets import *


# 基本参数
labels = ['entailment', 'not_entailment']
num_classes = len(labels)
maxlen = 128
batch_size = 32
epochs = 2


def load_data(filename):
    """加载数据
    格式：[(premise, hypothesis, 标签id)]
    """
    D = []
    with open(filename) as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            text1, text2, label = l['premise'], l['hypothesis'], l.get('label', 'entailment')
            D.append((text1, text2, labels.index(label)))
    return D


# 加载数据集
train_data = load_data(data_path + 'RTE/train.jsonl')
valid_data = load_data(data_path + 'RTE/val.jsonl')


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text1, text2, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append([0] * len(segment_ids))
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

# 构建模型
output = base.model.get_layer(last_layer).output
output = pooling_layer(output)
output = keras.layers.Dense(units=num_classes, activation='softmax', kernel_initializer=base.initializer)(output)
model = keras.models.Model(base.model.input, output)
model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate), metrics=['accuracy'])
model.summary()


class Evaluator(keras.callbacks.Callback):
    """保存验证集acc最好的模型
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = self.evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # model.save_weights('weights/RTE.weights')
        print(u'val_acc: %.5f, best_val_acc: %.5f\n' % (val_acc, self.best_val_acc))

    def evaluate(self, data):
        total, right = 0., 0.
        for x_true, y_true in data:
            y_pred = model.predict(x_true).argmax(axis=1)
            y_true = y_true[:, 0]
            total += len(y_true)
            right += (y_true == y_pred).sum()
        return right / total
 

def test_predict(in_file, out_file):
    """输出测试结果到文件
    结果文件可以提交到 https://super.gluebenchmark.com/ 评测。
    """
    test_data = load_data(in_file)
    test_generator = data_generator(test_data, batch_size)

    results = []
    for x_true, _ in tqdm(test_generator, ncols=0):
        y_pred = model.predict(x_true).argmax(axis=1)
        results.extend(y_pred)
    fw = open(out_file, 'w')
    with open(in_file) as fr:
        for l, r in zip(fr, results):
            l = json.loads(l)
            l = json.dumps({'idx': str(l['idx']), 'label': labels[r]})
            fw.write(l + '\n')
    fw.close()


if __name__ == '__main__':
    evaluator = Evaluator()

    model.fit(train_generator.forfit(), steps_per_epoch=len(train_generator), epochs=epochs, callbacks=[evaluator])

    # model.load_weights('weights/RTE.weights')
    # test_predict(in_file=data_path + 'RTE/test.jsonl', out_file='results/RTE.jsonl')

else:
    # model.load_weights('weights/RTE.weights')
    pass
