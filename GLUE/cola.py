# CoLA: The Corpus of Linguistic Acceptability(语言可接受性语料库)
# describe: 句子通顺度判断, 判断句子单词序列是否通顺
# metric: Matthews correlation coefficient

from snippets import *
from sklearn.metrics import matthews_corrcoef


num_classes = 2
maxlen = 128
batch_size = 32
epochs = 2


def load_data(filename):
    """加载数据
    单条格式: (文本, 标签id)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            _, label, _, text = l.strip().split('\t')
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
train_data = load_data(data_path + r'CoLA/train.tsv')
valid_data = load_data(data_path + r'CoLA/dev.tsv')

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

output = keras.layers.Dropout(rate=0.1)(bert.model.output)
output = keras.layers.Dense(units=2, activation='softmax', kernel_initializer=bert.initializer)(output)
model = keras.models.Model(bert.model.input, output)
model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(2e-5), metrics=['accuracy'])
model.summary()


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_mcc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc, mcc = self.evaluate(valid_generator)
        if mcc > self.best_val_mcc:
            self.best_val_mcc = mcc
            # model.save_weights('best_model_CoLA.weights')
        print(u'val_acc: %.5f, Matthews corr: %.5f, best_val_mcc: %.5f\n' % (val_acc, mcc, self.best_val_mcc))

    @staticmethod
    def evaluate(data):
        total, right = 0., 0.
        y_true_all = np.array([], dtype=int)
        y_pred_all = np.array([], dtype=int)
        for x_true, y_true in data:
            y_pred = model.predict(x_true).argmax(axis=1)
            y_true = y_true[:, 0]
            y_pred_all = np.append(y_pred_all, y_pred)
            y_true_all = np.append(y_true_all, y_true)
            total += len(y_true)
            right += (y_true == y_pred).sum()

        mcc = matthews_corrcoef(y_true_all, y_pred_all)
        return right / total, mcc


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

    model.fit(train_generator.forfit(), steps_per_epoch=len(train_generator), epochs=epochs, callbacks=[evaluator])

    # model.load_weights('best_model_CoLA.weights')
    # test_predict(in_file = './datasets/CoLA/test.tsv', out_file = './results/CoLA.tsv')

else:
    # model.load_weights('best_model_CoLA.weights')
    pass

