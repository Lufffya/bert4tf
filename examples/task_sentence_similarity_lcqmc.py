# 句子对分类任务, LCQMC数据集
# val_acc: 0.887071, test_acc: 0.870320

from snippets import *
from bert4tf2.backend import set_gelu
from bert4tf2.snippets import sequence_padding
from bert4tf2.snippets import DataGenerator


set_gelu('tanh')  # 切换gelu版本

maxlen = 128
batch_size = 64


def load_data(filename):
    """加载数据
    单条格式:(文本1, 文本2, 标签id)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text1, text2, label = l.strip().split('\t')
            D.append((text1, text2, int(label)))
    return D


# 加载数据集
train_data = load_data('datasets/lcqmc/lcqmc.train.data')
valid_data = load_data('datasets/lcqmc/lcqmc.valid.data')
test_data = load_data('datasets/lcqmc/lcqmc.test.data')

# 建立分词器
tokenizer = Tokenizer(dict_path_wwm_zh, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text1, text2, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert = build_bert_model(
    config_path=config_path_wwm_zh,
    checkpoint_path=checkpoint_path_wwm_zh,
    with_pool=True,
    return_keras_model=False
)

output = keras.layers.Dropout(rate=0.1)(bert.model.output)
output = keras.layers.Dense(units=2, activation='softmax', kernel_initializer=bert.initializer)(output)
model = keras.models.Model(bert.model.input, output)
# 用足够小的学习率, optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1})
model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(2e-5), metrics=['accuracy'])
model.summary()

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = self.evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # model.save_weights('best_model.weights')
        test_acc = self.evaluate(test_generator)
        print(u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' % (val_acc, self.best_val_acc, test_acc))
    
    @staticmethod
    def evaluate(data):
        total, right = 0., 0.
        for x_true, y_true in data:
            y_pred = model.predict(x_true).argmax(axis=1)
            y_true = y_true[:, 0]
            total += len(y_true)
            right += (y_true == y_pred).sum()
        return right / total


if __name__ == '__main__':
    evaluator = Evaluator()

    model.fit(train_generator.forfit(), steps_per_epoch=len(train_generator), epochs=20, callbacks=[evaluator])

    # model.load_weights('best_model.weights')
    print(u'final test acc: %05f\n' % (Evaluator.evaluate(test_generator)))

else:
    # model.load_weights('best_model.weights')
    pass
