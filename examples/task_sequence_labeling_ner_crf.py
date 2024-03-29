# 用CRF做中文命名实体识别
# 数据集 http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz
# 实测验证集的F1可以到96.48%，测试集的F1可以到95.38%

from snippets import *
from bert4tf2.snippets import sequence_padding, DataGenerator
from bert4tf2.snippets import to_array, ViterbiDecoder
from bert4tf2.layers import ConditionalRandomField


maxlen = 256
epochs = 1
batch_size = 16
bert_layers = 12
learning_rate = 2e-5  # bert_layers越小, 学习率应该要越大
crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率
categories = set()


def load_data(filename):
    """加载数据
    单条格式: [text, (start, end, label), (start, end, label), ...],
              意味着text[start:end + 1]是类型为label的实体.
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d = ['']
            for i, c in enumerate(l.split('\n')):
                char, flag = c.split(' ')
                d[0] += char
                if flag[0] == 'B':
                    d.append([i, i, flag[2:]])
                    categories.add(flag[2:])
                elif flag[0] == 'I':
                    d[-1][1] = i
            D.append(d)
    return D


# 标注数据
train_data = load_data(data_path + 'china-people-daily-ner-corpus/example.train')
valid_data = load_data(data_path + 'china-people-daily-ner-corpus/example.dev')
test_data = load_data(data_path + 'china-people-daily-ner-corpus/example.test')
categories = list(sorted(categories))

# 建立分词器
tokenizer = Tokenizer(dict_path_zh, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
            mapping = tokenizer.rematch(d[0], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            labels = np.zeros(len(token_ids))
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    labels[start] = categories.index(label) * 2 + 1
                    labels[start + 1:end + 1] = categories.index(label) * 2 + 2
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 后面的代码使用的是bert类型的模型，如果你用的是albert，那么前几行请改为：
# model = build_bert_model(config_path, checkpoint_path, model='albert')
# output_layer = 'Transformer-FeedForward-Norm'
# output = model.get_layer(output_layer).get_output_at(bert_layers - 1)

model = build_bert_model(config_path_zh, checkpoint_path_zh)

output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
output = model.get_layer(output_layer).output
output = keras.layers.Dense(len(categories) * 2 + 1)(output)
CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
output = CRF(output)
model = keras.models.Model(model.input, output)
model.compile(loss=CRF.sparse_loss, optimizer=keras.optimizers.Adam(learning_rate), metrics=[CRF.sparse_accuracy])
model.summary()


class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """
    def recognize(self, text):
        tokens = tokenizer.tokenize(text, maxlen=512)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        nodes = model.predict([token_ids, segment_ids])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], categories[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False
        return [(mapping[w[0]][0], mapping[w[-1]][-1], l) for w, l in entities]


NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        NER.trans = trans
        print(NER.trans)
        f1, precision, recall = self.evaluate(valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            # model.save_weights('./best_model.weights')
        print('valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' % (f1, precision, recall, self.best_val_f1))
        f1, precision, recall = self.evaluate(test_data)
        print('test:  f1: %.5f, precision: %.5f, recall: %.5f\n' % (f1, precision, recall))
    
    @staticmethod
    def evaluate(data):
        """评测函数
        """
        X, Y, Z = 1e-10, 1e-10, 1e-10
        for d in tqdm(data, ncols=100):
            R = set(NER.recognize(d[0]))
            T = set([tuple(i) for i in d[1:]])
            X += len(R & T)
            Y += len(R)
            Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return f1, precision, recall


if __name__ == '__main__':
    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)

    model.fit(train_generator.forfit(), steps_per_epoch=len(train_generator), epochs=epochs, callbacks=[evaluator])

else:
    # model.load_weights('./best_model.weights')
    # NER.trans = K.eval(CRF.trans)
    pass
