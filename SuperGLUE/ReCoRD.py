# SuperGLUE评测
# ReCoRD阅读理解
# 思路：基于滑动窗口和GlobalPointer

from snippets import *
from itertools import groupby
from bert4tf2.layers import GlobalPointer
from bert4tf2.snippets import lowercase_and_normalize


# 基本参数
maxlen = 512
stride = 128
batch_size = 4
epochs = 2


def stride_split(i, q, c, a, s):
    """滑动窗口分割context
    """
    # 标准转换
    q = lowercase_and_normalize(q)
    c = lowercase_and_normalize(c)
    a = lowercase_and_normalize(a)
    e = s + len(a)

    # 滑窗分割
    results, n = [], 0
    max_c_len = maxlen - len(q) - 3
    while True:
        l, r = n * stride, n * stride + max_c_len
        if l <= s < e <= r:
            results.append((i, q, c[l:r], a, s - l, e - l))
        else:
            results.append((i, q, c[l:r], '', -1, -1))
        if r >= len(c):
            return results
        n += 1


def load_data(filename, is_test = False):
    """加载数据
    格式：[(id, 问题, 篇章, 答案, start, end)]
    """
    D = []
    with open(filename) as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            qs, p  = l['qas'], l['passage']['text']
            entities = l['passage']['entities']
            entities_str = ""
            for ent in entities:
                ent_s = int(ent.get('start'))
                ent_e = int(ent.get('end'))
                ent_str = p[int(ent_s):int(ent_e)]
                entities_str += ent_str + ' '
            p += entities_str
            if is_test:
                for q in qs:
                    inx = q['idx']
                    q = q['query']
                    D.extend(stride_split(inx, q, p, 'aaa'*500, len(q)))
            else:
                for q in qs:
                    # print(q)
                    inx = q['idx']
                    ans = q['answers']
                    q = q['query']
                    for a in ans:
                        D.extend(stride_split(inx, q, p, a['text'], a['start']))
            # if len(D)>2000:
            #     break
    # print(D)
    return D


# 加载数据集
train_data = load_data(data_path + 'ReCoRD/train.jsonl')
valid_data = load_data(data_path + 'ReCoRD/val.jsonl')


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        batch_masks, batch_labels = [], []
        for is_end, (i, q, c, a, s, e) in self.sample(random):
            token_ids = tokenizer.encode(q)[0]
            mask = [1] + [0] * len(token_ids[:-1])
            if s == -1:
                token_ids.extend(tokenizer.encode(c)[0][1:])
                batch_labels.append([0, 0])
            else:
                cl_ids = tokenizer.encode(c[:s])[0][1:-1]
                a_ids = tokenizer.encode(c[s:e])[0][1:-1]
                cr_ids = tokenizer.encode(c[e:])[0][1:]
                start = len(token_ids) + len(cl_ids)
                end = start + len(a_ids) - 1
                batch_labels.append([start, end])
                token_ids.extend(cl_ids + a_ids + cr_ids)
            mask.extend([1] * (len(token_ids[:-1]) - len(mask)) + [0])
            batch_token_ids.append(token_ids)
            batch_segment_ids.append([0] * len(token_ids))
            batch_masks.append(mask)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_masks = sequence_padding(batch_masks)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids, batch_masks], batch_labels
                batch_token_ids, batch_segment_ids = [], []
                batch_masks, batch_labels = [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)


class CustomMasking(keras.layers.Layer):
    """自定义mask（主要用于mask掉question部分）
    """
    def compute_mask(self, inputs, mask=None):
        return K.greater(inputs[1], 0.5)

    def call(self, inputs, mask=None):
        return inputs[0]

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def globalpointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    b, l = K.shape(y_pred)[0], K.shape(y_pred)[1]
    # y_true需要重新明确一下shape和dtype
    y_true = K.reshape(y_true, (b, 2))
    y_true = K.cast(y_true, 'int32')
    y_true = y_true[:, 0] * l + y_true[:, 1]
    # 计算交叉熵
    y_pred = K.reshape(y_pred, (b, -1))
    return K.mean(K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True))


def globalpointer_accuracy(y_true, y_pred):
    """给GlobalPointer设计的准确率
    """
    b, l = K.shape(y_pred)[0], K.shape(y_pred)[1]
    # y_true需要重新明确一下shape和dtype
    y_true = K.reshape(y_true, (b, 2))
    y_true = K.cast(y_true, 'int32')
    y_true = y_true[:, 0] * l + y_true[:, 1]
    # 计算准确率
    y_pred = K.reshape(y_pred, (b, -1))
    y_pred = K.cast(K.argmax(y_pred, axis=1), 'int32')
    return K.mean(K.cast(K.equal(y_true, y_pred), K.floatx()))


# 构建模型
masks_in = keras.layers.Input(shape=(None,))
output = base.model.get_layer(last_layer).output
output = CustomMasking()([output, masks_in])
output = GlobalPointer(heads=1, head_size=base.attention_head_size, use_bias=False, kernel_initializer=base.initializer)(output)
output = keras.layers.Lambda(lambda x: x[:, 0])(output)
model = keras.models.Model(base.model.inputs + [masks_in], output)
model.compile(loss=globalpointer_crossentropy, optimizer=keras.optimizers.Adam(learning_rate),metrics=[globalpointer_accuracy])
model.summary()


class Evaluator(keras.callbacks.Callback):
    """保存验证集acc最好的模型
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = self.evaluate(valid_data, valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # model.save_weights('weights/ReCoRD.weights')
        print(u'val_acc: %.5f, best_val_acc: %.5f\n' % (val_acc, self.best_val_acc))

    def evaluate(self, data, generator):
        Y_scores = np.empty((0, 1))
        Y_start_end = np.empty((0, 2), dtype=int)
        Y_true = np.empty((0, 2), dtype=int)
        for x_true, y_true in tqdm(generator, ncols=0):
            y_pred = model.predict(x_true)
            y_pred[:, 0] -= np.inf
            y_pred[:, :, 0] -= np.inf
            y_pred = y_pred.reshape((x_true[0].shape[0], -1))
            y_start_end = y_pred.argmax(axis=1)[:, None]
            y_scores = np.take_along_axis(y_pred, y_start_end, axis=1)
            y_start = y_start_end // x_true[0].shape[1]
            y_end = y_start_end % x_true[0].shape[1]
            y_start_end = np.concatenate([y_start, y_end], axis=1)
            Y_scores = np.concatenate([Y_scores, y_scores], axis=0)
            Y_start_end = np.concatenate([Y_start_end, y_start_end], axis=0)
            Y_true = np.concatenate([Y_true, y_true], axis=0)

        total, right, n = 0., 0., 0
        for k, g in groupby(data, key=lambda d: d[0]):  # 按qid分组
            g = len(list(g))
            i = Y_scores[n:n + g].argmax() + n  # 取组内最高分答案
            y_true, y_pred = Y_true[i], Y_start_end[i]
            if (y_pred == y_true).all():
                right += 1
            total += 1
            n += g

        return right / total


def test_predict(in_file, out_file):
    """输出测试结果到文件
    结果文件可以提交到 https://super.gluebenchmark.com/ 评测。
    """
    test_data = load_data(in_file, is_test = True)
    test_generator = data_generator(test_data, batch_size)

    Y_scores = np.empty((0, 1))
    Y_start_end = np.empty((0, 2), dtype=int)
    for x_true, _ in tqdm(test_generator, ncols=0):
        y_pred = model.predict(x_true)
        y_pred[:, 0] -= np.inf
        y_pred[:, :, 0] -= np.inf
        y_pred = y_pred.reshape((x_true[0].shape[0], -1))
        y_start_end = y_pred.argmax(axis=1)[:, None]
        y_scores = np.take_along_axis(y_pred, y_start_end, axis=1)
        y_start = y_start_end // x_true[0].shape[1]
        y_end = y_start_end % x_true[0].shape[1]
        y_start_end = np.concatenate([y_start, y_end], axis=1)
        Y_scores = np.concatenate([Y_scores, y_scores], axis=0)
        Y_start_end = np.concatenate([Y_start_end, y_start_end], axis=0)

    results, n = {}, 0
    for k, g in groupby(test_data, key=lambda d: d[0]):  # 按qid分组
        g = len(list(g))
        i = Y_scores[n:n + g].argmax() + n  # 取组内最高分答案
        start, end = Y_start_end[i]
        q, c = test_data[i][1:3]
        q_tokens = tokenizer.tokenize(q)
        c_tokens = tokenizer.tokenize(c)[1:-1]
        mapping = tokenizer.rematch(c, c_tokens)  # 重匹配，直接在context取片段
        start, end = start - len(q_tokens), end - len(q_tokens)
        results[k] = c[mapping[start][0]:mapping[end][-1] + 1]
        n += g
    # print(results)
    fw = open(out_file, 'w', encoding='utf-8')
    with open(in_file) as fr:
        for l, r in zip(fr, results.values()):
            l = json.loads(l)
            for q in l['qas']:
                inx = q['idx']
                l = json.dumps({'idx': inx, 'label':str(r)})
                fw.write(l + '\n')
    fw.close()


if __name__ == '__main__':
    evaluator = Evaluator()

    model.fit(train_generator.forfit(), steps_per_epoch=len(train_generator), epochs=epochs, callbacks=[evaluator])

    # model.load_weights('weights/ReCoRD.weights')
    # test_predict(in_file=data_path + 'ReCoRD/test.jsonl', out_file='results/ReCoRD.jsonl')

else:
    # model.load_weights('weights/ReCoRD.weights')
    pass
