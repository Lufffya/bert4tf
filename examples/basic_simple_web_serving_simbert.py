# 利用自带的接口，将SimBERT的同义句生成搭建成Web服务。
# 基于bottlepy简单封装，仅作为临时测试使用，不保证性能。
# 具体用法请看 https://github.com/bojone/bert4keras/blob/8ffb46a16a79f87aa8cdf045df7994036b4be47d/bert4keras/snippets.py#L580

from snippets import *
from collections import Counter
from bert4tf.snippets import sequence_padding
from bert4tf.snippets import AutoRegressiveDecoder


maxlen = 32

# 建立分词器
tokenizer = Tokenizer(dict_path_sim_zh, do_lower_case=True)  # 建立分词器

# 建立加载模型
bert = build_bert_model(config_path_sim_zh, checkpoint_path_sim_zh, with_pool='linear', application='unilm', return_keras_model=False)

encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])
seq2seq = keras.models.Model(bert.model.inputs, bert.model.outputs[1])


class SynonymsGenerator(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return self.last_token(seq2seq).predict([token_ids, segment_ids])

    def generate(self, text, n=1, topp=0.95):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        output_ids = self.random_sample([token_ids, segment_ids], n, topp=topp)  # 基于随机采样
        return [tokenizer.decode(ids) for ids in output_ids]


synonyms_generator = SynonymsGenerator(start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen)


def gen_synonyms(text, n=100, k=20):
    """"含义： 产生sent的n个相似句，然后返回最相似的k个。
    做法：用seq2seq生成，并用encoder算相似度并排序。
    """
    r = synonyms_generator.generate(text, n)
    r = [i for i in set(r) if i != text]
    r = [text] + r
    X, S = [], []
    for t in r:
        x, s = tokenizer.encode(t)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = encoder.predict([X, S])
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    argsort = np.dot(Z[1:], -Z[0]).argsort()
    return [r[i + 1] for i in argsort[:k]]


class WebServing(object):
    """简单的Web接口
    用法：
        arguments = {'text': (None, True), 'n': (int, False)}
        web = WebServing(port=8864)
        web.route('/gen_synonyms', gen_synonyms, arguments)
        web.start()
        # 然后访问 http://127.0.0.1:8864/gen_synonyms?text=你好
    说明：
        基于bottlepy简单封装，仅作为临时测试使用，不保证性能。
        目前仅保证支持 Tensorflow 1.x + Keras <= 2.3.1。
        欢迎有经验的开发者帮忙改进。
    依赖：
        pip install bottle
        pip install paste
        （如果不用 server='paste' 的话，可以不装paste库）
    """
    def __init__(self, host='0.0.0.0', port=8000, server='paste'):

        import bottle

        self.host = host
        self.port = port
        self.server = server
        self.graph = tf.get_default_graph()
        self.sess = K.get_session()
        self.set_session = K.set_session
        self.bottle = bottle

    def wraps(self, func, arguments, method='GET'):
        """封装为接口函数
        参数：
            func：要转换为接口的函数，需要保证输出可以json化，即需要
                  保证 json.dumps(func(inputs)) 能被执行成功；
            arguments：声明func所需参数，其中key为参数名，value[0]为
                       对应的转换函数（接口获取到的参数值都是字符串
                       型），value[1]为该参数是否必须；
            method：GET或者POST。
        """
        def new_func():
            outputs = {'code': 0, 'desc': u'succeeded', 'data': {}}
            kwargs = {}
            for key, value in arguments.items():
                if method == 'GET':
                    result = self.bottle.request.GET.getunicode(key)
                else:
                    result = self.bottle.request.POST.getunicode(key)
                if result is None:
                    if value[1]:
                        outputs['code'] = 1
                        outputs['desc'] = 'lack of "%s" argument' % key
                        return json.dumps(outputs, ensure_ascii=False)
                else:
                    if value[0] is not None:
                        result = value[0](result)
                    kwargs[key] = result
            try:
                with self.graph.as_default():
                    self.set_session(self.sess)
                    outputs['data'] = func(**kwargs)
            except Exception as e:
                outputs['code'] = 2
                outputs['desc'] = str(e)
            return json.dumps(outputs, ensure_ascii=False)

        return new_func

    def route(self, path, func, arguments, method='GET'):
        """添加接口
        """
        func = self.wraps(func, arguments, method)
        self.bottle.route(path, method=method)(func)

    def start(self):
        """启动服务
        """
        self.bottle.run(host=self.host, port=self.port, server=self.server)


if __name__ == '__main__':
    arguments = {'text': (None, True), 'n': (int, False), 'k': (int, False)}
    web = WebServing(port=8864)
    web.route('/gen_synonyms', gen_synonyms, arguments)
    web.start()
    # 现在可以测试访问 http://127.0.0.1:8864/gen_synonyms?text=苹果多少钱一斤
