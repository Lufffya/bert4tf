import json, numpy
from bert4tf2.layers import *
from tensorflow.keras.models import Model
from typing import Union, Optional, List, Dict, Any
from tensorflow.python.framework.ops import Tensor


class BERT(object):
    """BERT模型

    Args:
        vocab_size (int): -- 词表大小
        hidden_size (int): -- 编码维度
        num_hidden_layers (int): -- Transformer总层数
        num_attention_heads (int): -- Attention的头数
        intermediate_size (int): -- FeedForward的隐层维度
        hidden_act (str): -- FeedForward隐层的激活函数
        hidden_dropout_prob (float): -- Dropout比例
        attention_probs_dropout_prob (float): -- Attention矩阵的Dropout比例
        max_position_embeddings (int): -- 序列最大长度
        type_vocab_size (int): -- Segment总数目
        custom_position_ids (bool, optional): -- 是否自行传入位置id(position id). Defaults False
        keep_tokens (int, optional): -- 要保留的词ID列表. Defaults None
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        intermediate_size: int,
        hidden_act: str,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
        max_position_embeddings: int,
        type_vocab_size: int,
        custom_position_ids: Optional[bool] = False,
        keep_tokens: Optional[int] = None,
        **kwargs):
        ### BERT Config ###
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob 
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        ### BERT Custom settings ###
        self.custom_position_ids = custom_position_ids
        self.embedding_size = hidden_size
        self.attention_head_size = hidden_size // num_attention_heads
        self.attention_key_size =  hidden_size // num_attention_heads
        self.attention_bias = None
        self.keep_tokens = keep_tokens
        if self.keep_tokens is not None:
            self.vocab_size = len(self.keep_tokens)
        # 默认使用截断正态分布初始化
        self.initializer = keras.initializers.TruncatedNormal(stddev=0.02)
        self.bert_layers = {} # 记录构建的BERT模型层, 用来设置Google官方的BERT权重
    
    def simplify(self, inputs: list) -> Union[Any, list]:
        """将list中的None过滤掉
        """
        inputs = [i for i in inputs if i is not None]
        if len(inputs) == 1: inputs = inputs[0]
        return inputs
    
    def build(
        self,
        with_mlm: Optional[bool] = False,
        with_pool: Optional[bool] = False,
        application: Optional[str] = 'encoder',
        additional_input_layers: Optional[Layer] = None,
        layer_norm_cond: Optional[Layer] = None,
        layer_norm_cond_hidden_size: Optional[int] = None,
        layer_norm_cond_hidden_act: Optional[str] = 'linear',
        **kwargs
        ) -> None:
        """构建BERT模型
        
        Args:
            with_mlm (bool, optional): -- 是否包含MLM部分. Defaults False
            with_pool (bool, optional): -- 是否包含Pool部分. Defaults False
            application (str, optional): -- 决定模型类型. Defaults encoder
            additional_input_layers (layer, optional): -- 额外的输入层, 如果外部传入了张量作为条件,
            则需要把条件张量所依赖的所有输入层都添加进来, 作为输入层, 才能构建最终的模型. Defaults None
            layer_norm_condition (layer, optional): -- 如果该参数非None, 则意味着它是一个张量, 
            shape=[batch_size, cond_size] 用来作为Layer Normalization的条件模型额外的输入层. Defaults None
            layer_norm_cond_hidden_size (int, optional): -- 如果该参数非None, 则意味着它是一个整数, 
            用于先将输入条件投影到更低维空间, 这是因为输入的条件可能维度很高, 直接投影到hidden_size(比如768)的话, 参数可能过多, 
            所以可以先投影到更低维空间, 然后升维. Defaults None
            layer_norm_cond_hidden_size (str, optional): -- 投影到更低维空间时的激活函数, 如果为None, 则不加激活函数(线性激活). Defaults linear
        """
        ### BERT Inputs ###
        ### BERT的输入是token_ids和segment_ids(但允许自行传入位置id, 以实现一些特殊需求)
        input_token = Input(shape=(None,), name='Input-Token')
        
        input_segment = Input(shape=(None,), name='Input-Segment')
        
        input_position = None
        if self.custom_position_ids:
            input_position = Input(shape=(None,), name='Input-Position')
        
        ### BERT Embedding ###
        ### BERT的embedding是token, position, segment三者embedding之和
        embedding_token = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token')
        self.bert_layers[embedding_token.name] = embedding_token
        x_token = embedding_token(input_token)

        embedding_segment = Embedding(
            input_dim=self.type_vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            name='Embedding-Segment')
        self.bert_layers[embedding_segment.name] = embedding_segment
        x_segment = embedding_segment(input_segment)
        
        x = Add(name='Embedding-Token-Segment')([x_token, x_segment])

        embedding_position = PositionEmbedding(
            input_dim=self.max_position_embeddings,
            output_dim=self.embedding_size,
            merge_mode='add',
            embeddings_initializer=self.initializer,
            custom_position_ids=self.custom_position_ids,
            name='Embedding-Position')
        self.bert_layers[embedding_position.name] = embedding_position
        x = embedding_position(self.simplify([x, input_position]))

        embedding_norm = LayerNormalization(
            conditional=(layer_norm_cond is not None),
            hidden_units=layer_norm_cond_hidden_size,
            hidden_activation=layer_norm_cond_hidden_act,
            hidden_initializer=self.initializer,
            name='Embedding-Norm')
        self.bert_layers[embedding_norm.name] = embedding_norm
        x = embedding_norm(self.simplify([x, layer_norm_cond]))

        x = Dropout(rate=self.hidden_dropout_prob, name='Embedding-Dropout')(x)
       
        if self.embedding_size != self.hidden_size:
            embedding_mapping = Dense(
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Embedding-Mapping')
            self.bert_layers[embedding_mapping.name] = embedding_mapping
            x = embedding_mapping(x)

        ### BERT的主体是基于Self-Attention的模块
        ### 顺序: Att --> Add --> LN --> FFN --> Add --> LN
        for i in range(self.num_hidden_layers):
            attention_name = 'Transformer-%d-MultiHeadSelfAttention' % i
            feed_forward_name = 'Transformer-%d-FeedForward' % i
            
            ### Self Attention ### 
            x_input, x, att_args = x, [x, x, x], {'a_bias': None}

            ### 为BERT模型添加一些trick
            if application == 'lm':
                att_args['a_bias'] = True
                x.append(self.compute_lm_attention_bias(input_token))
            elif application == 'unilm':
                att_args['a_bias'] = True
                x.append(self.compute_unilm_attention_bias(input_segment))

            multi_head_attention = MultiHeadAttention(
                heads=self.num_attention_heads,
                head_size=self.attention_head_size,
                out_dim=self.hidden_size,
                key_size=self.attention_key_size,
                attention_dropout=self.attention_probs_dropout_prob,
                kernel_initializer=self.initializer,
                name=attention_name)
            self.bert_layers[multi_head_attention.name] = multi_head_attention
            x = multi_head_attention(x, **att_args)

            x = Dropout(rate=self.hidden_dropout_prob, name='%s-Dropout' % attention_name)(x)

            x = Add(name='%s-Add' % attention_name)([x_input, x])

            attention_norm = LayerNormalization(
                conditional=(layer_norm_cond is not None),
                hidden_units=layer_norm_cond_hidden_size,
                hidden_activation=layer_norm_cond_hidden_act,
                hidden_initializer=self.initializer,
                name='%s-Norm' % attention_name)
            self.bert_layers[attention_norm.name] = attention_norm
            x = attention_norm(self.simplify([x, layer_norm_cond]))
            
            ### Feed Forward ###
            x_input = x
            
            feed_forward = FeedForward(
                units=self.intermediate_size,
                activation=self.hidden_act,
                kernel_initializer=self.initializer,
                name=feed_forward_name)
            self.bert_layers[feed_forward.name] = feed_forward
            x = feed_forward(x)
            
            x = Dropout(rate=self.hidden_dropout_prob, name='%s-Dropout' % feed_forward_name)(x)

            x = Add(name='%s-Add' % feed_forward_name)([x_input, x])
            
            feed_forward_norm = LayerNormalization(
                conditional=(layer_norm_cond is not None),
                hidden_units=layer_norm_cond_hidden_size,
                hidden_activation=layer_norm_cond_hidden_act,
                hidden_initializer=self.initializer,
                name='%s-Norm' % feed_forward_name)
            self.bert_layers[feed_forward_norm.name] = feed_forward_norm
            x = feed_forward_norm(self.simplify([x, layer_norm_cond]))
        
        ### 根据剩余参数决定输出
        ### Pooler部分(提取CLS向量)
        if with_pool:
            x = Lambda(function=lambda x: x[:, 0], name='Pooler')(x)
            pooler_dense = Dense(
                units=self.hidden_size,
                activation=('tanh' if with_pool is True else with_pool),
                kernel_initializer=self.initializer,
                name='Pooler-Dense')
            self.bert_layers[pooler_dense.name] = pooler_dense
            x = pooler_dense(x)
        
        ### Masked Language Model部分
        if with_mlm:
            mlm_dense = Dense(
                units=self.embedding_size,
                activation=self.hidden_act,
                kernel_initializer=self.initializer,
                name='MLM-Dense')
            self.bert_layers[mlm_dense.name] = mlm_dense
            x = mlm_dense(x)
            
            mlm_norm = LayerNormalization(
                conditional=(layer_norm_cond is not None),
                hidden_units=layer_norm_cond_hidden_size,
                hidden_activation=layer_norm_cond_hidden_act,
                hidden_initializer=self.initializer,
                name='MLM-Norm')
            self.bert_layers[mlm_norm.name] = mlm_norm
            x = mlm_norm(self.simplify([x, layer_norm_cond]))

            x = self.bert_layers['Embedding-Token'](x, **{'mode': 'dense'})
            
            mlm_bias = BiasAdd(name='MLM-Bias')
            self.bert_layers[mlm_bias.name] = mlm_bias
            x = mlm_bias(x)
            
            x = Activation(
                activation=('softmax' if with_mlm is True else with_mlm), 
                name='MLM-Activation')(x)
        
        ### 根据输入输出构建BERT模型
        if not isinstance(additional_input_layers, list):
            additional_input_layers = [additional_input_layers]
        inputs = self.simplify([input_token, input_segment, input_position] + additional_input_layers)
        outputs = [x]
        self.model = Model(inputs, outputs)

    def compute_lm_attention_bias(self, input):
        """定义下三角Attention Mask(语言模型用)
        """
        if self.attention_bias is None:
            
            def lm_mask(s):
                # 通过idxs序列的比较来得到对应的mask
                seq_len = K.shape(s)[1]
                idxs = K.arange(0, seq_len)
                mask = idxs[None, :] <= idxs[:, None]
                mask = K.cast(mask, K.floatx())
                return -(1 - mask[None, None]) * K.infinity()

            self.attention_bias = Lambda(
                function=lm_mask,
                name='Attention-LM-Mask')(input)

        return self.attention_bias
    
    def compute_unilm_attention_bias(self, input):
        """添加UniLM的Attention Mask(Seq2Seq模型用)
        UniLM: https://arxiv.org/abs/1905.03197
        """
        # 其中source和target的分区, 由segment_ids来表示.
        if self.attention_bias is None:

            def unilm_mask(s):
                # 通过idxs序列的比较来得到对应的mask
                idxs = K.cumsum(s, axis=1)
                mask = idxs[:, None, :] <= idxs[:, :, None]
                mask = K.cast(mask, K.floatx())
                return -(1 - mask[:, None]) * K.infinity()

            self.attention_bias = Lambda(
                function=unilm_mask,
                name='Attention-UniLM-Mask')(input)

        return self.attention_bias

    def load_weights_from_checkpoint(self, checkpoint: str) -> None:
        """根据mapping从checkpoint加载权重
        """
        mapping = self.variable_mapping()
        mapping = {k: v for k, v in mapping.items() if k in self.bert_layers}

        weight_value_pairs = []
        for layer_name, variables in mapping.items():
            weights, values = [], []
            layer = self.bert_layers[layer_name]
            # 允许跳过不存在的权重
            for w, v in zip(layer.trainable_weights, variables):
                try:
                    weights.append(w)
                    values.append(self.load_variable(checkpoint, v))
                except Exception as e: raise e

            for w, v in zip(weights, values):
                if v is not None: weight_value_pairs.append((w, v))

        K.batch_set_value(weight_value_pairs)

    def save_weights_as_checkpoint(self, filename: str, dtype=None) -> None:
        """根据mapping将权重保存为checkpoint格式
        """
        mapping = self.variable_mapping()
        mapping = {k: v for k, v in mapping.items() if k in self.layers}

        with tf.Graph().as_default():
            all_variables, all_values = [], []
            for layer, variables in mapping.items():
                layer = self.layers[layer]
                values = K.batch_get_value(layer.trainable_weights)
                for name, value in zip(variables, values):
                    variable, value = self.create_variable(name, value, dtype)
                    all_variables.append(variable)
                    all_values.append(value)
            with tf.Session() as sess:
                K.batch_set_value(zip(all_variables, all_values))
                saver = tf.train.Saver()
                saver.save(sess, filename)

    def load_variable(self, checkpoint: str, name: str) -> Any:
        """加载单个变量的函数
        """
        variable = tf.train.load_variable(checkpoint, name)
        if name in ['bert/embeddings/word_embeddings', 'cls/predictions/output_bias']:
            embeddings = variable.astype(K.floatx())  # 防止np.average报错
            if self.keep_tokens is not None:
                embeddings = embeddings[self.keep_tokens]
            return embeddings
        elif name == 'cls/seq_relationship/output_weights':
            return variable.T
        else:
            return variable
        
    def create_variable(self, name: str, value: numpy, dtype=None) -> Tensor:
        """在tensorflow中创建一个变量
        """
        if name == 'cls/seq_relationship/output_weights':
            value = value.T
        dtype = dtype or K.floatx()
        return K.variable(self.initializer(value.shape, dtype), dtype, name=name), value
    
    def variable_mapping(self) -> Dict[str, List[str]]:
        """映射到官方BERT权重格式
        """
        mapping = {
            'Embedding-Token': ['bert/embeddings/word_embeddings'],
            'Embedding-Segment': ['bert/embeddings/token_type_embeddings'],
            'Embedding-Position': ['bert/embeddings/position_embeddings'],
            'Embedding-Norm': [
                'bert/embeddings/LayerNorm/beta',
                'bert/embeddings/LayerNorm/gamma',
            ],
            'Embedding-Mapping': [
                'bert/encoder/embedding_hidden_mapping_in/kernel',
                'bert/encoder/embedding_hidden_mapping_in/bias',
            ],
            'Pooler-Dense': [
                'bert/pooler/dense/kernel',
                'bert/pooler/dense/bias',
            ],
            'NSP-Proba': [
                'cls/seq_relationship/output_weights',
                'cls/seq_relationship/output_bias',
            ],
            'MLM-Dense': [
                'cls/predictions/transform/dense/kernel',
                'cls/predictions/transform/dense/bias',
            ],
            'MLM-Norm': [
                'cls/predictions/transform/LayerNorm/beta',
                'cls/predictions/transform/LayerNorm/gamma',
            ],
            'MLM-Bias': ['cls/predictions/output_bias'],
        }

        for i in range(self.num_hidden_layers):
            prefix = 'bert/encoder/layer_%d/' % i
            mapping.update({
                'Transformer-%d-MultiHeadSelfAttention' % i: [
                    prefix + 'attention/self/query/kernel',
                    prefix + 'attention/self/query/bias',
                    prefix + 'attention/self/key/kernel',
                    prefix + 'attention/self/key/bias',
                    prefix + 'attention/self/value/kernel',
                    prefix + 'attention/self/value/bias',
                    prefix + 'attention/output/dense/kernel',
                    prefix + 'attention/output/dense/bias',
                ],
                'Transformer-%d-MultiHeadSelfAttention-Norm' % i: [
                    prefix + 'attention/output/LayerNorm/beta',
                    prefix + 'attention/output/LayerNorm/gamma',
                ],
                'Transformer-%d-FeedForward' % i: [
                    prefix + 'intermediate/dense/kernel',
                    prefix + 'intermediate/dense/bias',
                    prefix + 'output/dense/kernel',
                    prefix + 'output/dense/bias',
                ],
                'Transformer-%d-FeedForward-Norm' % i: [
                    prefix + 'output/LayerNorm/beta',
                    prefix + 'output/LayerNorm/gamma',
                ],
            })

        return mapping


def build_bert_model(config_path=None, checkpoint_path=None, return_keras_model=True, **kwargs) -> Union[Model, BERT]:
    """根据配置文件构建模型, 可选加载checkpoint权重
    """
    configs = {}
    if config_path is not None:
        configs.update(json.load(open(config_path)))
    configs.update(kwargs)

    if 'application' in configs and configs['application'] in ['lm', 'unilm']:
        configs['with_mlm'] = True

    bert = BERT(**configs)
    bert.build(**configs)

    if checkpoint_path is not None:
        bert.load_weights_from_checkpoint(checkpoint_path)

    if return_keras_model:
        return bert.model
    else:
        return bert
