# -*- coding: utf-8 -*-

import sys
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K


def infinity():
    """返回默认的代表无穷大的数值
    """
    return keras.utils.get_custom_objects().get('infinity', 1e12)


def sequence_masking(x, mask, value=0, axis=None):
    """为序列条件mask的函数
    mask: 形如(batch_size, seq_len)的0-1矩阵;
    value: mask部分要被替换成的值, 可以是'-inf'或'inf';
    axis: 序列所在轴, 默认为1;
    """
    if mask is None:
        return x
    else:
        x_dtype = K.dtype(x)
        if x_dtype == 'bool':
            x = K.cast(x, 'int32')
        if K.dtype(mask) != K.dtype(x):
            mask = K.cast(mask, K.dtype(x))
        if value == '-inf':
            value = -K.infinity()
        elif value == 'inf':
            value = K.infinity()
        if axis is None:
            axis = 1
        elif axis < 0:
            axis = K.ndim(x) + axis
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = K.expand_dims(mask, 1)
        value = K.cast(value, K.dtype(x))
        for _ in range(K.ndim(x) - K.ndim(mask)):
            mask = K.expand_dims(mask, K.ndim(mask))
        x = x * mask + value * (1 - mask)
        if x_dtype == 'bool':
            x = K.cast(x, 'bool')
        return x


def root_mean_square(x, axis=None, keepdims=False):
    """均方根, 相当于模长的变体
    """
    return K.sqrt(K.mean(K.square(x), axis=axis, keepdims=keepdims))


def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明:
        1. y_true和y_pred的shape一致, y_true的元素非0即1, 1表示对应的类为目标类, 0表示对应的类为非目标类;
        2. 请保证y_pred的值域是全体实数, 换言之一般情况下y_pred不用加激活函数, 尤其是不能加sigmoid或者softmax;
        3. 预测阶段则输出y_pred大于0的类;
        4. 详情请看: https://kexue.fm/archives/7359.
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * K.infinity()
    y_pred_pos = y_pred - (1 - y_true) * K.infinity()
    zeros = K.zeros_like(y_pred[..., :1])
    y_pred_neg = K.concatenate([y_pred_neg, zeros], axis=-1)
    y_pred_pos = K.concatenate([y_pred_pos, zeros], axis=-1)
    neg_loss = tf.reduce_logsumexp(y_pred_neg, axis=-1)
    pos_loss = tf.reduce_logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss


# sys.modules['keras'] = keras
# 添加到 keras.backend 上, 使其可以像 K.epsilon() 那样操作
K.infinity = infinity
sys.modules['keras.backend'] = K
custom_objects = {
    'root_mean_square': root_mean_square,
    'multilabel_categorical_crossentropy': multilabel_categorical_crossentropy,
}

keras.utils.get_custom_objects().update(custom_objects)
