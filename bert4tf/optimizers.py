import numpy as np
from bert4tf.backend import tf, keras, K
from bert4tf.backend import root_mean_square as rms
from bert4tf.snippets import string_matching, insert_arguments


class Adam(keras.optimizers.Optimizer):
    """重新定义Adam优化器, 便于派生出新的优化器(tensorflow的optimizer_v2类)
    """
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-6, bias_correction=True, **kwargs):
        kwargs['name'] = kwargs.get('name') or 'Adam'
        super(Adam, self).__init__(**kwargs)
        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon or K.epislon()
        self.bias_correction = bias_correction

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')

    def _resource_apply(self, grad, var, indices=None):
        # 准备变量
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        epsilon_t = K.cast(self.epsilon, var_dtype)
        local_step = K.cast(self.iterations + 1, var_dtype)
        beta_1_t_power = K.pow(beta_1_t, local_step)
        beta_2_t_power = K.pow(beta_2_t, local_step)

        # 更新公式
        if indices is None:
            m_t = K.update(m, beta_1_t * m + (1 - beta_1_t) * grad)
            v_t = K.update(v, beta_2_t * v + (1 - beta_2_t) * K.square(grad))
        else:
            mv_ops = [K.update(m, beta_1_t * m), K.update(v, beta_2_t * v)]
            with tf.control_dependencies(mv_ops):
                m_t = self._resource_scatter_add(m, indices, (1 - beta_1_t) * grad)
                v_t = self._resource_scatter_add(v, indices, (1 - beta_2_t) * K.square(grad))

        # 返回算子
        with tf.control_dependencies([m_t, v_t]):
            if self.bias_correction:
                m_t = m_t / (1.0 - beta_1_t_power)
                v_t = v_t / (1.0 - beta_2_t_power)
            var_t = var - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)
            return K.update(var, var_t)

    def _resource_apply_dense(self, grad, var):
        return self._resource_apply(grad, var)

    def _resource_apply_sparse(self, grad, var, indices):
        return self._resource_apply(grad, var, indices)

    def get_config(self):
        config = {
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'bias_correction': self.bias_correction,
        }
        base_config = super(Adam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AdaFactorBase(keras.optimizers.Optimizer):
    """AdaFactor优化器(基类)
    论文链接: https://arxiv.org/abs/1804.04235
    参考实现: https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/optimize.py
    """
    def __init__(
        self,
        learning_rate=1e-3,  # 可以为None
        beta1=0.0,
        beta2=None,
        epsilon1=1e-30,
        epsilon2=1e-3,
        multiply_by_parameter_scale=True,
        clipping_threshold=1.0,
        min_dim_size_to_factor=128,
        exclude_from_parameter_scale=None,
        **kwargs
    ):
        super(AdaFactorBase, self).__init__(**kwargs)
        self._learning_rate = learning_rate
        self.beta1 = beta1
        self._beta2 = beta2
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.multiply_by_parameter_scale = multiply_by_parameter_scale
        self.clipping_threshold = clipping_threshold
        self.min_dim_size_to_factor = min_dim_size_to_factor
        self.exclude_from_parameter_scale = exclude_from_parameter_scale or []

    @property
    def learning_rate(self):
        if self._learning_rate is None:
            iterations = K.cast(self.iterations + 1, K.floatx())
            learning_rate = K.minimum(1.0 / K.sqrt(iterations), 0.01)
            if self.multiply_by_parameter_scale:
                return learning_rate
            else:
                return learning_rate * 0.05
        else:
            if not hasattr(self, '__learning_rate'):
                with K.name_scope(self.__class__.__name__):
                    self.__learning_rate = K.variable(self._learning_rate, name='learning_rate')
            return self.__learning_rate

    @property
    def beta2(self):
        if self._beta2 is None:
            iterations = K.cast(self.iterations + 1, K.floatx())
            return 1.0 - K.pow(iterations, -0.8)
        else:
            return self._beta2

    def factored_shape(self, shape):
        if len(shape) < 2:
            return None
        shape = np.array(shape)
        indices = shape.argpartition(-2)
        if shape[indices[-2]] < self.min_dim_size_to_factor:
            return None
        shape1, shape2 = np.array(shape), np.array(shape)
        shape1[indices[-1]] = 1
        shape2[indices[-2]] = 1
        return shape1, indices[-1], shape2, indices[-2]

    def _do_parameter_scale(self, w):
        return self.multiply_by_parameter_scale and (not string_matching(w.name, self.exclude_from_parameter_scale))

    def get_config(self):
        config = {
            'learning_rate': self._learning_rate,
            'beta1': self.beta1,
            'beta2': self._beta2,
            'epsilon1': self.epsilon1,
            'epsilon2': self.epsilon2,
            'multiply_by_parameter_scale': self.multiply_by_parameter_scale,
            'clipping_threshold': self.clipping_threshold,
            'min_dim_size_to_factor': self.min_dim_size_to_factor,
            'exclude_from_parameter_scale': self.exclude_from_parameter_scale,
        }
        base_config = super(AdaFactorBase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AdaFactor(AdaFactorBase):
    """AdaFactor优化器(tf.keras版)
    论文链接: https://arxiv.org/abs/1804.04235
    参考实现: https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/optimize.py
    """
    def __init__(self, *args, **kwargs):
        kwargs['name'] = kwargs.get('name') or 'AdaFactor'
        super(AdaFactor, self).__init__(*args, **kwargs)

    def _create_slots(self, var_list):
        for var in var_list:
            if self.beta1 > 0.0:
                self.add_slot(var, 'm')
            shape = K.int_shape(var)
            factored_shape = self.factored_shape(shape)
            if factored_shape is None:
                self.add_slot(var, 'v')
            else:
                shape1, axis1, shape2, axis2 = factored_shape
                value1, value2 = np.zeros(shape1), np.zeros(shape2)
                self.add_slot(var, 'vr', value1)
                self.add_slot(var, 'vc', value2)

    def _decayed_lr(self, var_dtype):
        return self._learning_rate

    def _resource_apply(self, grad, var, indices=None):
        lr = self._decayed_lr(var.dtype.base_dtype)
        g2 = K.square(grad) + self.epsilon1
        shape = K.int_shape(var)
        factored_shape = self.factored_shape(shape)
        if factored_shape is None:
            v = self.get_slot(var, 'v')
            # 定义更新
            v_t = self.beta2 * v + (1.0 - self.beta2) * g2
            v_t = K.update(v, v_t)
        else:
            shape1, axis1, shape2, axis2 = factored_shape
            vr = self.get_slot(var, 'vr')
            vc = self.get_slot(var, 'vc')
            # 定义更新
            g2r = K.mean(g2, axis=axis1, keepdims=True)
            g2c = K.mean(g2, axis=axis2, keepdims=True)
            vr_t = self.beta2 * vr + (1.0 - self.beta2) * g2r
            vc_t = self.beta2 * vc + (1.0 - self.beta2) * g2c
            vr_t, vc_t = K.update(vr, vr_t), K.update(vc, vc_t)
            # 合成矩阵
            v_t = vr_t * vc_t / K.mean(vr_t, axis=axis2, keepdims=True)
        # 增量主体
        u = grad / K.sqrt(v_t + self.epsilon1)
        # 增量裁剪
        if self.clipping_threshold is not None:
            u = u / K.maximum(1.0, rms(u) / self.clipping_threshold)
        # 增量滑动
        if self.beta1 > 0.0:
            m = self.get_slot(var, 'm')
            # 定义更新
            m_t = self.beta1 * m + (1.0 - self.beta1) * u
            u = K.update(m, m_t)
        # 增量调整
        if self._do_parameter_scale(var):
            u = u * K.maximum(rms(var), self.epsilon2)
        # 更新参数
        return K.update(var, var - lr * u)

    def _resource_apply_dense(self, grad, var):
        return self._resource_apply(grad, var)

    def _resource_apply_sparse(self, grad, var, indices):
        grad = tf.IndexedSlices(grad, indices, K.shape(var))
        grad = tf.convert_to_tensor(grad)
        return self._resource_apply_dense(grad, var)


def export_to_custom_objects(base_extend_with):
    """装饰器, 用来将优化器放到custom_objects中
    """
    def new_extend_with(BaseOptimizer, name=None):
        NewOptimizer = base_extend_with(BaseOptimizer)

        if isinstance(name, str):
            NewOptimizer.__name__ = name

        name = NewOptimizer.__name__
        keras.utils.get_custom_objects()[name] = NewOptimizer

        return NewOptimizer

    return new_extend_with


@export_to_custom_objects
def extend_with_gradient_accumulation(BaseOptimizer):
    """返回新的优化器类, 加入梯度累积
    """
    class NewOptimizer(BaseOptimizer):
        """带有梯度累积的优化器
        """
        @insert_arguments(grad_accum_steps=2)
        def __init__(self, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)

        def _create_slots(self, var_list):
            super(NewOptimizer, self)._create_slots(var_list)
            for var in var_list:
                self.add_slot(var, 'ag')

        def _resource_apply(self, grad, var, indices=None):
            # 更新判据
            cond = K.equal(self.iterations % self.grad_accum_steps, 0)
            # 获取梯度
            ag = self.get_slot(var, 'ag')

            old_update = K.update

            def new_update(x, new_x):
                new_x = K.switch(cond, new_x, x)
                return old_update(x, new_x)

            K.update = new_update
            ag_t = ag / self.grad_accum_steps
            op = super(NewOptimizer, self)._resource_apply(ag_t, var)
            K.update = old_update

            # 累积梯度
            with tf.control_dependencies([op]):
                ag_t = K.switch(cond, K.zeros_like(ag), ag)
                with tf.control_dependencies([K.update(ag, ag_t)]):
                    if indices is None:
                        ag_t = K.update(ag, ag + grad)
                    else:
                        ag_t = self._resource_scatter_add(ag, indices, grad)

            return ag_t

        def get_config(self):
            config = {'grad_accum_steps': self.grad_accum_steps}
            base_config = super(NewOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NewOptimizer


custom_objects = {
    'Adam': Adam,
    'AdaFactor': AdaFactor,
}

keras.utils.get_custom_objects().update(custom_objects)