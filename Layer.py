import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

w_initializer = keras.initializers.VarianceScaling
b_initializer = keras.initializers.constant


class GuideMap(keras.layers.Layer):
    def __init__(self):
        super(GuideMap, self).__init__()
        self.conv = keras.layers.Conv2D(filters=1, kernel_size=1, kernel_initializer=tf.initializers.constant(1 / 3),
                                        bias_initializer=tf.initializers.constant(0))
        npts = 16
        nchans = 3

        identity = np.identity(nchans, dtype=np.float32) + np.random.randn(1).astype(np.float32) * 1e-4
        self.ccm = tf.Variable(name='ccm', initial_value=identity, dtype=tf.float32)
        self.ccm_bias = self.add_weight(name='ccm_bias', shape=[nchans, ], dtype=tf.float32,
                                        initializer=tf.initializers.constant(0))
        shifts_ = np.linspace(0, 1, npts, endpoint=False, dtype=np.float32)
        shifts_ = shifts_[np.newaxis, np.newaxis, np.newaxis, :]
        shifts_ = np.tile(shifts_, (1, 1, nchans, 1))
        slopes_ = np.zeros([1, 1, 1, nchans, npts], dtype=np.float32)
        slopes_[:, :, :, :, 0] = 1.0

        self.shifts = tf.Variable(name='shifts', dtype=tf.float32, initial_value=shifts_)
        self.slopes = tf.Variable(name='slopes', dtype=tf.float32, initial_value=slopes_)

    def call(self, inputs, **kwargs):
        nchans = inputs.get_shape().as_list()[-1]
        guidemap = tf.matmul(tf.reshape(inputs, [-1, nchans]), self.ccm, name='ccm_matmul')
        guidemap = tf.nn.bias_add(guidemap, self.ccm_bias, name='ccm_bias_add')
        guidemap = tf.reshape(guidemap, tf.shape(inputs))
        guidemap = tf.expand_dims(guidemap, axis=4)
        guidemap = tf.reduce_sum(self.slopes * tf.nn.relu(guidemap - self.shifts), axis=[4])
        guidemap = self.conv(guidemap)
        guidemap = tf.clip_by_value(guidemap, 0, 1)
        guidemap = tf.squeeze(guidemap, axis=[3, ])
        return guidemap


class Conv2D(keras.layers.Layer):

    def __init__(self, kernel_num,
                 kernel_size,
                 name,
                 stride=1,
                 norm='bn',
                 use_bias=True,
                 activation='relu',
                 padding='same'):
        super(Conv2D, self).__init__(filter)
        self.norm = norm
        if norm == 'bn':
            self.norm = keras.layers.BatchNormalization(name=name + '_bn')
            b_init = None
        elif norm == 'in':
            self.norm = InstanceNormalization(name=name + '_in')
            b_init = None
        else:
            if use_bias:
                b_init = b_initializer(0)
            else:
                b_init = None

        self.conv = keras.layers.Conv2D(
            filters=kernel_num,
            kernel_size=kernel_size,
            strides=stride,
            kernel_initializer=w_initializer(),
            bias_initializer=b_init,
            name=name + '_conv',
            activation=getattr(tf.nn, activation) if not activation == 'None' else None,
            padding=padding
        )

    def call(self, inputs, **kwargs):
        outputs = self.conv(inputs)
        if self.norm:
            outputs = self.norm(outputs)
        return outputs


class FullConnect(keras.layers.Layer):
    def __init__(self, num_output,
                 name,
                 use_bias=True,
                 is_bn=True,
                 activation='relu'):
        super(FullConnect, self).__init__()
        self.is_bn = is_bn
        if is_bn:
            self.bn = keras.layers.BatchNormalization(name=name + '_bn')
            b_init = None
        else:
            if use_bias:
                b_init = b_initializer(0)
            else:
                b_init = None
        self.fc = keras.layers.Dense(
            num_output,
            name=name + '_fc',
            kernel_initializer=w_initializer(),
            bias_initializer=b_init,
            activation=None if activation == 'None' else getattr(tf.nn, activation)
        )

    def call(self, inputs, **kwargs):
        if self.is_bn:
            outputs = self.bn(self.fc(inputs))
        else:
            outputs = self.fc(inputs)
        return outputs


class InstanceNormalization(keras.layers.Layer):
    """Instance normalization layer.
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model_604.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """

    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = keras.layers.InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = keras.backend.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = keras.backend.mean(inputs, reduction_axes, keepdims=True)
        stddev = keras.backend.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = keras.backend.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = keras.backend.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': keras.constraints.serialize(self.beta_constraint),
            'gamma_constraint': keras.constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
