import tensorflow as tf
from tensorflow.keras.layers import LSTM, Conv1D, Dense, Dropout, BatchNormalization, Activation, Lambda, Add, Input

def lstm_layer(inputs, lengths, state_size, keep_prob=1.0, return_final_state=False):
    """
    LSTM layer.

    Args:
        inputs: Tensor of shape [batch size, max sequence length, ...].
        lengths: Tensor of shape [batch size].
        state_size: LSTM state size.
        keep_prob: 1 - p, where p is the dropout probability.

    Returns:
        Tensor of shape [batch size, max sequence length, state_size] containing the lstm
        outputs at each timestep.

    """
    cell_fw = LSTM(state_size, return_sequences=True, return_state=return_final_state, dropout=1-keep_prob)
    outputs = cell_fw(inputs)
    return outputs


def temporal_convolution_layer(inputs, output_units, convolution_width, causal=False, dilation_rate=1, bias=True,
                               activation=None, dropout=None, scope='temporal-convolution-layer'):
    """
    Convolution over the temporal axis of sequence data using Keras layers.

    Args:
        inputs: Tensor of shape [batch size, max sequence length, input_units].
        output_units: Output channels for convolution.
        convolution_width: Number of timesteps to use in convolution.
        causal: Output at timestep t is a function of inputs at or before timestep t.
        dilation_rate:  Dilation rate along temporal axis.

    Returns:
        Tensor of shape [batch size, max sequence length, output_units].

    """
    if causal:
        shift = (convolution_width // 2) + ((dilation_rate - 1) // 2)
        pad = Lambda(lambda x: tf.pad(x, [[0, 0], [shift, 0], [0, 0]]))(inputs)
        inputs = pad

    conv_layer = Conv1D(filters=output_units, kernel_size=convolution_width, dilation_rate=dilation_rate, padding='same', use_bias=bias)
    z = conv_layer(inputs)

    if activation is not None:
        z = Activation(activation)(z)

    if dropout is not None:
        z = Dropout(dropout)(z)

    z = Lambda(lambda x: x[:, :-shift, :] if causal else x)(z)
    return z


def time_distributed_dense_layer(inputs, output_units, bias=True, activation=None, batch_norm=None, dropout=None,
                                 scope='time-distributed-dense-layer'):
    """
    Applies a shared dense layer to each timestep of a tensor of shape [batch_size, max_seq_len, input_units]
    to produce a tensor of shape [batch_size, max_seq_len, output_units].

    Args:
        inputs: Tensor of shape [batch size, max sequence length, ...].
        output_units: Number of output units.
        activation: activation function.
        dropout: dropout keep prob.

    Returns:
        Tensor of shape [batch size, max sequence length, output_units].

    """
    dense_layer = Dense(units=output_units, use_bias=bias)
    z = dense_layer(inputs)

    if activation is not None:
        z = Activation(activation)(z)

    if batch_norm is not None:
        z = BatchNormalization()(z)

    if dropout is not None:
        z = Dropout(dropout)(z)

    return z


def dense_layer(inputs, output_units, bias=True, activation=None, batch_norm=None, dropout=None):
    """
    Applies a dense layer to a 2D tensor of shape [batch_size, input_units]
    to produce a tensor of shape [batch_size, output_units].

    Args:
        inputs: Tensor of shape [batch size, input_units].
        output_units: Number of output units.
        activation: Activation function.
        dropout: Dropout rate.

    Returns:
        Tensor of shape [batch size, output_units].
    """
    dense_layer = Dense(units=output_units, use_bias=bias, activation=activation)
    z = dense_layer(inputs)

    if batch_norm is not None:
        z = BatchNormalization()(z)

    if dropout is not None:
        z = Dropout(dropout)(z)

    return z


def wavenet(x, dilations, filter_widths, skip_channels, residual_channels, scope='wavenet'):
    """
    A stack of causal dilated convolutions with parameterized residual and skip connections.

    Args:
        x: Input tensor of shape [batch size, max sequence length, input units].
        dilations: List of dilations for each layer. len(dilations) is the number of layers.
        filter_widths: List of filter widths. Same length as dilations.
        skip_channels: Number of channels to use for skip connections.
        residual_channels: Number of channels to use for residual connections.

    Returns:
        Tensor of shape [batch size, max sequence length, len(dilations)*skip_channels].
    """
    inputs = Input(shape=x.shape[1:])  # Create a Keras Input layer

    # wavenet uses 2x1 conv here
    inputs_proj = time_distributed_dense_layer(inputs, residual_channels, activation='tanh', scope='x-proj')

    skip_outputs = []
    for i, (dilation, filter_width) in enumerate(zip(dilations, filter_widths)):
        dilated_conv = temporal_convolution_layer(
            inputs=inputs_proj,
            output_units=2 * residual_channels,
            convolution_width=filter_width,
            causal=True,
            dilation_rate=dilation,
            scope='cnn-{}'.format(i)
        )
        conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=2)
        dilated_conv = tf.nn.tanh(conv_filter) * tf.nn.sigmoid(conv_gate)

        output_units = skip_channels + residual_channels
        outputs = time_distributed_dense_layer(dilated_conv, output_units, scope='cnn-{}-proj'.format(i))
        skips, residuals = tf.split(outputs, [skip_channels, residual_channels], axis=2)

        inputs_proj = Add()([inputs_proj, residuals])
        skip_outputs.append(skips)

    skip_outputs = Activation('relu')(Add()(skip_outputs))

    return skip_outputs


"""
Source of this Entmax 1.5 implementation: 
https://gist.github.com/BenjaminWegener/8fad40ffd80fbe9087d13ad464a48ca9 

It was developed by: 
https://gist.github.com/BenjaminWegener
"""


def entmax15(inputs, axis=-1):
    """
    Entmax 1.5 implementation, heavily inspired by
     * paper: https://arxiv.org/pdf/1905.05702.pdf
     * pytorch code: https://github.com/deep-spin/entmax
    :param inputs: similar to softmax logits, but for entmax1.5
    :param axis: entmax1.5 outputs will sum to 1 over this axis
    :return: entmax activations of same shape as inputs
    """
    @tf.custom_gradient
    def _entmax_inner(inputs):
        with tf.name_scope('entmax'):
            inputs = inputs / 2  # divide by 2 so as to solve actual entmax
            inputs -= tf.reduce_max(inputs, axis, keepdims=True)  # subtract max for stability

            threshold, _ = entmax_threshold_and_support(inputs, axis)
            outputs_sqrt = tf.nn.relu(inputs - threshold)
            outputs = tf.square(outputs_sqrt)

        def grad_fn(d_outputs):
            with tf.name_scope('entmax_grad'):
                d_inputs = d_outputs * outputs_sqrt
                q = tf.reduce_sum(d_inputs, axis=axis, keepdims=True) 
                q = q / tf.reduce_sum(outputs_sqrt, axis=axis, keepdims=True)
                d_inputs -= q * outputs_sqrt
                return d_inputs
    
        return outputs, grad_fn
    
    return _entmax_inner(inputs)


@tf.custom_gradient
def sparse_entmax15_loss_with_logits(labels, logits):
    """
    Computes sample-wise entmax1.5 loss
    :param labels: reference answers vector int64[batch_size] \in [0, num_classes)
    :param logits: output matrix float32[batch_size, num_classes] (not actually logits :)
    :returns: elementwise loss, float32[batch_size]
    """
    assert logits.shape.ndims == 2 and labels.shape.ndims == 1
    with tf.name_scope('entmax_loss'):
        p_star = entmax15(logits, axis=-1)
        omega_entmax15 = (1 - (tf.reduce_sum(p_star * tf.sqrt(p_star), axis=-1))) / 0.75
        p_incr = p_star - tf.one_hot(labels, depth=tf.shape(logits)[-1], axis=-1)
        loss = omega_entmax15 + tf.einsum("ij,ij->i", p_incr, logits)
    
    def grad_fn(grad_output):
        with tf.name_scope('entmax_loss_grad'):
            return None, grad_output[..., None] * p_incr

    return loss, grad_fn


@tf.custom_gradient
def entmax15_loss_with_logits(labels, logits):
    """
    Computes sample-wise entmax1.5 loss
    :param logits: "logits" matrix float32[batch_size, num_classes]
    :param labels: reference answers indicators, float32[batch_size, num_classes]
    :returns: elementwise loss, float32[batch_size]
    
    WARNING: this function does not propagate gradients through :labels:
    This behavior is the same as like softmax_crossentropy_with_logits v1
    It may become an issue if you do something like co-distillation
    """
    assert labels.shape.ndims == logits.shape.ndims == 2
    with tf.name_scope('entmax_loss'):
        p_star = entmax15(logits, axis=-1)
        omega_entmax15 = (1 - (tf.reduce_sum(p_star * tf.sqrt(p_star), axis=-1))) / 0.75
        p_incr = p_star - labels
        loss = omega_entmax15 + tf.einsum("ij,ij->i", p_incr, logits)

    def grad_fn(grad_output):
        with tf.name_scope('entmax_loss_grad'):
            return None, grad_output[..., None] * p_incr

    return loss, grad_fn


def top_k_over_axis(inputs, k, axis=-1, **kwargs):
    """ performs tf.nn.top_k over any chosen axis """
    with tf.name_scope('top_k_along_axis'):
        if axis == -1:
            return tf.nn.top_k(inputs, k, **kwargs)

        perm_order = list(range(inputs.shape.ndims))
        perm_order.append(perm_order.pop(axis))
        inv_order = [perm_order.index(i) for i in range(len(perm_order))]

        input_perm = tf.transpose(inputs, perm_order)
        input_perm_sorted, sort_indices_perm = tf.nn.top_k(
            input_perm, k=k, **kwargs)

        input_sorted = tf.transpose(input_perm_sorted, inv_order)
        sort_indices = tf.transpose(sort_indices_perm, inv_order)
    return input_sorted, sort_indices


def _make_ix_like(inputs, axis=-1):
    """ creates indices 0, ... , input[axis] unsqueezed to input dimensios """
    assert inputs.shape.ndims is not None
    rho = tf.cast(tf.range(1, tf.shape(inputs)[axis] + 1), dtype=inputs.dtype)
    view = [1] * inputs.shape.ndims
    view[axis] = -1
    return tf.reshape(rho, view)


def gather_over_axis(values, indices, gather_axis):
    """
    replicates the behavior of torch.gather for tf<=1.8;
    for newer versions use tf.gather with batch_dims
    :param values: tensor [d0, ..., dn]
    :param indices: int64 tensor of same shape as values except for gather_axis
    :param gather_axis: performs gather along this axis
    :returns: gathered values, same shape as values except for gather_axis
        If gather_axis == 2
        gathered_values[i, j, k, ...] = values[i, j, indices[i, j, k, ...], ...]
        see torch.gather for more detils
    """
    assert indices.shape.ndims is not None
    assert indices.shape.ndims == values.shape.ndims

    ndims = indices.shape.ndims
    gather_axis = gather_axis % ndims
    shape = tf.shape(indices)

    selectors = []
    for axis_i in range(ndims):
        if axis_i == gather_axis:
            selectors.append(indices)
        else:
            index_i = tf.range(tf.cast(shape[axis_i], dtype=indices.dtype), dtype=indices.dtype)
            index_i = tf.reshape(index_i, [-1 if i == axis_i else 1 for i in range(ndims)])
            index_i = tf.tile(index_i, [shape[i] if i != axis_i else 1 for i in range(ndims)])
            selectors.append(index_i)

    return tf.gather_nd(values, tf.stack(selectors, axis=-1))


def entmax_threshold_and_support(inputs, axis=-1):
    """
    Computes clipping threshold for entmax1.5 over specified axis
    NOTE this implementation uses the same heuristic as
    the original code: https://tinyurl.com/pytorch-entmax-line-203
    :param inputs: (entmax1.5 inputs - max) / 2
    :param axis: entmax1.5 outputs will sum to 1 over this axis
    """

    with tf.name_scope('entmax_threshold_and_support'):
        num_outcomes = tf.shape(inputs)[axis]
        inputs_sorted, _ = top_k_over_axis(inputs, k=num_outcomes, axis=axis, sorted=True)

        rho = _make_ix_like(inputs, axis=axis)

        mean = tf.cumsum(inputs_sorted, axis=axis) / rho

        mean_sq = tf.cumsum(tf.square(inputs_sorted), axis=axis) / rho
        delta = (1 - rho * (mean_sq - tf.square(mean))) / rho

        delta_nz = tf.nn.relu(delta)
        tau = mean - tf.sqrt(delta_nz)

        support_size = tf.reduce_sum(tf.cast(tf.less_equal(tau, inputs_sorted), tf.int64), axis=axis, keepdims=True)

        tau_star = gather_over_axis(tau, support_size - 1, axis)
    return tau_star, support_size
