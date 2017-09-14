import tensorflow as tf
import tensorflow.contrib.slim as slim






from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.layers import convolutional as convolutional_layers
from tensorflow.python.layers import core as core_layers
from tensorflow.python.layers import normalization as normalization_layers
from tensorflow.python.layers import pooling as pooling_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training import moving_averages


def batch_normalization(x,
                        mean,
                        variance,
                        offset,
                        scale,
                        variance_epsilon,
                        name=None,
                        noise_std=None):
  r"""Batch normalization.

  As described in http://arxiv.org/abs/1502.03167.
  Normalizes a tensor by `mean` and `variance`, and applies (optionally) a
  `scale` \\(\gamma\\) to it, as well as an `offset` \\(\beta\\):

  \\(\frac{\gamma(x-\mu)}{\sigma}+\beta\\)

  `mean`, `variance`, `offset` and `scale` are all expected to be of one of two
  shapes:

    * In all generality, they can have the same number of dimensions as the
      input `x`, with identical sizes as `x` for the dimensions that are not
      normalized over (the 'depth' dimension(s)), and dimension 1 for the
      others which are being normalized over.
      `mean` and `variance` in this case would typically be the outputs of
      `tf.nn.moments(..., keep_dims=True)` during training, or running averages
      thereof during inference.
    * In the common case where the 'depth' dimension is the last dimension in
      the input tensor `x`, they may be one dimensional tensors of the same
      size as the 'depth' dimension.
      This is the case for example for the common `[batch, depth]` layout of
      fully-connected layers, and `[batch, height, width, depth]` for
      convolutions.
      `mean` and `variance` in this case would typically be the outputs of
      `tf.nn.moments(..., keep_dims=False)` during training, or running averages
      thereof during inference.

  Args:
    x: Input `Tensor` of arbitrary dimensionality.
    mean: A mean `Tensor`.
    variance: A variance `Tensor`.
    offset: An offset `Tensor`, often denoted \\(\beta\\) in equations, or
      None. If present, will be added to the normalized tensor.
    scale: A scale `Tensor`, often denoted \\(\gamma\\) in equations, or
      `None`. If present, the scale is applied to the normalized tensor.
    variance_epsilon: A small float number to avoid dividing by 0.
    name: A name for this operation (optional).

  Returns:
    the normalized, scaled, offset tensor.
  """
  with ops.name_scope(name, "batchnorm", [x, mean, variance, scale, offset]):
    inv = math_ops.rsqrt(variance + variance_epsilon)
    if scale is not None:
      inv *= scale
    if noise_std:
        x = _noise(x, noise_std)
    return x * inv + (offset - mean * inv
                      if offset is not None else -mean * inv)


@slim.layers.add_arg_scope
def custom_batch_norm(inputs,
                      decay=0.999,
                      center=True,
                      scale=False,
                      epsilon=0.001,
                      activation_fn=None,
                      param_initializers=None,
                      param_regularizers=None,
                      updates_collections=ops.GraphKeys.UPDATE_OPS,
                      is_training=True,
                      reuse=None,
                      variables_collections=None,
                      outputs_collections=None,
                      trainable=True,
                      batch_weights=None,
                      data_format='NHWC',
                      zero_debias_moving_mean=False,
                      scope=None,
                      renorm=False,
                      renorm_clipping=None,
                      renorm_decay=0.99,
                      noise_std=None):
    """Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.

      "Batch Normalization: Accelerating Deep Network Training by Reducing
      Internal Covariate Shift"

      Sergey Ioffe, Christian Szegedy

    Can be used as a normalizer function for conv2d and fully_connected.

    Note: when training, the moving_mean and moving_variance need to be updated.
    By default the update ops are placed in `tf.GraphKeys.UPDATE_OPS`, so they
    need to be added as a dependency to the `train_op`. For example:

    ```python
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)
    ```

    One can set updates_collections=None to force the updates in place, but that
    can have a speed penalty, especially in distributed settings.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`. The normalization is over all but the last dimension if
        `data_format` is `NHWC` and the second dimension if `data_format` is
        `NCHW`.
      decay: Decay for the moving average. Reasonable values for `decay` are close
        to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
        Lower `decay` value (recommend trying `decay`=0.9) if model experiences
        reasonably good training performance but poor validation and/or test
        performance. Try zero_debias_moving_mean=True for improved stability.
      center: If True, add offset of `beta` to normalized tensor. If False, `beta`
        is ignored.
      scale: If True, multiply by `gamma`. If False, `gamma` is
        not used. When the next layer is linear (also e.g. `nn.relu`), this can be
        disabled since the scaling can be done by the next layer.
      epsilon: Small float added to variance to avoid dividing by zero.
      activation_fn: Activation function, default set to None to skip it and
        maintain a linear activation.
      param_initializers: Optional initializers for beta, gamma, moving mean and
        moving variance.
      param_regularizers: Optional regularizer for beta and gamma.
      updates_collections: Collections to collect the update ops for computation.
        The updates_ops need to be executed with the train_op.
        If None, a control dependency would be added to make sure the updates are
        computed in place.
      is_training: Whether or not the layer is in training mode. In training mode
        it would accumulate the statistics of the moments into `moving_mean` and
        `moving_variance` using an exponential moving average with the given
        `decay`. When it is not in training mode then it would use the values of
        the `moving_mean` and the `moving_variance`.
      reuse: Whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
      variables_collections: Optional collections for the variables.
      outputs_collections: Collections to add the outputs.
      trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
      batch_weights: An optional tensor of shape `[batch_size]`,
        containing a frequency weight for each batch item. If present,
        then the batch normalization uses weighted mean and
        variance. (This can be used to correct for bias in training
        example selection.)
      fused:  Use nn.fused_batch_norm if True, nn.batch_normalization otherwise.
      data_format: A string. `NHWC` (default) and `NCHW` are supported.
      zero_debias_moving_mean: Use zero_debias for moving_mean. It creates a new
        pair of variables 'moving_mean/biased' and 'moving_mean/local_step'.
      scope: Optional scope for `variable_scope`.
      renorm: Whether to use Batch Renormalization
        (https://arxiv.org/abs/1702.03275). This adds extra variables during
        training. The inference is the same for either value of this parameter.
      renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
        scalar `Tensors` used to clip the renorm correction. The correction
        `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
        `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
        dmax are set to inf, 0, inf, respectively.
      renorm_decay: Momentum used to update the moving means and standard
        deviations with renorm. Unlike `momentum`, this affects training
        and should be neither too small (which would add noise) nor too large
        (which would give stale estimates). Note that `decay` is still applied
        to get the means and variances for inference.

    Returns:
      A `Tensor` representing the output of the operation.

    Raises:
      ValueError: If `batch_weights` is not None and `fused` is True.
      ValueError: If `param_regularizers` is not None and `fused` is True.
      ValueError: If `data_format` is neither `NHWC` nor `NCHW`.
      ValueError: If the rank of `inputs` is undefined.
      ValueError: If rank or channels dimension of `inputs` is undefined.
    """

    layer_variable_getter = slim.layers._build_variable_getter()
    with variable_scope.variable_scope(
            scope, 'BatchNorm', [inputs], reuse=reuse,
            custom_getter=layer_variable_getter) as sc:
        inputs = ops.convert_to_tensor(inputs)

        # Determine whether we can use the core layer class.
        if (batch_weights is None and
                    updates_collections is ops.GraphKeys.UPDATE_OPS and
                not zero_debias_moving_mean):
            # Use the core layer class.
            axis = 1 if data_format == 'NCHW' else -1
            if not param_initializers:
                param_initializers = {}
            beta_initializer = param_initializers.get('beta',
                                                      init_ops.zeros_initializer())
            gamma_initializer = param_initializers.get('gamma',
                                                       init_ops.ones_initializer())
            moving_mean_initializer = param_initializers.get(
                'moving_mean', init_ops.zeros_initializer())
            moving_variance_initializer = param_initializers.get(
                'moving_variance', init_ops.ones_initializer())
            if not param_regularizers:
                param_regularizers = {}
            beta_regularizer = param_regularizers.get('beta')
            gamma_regularizer = param_regularizers.get('gamma')
            layer = normalization_layers.BatchNormalization(
                axis=axis,
                momentum=decay,
                epsilon=epsilon,
                center=center,
                scale=scale,
                beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer,
                moving_mean_initializer=moving_mean_initializer,
                moving_variance_initializer=moving_variance_initializer,
                beta_regularizer=beta_regularizer,
                gamma_regularizer=gamma_regularizer,
                trainable=trainable,
                renorm=renorm,
                renorm_clipping=renorm_clipping,
                renorm_momentum=renorm_decay,
                name=sc.name,
                _scope=sc,
                _reuse=reuse)
            outputs = layer.apply(inputs, training=is_training)

            # Add variables to collections.
            slim.layers._add_variable_to_collections(
                layer.moving_mean, variables_collections, 'moving_mean')
            slim.layers._add_variable_to_collections(
                layer.moving_variance, variables_collections, 'moving_variance')
            if layer.beta:
                slim.layers._add_variable_to_collections(layer.beta, variables_collections, 'beta')
            if layer.gamma:
                slim.layers._add_variable_to_collections(
                    layer.gamma, variables_collections, 'gamma')

            if activation_fn is not None:
                outputs = activation_fn(outputs)
            return utils.collect_named_outputs(outputs_collections,
                                               sc.original_name_scope, outputs)

        # Not supported by layer class: batch_weights argument,
        # and custom updates_collections. In that case, use the legacy BN
        # implementation.
        # Custom updates collections are not supported because the update logic
        # is different in this case, in particular w.r.t. "forced updates" and
        # update op reuse.
        if renorm:
            raise ValueError('renorm is not supported with batch_weights, '
                             'updates_collections or zero_debias_moving_mean')
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        if inputs_rank is None:
            raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        dtype = inputs.dtype.base_dtype
        if batch_weights is not None:
            batch_weights = ops.convert_to_tensor(batch_weights)
            inputs_shape[0:1].assert_is_compatible_with(batch_weights.get_shape())
            # Reshape batch weight values so they broadcast across inputs.
            nshape = [-1] + [1 for _ in range(inputs_rank - 1)]
            batch_weights = array_ops.reshape(batch_weights, nshape)

        if data_format == 'NCHW':
            moments_axes = [0] + list(range(2, inputs_rank))
            params_shape = inputs_shape[1:2]
            # For NCHW format, rather than relying on implicit broadcasting, we
            # explicitly reshape the params to params_shape_broadcast when computing
            # the moments and the batch normalization.
            params_shape_broadcast = list(
                [1, inputs_shape[1].value] + [1 for _ in range(2, inputs_rank)])
        else:
            moments_axes = list(range(inputs_rank - 1))
            params_shape = inputs_shape[-1:]
            params_shape_broadcast = None
        if not params_shape.is_fully_defined():
            raise ValueError('Inputs %s has undefined channels dimension %s.' % (
                inputs.name, params_shape))

        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        if not param_initializers:
            param_initializers = {}
        if center:
            beta_collections = utils.get_variable_collections(variables_collections,
                                                              'beta')
            beta_initializer = param_initializers.get('beta',
                                                      init_ops.zeros_initializer())
            beta = variables.model_variable('beta',
                                            shape=params_shape,
                                            dtype=dtype,
                                            initializer=beta_initializer,
                                            collections=beta_collections,
                                            trainable=trainable)
        if scale:
            gamma_collections = utils.get_variable_collections(variables_collections,
                                                               'gamma')
            gamma_initializer = param_initializers.get('gamma',
                                                       init_ops.ones_initializer())
            gamma = variables.model_variable('gamma',
                                             shape=params_shape,
                                             dtype=dtype,
                                             initializer=gamma_initializer,
                                             collections=gamma_collections,
                                             trainable=trainable)

        # Create moving_mean and moving_variance variables and add them to the
        # appropriate collections. We disable variable partitioning while creating
        # them, because assign_moving_average is not yet supported for partitioned
        # variables.
        partitioner = variable_scope.get_variable_scope().partitioner
        try:
            variable_scope.get_variable_scope().set_partitioner(None)
            moving_mean_collections = utils.get_variable_collections(
                variables_collections, 'moving_mean')
            moving_mean_initializer = param_initializers.get(
                'moving_mean', init_ops.zeros_initializer())
            moving_mean = variables.model_variable(
                'moving_mean',
                shape=params_shape,
                dtype=dtype,
                initializer=moving_mean_initializer,
                trainable=False,
                collections=moving_mean_collections)
            moving_variance_collections = utils.get_variable_collections(
                variables_collections, 'moving_variance')
            moving_variance_initializer = param_initializers.get(
                'moving_variance', init_ops.ones_initializer())
            moving_variance = variables.model_variable(
                'moving_variance',
                shape=params_shape,
                dtype=dtype,
                initializer=moving_variance_initializer,
                trainable=False,
                collections=moving_variance_collections)
        finally:
            variable_scope.get_variable_scope().set_partitioner(partitioner)

        # If `is_training` doesn't have a constant value, because it is a `Tensor`,
        # a `Variable` or `Placeholder` then is_training_value will be None and
        # `needs_moments` will be true.
        is_training_value = utils.constant_value(is_training)
        need_moments = is_training_value is None or is_training_value
        if need_moments:
            # Calculate the moments based on the individual batch.
            if batch_weights is None:
                if data_format == 'NCHW':
                    mean, variance = nn.moments(inputs, moments_axes, keep_dims=True)
                    mean = array_ops.reshape(mean, [-1])
                    variance = array_ops.reshape(variance, [-1])
                else:
                    mean, variance = nn.moments(inputs, moments_axes)
            else:
                if data_format == 'NCHW':
                    mean, variance = nn.weighted_moments(inputs, moments_axes,
                                                         batch_weights, keep_dims=True)
                    mean = array_ops.reshape(mean, [-1])
                    variance = array_ops.reshape(variance, [-1])
                else:
                    mean, variance = nn.weighted_moments(inputs, moments_axes,
                                                         batch_weights)

            moving_vars_fn = lambda: (moving_mean, moving_variance)
            if updates_collections is None:
                def _force_updates():
                    """Internal function forces updates moving_vars if is_training."""
                    update_moving_mean = moving_averages.assign_moving_average(
                        moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
                    update_moving_variance = moving_averages.assign_moving_average(
                        moving_variance, variance, decay, zero_debias=False)
                    with ops.control_dependencies([update_moving_mean,
                                                   update_moving_variance]):
                        return array_ops.identity(mean), array_ops.identity(variance)

                mean, variance = utils.smart_cond(is_training,
                                                  _force_updates,
                                                  moving_vars_fn)
            else:
                def _delay_updates():
                    """Internal function that delay updates moving_vars if is_training."""
                    update_moving_mean = moving_averages.assign_moving_average(
                        moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
                    update_moving_variance = moving_averages.assign_moving_average(
                        moving_variance, variance, decay, zero_debias=False)
                    return update_moving_mean, update_moving_variance

                update_mean, update_variance = utils.smart_cond(is_training,
                                                                _delay_updates,
                                                                moving_vars_fn)
                ops.add_to_collections(updates_collections, update_mean)
                ops.add_to_collections(updates_collections, update_variance)
                # Use computed moments during training and moving_vars otherwise.
                vars_fn = lambda: (mean, variance)
                mean, variance = utils.smart_cond(is_training, vars_fn, moving_vars_fn)
        else:
            mean, variance = moving_mean, moving_variance
        if data_format == 'NCHW':
            mean = array_ops.reshape(mean, params_shape_broadcast)
            variance = array_ops.reshape(variance, params_shape_broadcast)
            beta = array_ops.reshape(beta, params_shape_broadcast)
            if gamma is not None:
                gamma = array_ops.reshape(gamma, params_shape_broadcast)

        # Compute batch_normalization.
        outputs = batch_normalization(inputs, mean, variance, beta, gamma,
                                      epsilon, noise_std=noise_std)
        outputs.set_shape(inputs_shape)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope, outputs)









# Function for adding batch normalization beta parameter
def _add_bias(data):
    own_beta = tf.get_variable('own_beta', shape=data.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    return data + own_beta


# Function for scaling by batch normalization gamma parameter
def _apply_scale(data):
    own_gamma = tf.get_variable('own_gamma', shape=data.get_shape()[-1], initializer=tf.constant_initializer(1.0))
    return data * own_gamma


def _gamma_layer(data, activation_fn, is_training, is_unlabeled, noise_std, ema, bn_assigns):
    with tf.variable_scope('enc', reuse=not is_training):
        running_mean_enc = tf.get_variable('running_mean_enc', shape=[data.get_shape()[-1]], trainable=False,
                                           initializer=tf.constant_initializer(0.0))
        running_var_enc = tf.get_variable('running_var_enc', shape=[data.get_shape()[-1]], trainable=False,
                                          initializer=tf.constant_initializer(1.0))
    mean_enc, var_enc = tf.nn.moments(data, axes=[0])
    if is_unlabeled:
        assign_mean_enc = running_mean_enc.assign(mean_enc)
        assign_var_enc = running_var_enc.assign(var_enc)
        bn_assigns.append(ema.apply([running_mean_enc, running_var_enc]))
        with tf.control_dependencies([assign_mean_enc, assign_var_enc]):
            normalized_enc = (data - mean_enc) / tf.sqrt(var_enc + 1e-10)
    elif is_training:
        normalized_enc = (data - mean_enc) / tf.sqrt(var_enc + 1e-10)
    else:
        normalized_enc = (data - ema.average(running_mean_enc)) / tf.sqrt(ema.average(running_var_enc) + 1e-10)

    z_tilde = _noise(normalized_enc, noise_std)
    with tf.variable_scope('bn_correct', reuse=not is_training):
        bn_corrected_tilde = _apply_scale(_add_bias(z_tilde))
    h_tilde = activation_fn(bn_corrected_tilde)

    z = normalized_enc
    with tf.variable_scope('bn_correct', reuse=True):
        bn_corrected = _apply_scale(_add_bias(z))
    h = activation_fn(bn_corrected)

    with tf.variable_scope('dec', reuse=not is_training):
        running_mean_dec = tf.get_variable('running_mean_dec', shape=[data.get_shape()[-1]], trainable=False,
                                           initializer=tf.constant_initializer(0.0))
        running_var_dec = tf.get_variable('running_var_dec', shape=[data.get_shape()[-1]], trainable=False,
                                          initializer=tf.constant_initializer(1.0))
        mean_dec, var_dec = tf.nn.moments(h_tilde, axes=[0])
    if is_unlabeled:
        assign_mean_dec = running_mean_dec.assign(mean_dec)
        assign_var_dec = running_var_dec.assign(var_dec)
        bn_assigns.append(ema.apply([running_mean_dec, running_var_dec]))
        with tf.control_dependencies([assign_mean_dec, assign_var_dec]):
            normalized_dec = (h_tilde - mean_dec) / tf.sqrt(var_dec + 1e-10)
    elif is_training:
        normalized_dec = (h_tilde - mean_dec) / tf.sqrt(var_dec + 1e-10)
    else:
        normalized_dec = (h_tilde - ema.average(running_mean_dec)) / tf.sqrt(ema.average(running_var_dec) + 1e-10)

    with tf.variable_scope('g', reuse=not is_training):
        z_est = _g(z_tilde, normalized_dec)

    return h_tilde, h, z_est, z


def _leaky_relu(features, name=None):
    alpha = 0.1
    return tf.maximum(features, alpha * features)


def _g_m(u):
    a1 = tf.get_variable('a1', shape=u.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    a2 = tf.get_variable('a2', shape=u.get_shape()[-1], initializer=tf.constant_initializer(1.0))
    a3 = tf.get_variable('a3', shape=u.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    a4 = tf.get_variable('a4', shape=u.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    a5 = tf.get_variable('a5', shape=u.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    return a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5


def _g_v(u):
    a6 = tf.get_variable('a6', shape=u.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    a7 = tf.get_variable('a7', shape=u.get_shape()[-1], initializer=tf.constant_initializer(1.0))
    a8 = tf.get_variable('a8', shape=u.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    a9 = tf.get_variable('a9', shape=u.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    a10 = tf.get_variable('a10', shape=u.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    return a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10


# The combinator function described in the paper, initial values from https://github.com/CuriousAI/ladder/
def _g(z_crt, u):
    m = _g_m(u)
    return (z_crt - m) * _g_v(u) + m


def _noise(data, noise_std):
    new_noise = tf.random_normal(shape=tf.shape(data), mean=0.0, stddev=noise_std, dtype=tf.float32)
    result = tf.add(data, new_noise)
    result.set_shape(data.get_shape())
    return result


def cifar10_gamma(inputs, is_training, is_unlabeled, ema, bn_assigns, batch_norm_decay, noise_std):
    inputs = tf.cast(inputs, tf.float32)
    net = inputs
    with tf.variable_scope('model', reuse=not is_training):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=_leaky_relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training or is_unlabeled,
                                               'decay': batch_norm_decay}):
            net = slim.conv2d(net, 96, [3, 3], scope='conv1_1')
            net = slim.conv2d(net, 96, [3, 3], scope='conv1_2')
            net = slim.conv2d(net, 96, [3, 3], scope='conv1_3')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')

            net = slim.conv2d(net, 192, [3, 3], scope='conv2_1')
            net = slim.conv2d(net, 192, [3, 3], scope='conv2_2')
            net = slim.conv2d(net, 192, [3, 3], scope='conv2_3')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')

            net = slim.conv2d(net, 192, [3, 3], scope='conv3_1')
            net = slim.conv2d(net, 192, [1, 1], scope='conv3_2')
            net = slim.conv2d(net, 10, [1, 1], scope='conv3_3')
            net = slim.avg_pool2d(net, [7, 7], scope='pool3')

            net = slim.flatten(net, scope='flatten')

    logits_crt, logits_cln, z_crt, z_cln = _gamma_layer(net,
                                                        lambda x: x,
                                                        is_training=is_training,
                                                        is_unlabeled=is_unlabeled,
                                                        noise_std=noise_std,
                                                        ema=ema,
                                                        bn_assigns=bn_assigns)
    #return logits_crt, logits_cln, z_crt, z_cln
    return net, net, net, net


def cifar10_supervised_rasmus(inputs, is_training, batch_norm_decay=0.9):
    inputs = tf.cast(inputs, tf.float32)
    net = inputs
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=_leaky_relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training, 'decay': batch_norm_decay}):
        net = slim.conv2d(net, 96, [3, 3], scope='conv1_1')
        net = slim.conv2d(net, 96, [3, 3], scope='conv1_2')
        net = slim.conv2d(net, 96, [3, 3], scope='conv1_3')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        net = slim.conv2d(net, 192, [3, 3], scope='conv2_1')
        net = slim.conv2d(net, 192, [3, 3], scope='conv2_2')
        net = slim.conv2d(net, 192, [3, 3], scope='conv2_3')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        net = slim.conv2d(net, 192, [3, 3], scope='conv3_1')
        net = slim.conv2d(net, 192, [1, 1], scope='conv3_2')
        net = slim.conv2d(net, 10, [1, 1], scope='conv3_3')
        net = slim.avg_pool2d(net, [7, 7], scope='pool3')

        logits = slim.flatten(net, scope='flatten')
    return logits


def mnist_gamma(inputs, is_training, is_unlabeled, ema, bn_assigns, batch_norm_decay, noise_std):
    inputs = tf.cast(inputs, tf.float32)
    net = inputs
    with tf.variable_scope('model', reuse=not is_training):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=custom_batch_norm,
                            normalizer_params={'is_training': is_training,
                                               'decay': batch_norm_decay,
                                               'noise_std': noise_std}):
            net = slim.conv2d(net, 32, [5, 5], scope='conv1_1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')

            net = slim.conv2d(net, 64, [3, 3], scope='conv2_1')
            net = slim.conv2d(net, 64, [3, 3], scope='conv2_2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')

            net = slim.conv2d(net, 128, [3, 3], scope='conv3_1')
            net = slim.conv2d(net, 10, [1, 1], scope='conv3_2')
            net = slim.avg_pool2d(net, [7, 7], scope='pool3')

            net = slim.flatten(net, scope='flatten')

        net = tf.layers.dense(net, 10, use_bias=False, name='dense')

    logits_crt, logits_cln, z_crt, z_cln = _gamma_layer(net,
                                                        lambda x: x,
                                                        is_training=is_training,
                                                        is_unlabeled=is_unlabeled,
                                                        noise_std=noise_std,
                                                        ema=ema,
                                                        bn_assigns=bn_assigns)
    return logits_crt, logits_cln, z_crt, z_cln


def mnist_supervised_haeusser(inputs, emb_size=128, l2_weight_decay=1e-3):
    inputs = tf.cast(inputs, tf.float32)
    net = inputs
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.elu,
                        weights_regularizer=slim.l2_regularizer(l2_weight_decay)):
        net = slim.conv2d(net, 32, [3, 3], scope='conv1_1')
        net = slim.conv2d(net, 32, [3, 3], scope='conv1_2')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')  # 14

        net = slim.conv2d(net, 64, [3, 3], scope='conv2_1')
        net = slim.conv2d(net, 64, [3, 3], scope='conv2_2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')  # 7

        net = slim.conv2d(net, 128, [3, 3], scope='conv3_1')
        net = slim.conv2d(net, 128, [3, 3], scope='conv3_2')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')  # 3

        net = slim.flatten(net, scope='flatten')
        emb = slim.fully_connected(net, emb_size, scope='fc1')
        logits = slim.fully_connected(emb, 10, scope='fc2')
    return logits


def mnist_supervised_rasmus(inputs, is_training, batch_norm_decay=0.9):
    inputs = tf.cast(inputs, tf.float32)
    net = inputs
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training, 'decay': batch_norm_decay}):
        net = slim.conv2d(net, 32, [5, 5], scope='conv1_1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        net = slim.conv2d(net, 64, [3, 3], scope='conv2_1')
        net = slim.conv2d(net, 64, [3, 3], scope='conv2_2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        net = slim.conv2d(net, 128, [3, 3], scope='conv3_1')
        net = slim.conv2d(net, 10, [1, 1], scope='conv3_2')
        net = slim.avg_pool2d(net, [7, 7], scope='pool3')

        net = slim.flatten(net, scope='flatten')
        logits = slim.fully_connected(net, 10, scope='fc1')
    return logits
