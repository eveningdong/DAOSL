import collections
import tensorflow as tf

from functools import reduce

slim = tf.contrib.slim

def net_arg_scope(weight_decay=0.0001,
                  batch_norm_decay=0.997,
                  batch_norm_epsilon=1e-5,
                  batch_norm_scale=True,
                  activation_fn=tf.nn.relu,
                  pool_pad='SAME',
                  use_batch_norm=True,
                  is_training=True):
  """
  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.
    activation_fn: The activation function which is used in ResNet.
    use_batch_norm: Whether or not to use batch normalization.

  Returns:
    An `arg_scope` to use for models.
  """
  batch_norm_params = {
    'is_training': is_training,
    'decay': batch_norm_decay,
    'epsilon': batch_norm_epsilon,
    'scale': batch_norm_scale,
    'updates_collections': tf.GraphKeys.UPDATE_OPS,
    'fused': None,  # Use fused batch norm if possible.
  }

  with slim.arg_scope(
    [slim.conv2d, slim.fully_connected],
    weights_regularizer=slim.l2_regularizer(weight_decay),
    weights_initializer=slim.variance_scaling_initializer(),
    activation_fn=activation_fn,
    normalizer_fn=slim.batch_norm if use_batch_norm else None,
    normalizer_params=batch_norm_params,):
    # padding=conv_pad):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.max_pool2d], padding=pool_pad) as arg_sc:
        return arg_sc

def leaky_relu(x, 
               leak=0.2):
  return tf.maximum(x, x * leak)

def get_num_model_params():
  # Compute number of parameters in the model.
  num_params = 0
  from operator import mul
  for var in tf.trainable_variables():
    num_params += reduce(mul, var.get_shape().as_list(), 1)
  return num_params