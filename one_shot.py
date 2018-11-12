import net_utils
import tensorflow as tf
from config import * 

slim = tf.contrib.slim
net_arg_scope = net_utils.net_arg_scope


def embedding_network(inputs):
  """Feature Extractor
  :param image_input: Image input to produce embeddings for 
    [batch_size, 28, 28, num_channel]
  :return: Embeddings of size [batch_size, 64]
  """
  with tf.variable_scope('embedding', [inputs]) as sc:
    net = slim.conv2d(inputs, 64, [3, 3], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1')

    net = slim.conv2d(net, 64, [3, 3], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')

    net = slim.conv2d(net, 64, [3, 3], scope='conv3')
    net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool3')

    net = slim.conv2d(net, 64, [3, 3], scope='conv4')
    net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool4')
    net = slim.flatten(net)
  return net

def similarity_network(support_set, 
                       query_image):
  """
  This module calculates the distance between each of the support set 
    embeddings and the query
  image embeddings.
  :param support_set: The embeddings of the support set images, tensor of
    shape [batch_size, 64]
  :param input_image: The embedding of the query image, tensor of shape 
    [1, 64]
  :return: A tensor with similarities of shape [batch_size, 1]
  """
  if args.distance == 'cosine':
    with tf.variable_scope('cosine'):
      print('cosine')
      support_set_norm = tf.nn.l2_normalize(support_set, axis=1)
      query_norm = tf.nn.l2_normalize(query_image, axis=1)
      query_norm = tf.tile(query_norm, [tf.shape(support_set)[0], 1])
      cos_distance = tf.losses.cosine_distance(support_set_norm, query_norm,
        reduction=tf.losses.Reduction.NONE, axis=1)
      cos_sim = 1 - cos_distance
      return cos_sim
  elif args.distance == 'euclidean':
    with tf.variable_scope('euclidean'):
      print('euclidean')
      support_set_norm = tf.nn.l2_normalize(support_set, axis=1)
      query_norm = tf.nn.l2_normalize(query_image, axis=1)
      query_norm = tf.tile(query_norm, [tf.shape(support_set)[0], 1])
      euclidean = tf.reduce_sum((support_set_norm - query_norm)**2, axis=-1,
        keepdims=True)
      euclidean_sim = - euclidean
      return euclidean_sim


def nearest_neighbor_classifier(similarities, 
                                support_set_y):
  """
  Produces pdfs over the support set classes for the query set image.
  :param similarities: A tensor with cosine similarities of size 
    [batch_size, 1]
  :param support_set_y: A tensor with the one hot vectors of the querys 
    for each support set image [batch_size, num_classes]
  :return: Softmax pdf
  """
  with tf.variable_scope('nn_classifier'):
    softmax_similarities = tf.nn.softmax(similarities, axis=0)
    preds = tf.multiply(support_set_y, softmax_similarities)
    preds = tf.reduce_sum(preds, 1)
    preds = tf.reshape(preds, [1, -1])
  return preds


def matching_network(support_set_images, 
                     support_set_labels, 
                     query_image, 
                     query_label, 
                     reuse=None,
                     is_training=True):
  """
  Builds a matching network, the training and evaluation ops as well as data 
    augmentation routines.
  :param support_set_images: A tensor containing the support set images 
    [classes_per_set, samples_per_class, input_size, input_size, num_channel]
  :param support_set_labels: A tensor containing the support set labels 
    [classes_per_set, samples_per_class, 1]
  :param query_image: A tensor containing the query image (image to produce 
    label for) [1, input_size, input_size, num_channel]
  :param query_label: A tensor containing the query label [1, 1]
  :param reuse: Whether or not the network and its variables should be reused. 
    To be able to reuse 'scope' must be given.
  :param is_training: Whether the mode is in training mode.
  """
  with tf.variable_scope('matching_net', [support_set_images, 
    support_set_labels, query_image, query_label], reuse=reuse) as sc:
    with slim.arg_scope(net_arg_scope(pool_pad='VALID',
                                      is_training=is_training)):
      support_image_shape = tf.shape(support_set_images)
      support_set_images = tf.reshape(support_set_images, 
        shape=[support_image_shape[0]*support_image_shape[1], 
        support_image_shape[2], support_image_shape[3], 
        support_image_shape[4]])
      
      support_set_labels = tf.reshape(support_set_labels, 
        shape=[support_image_shape[0]*support_image_shape[1]])
      support_set_labels = tf.one_hot(support_set_labels, args.num_ways)
      
      input_images = tf.concat([support_set_images, query_image], axis=0)
      
      ## embedding
      embedded_images = embedding_network(inputs=input_images)
      
      ## nn classification
      # get similarity between support set embeddings and query embedding
      similarities = similarity_network(support_set=embedded_images[:-1],
        query_image=tf.expand_dims(embedded_images[-1], axis=0))
      
      # produce predictions for query probabilities
      preds = nearest_neighbor_classifier(similarities,
        support_set_y=support_set_labels)

      accuracy = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(preds, 1), tf.cast(query_label, tf.int64)),
        tf.float32))
      querys = tf.cast(tf.one_hot(query_label, args.num_ways), tf.int64)

      reg_loss = tf.add_n(tf.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES))
      cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=preds, labels=querys))
      cls_loss = cross_entropy + reg_loss

      return accuracy, cls_loss

if __name__ == '__main__':
  images = tf.placeholder(tf.float32, [1, 28, 28, 1])
  net = embedding_network(images)
  print(net)
  logits = slim.fully_connected(net, 2)
  print(logits)
  print(net_utils.get_num_model_params())