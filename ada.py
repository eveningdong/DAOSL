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

def discriminator(inputs, 
                  reuse=None):
  """Discriminator
  :param inputs: Embedded images for classification [batch_size, 64]
  :param reuse: whether or not the network and its variables should be reused. 
    To be able to reuse 'scope' must be given.
  :return: Embeddings of size [batch_size, 1]
  """
  with tf.variable_scope('disc', [inputs], reuse=reuse) as sc:
    net = inputs
    net = slim.fully_connected(net, 64)
    net = slim.fully_connected(net, 64)
    net = slim.fully_connected(net, 1, activation_fn=None)
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
  :return: A tensor with similarities of shape [num_ways, 1]
  """
  if args.distance == 'cosine':
    with tf.variable_scope('cosine'):
      support_set_norm = tf.nn.l2_normalize(support_set, axis=1)
      query_norm = tf.nn.l2_normalize(query_image, axis=1)
      query_norm = tf.tile(query_norm, [tf.shape(support_set)[0], 1])
      cos_distance = tf.losses.cosine_distance(support_set_norm, query_norm,
        reduction=tf.losses.Reduction.NONE, axis=1)
      cos_sim = 1 - cos_distance
      return cos_sim
  elif args.distance == 'euclidean':
    with tf.variable_scope('euclidean'):
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
  # print('support_set_y', support_set_y)
  if args.distance == "cosine":
    with tf.variable_scope('nn_classifier'):
      softmax_similarities = tf.transpose(tf.nn.softmax(similarities, axis=0))

      preds = tf.matmul(softmax_similarities, support_set_y)
  elif args.distance == "euclidean":
    with tf.variable_scope('nn_classifier'):
      softmax_similarities = tf.transpose(tf.nn.softmax(similarities, axis=0))
      preds = tf.matmul(softmax_similarities, support_set_y)
  return preds

def ada(support_set_images, 
        support_set_labels, 
        query_image, 
        query_label, 
        target_images,
        reuse=None, 
        is_training=True):
  """
  Builds a matching network-based adversarial domain adaption framework, the training and evaluation ops as well as data augmentation routines.
  :param support_set_images: A tensor containing the support set images 
    [classes_per_set, samples_per_class, input_size, input_size, num_channel]
  :param support_set_labels: A tensor containing the support set labels 
    [classes_per_set, samples_per_class, 1]
  :param query_image: A tensor containing the query image (image to produce label for) [1, input_size, input_size, num_channel]
  :param query_label: A tensor containing the query label [1, 1]
  :param target_images: A tensor containing the target images 
    [classes_per_set, samples_per_class, input_size, input_size, num_channel]
  :param reuse: Whether or not the network and its variables should be reused. 
    To be able to reuse 'scope' must be given.
  :param is_training: Whether the mode is in training mode.
  """
  with tf.variable_scope('matching_net_ada', [support_set_images, 
    support_set_labels, query_image, query_label, target_images], 
    reuse=reuse) as sc:
    with slim.arg_scope(net_arg_scope(pool_pad='VALID',
                                      is_training=is_training)):
      support_image_shape = tf.shape(support_set_images)
      support_set_images = tf.reshape(support_set_images, 
        shape=[support_image_shape[0]*support_image_shape[1], 
        support_image_shape[2], support_image_shape[3], 
        support_image_shape[4]])
      num_support = tf.shape(support_set_images)[0]

      target_image_shape = tf.shape(target_images)
      target_images = tf.reshape(target_images,
        shape=[target_image_shape[0]*target_image_shape[1],
        target_image_shape[2], target_image_shape[3],
        target_image_shape[4]])
      num_target = tf.shape(target_images)[0]
      
      support_set_labels = tf.reshape(support_set_labels, 
        shape=[support_image_shape[0]*support_image_shape[1]])
      support_set_labels = tf.one_hot(support_set_labels, args.num_ways)
      
      input_images = tf.concat([support_set_images, query_image, target_images]
        , axis=0)
      
      ## embedding
      embedded_images = embedding_network(inputs=input_images)
      embedded_support = embedded_images[:num_support]
      embedded_query = tf.expand_dims(embedded_images[num_support], axis=0)
      embedded_source = embedded_images[:num_support+1]
      embedded_target = embedded_images[num_support+1:]

      ## domain classification
      disc_target_logits = discriminator(embedded_target)
      disc_target_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_target_logits, 
          labels=tf.zeros_like(disc_target_logits)))
      disc_source_logits = discriminator(embedded_source, reuse=True)
      disc_source_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_source_logits, 
          labels=tf.ones_like(disc_source_logits)))

      ## one-shot classification
      # get similarity between support set embeddings and query embedding
      similarities = similarity_network(support_set=embedded_support,
        query_image=embedded_query)
      
      # produce predictions for query probabilities
      preds = nearest_neighbor_classifier(similarities,
        support_set_y=support_set_labels)
      
      ## accuracy and loss
      # disc
      reg_loss = tf.add_n(tf.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES))
      disc_loss = disc_source_loss + disc_target_loss + reg_loss

      # one-shot
      accuracy = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(preds, 1), tf.cast(query_label, tf.int64)),
        tf.float32))
      querys = tf.cast(tf.one_hot(query_label, args.num_ways), tf.int64)
      cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=preds, labels=querys)) 

      adv_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_target_logits,
          labels=tf.ones_like(disc_target_logits)))

      cls_loss = args.la * adv_loss + cross_entropy + reg_loss

      return accuracy, disc_loss, cls_loss


if __name__ == '__main__':
  train_support_images = tf.placeholder(tf.float32, [5, 1, 32, 32, 3])
  train_support_labels = tf.placeholder(tf.int32, [5, 1, 1])
  train_query_image = tf.placeholder(tf.float32, [1, 32, 32, 3])
  train_query_label = tf.placeholder(tf.int32, [1])

  val_support_images = tf.placeholder(tf.float32, [5, 1, 32, 32, 3])
  val_support_labels = tf.placeholder(tf.int32, [5, 1, 1])
  val_target_image = tf.placeholder(tf.float32, [1, 32, 32, 3])
  val_target_label = tf.placeholder(tf.int32, [1])

  accuracy, disc_loss, cls_loss = ada(train_support_images,
    train_support_labels, train_query_image, train_query_label, 
    val_support_images)

  accuracy, disc_loss, cls_loss = ada(val_support_images,
    val_support_labels, val_target_image, val_target_label, 
    val_support_images, reuse=True, is_training=False)


  
