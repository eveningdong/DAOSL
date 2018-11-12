import numpy as np
from config import *
from scipy.ndimage import rotate

class CharDataset():
  def __init__(self, 
               classes_per_set=5, 
               samples_per_class=1, 
               seed=706, 
               source_file='omniglot.npy', 
               target_file='emnist.npy'):
    """
    Constructs an N-Shot omniglot Dataset
    :param batch_size: Experiment batch_size
    :param classes_per_set: Integer indicating the number of classes per set
    :param samples_per_class: Integer indicating samples per class
    e.g. For a 20-way, 1-shot learning task, use classes_per_set=20 and samples_per_class=1
       For a 5-way, 10-shot learning task, use classes_per_set=5 and samples_per_class=10
    """
    np.random.seed(seed)
    self.x_train = np.load(source_file)
    self.x_train = self.x_train.astype(np.float32)
    _num_train_classes = self.x_train.shape[0]
    self.x_train = np.reshape(self.x_train, newshape=(_num_train_classes,
      20, args.input_size, args.input_size, 1))
    
    self.x_val = np.load(target_file)
    self.x_val = self.x_val.astype(np.float32)
    _num_val_classes = self.x_val.shape[0]
    self.x_val = np.reshape(self.x_val, newshape=(_num_val_classes, args.num_target_examples, 
      args.input_size, args.input_size, 1))
      
    self.classes_per_set = classes_per_set
    self.samples_per_class = samples_per_class
    self.indexes = {"train": 0, "val": 0}
    self.datasets = {"train": self.x_train, "val": self.x_val} #original data 

  def sample_new_batch(self, data_pack):
    """
    Collects 1000 batches data for N-shot learning
    :param data_pack: Data pack to use (any one of train, val, test)
    :return: A list with [support_set_x, support_set_y, query_x, query_y] ready to be fed to our networks
    """
    support_set_x = np.zeros((self.classes_per_set, 
      self.samples_per_class, data_pack.shape[2],
      data_pack.shape[3], data_pack.shape[4]), dtype=np.float32)
    support_set_y = np.zeros((self.classes_per_set, 
      self.samples_per_class, 1), dtype=np.float32)
    query_x = np.zeros((1, data_pack.shape[2], 
      data_pack.shape[3], data_pack.shape[4]), dtype=np.float32)
    query_y = np.zeros((1,), dtype=np.float32)
    
    classes_idx = np.arange(data_pack.shape[0])
    samples_idx = np.arange(data_pack.shape[1])
    choose_classes = np.random.choice(classes_idx, 
      size=self.classes_per_set, replace=False)
    choose_label = np.random.choice(self.classes_per_set, size=1)

    for _idx in range(self.classes_per_set):
      if _idx == choose_label:
        choose_samples = np.random.choice(samples_idx, 
          size=self.samples_per_class+1, replace=False)
        x_temp = data_pack[choose_classes[_idx], choose_samples]
        support_set_x[_idx, :self.samples_per_class, :] = x_temp[:self.samples_per_class]
        query_x[0, :] = x_temp[self.samples_per_class]        
        query_y[0] = _idx
      else:
        choose_samples = np.random.choice(samples_idx, 
          size=self.samples_per_class, replace=False)
        x_temp = data_pack[choose_classes[_idx], choose_samples]
        support_set_x[_idx, :self.samples_per_class, :] = x_temp[:self.samples_per_class]

      support_set_y[_idx, :, :] = _idx       

    return support_set_x, support_set_y, query_x, query_y

  def get_batch(self, dataset_name, augment=False):
    """
    Gets next batch from the dataset with name.
    :param dataset_name: The name of the dataset (one of "train", "val", "test")
    :return:
    """
    # if self.num_batches_since_shuffle[dataset_name] >= 50:
    #   samples_idx = np.arange(self.datasets[dataset_name].shape[1])
    #   np.random.shuffle(samples_idx)
    #   self.datasets[dataset_name] = self.datasets[dataset_name][:, samples_idx]
    #   self.num_batches_since_shuffle[dataset_name] = 0
    # else:
    #   self.num_batches_since_shuffle[dataset_name] += 1
    x_support_set, y_support_set, x_query, y_query = self.sample_new_batch(self.datasets[dataset_name])
    if augment:
      k = np.random.randint(0, 4, size=(self.classes_per_set))
      x_augmented_support_set = []

      for c in range(self.classes_per_set):
        x_temp_support_set = self.rotate_batch(x_support_set[c], 
          axis=(1, 2), k=k[c])
        if y_query == y_support_set[c, 0]:
          x_temp_query = self.rotate_batch(x_query, 
            axis=(1, 2), k=k[c])

        x_augmented_support_set.append(x_temp_support_set) 

      x_support_set = np.array(x_augmented_support_set)
      x_query = x_temp_query

    return x_support_set, y_support_set, x_query, y_query

  def rotate_batch(self, x_batch, axis, k):
    x_batch = rotate(x_batch, k*90, reshape=False, axes=axis, 
      mode="nearest")
    return x_batch

  def get_train_batch(self, augment=False):
    """
    Get next training batch
    :return: Next training batch
    """
    return self.get_batch("train", augment)

  def get_val_batch(self, augment=False):

    """
    Get next val batch
    :return: Next val batch
    """
    return self.get_batch("val", augment)


if __name__ == '__main__':
  data = CharDataset(
    classes_per_set=5, 
    samples_per_class=1,
    seed=args.random_seed,
    source_file=args.source+'.npy',
    target_file=args.target+'.npy')

  x_support_set, y_support_set, x_query, y_query = data.get_train_batch(
    augment=False)
  print('x_support', x_support_set.shape)
  print('y_support', y_support_set.shape)
  print('x_query', x_query.shape)
  print('y_query', y_query.shape)

  np.save('x_support.npy', x_support_set)
  np.save('y_support.npy', y_support_set)
  np.save('x_query.npy', x_query)
  np.save('y_query.npy', y_query)

  x_support_set, y_support_set, x_query, y_query = data.get_val_batch(
    augment=False)
  print('x_support', x_support_set.shape)
  print('y_support', y_support_set.shape)
  print('x_query', x_query.shape)
  print('y_query', y_query.shape)

  np.save('x_support.npy', x_support_set)
  np.save('y_support.npy', y_support_set)
  np.save('x_query.npy', x_query)
  np.save('y_query.npy', y_query)
