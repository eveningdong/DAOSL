import cv2
import json
import numpy as np
import os
import pandas as pd
import pickle

from config import *
from scipy.ndimage import rotate

omniglot = {
  'sim':[    
    'Anglo-Saxon_Futhorc',
    'Armenian',
    'Asomtavruli_(Georgian)',
    'Cyrillic',
    'Greek',
    'Hebrew',
    'Latin',
    'Mkhedruli_(Georgian)',
  ],
  'dis':[
    'Balinese',
    'Bengali',
    'Grantha',
    'Gujarati',
    'Gurmukhi',
    'Kannada',
    'Malayalam',
    'Oriya',
  ],
  'chars':[
    'Anglo-Saxon_Futhorc',
    'Armenian',
    'Asomtavruli_(Georgian)',
    'Cyrillic',
    'Greek',
    'Hebrew',
    'Latin',
    'Mkhedruli_(Georgian)',
    'Balinese',
    'Bengali',
    'Grantha',
    'Gujarati',
    'Gurmukhi',
    'Kannada',
    'Malayalam',
    'Oriya',
  ],
  'fic':[
    'Alphabet of the Magi',
    'Angelic',
    'Arcadian',
    'Atlantean',
    'Aurek-Besh',
    'Futurama',
    'Tengwar',
    'ULOG',
  ],
  'chars3':[
    'Anglo-Saxon_Futhorc',
    'Armenian',
    'Asomtavruli_(Georgian)',
    'Cyrillic',
    'Greek',
    'Hebrew',
    'Latin',
    'Mkhedruli_(Georgian)',
    'Balinese',
    'Bengali',
    'Grantha',
    'Gujarati',
    'Gurmukhi',
    'Kannada',
    'Malayalam',
    'Oriya',
    'Alphabet of the Magi',
    'Angelic',
    'Arcadian',
    'Atlantean',
    'Aurek-Besh',
    'Futurama',
    'Tengwar',
    'ULOG',
  ]
}

def convert_omniglot():
  data_path = os.path.join(args.data_dir, args.data_name)
  langs = os.listdir(data_path)
  langs = sorted(langs)
  groups = {}
  data = np.zeros((1597, 20, args.input_size, args.input_size), dtype=np.uint8)
  cnt = 0
  for i, lang in enumerate(langs):
    lang_path = os.path.join(data_path, lang)
    chars = os.listdir(lang_path)
    chars = sorted(chars)
    num_chars = len(chars)
    groups[i] = list(range(cnt, cnt+num_chars))
    
    for j, char in enumerate(chars):
      char_path = os.path.join(lang_path, char)
      samples = os.listdir(char_path)
      samples = sorted(samples)
      assert len(samples) == 20

      for k, sample in enumerate(samples):
        sample_path = os.path.join(char_path, sample)
        print(sample_path)
        img = cv2.imread(sample_path, 0)
        img = cv2.resize(img, (args.input_size, args.input_size))
        data[cnt, k] = img
      
      cnt += 1
    assert cnt - 1 == groups[i][-1]
  print('Total chars:', cnt)
  print('data shape', data.shape)
  return data, groups

def convert_omniglot_subgroup(group_name):
  data_path = os.path.join(args.data_dir, 'omniglot')
  all_langs = os.listdir(data_path)
  langs = [lang for lang in all_langs if lang in omniglot[group_name]]
  print(langs)
  langs = sorted(langs)
  data = []
  cnt = 0
  for i, lang in enumerate(langs):
    lang_path = os.path.join(data_path, lang)
    chars = os.listdir(lang_path)
    chars = sorted(chars)
    num_chars = len(chars)
    
    
    for j, char in enumerate(chars):
      sub_data = []
      char_path = os.path.join(lang_path, char)
      samples = os.listdir(char_path)
      samples = sorted(samples)
      assert len(samples) == 20

      for k, sample in enumerate(samples):
        sample_path = os.path.join(char_path, sample)
        # print(sample_path)
        img = cv2.imread(sample_path, 0)
        img = cv2.resize(img, (args.input_size, args.input_size))
        sub_data.append(img)

      sub_data = np.array(sub_data)
      print('sub_data shape', sub_data.shape)

      cnt += 1
      data.append(sub_data)
  
  data = np.array(data)
  print('data shape', data.shape)
  return data

def convert_capitals():
  data_path = os.path.join(args.data_dir, 'emnist')
  data = np.zeros((26, 20, 28, 28))

  for i in range(10, 36):
    char_path = os.path.join(data_path, str(i))
    samples = os.listdir(char_path)

    for j, sample in enumerate(samples):
      if j < 20:
        sample_path = os.path.join(char_path, sample)
        print(sample_path)
        img = cv2.imread(sample_path, 0)
        img = cv2.resize(img, (args.input_size, args.input_size))
        data[i-10, j] = img

  print('data shape', data.shape)
  return data

def convert_emnist(num_examples=20):
  data_path = os.path.join(args.data_dir, args.data_name)
  chars = os.listdir(data_path)
  chars = sorted(chars)
  data = np.zeros((62, num_examples, args.input_size, args.input_size), dtype=np.uint8)
  for i, char in enumerate(chars):
    char_path = os.path.join(data_path, char)
    samples = os.listdir(char_path)
    
    for j, sample in enumerate(samples):
      if j < num_examples:
        sample_path = os.path.join(char_path, sample)
        print(sample_path)
        img = cv2.imread(sample_path, 0)
        img = cv2.resize(img, (args.input_size, args.input_size))
        data[i, j] = img

  print('data shape', data.shape)
  return data

if __name__ == '__main__':
  if args.data_name == 'omniglot':
    data, groups = convert_omniglot()
    np.save('{}.npy'.format(args.data_name), data)
  
  elif args.data_name == 'emnist':
    data = convert_emnist(args.num_target_examples)
    np.save('{}_{}.npy'.format(args.data_name, args.num_target_examples), data)

  elif args.data_name == 'digits':
    data = convert_digits()
    np.save('{}.npy'.format(args.data_name), data)

  elif args.data_name == 'capitals':
    data = convert_capitals()
    np.save('{}.npy'.format(args.data_name), data)

  else:
    if args.data_name in omniglot.keys():
      data = convert_omniglot_subgroup(args.data_name)
      np.save('{}.npy'.format(args.data_name), data)
