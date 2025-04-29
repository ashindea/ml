from typing import cast
import numpy as np
import pandas as pd
import tensorflow as tf
from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import INPUT_CIK_NUM
from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import INPUT_TEXT_NAME_1
from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import INPUT_TEXT_NAME_2
from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import OUTPUT_LABEL_NAME
from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import TOKENS_LENGTH


def fetch_data():
  texts = []
  binary_arrays = []
  for text in TEXTS:
    # Generate some example text (replace with your logic)
    texts.append(text)

    # Generate a random binary array of length 7
    binary_array = np.random.randint(
        0, 2, size=7
    )  # Generates array of 1s and 0s
    binary_arrays.append(binary_array)

  df = pd.DataFrame({INPUT_TEXT_NAME_1: texts, 'binary_array': binary_arrays})
  return df


def get_text_values():
  # return np.array(['This si a sample text'])
  return tf.convert_to_tensor(
      [
          'This si a sample text',
          'This is another sample text',
          'This is a third sample text',
      ],
      dtype=tf.string,
  )


def get_features_labels_df(size=3):
  data = {
      INPUT_TEXT_NAME_1: TEXT_ALL[:size],
      INPUT_TEXT_NAME_2: TEXT_ALL[:size],
      INPUT_CIK_NUM: INPUT_CIK_LIST_ALL[:size],
      OUTPUT_LABEL_NAME: LABELS_ALL[:size],
  }

  # Create the pandas DataFrame with the sample data
  train_df = pd.DataFrame(data)

  # Ensure the 'Input_CIK_num' column is explicitly float type (good practice)
  # Although pandas usually infers correctly from float literals like 12345.0
  train_df[INPUT_CIK_NUM] = train_df[INPUT_CIK_NUM].astype(float)

  print(f'\n * 3 * \n train_df:\n {train_df}')

  train_dataset = {name: np.array(value) for name, value in train_df.items()}

  return train_dataset


def get_features_labels(size=3):
  num_classes = 7

  if size == 1:
    texts = np.array(ONE_TEXT)
    labels = np.array(4)
    return texts, labels
  else:
    texts = np.array(THREE_TEXT)
    labels = np.array(
        [[4], [5], [3], [2], [1], [0], [6], [3], [1], [0], [6], [2]]
    )
    #  labels =  np.random.randint(0, num_classes, size=len(texts))
    return texts, labels


LABELS_ALL = [4, 5, 3, 2, 1, 0, 6, 3, 1, 0, 6, 2]
INPUT_CIK_LIST_ALL = [
    131878.0,
    237878.0,
    857.0,
    57756.0,
    59578801.0,
    456485.0,
    356565.0,
    895654.0,
    478988.0,
    4679647.0,
    23565.0,
    7165.0,
]
TEXT_ALL = [
    'This is a sample text',
    'This is another sample text',
    'This is a third sample text',
    'This is a sample text 3 ',
    'This is another sample text 56',
    'This is a third sample text 46',
    'This is a sample text 3',
    'This is another sample text 23',
    'This is a third sample text 64',
    'This is a sample text r',
    'This is another sample text 24',
    'This is a third sample text 78',
]
ONE_TEXT = 'This is a sample text'
THREE_TEXT = [
    ['This is a sample text'],
    ['This is another sample text'],
    ['This is a third sample text'],
    ['This is a sample text 3 '],
    ['This is another sample text 56'],
    ['This is a third sample text 46'],
    ['This is a sample text 3'],
    ['This is another sample text 23'],
    ['This is a third sample text 64'],
    ['This is a sample text r'],
    ['This is another sample text 24'],
    ['This is a third sample text 78'],
]
TEXTS = [
    'sds',
    'vcw wed',
    'vcw wed',
]
