import sys
import traceback
from typing import cast
from absl import app
import keras_hub.src.models.gemma.gemma_causal_lm
import keras_hub.src.models.gemma.gemma_causal_lm_preprocessor
import keras_nlp
# from  ads.publisher.quality.micro_models.tensorflow.util.my_common import my_common
# from google3.ads.publisher.quality.micro_models.tensorflow.util import my_common
# from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import TEST_TEXT
# from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import generate_text
# from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import USE_NLP
# from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import download_backbone
# from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import TEST_TOKEN
# from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import test_output_from_model
import numpy as np
# import kagglehub
##
## blaze build ads/publisher/quality/micro_models/tensorflow/util:test_any
## blaze run ads/publisher/quality/micro_models/tensorflow/util:test_any 2>&1 | tee /tmp/mltest_any.out
##
import numpy
import tensorflow as tf
# from transformers import BertTokenizer
# import kagglehub
import tensorflow.keras as keras

TEXT = (
    'The model works on sequences of subword tokens, represented as integers in'
    ' the range [0, 256000), such as 1938 (=> “▁want”). Since the input comes'
    ' in as a single long string, we use a tokenizer to first split the string'
    ' into tokens, second to map these tokens to numeric IDs..'
)


# Create a custom Layer class
class GemmaLayer(tf.keras.layers.Layer):

  def __init__(self, gemma_model, **kwargs):
    super(GemmaLayer, self).__init__(**kwargs)
    self.gemma_model = gemma_model

  def call(self, inputs, attention_mask=None):
    # Assuming 'inputs' are token IDs (batched)
    outputs = self.gemma_model(inputs, attention_mask=attention_mask)
    # Extract the last hidden states (or other relevant output)
    last_hidden_states = (
        outputs.last_hidden_state
    )  # Or outputs[0], depending on Gemma version/config
    return last_hidden_states  # Pass the hidden states to next layer

  def get_config(self):  # Important for saving/loading the model
    config = super().get_config()
    return config


def test_tokenizer():
  tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(
      'gemma_1.1_instruct_2b_en'
  )
  print(f'tokenizer: {tokenizer}')
  print('tokenizer: \n' + str(tokenizer))
  print(f'Tokenizing text: {TEXT}')
  result_tokenized = tokenizer(TEXT)
  print(f'result_tokenized: \n{result_tokenized}\n')
  print('detokenized: ' + str(tokenizer.detokenize(result_tokenized.numpy())))


def test_gemma_backbone():
  input_data = {
      'token_ids': numpy.array([[234, 4, 6, 35, 0, 0]]),
      # "token_ids": numpy.ones(shape=(1, 12), dtype="int32"),
      'padding_mask': numpy.array([[1, 1, 1, 1, 0, 0]]),
  }

  # Pretrained Gemma decoder.
  model = keras_nlp.models.GemmaBackbone.from_preset('gemma_2b_en')
  model.summary()
  token_embedding = model.token_embedding
  print(f'token_embedding: {token_embedding} ')
  outputs = model(input_data)
  print(f'outputs: {outputs} \n of shape: {outputs.shape}')


def test_flatten():
  inputs = keras.Input(shape=((None, 2048)))
  outputs = inputs
  # outputs = keras.layers.Flatten()(outputs)
  outputs = keras.layers.Reshape(([2048]))(outputs)
  # outputs = keras.layers.GlobalAveragePooling1D()(outputs)
  outputs = keras.layers.Dense(10, activation='softmax')(outputs)
  # outputs = tf.keras.layers.Softmax()(outputs)
  model = keras.Model(inputs=inputs, outputs=outputs)
  model.build(input_shape=[(None, 2048)])
  model.compile()
  print(f'model.built: {model.built}')
  model.summary(expand_nested=True, show_trainable=True, print_fn=print)
  # print(f'outputs: {outputs} \n of shape: {outputs.shape}')


TOKENS_LENGTH = 150


def test_dataset():
  num_classes = 7
  texts = np.array([
      'This is a sample text',
      'This is another sample text',
      'This is a third sample text',
  ])
  labels = np.random.randint(0, num_classes, size=len(texts))
  print(f'texts: {texts}')
  print(f'labels: {labels}')
  dataset = tf.data.Dataset.from_tensor_slices((texts, labels))
  print(f'x: {dataset}')
  for string, integer in dataset:
    print(
        f"String: {string.numpy().decode('utf-8')}, Integer: {integer.numpy()}"
    )


def main(argv):
  test_dataset()


if __name__ == '__main__':
  app.run(main)
