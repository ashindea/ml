# Run instructions:
#
#   g4d CS-util-BUILD-2024-11-30_121351
# blaze run ads/publisher/quality/micro_models/tensorflow/util:price_predict 2>&1 | tee /tmp/ml.out

# LInk to the NN from MLCC colab
# https://paste.googleplex.com/5627877981421568


from datetime import datetime
import sys
import time
import traceback
from typing import cast
from absl import app
import keras_hub.src.models.gemma.gemma_causal_lm
import keras_hub.src.models.gemma.gemma_causal_lm_preprocessor
import keras_nlp
import numpy as np
# import kagglehub
##
## blaze build ads/publisher/quality/micro_models/tensorflow/util:price_predict
## blaze run ads/publisher/quality/micro_models/tensorflow/util:price_predict
## blaze run ads/publisher/quality/micro_models/tensorflow/util:price_predict 2>&1 | tee /tmp/ml.out
##
import numpy
import pandas as pd
import tensorflow as tf
# from transformers import BertTokenizer
# import kagglehub
import tensorflow.keras as keras
# from  ads.publisher.quality.micro_models.tensorflow.util.my_common import my_common
from google3.ads.publisher.quality.micro_models.tensorflow.util import my_common
from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import add_dense_layers
from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import add_reversible_embedding
from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import add_softmax_layer
from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import download_backbone
from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import generate_text
from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import INPUT_CIK_NUM
from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import INPUT_TEXT_NAME_1
from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import INPUT_TEXT_NAME_2
from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import is_symbolic_tensor
from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import NUM_OUTPUT_CLASSES
from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import OUTPUT_LABEL_NAME
from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import plot_curve
from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import TEST_CIK
from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import test_output_from_model
from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import TEST_TEXT
from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import TEST_TOKEN
from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import TOKENS_LENGTH
from google3.ads.publisher.quality.micro_models.tensorflow.util.my_common import USE_NLP
from google3.ads.publisher.quality.micro_models.tensorflow.util.pp_data_fetcher import get_features_labels
from google3.ads.publisher.quality.micro_models.tensorflow.util.pp_data_fetcher import get_features_labels_df
from google3.ads.publisher.quality.micro_models.tensorflow.util.pp_data_fetcher import get_text_values
from google3.ads.publisher.quality.micro_models.tensorflow.util.pp_data_fetcher import INPUT_CIK_LIST_ALL
from google3.ads.publisher.quality.micro_models.tensorflow.util.pp_data_fetcher import TEXT_ALL


print('imported keras ver:' + str(keras.__version__))
print('imported tensorflow ver:' + str(tf.__version__))
GEMMA_MODEL = 'gemma_2b_en'  #
GEMMA_TOKENIZER = 'gemma_1.1_instruct_2b_en'
ENV_GOOGLE3 = 'g3'
ENV_KAGGLE = 'kaggle'
ENV = ENV_GOOGLE3
GEMMA_LAYER = None


# def get_home_path():
#   if ENV == ENV_GOOGLE3:
#     return '/usr/local/google/home/abhishinde/ml/models/gemma_causal_lm/'
#   else:
#     return '/home/models/gemma_causal_lm/'


def get_saved_model_path():
  if ENV == ENV_GOOGLE3:
    return (
        '/usr/local/google/home/abhishinde/ml/models/gemma_causal_lm/'
        + GEMMA_MODEL
        + '.keras'
    )
  else:
    return '/home/models/gemma_causal_lm/' + GEMMA_MODEL + '.keras'


def download_model():
  if USE_NLP:
    return keras_nlp.models.GemmaCausalLM.from_preset(GEMMA_MODEL)
  else:
    return keras_hub.models.GemmaCausalLM.from_preset(GEMMA_MODEL)


def get_tokenizer(model=None):
  # tokenizer = model.get_layer(name='gemma_causal_lm_preprocessor', index=None)
  if USE_NLP:
    tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(GEMMA_TOKENIZER)
  else:
    tokenizer = keras_hub.models.GemmaTokenizer.from_preset(GEMMA_TOKENIZER)
  tokenizer.sequence_length = TOKENS_LENGTH
  print('Retrieving tokenizer: ' + str(tokenizer))
  return tokenizer


# def download_and_save_model():
#   print('getting from preset LM')
#   # For keras hub internal
#   model = download_model()
#   print('got from preset LM')
#   model.summary()

#   print('Got base model. Saving to ' + str(get_saved_model_path()))
#   # model.save(get_saved_model_path())
#   tf.keras.saving.save_model(model, get_saved_model_path())
#   return model


def get_gemma_layer(load_from_disk=False, lora_rank=0):
  if load_from_disk:
    print(f'Loading from: {get_saved_model_path()}')
    # model = tf.keras.models.load_model(get_saved_model_path(), compile=False)
    model = tf.keras.saving.load_model(get_saved_model_path())

    print('Model loaded successfully')
  else:
    print('Downloading model')
    model = download_model()
    # my_common.print_model_layer_info('Downloaded base model', model)

  if lora_rank > 0:
    print('Lora rank: ' + str(lora_rank))
    model.backbone.enable_lora(rank=4)

  GEMMA_LAYER = model.get_layer('gemma_backbone')
  print('GEMMA_LAYER layers cnt ' + str(len(GEMMA_LAYER.layers)))
  my_common.print_layer_info(GEMMA_LAYER)

  return GEMMA_LAYER


def predict(model, input):
  print(f'predicting results for input: {input} ')
  results = model.predict(input, batch_size=1)  ##, max_length=30)
  # print('results : ' + str(results))
  return results


def print_layer_weights(layer):
  print(f'Layer weights for {layer.name}: ')
  for weight in layer.weights:
    print_weight(weight)


def print_weight(weight):
  print(
      f'Name: {weight.name} Shape: {weight.shape} dtype: {weight.dtype} weight:'
      f' {weight.value}'
  )
  # print(f" initializer: {weight.initializer} ", end=" ")
  # print(f"regularizer: {weight.regularizer} constraint: {weight.constraint} ", end=" ")
  # print(f"use_resource: {weight.use_resource} ")


def model_to_layer(model):
  class ModelAsLayer(tf.keras.layers.Layer):

    def __init__(self, model, **kwargs):
      super().__init__(**kwargs)
      self.model = model

    def call(self, inputs, training=None, mask=None):
      print(f'ModelAsLayer inputs {inputs}')
      print(
          'ModelAsLayer inputs value '
          f'{inputs.numpy() if hasattr(inputs, "numpy") else "none"}'
      )
      outputs = self.model(inputs, training=training, mask=mask)
      print(f'ModelAsLayer INtermediate outputs {outputs}')
      my_common.print_layer_info(outputs)
      last_hidden_states = outputs
      # Or outputs[0].last_hidden_state, depending on Gemma version/config
      return last_hidden_states  # Pass the hidden states to next layer

    def get_config(self):  # Important for saving/loading the model
      config = super().get_config()
      return config

  return ModelAsLayer(model)


# Tokenization layer
class TokenizerLayer(tf.keras.layers.Layer):  # Same as before

  def __init__(self, tokenizer, **kwargs):
    super(TokenizerLayer, self).__init__(**kwargs)
    self.tokenizer = tokenizer

  def call(self, inputs):
    # Tokenize the input strings (batched)
    print('Input to Tokenize layer: ', inputs)
    tokens = self.tokenizer(inputs)
    print('In Tokenize call encoded tokens: ', tokens)

    if isinstance(tokens, tf.RaggedTensor):
      tensor = tokens.to_tensor()  # Convert to dense tensor
      mask = tf.cast(tf.sequence_mask(tokens.row_lengths()), dtype=tf.int32)
      print('In Tokenize call encoded tensor:', tensor, ' mask:', mask)

      return tensor, mask
    else:
      # ragged_tensor = tf.ragged.constant(tf.reshape(tokens, (None, TOKENS_LENGTH)))
      # mask = tf.cast(tf.sequence_mask(TOKENS_LENGTH), dtype=tf.int32)
      # need numpy.array([[1, 1, 1, 1, 0, 0]])
      mask = tf.convert_to_tensor(numpy.ones((1, TOKENS_LENGTH)))
      # mask = tf.convert_to_tensor(numpy.ones(tokens.shape))
      # mask = tf.convert_to_tensor(numpy.array([1] * TOKENS_LENGTH))

      return tokens, mask
    # print('In call encoded tokens.numpy: ', tokens.numpy())
    # padding_mask = tf.convert_to_tensor(numpy.array([1] * len(tokens.numpy())))##ones(len(tokens.numpy())))
    # , padding_mask ##["input_ids"], encoded["attention_mask"] # Return input IDs and attention mask

  def get_config(self):  # For saving/loading
    config = super().get_config()
    return config


def create_inputs():
  return {
      INPUT_TEXT_NAME_1: tf.keras.layers.Input(
          shape=(), dtype=tf.string, name=INPUT_TEXT_NAME_1
      ),
      INPUT_TEXT_NAME_2: tf.keras.layers.Input(
          shape=(), dtype=tf.string, name=INPUT_TEXT_NAME_2
      ),
      INPUT_CIK_NUM: tf.keras.layers.Input(
          shape=(1,),
          dtype=tf.float32,
          name=INPUT_CIK_NUM,
      ),
  }


def create_model_from_tokenizer_layer(
    input_text_layer,
    layer,
    tokenizer_layer=None,
    reversible_embedding=None,
):
  print('Creating model from tokenizer Layer: ', str(layer))

  # input_text_layer = tf.keras.layers.Input(shape=(),
  #                                           dtype=tf.string,
  #                                           name=INPUT_TEXT_NAME)
  # Tokenizer layer needs to output also a padding mask of shape (None, None)?
  tokenizer_layer.trainable = False
  layer.trainable = False
  tensor, mask = tokenizer_layer(input_text_layer)
  # padding_mask = tf.convert_to_tensor(numpy.array([1] * 5))
  # print(f'Create model: tokens in put:{tok´
  # How to pass mltple inputs to the next layer?
  pre_outputs = layer(inputs=[tensor, mask], training=True)  # , mask=mask)
  outputs = add_reversible_embedding(pre_outputs, reversible_embedding)
  outputs = keras.layers.Flatten()(outputs)
  return outputs

def add_layers_and_softmax(
    input_cik, flattened_output_from_text1,
        flattened_output_from_text2, num_layers=2, num_units=10, num_classes=7
):
  outputs = add_dense_layers(
      input_cik, flattened_output_from_text1, flattened_output_from_text2, 
      num_layers=num_layers, num_units=num_units
  )
  outputs = add_softmax_layer(outputs, num_classes=num_classes)
  return outputs


def build_and_train_model(
    model,
    train_features,
    train_label,
    epochs=10,
    learning_rate=0.001,
    validation_split=0.2,
    batch_size=100,
):
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'],
  )
  # x = tf.data.Dataset.from_tensor_slices(train_features)
  x = tf.data.Dataset.from_tensor_slices((train_features, train_label))

  print(
      f'\n\n\n',
      '+' * 20,
      '\nTRAINING with features '
      # f'shape: {train_features.shape} ',
      f' and features.class: {train_features.__class__} '
      f' and features:\n {train_features}',
      f' \n and labels shape: {train_label.shape}:\n ',
      f' \n and labels class: {train_label.__class__}:\n ',
      train_label,
  )
  history = model.fit(
      x=x,
      y=None,
      batch_size=batch_size,
      epochs=epochs,
      shuffle=True,
      verbose=2,
      validation_split=validation_split,
  )

  # To track the progression of training, gather a snapshot
  # of the model's metrics at each epoch.
  epochs = history.epoch
  hist = pd.DataFrame(history.history)
  print(f'Training history: epochs {epochs} and history: \n{hist}')

  return epochs, hist


def build_and_train_model_df(
    model,
    train_dataset,
    label_name=OUTPUT_LABEL_NAME,
    epochs=10,
    learning_rate=0.001,
    validation_split=0.2,
    batch_size=100,
):
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'],
  )
  # x = tf.data.Dataset.from_tensor_slices(train_features)
  # x = tf.data.Dataset.from_tensor_slices((train_features, train_label))
  features = {name: np.array(value) for name, value in train_dataset.items()}
  label = np.array(features.pop(label_name))

  print(
      f'\n\n\n',
      '+' * 20,
      '\nTRAINING with features '
      # f'shape: {train_features.shape} ',
      f' and features.class: {features.__class__}  and features:\n {features}',
      f' \n and labels shape: {label.shape}:\n ',
      f' \n and labels class: {label.__class__}:\n ',
      f' \n and labels: {label}',
  )
  history = model.fit(
      x=features,
      y=label,
      batch_size=batch_size,
      epochs=epochs,
      shuffle=True,
      validation_split=validation_split,
  )

  # history = model.fit(x=x, y=None, batch_size=batch_size,
  #               epochs=epochs, shuffle=True, verbose=2,
  #               validation_split=validation_split)

  # To track the progression of training, gather a snapshot
  # of the model's metrics at each epoch.
  epochs = history.epoch
  hist = pd.DataFrame(history.history)
  print(f'Training history: epochs {epochs} and history: \n{hist}')

  return epochs, hist


def test_tokenizer_layer(tokenizer_layer, text):
  print('Text tokenizer layer: ', tokenizer_layer, ' withtext: ', text)
  tokens = tokenizer_layer(text)
  print('Result tokens: ', tokens)
  # print('Result padding_mask: ', padding_mask)
  # result_numpy = tokens.numpy()
  # print('Result tokens numpy: ', result_numpy)
  # tensor_from_numpy = tf.convert_to_tensor(result_numpy)  # or just tf.constant(numpy_array)
  # print("Tensor from NumPy (Example 1):\n", tensor_from_numpy)
  # tensor_from_numpy2 = tf.convert_to_tensor([1,1,1,1,0])  # or just tf.constant(numpy_array)
  # print("Tensor from NumPy (Example 2):\n", tensor_from_numpy2)
  # print('Result attention_mask: ', attention_mask)


def test_tokenizer(tokenizer, text='The quick brown fox tripped.'):
  print('tokenizer: \n' + str(tokenizer))

  result_tokenized = tokenizer(text)
  print(
      f'result_tokenized: \n{result_tokenized}\n'
      f'vector:{result_tokenized.numpy()}'
  )
  print('detokenized: ' + str(tokenizer.detokenize(result_tokenized.numpy())))


def test_model(
    description, model, input_text_list, input_cik_list, prompt=False
):
  print('\n',
      '+' * 20, 'Testing', '+' * 20, '\n',
      f'Test: model {description} name:{model.name} Layers:'
      f' {model.layers[0]} and  {model.layers[1]}',
  )

  # input = {INPUT_TEXT_NAME: input_text_list[0], INPUT_CIK_NUM: input_cik_list[0]}
  data = {
      # INPUT_TEXT_NAME: [['This is a sample text'], ['This is another sample text'], ['This is a third sample text']], ##input_text_list[:3],
      # INPUT_CIK_NUM: [[12.4], [23.5], [34.6]] ##input_cik_list[:3]
      INPUT_TEXT_NAME_1: input_text_list,
      INPUT_TEXT_NAME_2: input_text_list,
      INPUT_CIK_NUM: input_cik_list,
  }

  # Create the pandas DataFrame with the sample data
  test_df = pd.DataFrame(data)

  # Ensure the 'Input_CIK_num' column is explicitly float type (good practice)
  # Although pandas usually infers correctly from float literals like 12345.0
  test_df[INPUT_CIK_NUM] = test_df[INPUT_CIK_NUM].astype(float)
  input = {name: np.array(value) for name, value in test_df.items()}

  # input = [input_text_list, input_cik_list]

  print(f'input DF : {input}')
  results = predict(model=model, input=input)
  print(f'Results for {description} of shape {results.shape}:\n {results}')
  print('\n','+' * 20, 'Done testing', '+' * 20, '\n')

  while True and prompt:
    print('Enter your text input: ')
    text_input = sys.stdin.readline()
    if not text_input or text_input == '\n':
      break
    results = predict(model=model, input=[text_input])
    print(f'results of shape {results.shape} for user input:\n {results}')

  # results = model.predict(input)##, max_length=30)


def transfer_weights(gemma_layer, base_model_as_layer):
  print('transferring weights from gemma_layer to base_model_as_layer')
  for weight in gemma_layer.weights:
    base_model_as_layer.add_weight(
        name=weight.name, shape=weight.shape, dtype=weight.dtype
    )

  for index in range(len(gemma_layer.weights)):
    base_model_as_layer.weights[index].assign(gemma_layer.weights[index])
  print('weights transferred and assigned')


def train_model(model_from_layer):
  start = datetime.now()  # time.time()

  train_dataset = get_features_labels_df(size=5)
  epochs, hist = build_and_train_model_df(
      model_from_layer,
      train_dataset,
      label_name=OUTPUT_LABEL_NAME,
      epochs=10,
      learning_rate=0.001,
      validation_split=None,
      batch_size=1,
  )

  plot_curve(epochs, hist, ['accuracy'])
  end = datetime.now()  # time.time()
  startStr = start.strftime('%m/%d/%Y, %H:%M:%S')
  endStr = end.strftime('%m/%d/%Y, %H:%M:%S')
  print(
      f'Training time from start :{startStr} to end {endStr} is: '
      + str(end - start)
  )

  print('!!!!!!! Success training!!!!!!!')


def get_and_test_tokenizer_layer(run_tests=True):
  tokenizer = get_tokenizer()
  if run_tests:
    test_tokenizer(tokenizer, 'The quick brown fox tripped.')
  tokenizer_layer = TokenizerLayer(tokenizer)
  my_common.print_layer_info(tokenizer_layer)
  if run_tests:
    test_tokenizer_layer(tokenizer_layer, 'The quick brown fox tripped.')
  return tokenizer_layer

def create_output_layer_for_text(input_text_layer):
  tokenizer_layer = get_and_test_tokenizer_layer(run_tests=False)
  gemma_layer = get_gemma_layer(load_from_disk=False, lora_rank=3)

  base_model_as_layer1 = model_to_layer(gemma_layer)  # (base_model)
  transfer_weights(gemma_layer, base_model_as_layer1)

  return create_model_from_tokenizer_layer(
      input_text_layer,
      base_model_as_layer1,
      tokenizer_layer=tokenizer_layer,
      reversible_embedding=None,
  )


def main(argv):
  mainstart = datetime.now()
  try:
    print('***************************\nStarting\n***************************')
    inputs = create_inputs()

    flattened_output_from_text1 = create_output_layer_for_text(
        inputs[INPUT_TEXT_NAME_1]
    )
    flattened_output_from_text2 = create_output_layer_for_text(
        inputs[INPUT_TEXT_NAME_2]
    )

    outputs = add_layers_and_softmax(
        inputs[INPUT_CIK_NUM],
        flattened_output_from_text1,
        flattened_output_from_text2,
        num_layers=2,
        num_units=10,
        num_classes=NUM_OUTPUT_CLASSES,
    )

    model_from_layer = tf.keras.Model(inputs=inputs, outputs=outputs)

    my_common.print_model_info('New model from tok layers', model_from_layer)

    train_model(model_from_layer)
    test_model(
        'model_from_layer', model_from_layer, input_text_list=TEXT_ALL[:3],
        input_cik_list=INPUT_CIK_LIST_ALL[:3],##/*[TEST_CIK]*/
        prompt=False
    )
    mainend = datetime.now()

  except Exception as e:
    print('*******ERROR***************************')
    mainend = datetime.now()

    print(e)
    print(e.__traceback__)
    # traceback.format_exc()
    # print(sys.exc_info()[0])
    print(f'Time taken: {mainend - mainstart}')
    print(mainstart.strftime('%Y-%m-%d %H:%M:%S'))
    print(mainend.strftime('%Y-%m-%d %H:%M:%S'))
    # raise sys.exc_info()[0]


if __name__ == '__main__':
  app.run(main)
  # my_common.get_reversible_embedding_from_model(base_model)
  # gemma_layer = download_backbone(GEMMA_MODEL)
  # my_common.print_model_info('Downloaded gemma_layer', gemma_layer)
  # test_output_from_model(description='Gemma layer', model=gemma_layer,
  #                        tokenizer=tokenizer_layer, text_data=TEST_TEXT,
  #                        reversible_embedding=reversible_embedding)


# # Load the base gemma_1.1_instruct_2b_en model
# model = tf.saved_model.load(path)
