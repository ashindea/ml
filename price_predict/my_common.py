import sys
import time
import traceback
import keras_hub.src.models.gemma.gemma_causal_lm
import keras_hub.src.models.gemma.gemma_causal_lm_preprocessor
import keras_nlp
from matplotlib import pyplot as plt
import numpy
import tensorflow as tf
# from transformers import BertTokenizer
# import kagglehub
import tensorflow.keras as keras

TEST_TEXT = "What is keras now and then"
TEST_CIK = 101.0
TEST_TOKEN = {
    "token_ids": numpy.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype="int32"
    ),
    # "token_ids": numpy.ones(shape=(1, 12), dtype="int32"),
    "padding_mask": numpy.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
}
# [tf.convert_to_tensor([[23, 23 , 45, 23]]),tf.convert_to_tensor([[1,1,1,1]])]
USE_NLP = True
TOKENS_LENGTH = 150
INPUT_TEXT_NAME = "input_texts"
INPUT_TEXT_NAME_1 = "input_text_1"
INPUT_TEXT_NAME_2 = "input_text_2"
INPUT_CIK_NUM = "input_ciks"
NUM_OUTPUT_CLASSES = 7
OUTPUT_LABEL_NAME = "output_label"


def is_symbolic_tensor(tensor):
  return isinstance(tensor, tf.keras.KerasTensor) or isinstance(
      tensor, tf.compat.v1.placeholder
  )


def get_reversible_embedding_from_model(model):
  rev_embedding = model.get_layer(name="token_embedding")
  print("rev_embedding: " + str(rev_embedding))
  print(
      "rev_embedding:"
      f" {rev_embedding.input_dim} output_dim:{rev_embedding.output_dim}"
  )
  print(
      f"tie_weights tie_weights {rev_embedding.tie_weights}",
      f"embeddings_initializer  {rev_embedding.embeddings_initializer}",
      f"embeddings_regularizer {rev_embedding.embeddings_regularizer}",
      f"embeddings_constraint {rev_embedding.embeddings_constraint}",
      f"mask_zero {rev_embedding.mask_zero}",
      f"reverse_dtype {rev_embedding.reverse_dtype}",
      f"logit_soft_cap {rev_embedding.logit_soft_cap}",
  )
  return keras_hub.layers.ReversibleEmbedding(
      input_dim=rev_embedding.input_dim,
      output_dim=rev_embedding.output_dim,
      embeddings_initializer=keras.initializers.VarianceScaling(),
      # keras.initializers.TruncatedNormal(stddev=0.02),
      name="reversible_embedding",
  )


def get_tokenized_input(tokenizer, text):
  token_ids = tokenizer(text)
  padding_mask = numpy.array([1] * len(token_ids))
  return {
      "token_ids": token_ids,
      "padding_mask": padding_mask,
  }


def add_reversible_embedding(outputs, reversible_embedding):
  if reversible_embedding:
    outputs = reversible_embedding(outputs, reverse=True)
  return outputs


def add_dense_layers(input_cik, flattened_text_1, flattened_text_2, num_layers=1, num_units=10):
  if not num_layers or not num_units or num_layers == 0 or num_units == 0:
    return flattened_text_1
  print(f" Adding dense layers: {num_layers} {num_units}")
  # input_cik2 = tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
  #                                   name=INPUT_CIK_NUM)
  # preprocessing_layers = outputs_text
  preprocessing_layers = tf.keras.layers.Concatenate()(
      [input_cik, flattened_text_1, flattened_text_2]
  )
  # pre_outputs = layer(inputs=[input_cik, outputs_text], training=True)#, mask=mask)

  outputs = keras.layers.Dense(num_units, activation="relu")(
      preprocessing_layers
  )  ##(outputs_text)##
  print(f"Added preprocessing_layers layer")
  for _ in range(num_layers - 1):
    outputs = keras.layers.Dense(num_units, activation="relu")(outputs)
  print(f"Added all dense layers")
  return outputs


def add_softmax_layer(outputs, num_classes=NUM_OUTPUT_CLASSES):

  print(f"Adding softmax layer: {num_classes}")
  if num_classes and num_classes > 0:
    # outputs = keras.layers.GlobalAveragePooling1D()(outputs)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(outputs)
    outputs = tf.keras.layers.Softmax()(outputs)
    print(f"Added softmax layer")
  return outputs


def test_output_from_model(
    description, model, tokenizer, text_data, reversible_embedding=None
):
  print(f"Get output from model: {description} for text: {text_data}")
  input_data = tokenizer(text_data)
  outputs = model(input_data)
  print(
      f"outputs before reversible_embedding: {outputs} \n of shape:"
      f" {outputs.shape}"
  )
  if reversible_embedding:
    outputs = reversible_embedding(outputs, reverse=True)
    print(
        f"outputs after reversible_embedding: {outputs} \n of shape:"
        f" {outputs.shape}"
    )
  return outputs


def print_model_layer_info(description, model):
  print(f"\nCmn Print model layer info for: {description}\n")
  for layer in model.layers:
    print_layer_info(layer)


def print_model_info(description, model):
  print(
      f"Cmn model info: for {description} for name : {model.name}. Model built:"
      f" {model.built}"
  )
  if hasattr(model, "preprocessor"):
    print(
        f"->Preprocessor:-> {model.preprocessor} of class"
        f" {type(model.preprocessor).__class__}"
    )
  try:
    print("Show model.summary():")
    model.summary(expand_nested=True, show_trainable=True, print_fn=print)
    print("Done model.summary():")
  except ValueError as ve:
    print("Error in model.summary():")
    print(ve)
  except Exception as e:
    print(e)
    traceback.format_exc()
    # print(sys.exc_info()[2])
    raise sys.exc_info()[0]
  print_model_layer_info(description, model)


def print_layer_info(layer):

  print(
      f"Cmn LAYER info Name:{layer.name} (Classname:{layer.__class__.__name__})"
  )
  if hasattr(layer, "input_shape"):
    print(f"Input->: {layer.input_shape} -> ", end=" ")
  if hasattr(layer, "output_shape"):
    print(f"->Output:-> {layer.output_shape}", end=" ")
  if hasattr(layer, "activation"):
    print(f" \nActivation: {layer.activation}")
  if hasattr(layer, "weights"):
    print(
        f"  Weights: count:{len(layer.weights)} : "
        f"{[w.shape for w in layer.weights[:1]]}"
        f"..{[w.shape for w in layer.weights[-1:]]}"
        if len(layer.weights) > 2
        else ""
    )
  print("-" * 20)


def download_backbone(gemma_model):
  if USE_NLP:
    return keras_nlp.models.GemmaBackbone.from_preset(gemma_model)
  else:
    return keras_hub.models.GemmaBackbone.from_preset(gemma_model)


def generate_text(model, input_text_list):
  # See example use at https://www.kaggle.com/models/keras/gemma

  start = time.time()
  model.compile(sampler="top_k")

  print(f"Generating for batch input: {input_text_list}")
  print("Model class name: " + str(model.__class__))
  # Generate with batched prompts.
  results = model.generate(input_text_list, max_length=100)
  print("Generated in " + str(time.time() - start) + " :" + str(results))


def copy_reversible_embedding(source_layer):
  # """Deep copies a ReversibleEmbedding layer.
  print("copy_reversible_embedding source layer: " + str(source_layer))
  return keras_hub.layers.ReversibleEmbedding(
      input_dim=source_layer.input_dim,
      output_dim=source_layer.output_dim,
      embeddings_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
      name="reversible_embedding",
  )


def unpack_pack_model(model, new_layer=None):
  # """Unpacks a model and returns a list of layers.
  new_layers = []
  input_tensor = model.input

  x = input_tensor
  for layer in model.layers:
    new_layers.append(layer)

  if new_layer:
    new_layers.append(new_layer)

  # x = new_layers(x)
  for layer in new_layers[1:]:
    x = layer(x)

  new_model = tf.keras.Model(inputs=input_tensor, outputs=x)

  # Copy weights from the old model to the corresponding layer in the new model
  for old_layer, new_layer in zip(model.layers, new_model.layers):
    if hasattr(old_layer, "weights") and hasattr(new_layer, "weights"):
      for old_weight, new_weight in zip(old_layer.weights, new_layer.weights):
        new_weight.assign(old_weight)

  return new_model


def create_model_from_model_by_layer(
    model, tokenizer=None, exclude_names=None, num_classes=None
):
  print("Layers input shape: input_shape: " + str(model))
  print("source models layer: " + str(model.summary()))
  new_model = tf.keras.Sequential()
  for layer in model.layers:
    if isinstance(layer, tf.keras.layers.InputLayer):
      continue  # Skip the input layer
    if exclude_names and layer.name in exclude_names:
      continue
    print("adding layer: " + str(layer))
    new_model.add(layer)
  print("new model from layers:\n" + str(new_model.summary()))
  return new_model


# @title Define the plotting function
def plot_curve(epochs, hist, list_of_metrics):
  """Plot a curve of one or more classification metrics vs. epoch."""
  # list_of_metrics should be one of the names shown in:
  # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Value")

  for m in list_of_metrics:
    x = hist[m]
    plt.plot(epochs[1:], x[1:], label=m)

  plt.legend()


print("Loaded the plot_curve function.")


def create_model_from_layer(layer, num_classes=None):
  print("Creating model from Layer input shape: input_shape: " + str(layer))
  # None here means variable sequence length

  # input1 = tf.keras.layers.Input(type_spec=tf.TypeSpec())##(shape=input_shape)
  # input2 = tf.keras.layers.Input(type_spec=tf.TypeSpec())##(shape=input_shape)
  # input2 = tf.keras.layers.Input(type_spec=tf.TypeSpec(shape=None, dtype=tf.string))##(shape=input_shape)
  # input2 = tf.keras.layers.Input(dtype=tf.string)##shape=input_shape)
  # combined = tf.keras.layers.Concatenate()([input_layer1, input_layer2])
  # combined={'token_ids': input_layer1, 'padding_mask': input_layer2}
  combined = [
      tf.keras.layers.Input(shape=(None,), name="token_ids"),
      tf.keras.layers.Input(shape=(None,), name="padding_mask"),
  ]
  # outputs = layer([input_layer1, input_layer2])
  outputs = layer(combined)

  if num_classes:
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(outputs)

  model_base = tf.keras.Model(inputs=combined, outputs=outputs)
  print(f"Model created with inputs:\n {combined}")

  model = model_base
  model.build(input_shape=[(None,), (None,)])
  return model
  # if tokenizer:
  #   model = tf.keras.Sequential([
  #       tokenizer,
  #       model_base
  #   ])
  # else:


def deep_copy_layers(
    source_model, target_model, layer_names=None, exclude_names=None
):
  # """Deep copies layers (including weights) from a source model to a target model.

  # Args:
  #     source_model: The source tf.keras.Model.
  #     target_model: The target tf.keras.Model (can be Sequential or Functional).
  #     layer_names (optional): A list of layer names to copy. If None, copies all
  #                             layers (excluding InputLayer).
  #     exclude_names (optional): A list of layer names to exclude from copying.

  # Raises:
  #     ValueError: If layer_names are provided and a layer is not found.
  # """

  layers_to_copy = []

  if layer_names is not None:
    for layer_name in layer_names:
      for layer in source_model.layers:
        if layer.name == layer_name:
          layers_to_copy.append(layer)
          break
      else:  # No matching layer found
        raise ValueError(
            f"Layer with name '{layer_name}' not found in the source model."
        )

  else:  # Copy all layers (excluding InputLayer)
    for layer in source_model.layers:
      if isinstance(layer, tf.keras.layers.InputLayer):
        continue  # Skip input layers
      if exclude_names and layer.name in exclude_names:
        continue
      layers_to_copy.append(layer)

  # Deep copy and add to target model
  for source_layer in layers_to_copy:
    # 1. Get the config of the source layer
    config = source_layer.get_config()

    # 2. Create a *new* layer in the target model based on the config
    try:  # Try to initialize the layer with the config
      # Most layers can be initialized in this way
      target_layer = type(source_layer).from_config(config)
    except NotImplementedError:
      # Handle the case where from_config is not implemented
      if isinstance(source_layer, keras_nlp.layers.ReversibleEmbedding):
        target_layer = copy_reversible_embedding(source_layer)
      if isinstance(source_layer, tf.keras.layers.Conv2D):
        target_layer = tf.keras.layers.Conv2D(
            filters=source_layer.filters,
            kernel_size=source_layer.kernel_size,
            strides=source_layer.strides,
            padding=source_layer.padding,
            data_format=source_layer.data_format,
            dilation_rate=source_layer.dilation_rate,
            groups=source_layer.groups,
            activation=source_layer.activation,
            use_bias=source_layer.use_bias,
            kernel_initializer=source_layer.kernel_initializer,
            bias_initializer=source_layer.bias_initializer,
            kernel_regularizer=source_layer.kernel_regularizer,
            bias_regularizer=source_layer.bias_regularizer,
            activity_regularizer=source_layer.activity_regularizer,
            kernel_constraint=source_layer.kernel_constraint,
            bias_constraint=source_layer.bias_constraint,
        )
      elif isinstance(source_layer, tf.keras.layers.Dense):
        target_layer = tf.keras.layers.Dense(
            units=source_layer.units,
            activation=source_layer.activation,
            use_bias=source_layer.use_bias,
            kernel_initializer=source_layer.kernel_initializer,
            bias_initializer=source_layer.bias_initializer,
            kernel_regularizer=source_layer.kernel_regularizer,
            bias_regularizer=source_layer.bias_regularizer,
            activity_regularizer=source_layer.activity_regularizer,
            kernel_constraint=source_layer.kernel_constraint,
            bias_constraint=source_layer.bias_constraint,
        )
      elif isinstance(source_layer, tf.keras.layers.MaxPooling2D):
        target_layer = tf.keras.layers.MaxPooling2D(
            pool_size=source_layer.pool_size,
            strides=source_layer.strides,
            padding=source_layer.padding,
            data_format=source_layer.data_format,
        )
      elif isinstance(source_layer, tf.keras.layers.Dropout):
        target_layer = tf.keras.layers.Dropout(rate=source_layer.rate)
      elif isinstance(source_layer, tf.keras.layers.Flatten):
        target_layer = tf.keras.layers.Flatten(
            data_format=source_layer.data_format
        )
      else:
        raise ValueError(
            f"Layer type '{type(source_layer).__name__}' is not"
            " supported for deep copying."
        )

    # 3. Copy the weights
    for source_weight, target_weight in zip(
        source_layer.weights, target_layer.weights
    ):
      target_weight.assign(source_weight.numpy())  # Assigning the weight value

    # 4. Add the new layer to the target model
    print("Adding layer: " + str(target_layer.__class__))
    target_model.add(target_layer)
    return target_model


def run_example():
  # Example Usage:

  # 1. Create two models (source and target)
  source_input = tf.keras.layers.Input(shape=(100,))
  source_x = tf.keras.layers.Dense(64, activation="relu", name="dense_1")(
      source_input
  )
  source_x = tf.keras.layers.Dense(32, activation="relu", name="dense_2")(
      source_x
  )
  source_output = tf.keras.layers.Dense(
      10, activation="softmax", name="dense_3"
  )(source_x)
  source_model = tf.keras.Model(inputs=source_input, outputs=source_output)

  target_model = tf.keras.Sequential()  # Target can be Sequential or Functional

  # 2. Deep copy the layers
  deep_copy_layers(source_model, target_model)

  target_model.summary()

  # Verify weights (after copying)
  print("Source Model Weights:")
  for layer in source_model.layers:
    if hasattr(layer, "weights"):
      for weight in layer.weights:
        print(
            f"{layer.name}/{weight.name}: {weight.numpy()[:5]}"
        )  # Print first 5 elements

  print("\nTarget Model Weights:")
  for layer in target_model.layers:
    if hasattr(layer, "weights"):
      for weight in layer.weights:
        print(
            f"{layer.name}/{weight.name}: {weight.numpy()[:5]}"
        )  # Print first 5 elements

  # 3. Add more layers to the target model (if needed)
  target_output = tf.keras.layers.Dense(1, activation="sigmoid")(
      target_model.output
  )
  target_model_final = tf.keras.Model(
      inputs=target_model.input, outputs=target_output
  )  # Create a new functional model
  target_model_final.summary()

  # Example with layer names:
  target_model_2 = tf.keras.Sequential()
  deep_copy_layers(
      source_model, target_model_2, layer_names=["dense_1", "dense_3"]
  )
  target_model_2.summary()

  # Example with exclude names:
  target_model_3 = tf.keras.Sequential()
  deep_copy_layers(source_model, target_model_3, exclude_names=["dense_2"])
  target_model_3.summary()
