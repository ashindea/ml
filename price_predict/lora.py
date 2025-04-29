import traceback
from absl import app
import keras_hub
import keras_hub.src.models.gemma.gemma_causal_lm
import keras_hub.src.models.gemma.gemma_causal_lm_preprocessor
import keras_nlp
import numpy
import tensorflow as tf
import tensorflow as tf
# from transformers import BertTokenizer
# import kagglehub
import tensorflow.keras as keras
# import tensorflow_datasets as tfds
from google3.ads.publisher.quality.micro_models.tensorflow.util import my_common

# keras.config.set_dtype_policy("bfloat16")


def get_model(quantize=True, lora_rank=4):
  preprocessor = keras_nlp.models.GemmaCausalLMPreprocessor.from_preset(
      "gemma_1.1_instruct_2b_en", sequence_length=256
  )
  gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(
      "gemma_1.1_instruct_2b_en", preprocessor=preprocessor
  )
  if lora_rank > 0:
    gemma_lm.backbone.enable_lora(rank=lora_rank)
    if quantize:
      gemma_lm.quantize("int8")
  gemma_lm.summary(
      expand_nested=False, show_trainable=True
  )  ##, print_fn=print)
  # my_common.print_model_info('New model from tok layers', gemma_lm)
  return gemma_lm


def main(argv):
  get_model(quantize=True, lora_rank=3)


if __name__ == "__main__":
  app.run(main)
