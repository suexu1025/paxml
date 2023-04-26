"""Trains and evaluates C4 NWP model."""

from collections.abc import Sequence
import collections
import os.path
import functools
from typing import Optional
#import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
import tqdm
from typing import Dict, List, Optional

#from google3.intelligence.federated.research.adaptive_dpftrl.tasks import stackoverflow_models as models
#import google3.learning.deepmind.xmanager2.client.google as xm  # pylint: disable=unused-import

import seqio
import t5.data
from t5.data import preprocessors as t5_preprocessors

IRRELEVANT_FLAGS = frozenset(iter(flags.FLAGS))

flags.DEFINE_string(
    'experiment_name', 'c4', 'The name of this experiment. Will be'
    'append to  --root_output_dir to separate experiment results.')
flags.DEFINE_string('root_output_dir', '/tmp/dpftrl/c4',
                    'Root directory for writing experiment output.')
flags.DEFINE_integer('rounds_per_checkpoint', 100,
                     'How often to checkpoint the global model.')
flags.DEFINE_integer(
    'rounds_per_eval', 20,
    'How often to evaluate the global model on the validation dataset.')
flags.DEFINE_integer('clients_per_thread', 1, 'TFF executor configuration.')

# Training
flags.DEFINE_integer('clients_per_round', 100,
                     'How many clients to sample per round.')
flags.DEFINE_integer('client_epochs_per_round', 1,
                     'Number of epochs in the client to take per round.')
flags.DEFINE_integer('client_batch_size', 16, 'Batch size used on the client.')
flags.DEFINE_integer('total_rounds', 10, 'Number of total training rounds.')
flags.DEFINE_integer(
    'total_epochs', None,
    'If not None, use shuffling of clients instead of random sampling.')

# Optimizer
flags.DEFINE_enum('client_optimizer', 'sgd', ['sgd'], 'Client optimzier')
flags.DEFINE_enum('server_optimizer', 'dpftrlm', ['dpftrlm'],
                  'Server optimizer in federated optimizaiotn.')
flags.DEFINE_float('client_lr', 0.02, 'Client learning rate.')
flags.DEFINE_float('server_lr', 1.0, 'Server learning rate.')
flags.DEFINE_float('server_momentum', 0.9, 'Server momentum for SGDM.')

# Differential privacy
flags.DEFINE_float('clip_norm', 1.0, 'Clip L2 norm.')
flags.DEFINE_float('noise_multiplier', 0.01,
                   'Noise multiplier for DP algorithm.')
flags.DEFINE_float(
    'unclip_quantile', None,
    'Target quantile for adaptive clipping. If `None`, use fixed clipping.')
flags.DEFINE_integer('restart_warmup', 50,
                     'Number of rounds when the first restart happens.')

# Data
flags.DEFINE_integer('sequence_length', 20, 'Max sequence length to use.')
flags.DEFINE_integer('max_elements_per_user', 256, 'Max number of training '
                     'sentences to use per user.')
flags.DEFINE_integer(
    'num_validation_examples', 10000, 'Number of examples '
    'to use from test set for per-round validation.')

# Model
flags.DEFINE_enum('model', 'lstm', ['lstm', 'transformer'],
                  'NN model architecture.')
flags.DEFINE_integer('vocab_size', 32000, 'Size of vocab to use.')
flags.DEFINE_integer('num_oov_buckets', 1,
                     'Number of out of vocabulary buckets.')
flags.DEFINE_integer('embedding_size', 96,
                     'Dimension of word embedding to use.')
flags.DEFINE_integer(
    'num_layers', 1,
    'Number of stacked recurrent layers or transformer blocks.')
# LSTM
flags.DEFINE_integer('latent_size', 670,
                     'Dimension of latent size to use in recurrent cell')
flags.DEFINE_boolean(
    'shared_embedding', False,
    'Boolean indicating whether to tie input and output embeddings.')
# Transformer
flags.DEFINE_integer('dim_model', 512,
                     'Dimension of features of MultiHeadAttention layers.')
flags.DEFINE_integer('dim_hidden', 2048,
                     'Dimension of hidden layers of the FFN.')
flags.DEFINE_integer('num_heads', 8, 'Number of attention heads.')
flags.DEFINE_float('dropout', 0, 'Dropout rate.')
# make tff happy
flags.DEFINE_integer('repeat', 0, 'Repeat index.')
flags.DEFINE_string(
    'worker_bns', '',
    'If not empty and not None, runs with a TFF RemoteExecutor, otherwise uses '
    'local machine for execution. This sets the task bns address (go/bnsgdh) '
    'of worker for Borg executor. Generally set outside the binary by the '
    'XManager launcher script.')
flags.DEFINE_boolean('cpp_runtime', False,
                     'Flag for TFF experimental CPP runtime.')

HPARAM_FLAGS = [f for f in flags.FLAGS if f not in IRRELEVANT_FLAGS]
FLAGS = flags.FLAGS

TRAIN_DATADIR = '/cns/gc-d/home/qinwen/xm/'
EVAL_DATADIR = '/cns/gc-d/home/qinwen/xm/'


# Vocabulary (shared by encoder and decoder)
# sentencepiece_model_file = 'gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model'
# # END GOOGLE-INTERNAL

# vocab = seqio.SentencePieceVocabulary(sentencepiece_model_file)
# LAMDA_SPM_PATH = '/cns/ys-d/home/lamda-data/vocab/meena_0611.32000.model'
# LAMDA_VOCABULARY = t5.data.SentencePieceVocabulary(LAMDA_SPM_PATH)

# MEENA_OUTPUT_FEATURES_LM = {
#     "targets": t5.data.Feature(vocabulary=LAMDA_VOCABULARY, add_eos=True)
# }
  
GPT_SPM_PATH = (
    'gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model'
)
GPT_EOS_ID = 1
GPT_VOCABULARY = t5.data.SentencePieceVocabulary(GPT_SPM_PATH)
PASS_THROUGH_VOCABULARY = t5.data.PassThroughVocabulary(size=50257)

C4_GPT_TRAIN_FEATURES_LM = {
    'targets': t5.data.Feature(vocabulary=GPT_VOCABULARY, add_eos=False)
}
C4_GPT_EVAL_FEATURES_LM = {
    'targets': t5.data.Feature(
        vocabulary=PASS_THROUGH_VOCABULARY, add_eos=False
    )
}

class TaskRegistry(t5.data.TaskRegistry):
  """Task registry with extra tracking."""
  TASK_NAMES = []

  @classmethod
  def add_versioned_tfds_task(cls,
                              name: str,
                              *,
                              versions: List[str],
                              pinned_version: Optional[str] = None,
                              tfds_name: str,
                              tfds_data_dir: Optional[str] = None,
                              **kwargs) -> List[seqio.Task]:
    tasks = []
    for version in versions:
      tasks.append(
          cls.add(
              f'{name}_{version}',
              seqio.Task,
              source=seqio.TfdsDataSource(
                  tfds_name=f'{tfds_name}:{version}',
                  tfds_data_dir=tfds_data_dir,
              ),
              **kwargs,
          ))
    if pinned_version is not None:
      tasks.append(
          cls.add(
              name,
              seqio.Task,
              source=seqio.TfdsDataSource(
                  tfds_name=f'{tfds_name}:{pinned_version}',
                  tfds_data_dir=tfds_data_dir,
              ),
              **kwargs,
          ))
    return tasks
  
def get_train_dataset():

  # TaskRegistry = seqio.TaskRegistry
  # TaskRegistry.add(
  #     'c4_v220_full_lm_trim',
  #     source=seqio.TfdsDataSource(tfds_name='c4/en:2.2.0'),
  #     preprocessors=[
  #         functools.partial(
  #             preprocessors.rekey, key_map={
  #                 'inputs': None,
  #                 'targets': 'text'
  #             }), seqio.preprocessors.tokenize,
  #         seqio.CacheDatasetPlaceholder(), preprocessors.full_lm,
  #         preprocessors.trim_and_pad_dataset
  #     ],
  #     output_features={
  #         'targets': seqio.Feature(vocabulary=vocab, add_eos=True)
  #     },
  #     metric_fns=[])

  # dataset = seqio.get_mixture_or_task('c4_v220_full_lm_trim').get_dataset(
  #     sequence_length={'targets': FLAGS.sequence_length + 1},
  #     split='train',
  #     shuffle=True,
  #     num_epochs=1,
  #     shard_info=seqio.ShardInfo(index=0, num_shards=10),
  #     use_cached=False,
  #     seed=42)
  TaskRegistry.add_versioned_tfds_task(
      'cnn_dailymail',
      versions=['3.4.0'],
      pinned_version='3.4.0',
      tfds_name='cnn_dailymail',
      tfds_data_dir=TRAIN_DATADIR,
      preprocessors=[
          functools.partial(
              t5_preprocessors.rekey,
              key_map={
                  'article': 'text',
                  'highlights': 'text',
                  'publisher': 'text',
                  'id': 'text'
              },
          ),
          seqio.preprocessors.tokenize,
          # functools.partial(
          #     t5_preprocessors.reduce_concat_tokens,
          #     batch_size=4096,
          # ),
          # t5_preprocessors.split_tokens_to_targets_length,
      ],
      output_features=C4_GPT_TRAIN_FEATURES_LM,
      metric_fns=[],
      shuffle_buffer_size=10000,
  )

  dataset = seqio.get_mixture_or_task('cnn_dailymail').get_dataset(
      sequence_length={'targets': FLAGS.sequence_length + 1},
      split='train',
      shuffle=True,
      num_epochs=1,
      shard_info=seqio.ShardInfo(index=0, num_shards=10),
      use_cached=False,
      seed=42)

  # Print the first 5 examples.
  for _, ex in zip(range(5), dataset.as_numpy_iterator()):
    print(ex)
    # print(vocab.decode(ex['targets']))

  batch_size = 32
  train_dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

  return train_dataset


# def pretrain():
#   # """Pre-Train on the C4 NWP task."""
#   # logging.info('Show FLAGS for debugging:')
#   # for f in HPARAM_FLAGS:
#   #   logging.info('%s=%s', f, FLAGS[f].value)

#   # hparam_dict = collections.OrderedDict([
#   #     (name, FLAGS[name].value) for name in HPARAM_FLAGS
#   # ])

#   # model = models.create_recurrent_model(
#   #     vocab_size=FLAGS.vocab_size,
#   #     embedding_size=FLAGS.embedding_size,
#   #     latent_size=FLAGS.latent_size,
#   #     num_layers=FLAGS.num_layers,
#   #     shared_embedding=FLAGS.shared_embedding)

#   print(FLAGS.vocab_size)
#   print(FLAGS.embedding_size)
#   print(FLAGS.latent_size)
#   print(FLAGS.num_layers)
#   print(FLAGS.shared_embedding)

#   # Instantiate an optimizer to train the model.
#   optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
#   # Instantiate a loss function.
#   loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#   train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

#   train_dataset = get_train_dataset()

#   for step, (batch_train) in enumerate(tqdm.tqdm(train_dataset)):
#     # print(batch_train['targets'].shape)
#     x = batch_train['targets'][:, :FLAGS.sequence_length]
#     y = batch_train['targets'][:, 1:FLAGS.sequence_length + 1]
#     # print(x.shape)
#     with tf.GradientTape() as tape:
#       logits = model(x, training=True)  # Logits for this minibatch
#       # print(logits.shape)
#       loss_value = loss_fn(y, logits)
#       # print(loss_value.shape)

#     grads = tape.gradient(loss_value, model.trainable_weights)
#     optimizer.apply_gradients(zip(grads, model.trainable_weights))

#     #   # Update training metric.
#     train_acc_metric.update_state(y, logits)

#     # Log every 200 batches.
#     if step % 200 == 0:
#       print('Training loss (for one batch) at step %d: %.4f' %
#             (step, float(loss_value)))

#     if step % 100000 == 0:
#       model.save(
#           os.path.join(FLAGS.root_output_dir, 'pretrained_c4_lstm' + str(step)))
#   #     print("Seen so far: %d samples" % ((step + 1) * batch_size))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  #pretrain()
  train_dataset = get_train_dataset()
  for step, (batch_train) in enumerate(tqdm.tqdm(train_dataset)):
       x = batch_train['article'][:, :FLAGS.sequence_length]
       y = batch_train['hightlights'][:, 1:FLAGS.sequence_length + 1] 
       print(x)
       print(y)

if __name__ == '__main__':
  app.run(main)
