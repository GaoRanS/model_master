# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Preprocess dataset and construct any necessary artifacts."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import contextlib
import gc
import hashlib
import multiprocessing
import json
import os
import pickle
import signal
import socket
import subprocess
import threading
import time
import timeit
import typing

# pylint: disable=wrong-import-order
from absl import app as absl_app
from absl import flags
import numpy as np
import pandas as pd
import six
import tensorflow as tf
# pylint: enable=wrong-import-order

from official.datasets import movielens
from official.recommendation import constants as rconst
from official.recommendation import data_pipeline
from official.recommendation import stat_utils
from official.recommendation import popen_helper
from official.utils.logs import mlperf_helper


DATASET_TO_NUM_USERS_AND_ITEMS = {
    "ml-1m": (6040, 3706),
    "ml-20m": (138493, 26744)
}


# Number of batches to run per epoch when using synthetic data. At high batch
# sizes, we run for more batches than with real data, which is good since
# running more batches reduces noise when measuring the average batches/second.
SYNTHETIC_BATCHES_PER_EPOCH = 2000


class NCFDataset(object):
  """Container for training and testing data."""

  def __init__(self, user_map, item_map, num_data_readers,
               num_train_positives, deterministic=False):
    # type: (dict, dict, int, int, bool) -> None
    """Assign key values for recommendation dataset.

    Args:
      user_map: Dict mapping raw user ids to regularized ids.
      item_map: Dict mapping raw item ids to regularized ids.
      num_data_readers: The number of reader Datasets used during training.
      cache_paths: Object containing locations for various cache files.
      num_train_positives: The number of positive training examples in the
        dataset.
      deterministic: Operations should use deterministic, order preserving
        methods, even at the cost of performance.
    """

    self.user_map = {int(k): int(v) for k, v in user_map.items()}
    self.item_map = {int(k): int(v) for k, v in item_map.items()}
    self.num_users = len(user_map)
    self.num_items = len(item_map)
    self.num_data_readers = num_data_readers
    self.num_train_positives = num_train_positives
    self.deterministic = deterministic


def _filter_index_sort(raw_rating_path, match_mlperf):
  # type: (str, bool) -> (pd.DataFrame, dict, dict)
  """Read in data CSV, and output structured data.

  This function reads in the raw CSV of positive items, and performs three
  preprocessing transformations:

  1)  Filter out all users who have not rated at least a certain number
      of items. (Typically 20 items)

  2)  Zero index the users and items such that the largest user_id is
      `num_users - 1` and the largest item_id is `num_items - 1`

  3)  Sort the dataframe by user_id, with timestamp as a secondary sort key.
      This allows the dataframe to be sliced by user in-place, and for the last
      item to be selected simply by calling the `-1` index of a user's slice.

  While all of these transformations are performed by Pandas (and are therefore
  single-threaded), they only take ~2 minutes, and the overhead to apply a
  MapReduce pattern to parallel process the dataset adds significant complexity
  for no computational gain. For a larger dataset parallelizing this
  preprocessing could yield speedups. (Also, this preprocessing step is only
  performed once for an entire run.

  Args:
    raw_rating_path: The path to the CSV which contains the raw dataset.
    match_mlperf: If True, change the sorting algorithm to match the MLPerf
      reference implementation.

  Returns:
    A filtered, zero-index remapped, sorted dataframe, a dict mapping raw user
    IDs to regularized user IDs, and a dict mapping raw item IDs to regularized
    item IDs.
  """
  with tf.gfile.Open(raw_rating_path) as f:
    df = pd.read_csv(f)

  # Get the info of users who have more than 20 ratings on items
  grouped = df.groupby(movielens.USER_COLUMN)
  df = grouped.filter(
      lambda x: len(x) >= rconst.MIN_NUM_RATINGS) # type: pd.DataFrame

  original_users = df[movielens.USER_COLUMN].unique()
  original_items = df[movielens.ITEM_COLUMN].unique()

  mlperf_helper.ncf_print(key=mlperf_helper.TAGS.PREPROC_HP_MIN_RATINGS,
                          value=rconst.MIN_NUM_RATINGS)

  # Map the ids of user and item to 0 based index for following processing
  tf.logging.info("Generating user_map and item_map...")
  user_map = {user: index for index, user in enumerate(original_users)}
  item_map = {item: index for index, item in enumerate(original_items)}

  df[movielens.USER_COLUMN] = df[movielens.USER_COLUMN].apply(
      lambda user: user_map[user])
  df[movielens.ITEM_COLUMN] = df[movielens.ITEM_COLUMN].apply(
      lambda item: item_map[item])

  num_users = len(original_users)
  num_items = len(original_items)

  mlperf_helper.ncf_print(key=mlperf_helper.TAGS.PREPROC_HP_NUM_EVAL,
                          value=rconst.NUM_EVAL_NEGATIVES)
  mlperf_helper.ncf_print(
      key=mlperf_helper.TAGS.PREPROC_HP_SAMPLE_EVAL_REPLACEMENT,
      value=match_mlperf)

  assert num_users <= np.iinfo(np.int32).max
  assert num_items <= np.iinfo(np.uint16).max
  assert df[movielens.USER_COLUMN].max() == num_users - 1
  assert df[movielens.ITEM_COLUMN].max() == num_items - 1

  # This sort is used to shard the dataframe by user, and later to select
  # the last item for a user to be used in validation.
  tf.logging.info("Sorting by user, timestamp...")

  if match_mlperf:
    # This sort is equivalent to the non-MLPerf sort, except that the order of
    # items with the same user and timestamp are sometimes different. For some
    # reason, this sort results in a better hit-rate during evaluation, matching
    # the performance of the MLPerf reference implementation.
    df.sort_values(by=movielens.TIMESTAMP_COLUMN, inplace=True)
    df.sort_values([movielens.USER_COLUMN, movielens.TIMESTAMP_COLUMN],
                   inplace=True, kind="mergesort")
  else:
    df.sort_values([movielens.USER_COLUMN, movielens.TIMESTAMP_COLUMN],
                   inplace=True)

  df = df.reset_index()  # The dataframe does not reconstruct indicies in the
                         # sort or filter steps.

  return df, user_map, item_map


def instantiate_pipeline(dataset, data_dir, num_data_readers, match_mlperf,
                         deterministic, params):
  # type: (str, str, typing.Optional[int], bool, bool, dict) -> (NCFDataset, typing.Callable)
  """Load and digest data CSV into a usable form.

  Args:
    dataset: The name of the dataset to be used.
    data_dir: The root directory of the dataset.
    num_data_readers: The number of parallel processes which will request
      data during training.
    match_mlperf: If True, change the behavior of the cache construction to
      match the MLPerf reference implementation.
    deterministic: Try to enforce repeatable behavior, even at the cost of
      performance.
  """
  tf.logging.info("Beginning data preprocessing.")

  st = timeit.default_timer()
  raw_rating_path = os.path.join(data_dir, dataset, movielens.RATINGS_FILE)

  df, user_map, item_map = _filter_index_sort(raw_rating_path, match_mlperf)
  num_users, num_items = DATASET_TO_NUM_USERS_AND_ITEMS[dataset]

  if num_users != len(user_map):
    raise ValueError("Expected to find {} users, but found {}".format(
        num_users, len(user_map)))
  if num_items != len(item_map):
    raise ValueError("Expected to find {} items, but found {}".format(
        num_items, len(item_map)))

  ncf_dataset = NCFDataset(user_map=user_map, item_map=item_map,
                           num_data_readers=num_data_readers,
                           num_train_positives=len(df) - len(user_map),
                           deterministic=deterministic)

  # TODO(robieta): dump to file for MLPerf cache clear.
  grouped = df.groupby(movielens.USER_COLUMN, group_keys=False)
  eval_df, train_df = grouped.tail(1), grouped.apply(lambda x: x.iloc[:-1])
  producer = data_pipeline.MaterializedDataConstructor(
      maximum_number_epochs=params["train_epochs"],
      num_users=num_users,
      num_items=num_items,
      train_pos_users=train_df[movielens.USER_COLUMN].values.astype(rconst.USER_DTYPE),
      train_pos_items=train_df[movielens.ITEM_COLUMN].values.astype(rconst.ITEM_DTYPE),
      train_batch_size=params["batch_size"],
      batches_per_train_step=params["batches_per_step"],
      num_train_negatives=params["num_neg"],
      eval_pos_users=eval_df[movielens.USER_COLUMN].values.astype(rconst.USER_DTYPE),
      eval_pos_items=eval_df[movielens.ITEM_COLUMN].values.astype(rconst.ITEM_DTYPE),
      eval_batch_size=params["eval_batch_size"],
      batches_per_eval_step=params["batches_per_step"]
  )

  run_time = timeit.default_timer() - st
  tf.logging.info("Data preprocessing complete. Time: {:.1f} sec."
                  .format(run_time))

  return ncf_dataset, producer











def _train_map_fn(x):
    return {
        movielens.USER_COLUMN: x[0],
        movielens.ITEM_COLUMN: x[1],
    }, x[2]


def _eval_map_fn(x):
    return {
        movielens.USER_COLUMN: x[0],
        movielens.ITEM_COLUMN: x[1],
        rconst.DUPLICATE_MASK: x[2],
    }


def make_input_fn(producer, is_training, use_tpu):
  # type: (data_pipeline.BaseDataConstructor, bool, bool)
  if use_tpu:
    raise NotImplementedError["TODO(robieta): StreamingFilesDataset"]

  if isinstance(producer, data_pipeline.DummyConstructor):
    return make_synthetic_input_fn(is_training=is_training)

  if is_training:
    num_batches = producer.train_batches_per_epoch
    generator = producer.training_generator
    batch_size = producer.train_batch_size

    # users, items, labels
    output_types = (rconst.USER_DTYPE, rconst.ITEM_DTYPE, rconst.LABEL_DTYPE)
    output_shapes= tuple([tf.TensorShape([batch_size]) for _ in range(3)])
  else:
    num_batches = producer.eval_batches_per_epoch
    generator = producer.eval_generator
    batch_size = producer.eval_batch_size

    # users, items, duplicate_mask
    output_types = (rconst.USER_DTYPE, rconst.ITEM_DTYPE,
                    rconst.DUPE_MASK_DTYPE)
    output_shapes= tuple([tf.TensorShape([batch_size]) for _ in range(3)])

  def input_fn(params):
    param_batch_size = (params["batch_size"] if is_training else
                        params["eval_batch_size"])
    if batch_size != param_batch_size:
      raise ValueError("producer batch size ({}) differs from params batch "
                       "size ({})".format(batch_size, param_batch_size))

    dataset = tf.data.Dataset.from_generator(
        generator=generator, output_types=output_types,
        output_shapes=output_shapes)

    dataset = dataset.map(_train_map_fn if is_training else _eval_map_fn)
    dataset = dataset.prefetch(16)

    return dataset

  return input_fn, num_batches







# def make_deserialize(params, batch_size, training=False):
#   """Construct deserialize function for training and eval fns."""
#   feature_map = {
#       movielens.USER_COLUMN: tf.FixedLenFeature([], dtype=tf.string),
#       movielens.ITEM_COLUMN: tf.FixedLenFeature([], dtype=tf.string),
#   }
#   if training:
#     feature_map["labels"] = tf.FixedLenFeature([], dtype=tf.string)
#   else:
#     feature_map[rconst.DUPLICATE_MASK] = tf.FixedLenFeature([], dtype=tf.string)
#
#   def deserialize(examples_serialized):
#     """Called by Dataset.map() to convert batches of records to tensors."""
#     features = tf.parse_single_example(examples_serialized, feature_map)
#     users = tf.reshape(tf.decode_raw(
#         features[movielens.USER_COLUMN], tf.int32), (batch_size,))
#     items = tf.reshape(tf.decode_raw(
#         features[movielens.ITEM_COLUMN], tf.uint16), (batch_size,))
#
#     if params["use_tpu"] or params["use_xla_for_gpu"]:
#       items = tf.cast(items, tf.int32)  # TPU and XLA disallows uint16 infeed.
#
#     if not training:
#       dupe_mask = tf.reshape(tf.cast(tf.decode_raw(
#           features[rconst.DUPLICATE_MASK], tf.int8), tf.bool), (batch_size,))
#       return {
#           movielens.USER_COLUMN: users,
#           movielens.ITEM_COLUMN: items,
#           rconst.DUPLICATE_MASK: dupe_mask,
#       }
#
#     labels = tf.reshape(tf.cast(tf.decode_raw(
#         features["labels"], tf.int8), tf.bool), (batch_size,))
#
#     return {
#         movielens.USER_COLUMN: users,
#         movielens.ITEM_COLUMN: items,
#     }, labels
#   return deserialize
#
#
# def hash_pipeline(dataset, deterministic):
#   # type: (tf.data.Dataset, bool) -> None
#   """Utility function for detecting non-determinism in the data pipeline.
#
#   Args:
#     dataset: a tf.data.Dataset generated by the input_fn
#     deterministic: Does the input_fn expect the dataset to be deterministic.
#       (i.e. fixed seed, sloppy=False, etc.)
#   """
#   if not deterministic:
#     tf.logging.warning("Data pipeline is not marked as deterministic. Hash "
#                        "values are not expected to be meaningful.")
#
#   batch = dataset.make_one_shot_iterator().get_next()
#   md5 = hashlib.md5()
#   count = 0
#   first_batch_hash = b""
#   with tf.Session() as sess:
#     while True:
#       try:
#         result = sess.run(batch)
#         if isinstance(result, tuple):
#           result = result[0]  # only hash features
#       except tf.errors.OutOfRangeError:
#         break
#
#       count += 1
#       md5.update(memoryview(result[movielens.USER_COLUMN]).tobytes())
#       md5.update(memoryview(result[movielens.ITEM_COLUMN]).tobytes())
#       if count == 1:
#         first_batch_hash = md5.hexdigest()
#   overall_hash = md5.hexdigest()
#   tf.logging.info("Batch count: {}".format(count))
#   tf.logging.info("  [pipeline_hash] First batch hash: {}".format(
#       first_batch_hash))
#   tf.logging.info("  [pipeline_hash] All batches hash: {}".format(overall_hash))
#
#
# def make_input_fn(
#     ncf_dataset,       # type: typing.Optional[NCFDataset]
#     is_training,       # type: bool
#     record_files=None  # type: typing.Optional[tf.Tensor]
#     ):
#   # type: (...) -> (typing.Callable, str, int)
#   """Construct training input_fn for the current epoch."""
#
#   if ncf_dataset is None:
#     return make_synthetic_input_fn(is_training)
#
#   if record_files is not None:
#     epoch_metadata = None
#     batch_count = None
#     record_dir = None
#   else:
#     epoch_metadata, record_dir, template = get_epoch_info(is_training,
#                                                           ncf_dataset)
#     record_files = os.path.join(record_dir, template.format("*"))
#     # This value is used to check that the batch count from the subprocess
#     # matches the batch count expected by the main thread.
#     batch_count = epoch_metadata["batch_count"]
#
#
#   def input_fn(params):
#     """Generated input_fn for the given epoch."""
#     if is_training:
#       batch_size = params["batch_size"]
#     else:
#       # Estimator has "eval_batch_size" included in the params, but TPUEstimator
#       # populates "batch_size" to the appropriate value.
#       batch_size = params.get("eval_batch_size") or params["batch_size"]
#
#     if epoch_metadata and epoch_metadata["batch_size"] != batch_size:
#       raise ValueError(
#           "Records were constructed with batch size {}, but input_fn was given "
#           "a batch size of {}. This will result in a deserialization error in "
#           "tf.parse_single_example."
#           .format(epoch_metadata["batch_size"], batch_size))
#     record_files_ds = tf.data.Dataset.list_files(record_files, shuffle=False)
#
#     interleave = tf.contrib.data.parallel_interleave(
#         tf.data.TFRecordDataset,
#         cycle_length=4,
#         block_length=100000,
#         sloppy=not ncf_dataset.deterministic,
#         prefetch_input_elements=4,
#     )
#
#     deserialize = make_deserialize(params, batch_size, is_training)
#     dataset = record_files_ds.apply(interleave)
#     dataset = dataset.map(deserialize, num_parallel_calls=4)
#     dataset = dataset.prefetch(32)
#
#     if params.get("hash_pipeline"):
#       hash_pipeline(dataset, ncf_dataset.deterministic)
#
#     return dataset
#
#   return input_fn, record_dir, batch_count
#
#
# def _check_subprocess_alive(ncf_dataset, directory):
#   if (not tf.gfile.Exists(ncf_dataset.cache_paths.subproc_alive) and
#       not tf.gfile.Exists(directory)):
#     # The generation subprocess must have been alive at some point, because we
#     # earlier checked that the subproc_alive file existed.
#     raise ValueError("Generation subprocess unexpectedly died. Data will not "
#                      "be available; exiting to avoid waiting forever.")
#
#
#
# def get_epoch_info(is_training, ncf_dataset):
#   """Wait for the epoch input data to be ready and return various info about it.
#
#   Args:
#     is_training: If we should return info for a training or eval epoch.
#     ncf_dataset: An NCFDataset.
#
#   Returns:
#     epoch_metadata: A dict with epoch metadata.
#     record_dir: The directory with the TFRecord files storing the input data.
#     template: A string template of the files in `record_dir`.
#       `template.format('*')` is a glob that matches all the record files.
#   """
#   if is_training:
#     train_epoch_dir = ncf_dataset.cache_paths.train_epoch_dir
#     _check_subprocess_alive(ncf_dataset, train_epoch_dir)
#
#     while not tf.gfile.Exists(train_epoch_dir):
#       tf.logging.info("Waiting for {} to exist.".format(train_epoch_dir))
#       time.sleep(1)
#
#     train_data_dirs = tf.gfile.ListDirectory(train_epoch_dir)
#     while not train_data_dirs:
#       tf.logging.info("Waiting for data folder to be created.")
#       time.sleep(1)
#       train_data_dirs = tf.gfile.ListDirectory(train_epoch_dir)
#     train_data_dirs.sort()  # names are zfilled so that
#                             # lexicographic sort == numeric sort
#     record_dir = os.path.join(train_epoch_dir, train_data_dirs[0])
#     template = rconst.TRAIN_RECORD_TEMPLATE
#   else:
#     record_dir = ncf_dataset.cache_paths.eval_data_subdir
#     _check_subprocess_alive(ncf_dataset, record_dir)
#     template = rconst.EVAL_RECORD_TEMPLATE
#
#   ready_file = os.path.join(record_dir, rconst.READY_FILE)
#   while not tf.gfile.Exists(ready_file):
#     tf.logging.info("Waiting for records in {} to be ready".format(record_dir))
#     time.sleep(1)
#
#   with tf.gfile.Open(ready_file, "r") as f:
#     epoch_metadata = json.load(f)
#   return epoch_metadata, record_dir, template


def make_synthetic_input_fn(is_training):
  """Construct training input_fn that uses synthetic data."""
  def input_fn(params):
    """Generated input_fn for the given epoch."""
    batch_size = (params["batch_size"] if is_training else
                  params["eval_batch_size"] or params["batch_size"])
    num_users = params["num_users"]
    num_items = params["num_items"]

    users = tf.random_uniform([batch_size], dtype=tf.int32, minval=0,
                              maxval=num_users)
    items = tf.random_uniform([batch_size], dtype=tf.int32, minval=0,
                              maxval=num_items)

    if is_training:
      labels = tf.random_uniform([batch_size], dtype=tf.int32, minval=0,
                                 maxval=2)
      data = {
          movielens.USER_COLUMN: users,
          movielens.ITEM_COLUMN: items,
      }, labels
    else:
      dupe_mask = tf.cast(tf.random_uniform([batch_size], dtype=tf.int32,
                                            minval=0, maxval=2), tf.bool)
      data = {
          movielens.USER_COLUMN: users,
          movielens.ITEM_COLUMN: items,
          rconst.DUPLICATE_MASK: dupe_mask,
      }

    dataset = tf.data.Dataset.from_tensors(data).repeat(
        SYNTHETIC_BATCHES_PER_EPOCH)
    dataset = dataset.prefetch(32)
    return dataset

  return input_fn, SYNTHETIC_BATCHES_PER_EPOCH
