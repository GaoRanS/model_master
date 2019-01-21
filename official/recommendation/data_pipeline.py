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
"""Asynchronous data producer for the NCF pipeline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import collections
import functools
import gc
import mmap
import multiprocessing
import os
import sys
import tempfile
import threading
import time
import timeit
import traceback
import typing
import uuid

import numpy as np
import six
from six.moves import queue
import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu.datasets import StreamingFilesDataset

from official.datasets import movielens
from official.recommendation import constants as rconst
from official.recommendation import popen_helper
from official.recommendation import stat_utils


SUMMARY_TEMPLATE = """General:
{spacer}Num users: {num_users}
{spacer}Num items: {num_items}

Training:
{spacer}Positive count:          {train_pos_ct}
{spacer}Batch size:              {train_batch_size} {multiplier}
{spacer}Batch count per epoch:   {train_batch_ct}

Eval:
{spacer}Positive count:          {eval_pos_ct}
{spacer}Batch size:              {eval_batch_size} {multiplier}
{spacer}Batch count per epoch:   {eval_batch_ct}"""


_TRAIN_FEATURE_MAP = {
    movielens.USER_COLUMN: tf.FixedLenFeature([], dtype=tf.string),
    movielens.ITEM_COLUMN: tf.FixedLenFeature([], dtype=tf.string),
    rconst.MASK_START_INDEX: tf.FixedLenFeature([1], dtype=tf.string),
    "labels": tf.FixedLenFeature([], dtype=tf.string),
}


_EVAL_FEATURE_MAP = {
    movielens.USER_COLUMN: tf.FixedLenFeature([], dtype=tf.string),
    movielens.ITEM_COLUMN: tf.FixedLenFeature([], dtype=tf.string),
    rconst.DUPLICATE_MASK: tf.FixedLenFeature([], dtype=tf.string)
}


def batch_to_file(data):
  # type: (dict) -> str
  fpath = os.path.join(rconst.BATCH_FILE_DIR, str(uuid.uuid4()))
  feature_dict = {
    k: tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[memoryview(v).tobytes()])) for k, v in data.items()}

  data_bytes = tf.train.Example(
      features=tf.train.Features(feature=feature_dict)).SerializeToString()

  with tf.gfile.Open(fpath, "wb") as f:
    f.write(data_bytes)

  return fpath


class DatasetManager(object):
  """Helper class for handling TensorFlow specific data tasks.

  This class takes the (relatively) framework agnostic work done by the data
  constructor classes and handles the TensorFlow specific portions (TFRecord
  management, tf.Dataset creation, etc.).
  """
  def __init__(self, is_training, stream_files, batches_per_epoch,
               shard_root=None, deterministic=False):
    # type: (bool, bool, int, typing.Optional[str], bool) -> None
    """Constructs a `DatasetManager` instance.
    Args:
      is_training: Boolean of whether the data provided is training or
        evaluation data. This determines whether to reuse the data
        (if is_training=False) and the exact structure to use when storing and
        yielding data.
      stream_files: Boolean indicating whether data should be serialized and
        written to file shards.
      batches_per_epoch: The number of batches in a single epoch.
      shard_root: The base directory to be used when stream_files=True.
      deterministic: Forgo non-deterministic speedups. (i.e. sloppy=True)
    """
    self._is_training = is_training
    self._deterministic = deterministic
    self._stream_files = stream_files
    self._writers = []
    self._write_locks = [threading.RLock() for _ in
                         range(rconst.NUM_FILE_SHARDS)] if stream_files else []
    self._batches_per_epoch = batches_per_epoch
    self._epochs_completed = 0
    self._epochs_requested = 0
    self._shard_root = shard_root

    self._result_queue = queue.Queue()
    self._result_reuse = []

  @property
  def current_data_root(self):
    subdir = (rconst.TRAIN_FOLDER_TEMPLATE.format(self._epochs_completed)
              if self._is_training else rconst.EVAL_FOLDER)
    return os.path.join(self._shard_root, subdir)

  def buffer_reached(self):
    # Only applicable for training.
    return (self._epochs_completed - self._epochs_requested >=
            rconst.CYCLES_TO_BUFFER and self._is_training)

  @staticmethod
  def _serialize(data):
    """Convert NumPy arrays into a TFRecords entry."""

    feature_dict = {
        k: tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[memoryview(v).tobytes()])) for k, v in data.items()}

    return tf.train.Example(
        features=tf.train.Features(feature=feature_dict)).SerializeToString()

  def _deserialize(self, serialized_data, batch_size):
    """Convert serialized TFRecords into tensors.

    Args:
      serialized_data: A tensor containing serialized records.
      batch_size: The data arrives pre-batched, so batch size is needed to
        deserialize the data.
    """
    feature_map = _TRAIN_FEATURE_MAP if self._is_training else _EVAL_FEATURE_MAP
    features = tf.parse_single_example(serialized_data, feature_map)

    users = tf.reshape(tf.decode_raw(
        features[movielens.USER_COLUMN], rconst.USER_DTYPE), (batch_size,))
    items = tf.reshape(tf.decode_raw(
        features[movielens.ITEM_COLUMN], rconst.ITEM_DTYPE), (batch_size,))

    def decode_binary(data_bytes):
      # tf.decode_raw does not support bool as a decode type. As a result it is
      # necessary to decode to int8 (7 of the bits will be ignored) and then
      # cast to bool.
      return tf.reshape(tf.cast(tf.decode_raw(data_bytes, tf.int8), tf.bool),
                        (batch_size,))

    if self._is_training:
      mask_start_index = tf.decode_raw(
          features[rconst.MASK_START_INDEX], tf.int32)[0]
      valid_point_mask = tf.less(tf.range(batch_size), mask_start_index)

      return {
          movielens.USER_COLUMN: users,
          movielens.ITEM_COLUMN: items,
          rconst.VALID_POINT_MASK: valid_point_mask,
      }, decode_binary(features["labels"])

    return {
        movielens.USER_COLUMN: users,
        movielens.ITEM_COLUMN: items,
        rconst.DUPLICATE_MASK: decode_binary(features[rconst.DUPLICATE_MASK]),
    }

  def put(self, fpath):
    if self._stream_files:
      raise NotImplementedError("Shared file writers have not been migrated in "
                                "this prototype.")

    return self._result_queue.put(fpath)

  def start_construction(self):
    if self._stream_files:
      tf.gfile.MakeDirs(self.current_data_root)
      template = os.path.join(self.current_data_root, rconst.SHARD_TEMPLATE)
      self._writers = [tf.io.TFRecordWriter(template.format(i))
                       for i in range(rconst.NUM_FILE_SHARDS)]

  def end_construction(self):
    if self._stream_files:
      [writer.close() for writer in self._writers]
      self._writers = []
      self._result_queue.put(self.current_data_root)

    self._epochs_completed += 1

  def data_generator(self, epochs_between_evals):
    """Yields examples during local training."""
    assert not self._stream_files
    assert self._is_training or epochs_between_evals == 1

    if self._is_training:
      qsize_counter, idx = 0, 0
      for idx in range(self._batches_per_epoch * epochs_between_evals):
        qsize_counter += self._result_queue.qsize()
        batch_path = self._result_queue.get(timeout=300)
        with tf.gfile.Open(batch_path, "rb") as f:
          result = f.read()
        tf.gfile.Remove(batch_path)
        yield result
      tf.logging.info("Mean qsize: {:.1f}".format(qsize_counter / (idx + 1)))
    else:
      if self._result_reuse:
        assert len(self._result_reuse) == self._batches_per_epoch

        for batch_path in self._result_reuse:
          with tf.gfile.Open(batch_path, "rb") as f:
            result = f.read()
          yield result
      else:
        # First epoch.
        for _ in range(self._batches_per_epoch * epochs_between_evals):
          batch_path = self._result_queue.get(timeout=300)
          with tf.gfile.Open(batch_path, "rb") as f:
            result = f.read()
          self._result_reuse.append(batch_path)
          yield result


  def get_dataset(self, batch_size, epochs_between_evals):
    """Construct the dataset to be used for training and eval.

    For local training, data is provided through Dataset.from_generator. For
    remote training (TPUs) the data is first serialized to files and then sent
    to the TPU through a StreamingFilesDataset.

    Args:
      batch_size: The per-device batch size of the dataset.
      epochs_between_evals: How many epochs worth of data to yield.
        (Generator mode only.)
    """
    self._epochs_requested += 1
    if self._stream_files:
      raise NotImplementedError("StreamingFilesDataset has been removed from "
                                "this prototype.")

    else:
      data_generator = functools.partial(
          self.data_generator, epochs_between_evals=epochs_between_evals)
      dataset = tf.data.Dataset.from_generator(
          generator=data_generator, output_types=tf.string,
          output_shapes=tf.TensorShape([]))

      map_fn = functools.partial(self._deserialize, batch_size=batch_size)
      dataset = dataset.map(map_fn, num_parallel_calls=16)

    return dataset.prefetch(16)

  def make_input_fn(self, batch_size):
    """Create an input_fn which checks for batch size consistency."""

    def input_fn(params):
      param_batch_size = (params["batch_size"] if self._is_training else
                          params["eval_batch_size"])
      if batch_size != param_batch_size:
        raise ValueError("producer batch size ({}) differs from params batch "
                         "size ({})".format(batch_size, param_batch_size))

      epochs_between_evals = (params.get("epochs_between_evals", 1)
                              if self._is_training else 1)
      return self.get_dataset(batch_size=batch_size,
                              epochs_between_evals=epochs_between_evals)

    return input_fn


_WORKER_CACHE = {}
MMAPSPEC = collections.namedtuple("MMapSpec", ["buffer", "dtype", "shape"])
def to_mmap(x, name):
  # type: (np.ndarray, str) -> MMAPSPEC
  buffer_path = os.path.join(rconst.MMAP_CACHE, "{}.buffer".format(name))
  if tf.gfile.Exists(buffer_path):
    raise ValueError("{} exists".format(buffer_path))

  x.tofile(buffer_path)
  atexit.register(tf.gfile.Remove, filename=buffer_path)
  return MMAPSPEC(buffer=buffer_path, dtype=x.dtype, shape=x.shape)


def from_mmap(spec):
  # type: (MMAPSPEC) -> np.ndarray
  f = os.open(spec.buffer, os.O_RDONLY)
  atexit.register(os.close, fd=f)
  buffer = mmap.mmap(f, 0, access=mmap.ACCESS_READ)
  x = np.frombuffer(buffer, dtype=spec.dtype).reshape(spec.shape)  # type: np.ndarray
  x.flags.writeable = False
  return x


def worker_init_fn(lookup_variables):
  # type: (dict) -> None
  try:
    for key, value in lookup_variables.items():
      if isinstance(value, MMAPSPEC):
        value = from_mmap(value)

      _WORKER_CACHE[key] = value
  except Exception:
    traceback.print_exc()
    sys.stderr.flush()
    raise


class BaseDataConstructor(threading.Thread):
  """Data constructor base class.

  This class manages the control flow for constructing data. It is not meant
  to be used directly, but instead subclasses should implement the following
  two methods:

    self.construct_lookup_variables
    self.lookup_negative_items

  """
  def __init__(self,
               maximum_number_epochs,   # type: int
               num_users,               # type: int
               num_items,               # type: int
               user_map,                # type: dict
               item_map,                # type: dict
               train_pos_users,         # type: np.ndarray
               train_pos_items,         # type: np.ndarray
               train_batch_size,        # type: int
               batches_per_train_step,  # type: int
               num_train_negatives,     # type: int
               eval_pos_users,          # type: np.ndarray
               eval_pos_items,          # type: np.ndarray
               eval_batch_size,         # type: int
               batches_per_eval_step,   # type: int
               stream_files,            # type: bool
               deterministic=False,     # type: bool
               use_permutation=True     # type: bool
              ):
    # General constants
    self._maximum_number_epochs = maximum_number_epochs
    self._num_users = num_users
    self._num_items = num_items
    self.user_map = user_map
    self.item_map = item_map
    self._train_pos_users = train_pos_users
    self._train_pos_items = train_pos_items
    self.train_batch_size = train_batch_size
    self._num_train_negatives = num_train_negatives
    self._batches_per_train_step = batches_per_train_step
    self._eval_pos_users = eval_pos_users
    self._eval_pos_items = eval_pos_items
    self.eval_batch_size = eval_batch_size
    self._use_permutation = use_permutation

    # Training
    if self._train_pos_users.shape != self._train_pos_items.shape:
      raise ValueError(
          "User positives ({}) is different from item positives ({})".format(
              self._train_pos_users.shape, self._train_pos_items.shape))

    (self._train_pos_count,) = self._train_pos_users.shape
    self._elements_in_epoch = (1 + num_train_negatives) * self._train_pos_count
    self.train_batches_per_epoch = self._count_batches(
        self._elements_in_epoch, train_batch_size, batches_per_train_step)

    # Evaluation
    if eval_batch_size % (1 + rconst.NUM_EVAL_NEGATIVES):
      raise ValueError("Eval batch size {} is not divisible by {}".format(
          eval_batch_size, 1 + rconst.NUM_EVAL_NEGATIVES))
    self._eval_users_per_batch = int(
        eval_batch_size // (1 + rconst.NUM_EVAL_NEGATIVES))
    self._eval_elements_in_epoch = num_users * (1 + rconst.NUM_EVAL_NEGATIVES)
    self.eval_batches_per_epoch = self._count_batches(
        self._eval_elements_in_epoch, eval_batch_size, batches_per_eval_step)

    if stream_files:
      self._shard_root = tempfile.mkdtemp(prefix="ncf_")
      atexit.register(tf.gfile.DeleteRecursively, dirname=self._shard_root)
    else:
      self._shard_root = None

    self._train_dataset = DatasetManager(
        True, stream_files, self.train_batches_per_epoch, self._shard_root,
        deterministic)
    self._eval_dataset = DatasetManager(
        False, stream_files, self.eval_batches_per_epoch, self._shard_root,
        deterministic)

    # Threading details
    super(BaseDataConstructor, self).__init__()
    self.daemon = True
    self._stop_loop = False
    self._fatal_exception = None
    self.deterministic = deterministic

    # Share constants with workers.
    self.initialize_mmap_cache()
    self._lookup_variables = {
      "num_users": self._num_users,
      "num_items": self._num_items,
      "elements_in_epoch": self._elements_in_epoch,
      "train_pos_count": self._train_pos_count,
      "train_batch_size": self.train_batch_size,
      "train_pos_users": to_mmap(self._train_pos_users, "train_pos_users"),
      "train_pos_items": to_mmap(self._train_pos_items, "train_pos_items"),
      "eval_users_per_batch": self._eval_users_per_batch,
      "eval_pos_users": self._eval_pos_users,
      "eval_pos_items": self._eval_pos_items,
    }
    self.construct_lookup_variables()
    gc.collect()

    self._master_pool_worker_count = 32
    self._master_pool = popen_helper.get_forkpool(
        self._master_pool_worker_count, init_worker=worker_init_fn,
        initargs=(self._lookup_variables,), closing=False)

    # atexit will run functions in reverse order of registration.
    atexit.register(self._master_pool.join)
    atexit.register(self._master_pool.terminate)


  def __str__(self):
    multiplier = ("(x{} devices)".format(self._batches_per_train_step)
                  if self._batches_per_train_step > 1 else "")
    summary = SUMMARY_TEMPLATE.format(
        spacer="  ", num_users=self._num_users, num_items=self._num_items,
        train_pos_ct=self._train_pos_count,
        train_batch_size=self.train_batch_size,
        train_batch_ct=self.train_batches_per_epoch,
        eval_pos_ct=self._num_users, eval_batch_size=self.eval_batch_size,
        eval_batch_ct=self.eval_batches_per_epoch, multiplier=multiplier)
    return super(BaseDataConstructor, self).__str__() + "\n" + summary

  def initialize_mmap_cache(self):
    if tf.gfile.Exists(rconst.MMAP_CACHE):
      tf.gfile.DeleteRecursively(rconst.MMAP_CACHE)

    tf.gfile.MakeDirs(rconst.MMAP_CACHE)
    tf.gfile.MakeDirs(rconst.BATCH_FILE_DIR)

  @staticmethod
  def _count_batches(example_count, batch_size, batches_per_step):
    """Determine the number of batches, rounding up to fill all devices."""
    x = (example_count + batch_size - 1) // batch_size
    return (x + batches_per_step - 1) // batches_per_step * batches_per_step

  def stop_loop(self):
    self._stop_loop = True

  def construct_lookup_variables(self):
    """Perform any one time pre-compute work."""
    raise NotImplementedError

  def lookup_negative_items(self, **kwargs):
    """Randomly sample negative items for given users."""
    raise NotImplementedError

  def _run(self):
    atexit.register(self.stop_loop)
    self._start_shuffle_iterator()
    self._construct_training_epoch()
    self._construct_eval_epoch()
    for _ in range(self._maximum_number_epochs - 1):
      self._construct_training_epoch()
    self.stop_loop()

  def run(self):
    try:
      self._run()
    except Exception as e:
      # The Thread base class swallows stack traces, so unfortunately it is
      # necessary to catch and re-raise to get debug output
      traceback.print_exc()
      self._fatal_exception = e
      sys.stdout.flush()
      sys.stderr.flush()
      raise

  def _start_shuffle_iterator(self):
    if self._use_permutation:
      raise NotImplementedError("Full shuffle support is not in this prototype.")

  @staticmethod
  def _get_training_batch(args):
    """Construct a single batch of training data.
    """
    i, lookup_negative_items = args

    train_batch_size = _WORKER_CACHE["train_batch_size"]
    _train_pos_count = _WORKER_CACHE["train_pos_count"]
    _train_pos_users = _WORKER_CACHE["train_pos_users"]
    _train_pos_items = _WORKER_CACHE["train_pos_items"]
    _num_users = _WORKER_CACHE["num_users"]
    _num_items = _WORKER_CACHE["num_items"]
    _elements_in_epoch = _WORKER_CACHE["elements_in_epoch"]

    np.random.seed()
    batch_indices = np.random.randint(low=0, high=_elements_in_epoch,
                                      size=(train_batch_size,), dtype=np.int64)
    (mask_start_index,) = batch_indices.shape

    batch_ind_mod = np.mod(batch_indices, _train_pos_count)
    users = _train_pos_users[batch_ind_mod]

    negative_indices = np.greater_equal(batch_indices, _train_pos_count)
    negative_users = users[negative_indices]

    negative_items = lookup_negative_items(negative_users=negative_users)

    items = _train_pos_items[batch_ind_mod]
    items[negative_indices] = negative_items

    labels = np.logical_not(negative_indices)

    # Pad last partial batch
    pad_length = train_batch_size - mask_start_index
    if pad_length:
      # We pad with arange rather than zeros because the network will still
      # compute logits for padded examples, and padding with zeros would create
      # a very "hot" embedding key which can have performance implications.
      user_pad = np.arange(pad_length, dtype=users.dtype) % _num_users
      item_pad = np.arange(pad_length, dtype=items.dtype) % _num_items
      label_pad = np.zeros(shape=(pad_length,), dtype=labels.dtype)
      users = np.concatenate([users, user_pad])
      items = np.concatenate([items, item_pad])
      labels = np.concatenate([labels, label_pad])

    return batch_to_file({
      movielens.USER_COLUMN: users,
      movielens.ITEM_COLUMN: items,
      rconst.MASK_START_INDEX: np.array(mask_start_index, dtype=np.int32),
      "labels": labels,
    })

  def _wait_to_construct_train_epoch(self):
    count = 0
    while self._train_dataset.buffer_reached() and not self._stop_loop:
      time.sleep(0.01)
      count += 1
      if count >= 100 and np.log10(count) == np.round(np.log10(count)):
        tf.logging.info(
            "Waited {} times for training data to be consumed".format(count))

  def _construct_training_epoch(self):
    """Loop to construct a batch of training data."""
    self._wait_to_construct_train_epoch()
    start_time = timeit.default_timer()
    if self._stop_loop:
      return

    self._train_dataset.start_construction()
    map_args = ((i, self.lookup_negative_items) for i
                in range(self.train_batches_per_epoch))

    for batch_fpath in self._master_pool.imap_unordered(
        self._get_training_batch, map_args):
      self._train_dataset.put(batch_fpath)

    self._train_dataset.end_construction()

    tf.logging.info("Epoch construction complete. Time: {:.1f} seconds".format(
        timeit.default_timer() - start_time))

  @staticmethod
  def _assemble_eval_batch(users, positive_items, negative_items,
                           users_per_batch):
    """Construct duplicate_mask and structure data accordingly.

    The positive items should be last so that they lose ties. However, they
    should not be masked out if the true eval positive happens to be
    selected as a negative. So instead, the positive is placed in the first
    position, and then switched with the last element after the duplicate
    mask has been computed.

    Args:
      users: An array of users in a batch. (should be identical along axis 1)
      positive_items: An array (batch_size x 1) of positive item indices.
      negative_items: An array of negative item indices.
      users_per_batch: How many users should be in the batch. This is passed
        as an argument so that ncf_test.py can use this method.

    Returns:
      User, item, and duplicate_mask arrays.
    """
    items = np.concatenate([positive_items, negative_items], axis=1)

    # We pad the users and items here so that the duplicate mask calculation
    # will include padding. The metric function relies on all padded elements
    # except the positive being marked as duplicate to mask out padded points.
    if users.shape[0] < users_per_batch:
      pad_rows = users_per_batch - users.shape[0]
      padding = np.zeros(shape=(pad_rows, users.shape[1]), dtype=np.int32)
      users = np.concatenate([users, padding.astype(users.dtype)], axis=0)
      items = np.concatenate([items, padding.astype(items.dtype)], axis=0)

    duplicate_mask = stat_utils.mask_duplicates(items, axis=1).astype(np.bool)

    items[:, (0, -1)] = items[:, (-1, 0)]
    duplicate_mask[:, (0, -1)] = duplicate_mask[:, (-1, 0)]

    assert users.shape == items.shape == duplicate_mask.shape
    return users, items, duplicate_mask

  @staticmethod
  def _get_eval_batch(args):
    """Construct a single batch of evaluation data.

    Args:
      i: The index of the batch.
    """
    i, lookup_negative_items, _assemble_eval_batch = args

    _eval_users_per_batch = _WORKER_CACHE["eval_users_per_batch"]
    _eval_pos_users = _WORKER_CACHE["eval_pos_users"]
    _eval_pos_items = _WORKER_CACHE["eval_pos_items"]

    low_index = i * _eval_users_per_batch
    high_index = (i + 1) * _eval_users_per_batch
    users = np.repeat(_eval_pos_users[low_index:high_index, np.newaxis],
                      1 + rconst.NUM_EVAL_NEGATIVES, axis=1)
    positive_items = _eval_pos_items[low_index:high_index, np.newaxis]
    negative_items = (lookup_negative_items(negative_users=users[:, :-1])
                      .reshape(-1, rconst.NUM_EVAL_NEGATIVES))

    users, items, duplicate_mask = _assemble_eval_batch(
        users, positive_items, negative_items, _eval_users_per_batch)

    return batch_to_file({
      movielens.USER_COLUMN: users.flatten(),
      movielens.ITEM_COLUMN: items.flatten(),
      rconst.DUPLICATE_MASK: duplicate_mask.flatten(),
    })


  def _construct_eval_epoch(self):
    """Loop to construct data for evaluation."""
    if self._stop_loop:
      return

    start_time = timeit.default_timer()

    self._eval_dataset.start_construction()
    map_args = ((i, self.lookup_negative_items, self._assemble_eval_batch)
                for i in range(self.eval_batches_per_epoch))

    for batch_fpath in self._master_pool.imap_unordered(
        self._get_eval_batch, map_args):
      self._eval_dataset.put(batch_fpath)

    self._eval_dataset.end_construction()

    tf.logging.info("Eval construction complete. Time: {:.1f} seconds".format(
        timeit.default_timer() - start_time))

  def make_input_fn(self, is_training):
    # It isn't feasible to provide a foolproof check, so this is designed to
    # catch most failures rather than provide an exhaustive guard.
    if self._fatal_exception is not None:
      raise ValueError("Fatal exception in the data production loop: {}"
                       .format(self._fatal_exception))

    return (
        self._train_dataset.make_input_fn(self.train_batch_size) if is_training
        else self._eval_dataset.make_input_fn(self.eval_batch_size))


class DummyConstructor(threading.Thread):
  """Class for running with synthetic data."""
  def run(self):
    pass

  def stop_loop(self):
    pass

  @staticmethod
  def make_input_fn(is_training):
    """Construct training input_fn that uses synthetic data."""

    def input_fn(params):
      """Generated input_fn for the given epoch."""
      batch_size = (params["batch_size"] if is_training else
                    params["eval_batch_size"])
      num_users = params["num_users"]
      num_items = params["num_items"]

      users = tf.random_uniform([batch_size], dtype=tf.int32, minval=0,
                                maxval=num_users)
      items = tf.random_uniform([batch_size], dtype=tf.int32, minval=0,
                                maxval=num_items)

      if is_training:
        valid_point_mask = tf.cast(tf.random_uniform(
            [batch_size], dtype=tf.int32, minval=0, maxval=2), tf.bool)
        labels = tf.cast(tf.random_uniform(
            [batch_size], dtype=tf.int32, minval=0, maxval=2), tf.bool)
        data = {
            movielens.USER_COLUMN: users,
            movielens.ITEM_COLUMN: items,
            rconst.VALID_POINT_MASK: valid_point_mask,
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
          rconst.SYNTHETIC_BATCHES_PER_EPOCH * params["batches_per_step"])
      dataset = dataset.prefetch(32)
      return dataset

    return input_fn


class MaterializedDataConstructor(BaseDataConstructor):
  def __init__(self, *args, **kwargs):
    super(MaterializedDataConstructor, self).__init__(*args, **kwargs)
    raise NotImplementedError




class BisectionDataConstructor(BaseDataConstructor):
  """Use bisection to index within positive examples.

  This class tallies the number of negative items which appear before each
  positive item for a user. This means that in order to select the ith negative
  item for a user, it only needs to determine which two positive items bound
  it at which point the item id for the ith negative is a simply algebraic
  expression.
  """
  def __init__(self, *args, **kwargs):
    super(BisectionDataConstructor, self).__init__(*args, **kwargs)
    self.index_bounds = None
    self._sorted_train_pos_items = None

  def _index_segment(self, user):
    lower, upper = self.index_bounds[user:user+2]
    items = self._sorted_train_pos_items[lower:upper]

    negatives_since_last_positive = np.concatenate(
        [items[0][np.newaxis], items[1:] - items[:-1] - 1])

    return np.cumsum(negatives_since_last_positive)

  def construct_lookup_variables(self):
    start_time = timeit.default_timer()
    inner_bounds = np.argwhere(self._train_pos_users[1:] -
                               self._train_pos_users[:-1])[:, 0] + 1
    (upper_bound,) = self._train_pos_users.shape
    self.index_bounds = np.array([0] + inner_bounds.tolist() + [upper_bound],
                                 dtype=np.int64)

    # Later logic will assume that the users are in sequential ascending order.
    assert np.array_equal(self._train_pos_users[self.index_bounds[:-1]],
                          np.arange(self._num_users))

    self._sorted_train_pos_items = self._train_pos_items.copy()

    for i in range(self._num_users):
      lower, upper = self.index_bounds[i:i+2]
      self._sorted_train_pos_items[lower:upper].sort()

    total_negatives = np.concatenate([
        self._index_segment(i) for i in range(self._num_users)])

    # Share constants with workers.
    self._lookup_variables["index_bounds"] = to_mmap(self.index_bounds, "index_bounds")
    self._lookup_variables["total_negatives"] = to_mmap(total_negatives, "total_negatives")
    self._lookup_variables["sorted_train_pos_items"] = to_mmap(self._sorted_train_pos_items, "sorted_train_pos_items")

    tf.logging.info("Negative total vector built. Time: {:.1f} seconds".format(
        timeit.default_timer() - start_time))

  @staticmethod
  def lookup_negative_items(negative_users, **kwargs):
    _index_bounds = _WORKER_CACHE["index_bounds"]
    _total_negatives = _WORKER_CACHE["total_negatives"]
    _sorted_train_pos_items = _WORKER_CACHE["sorted_train_pos_items"]
    _num_items = _WORKER_CACHE["num_items"]

    output = np.zeros(shape=negative_users.shape, dtype=rconst.ITEM_DTYPE) - 1

    left_index = _index_bounds[negative_users]
    right_index = _index_bounds[negative_users + 1] - 1

    num_positives = right_index - left_index + 1
    num_negatives = _num_items - num_positives
    neg_item_choice = stat_utils.very_slightly_biased_randint(num_negatives)

    # Shortcuts:
    # For points where the negative is greater than or equal to the tally before
    # the last positive point there is no need to bisect. Instead the item id
    # corresponding to the negative item choice is simply:
    #   last_postive_index + 1 + (neg_choice - last_negative_tally)
    # Similarly, if the selection is less than the tally at the first positive
    # then the item_id is simply the selection.
    #
    # Because MovieLens organizes popular movies into low integers (which is
    # preserved through the preprocessing), the first shortcut is very
    # efficient, allowing ~60% of samples to bypass the bisection. For the same
    # reason, the second shortcut is rarely triggered (<0.02%) and is therefore
    # not worth implementing.
    use_shortcut = neg_item_choice >= _total_negatives[right_index]
    output[use_shortcut] = (
        _sorted_train_pos_items[right_index] + 1 +
        (neg_item_choice - _total_negatives[right_index])
    )[use_shortcut]

    if np.all(use_shortcut):
      # The bisection code is ill-posed when there are no elements.
      return output

    not_use_shortcut = np.logical_not(use_shortcut)
    left_index = left_index[not_use_shortcut]
    right_index = right_index[not_use_shortcut]
    neg_item_choice = neg_item_choice[not_use_shortcut]

    num_loops = np.max(
        np.ceil(np.log2(num_positives[not_use_shortcut])).astype(np.int32))

    for i in range(num_loops):
      mid_index = (left_index + right_index) // 2
      right_criteria = _total_negatives[mid_index] > neg_item_choice
      left_criteria = np.logical_not(right_criteria)

      right_index[right_criteria] = mid_index[right_criteria]
      left_index[left_criteria] = mid_index[left_criteria]

    # Expected state after bisection pass:
    #   The right index is the smallest index whose tally is greater than the
    #   negative item choice index.

    assert np.all((right_index - left_index) <= 1)

    output[not_use_shortcut] = (
        _sorted_train_pos_items[right_index] -
        (_total_negatives[right_index] - neg_item_choice)
    )

    assert np.all(output >= 0)

    return output


def get_constructor(name):
  if name == "bisection":
    return BisectionDataConstructor
  if name == "materialized":
    return MaterializedDataConstructor
  raise ValueError("Unrecognized constructor: {}".format(name))
