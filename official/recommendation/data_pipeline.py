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

import multiprocessing.dummy
import os
import pickle
import struct
import threading
import time
import timeit

import numpy as np
from six.moves import queue
import tensorflow as tf

from official.datasets import movielens
from official.recommendation import constants as rconst
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

class BaseDataConstructor(threading.Thread):
  def __init__(self,
               maximum_number_epochs,   # type: int
               num_users,               # type: int
               num_items,               # type: int
               train_pos_users,         # type: np.ndarray
               train_pos_items,         # type: np.ndarray
               train_batch_size,        # type: int
               batches_per_train_step,  # type: int
               num_train_negatives,     # type: int
               eval_pos_users,          # type: np.ndarray
               eval_pos_items,          # type: np.ndarray
               eval_batch_size,         # type: int
               batches_per_eval_step,   # type: int
              ):
    # General constants
    self._maximum_number_epochs = maximum_number_epochs
    self._num_users = num_users
    self._num_items = num_items

    # Training
    self._train_pos_users = train_pos_users
    self._train_pos_items = train_pos_items
    assert self._train_pos_users.shape == self._train_pos_items.shape
    self._train_pos_count = self._train_pos_users.shape[0]
    self.train_batch_size = train_batch_size
    self._num_train_negatives = num_train_negatives
    self._elements_in_epoch = (1 + num_train_negatives) * self._train_pos_count
    self._batches_per_train_step = batches_per_train_step
    self.train_batches_per_epoch = self._count_batches(
        self._elements_in_epoch, train_batch_size, batches_per_train_step)

    # Evaluation
    self._eval_pos_users = eval_pos_users
    self._eval_pos_items = eval_pos_items
    self.eval_batch_size = eval_batch_size
    if eval_batch_size % (1 + rconst.NUM_EVAL_NEGATIVES):
      raise ValueError("Eval batch size {} is not divisible by {}".format(
          eval_batch_size, 1 + rconst.NUM_EVAL_NEGATIVES))
    self._eval_users_per_batch = int(
        eval_batch_size // (1 + rconst.NUM_EVAL_NEGATIVES))
    self._eval_elements_in_epoch = num_users * (1 + rconst.NUM_EVAL_NEGATIVES)
    self.eval_batches_per_epoch = self._count_batches(
        self._eval_elements_in_epoch, eval_batch_size, batches_per_eval_step)

    # Intermediate artifacts
    self._current_epoch_order = np.empty(shape=(0,))
    self._shuffle_producer = stat_utils.AsyncPermuter(
        self._elements_in_epoch, num_workers=3,
        num_to_produce=maximum_number_epochs)
    self._training_queue = queue.Queue()
    self._eval_results = None
    self._eval_batches = None

    # Threading details
    self._current_epoch_order_lock = threading.Lock()
    super(BaseDataConstructor, self).__init__()
    self.daemon = True
    self._stop_loop = False

    # Generator annotations
    self.train_gen_types = (rconst.USER_DTYPE, rconst.ITEM_DTYPE,
                            rconst.LABEL_DTYPE, np.int32)
    self.train_gen_shapes = (tf.TensorShape([self.train_batch_size]),
                             tf.TensorShape([self.train_batch_size]),
                             tf.TensorShape([self.train_batch_size]),
                             tf.TensorShape([]))

    self.eval_gen_types = (rconst.USER_DTYPE, rconst.ITEM_DTYPE,
                           rconst.DUPE_MASK_DTYPE)
    self.eval_gen_shapes = tuple([tf.TensorShape([self.eval_batch_size])
                                  for _ in range(3)])

  def __repr__(self):
    summary = SUMMARY_TEMPLATE.format(
        spacer="  ", num_users=self._num_users, num_items=self._num_items,
        train_pos_ct=self._train_pos_count,
        train_batch_size=self.train_batch_size,
        train_batch_ct=self.train_batches_per_epoch,
        eval_pos_ct=self._num_users, eval_batch_size=self.eval_batch_size,
        eval_batch_ct=self.eval_batches_per_epoch,
        multiplier = "(x{} devices)".format(self._batches_per_train_step) if
        self._batches_per_train_step > 1 else "")
    return super(BaseDataConstructor, self).__repr__() + "\n" + summary

  def _count_batches(self, example_count, batch_size, batches_per_step):
    x = (example_count + batch_size - 1) // batch_size
    return (x + batches_per_step - 1) // batches_per_step * batches_per_step

  def stop_loop(self):
    self._shuffle_producer.stop_loop()
    self._stop_loop = True

  def _get_order_chunk(self):
    with self._current_epoch_order_lock:
      batch_indices = self._current_epoch_order[:self.train_batch_size]
      self._current_epoch_order = self._current_epoch_order[self.train_batch_size:]

      num_extra = self.train_batch_size - batch_indices.shape[0]
      if num_extra:
        batch_indices = np.concatenate([batch_indices,
                                        self._current_epoch_order[:num_extra]])
        self._current_epoch_order = self._current_epoch_order[num_extra:]

      return batch_indices

  def construct_lookup_variables(self):
    raise NotImplementedError

  def lookup_negative_items(self, **kwargs):
    raise NotImplementedError

  def run(self):
    self._shuffle_producer.start()
    self.construct_lookup_variables()
    self._construct_training_epoch()
    self._construct_eval_epoch()
    for _ in range(self._maximum_number_epochs - 1):
      self._construct_training_epoch()

  def _get_training_batch(self, _):
    batch_indices = self._get_order_chunk()

    batch_ind_mod = np.mod(batch_indices, self._train_pos_count)
    users = self._train_pos_users[batch_ind_mod]

    negative_indices = np.greater_equal(batch_indices, self._train_pos_count)
    negative_users = users[negative_indices]

    negative_items = self.lookup_negative_items(negative_users=negative_users)

    items = self._train_pos_items[batch_ind_mod]
    items[negative_indices] = negative_items

    labels = np.logical_not(negative_indices).astype(rconst.LABEL_DTYPE)

    # Pad last partial batch
    pad_length = self.train_batch_size - batch_indices.shape[0]
    if pad_length:
      # We pad with arange rather than zeros because the network will still
      # compute logits for padded examples, and padding with zeros would create
      # a very "hot" embedding key which can have performance implications.
      user_pad = np.arange(pad_length, dtype=users.dtype) % self._num_users
      item_pad = np.arange(pad_length, dtype=items.dtype) % self._num_items
      label_pad = np.zeros(shape=(pad_length,), dtype=labels.dtype)
      users = np.concatenate([users, user_pad])
      items = np.concatenate([items, item_pad])
      labels = np.concatenate([labels, label_pad])

    self._training_queue.put((users, items, labels, batch_indices.shape[0]))

  def _wait_to_construct_train_epoch(self):
    threshold = rconst.CYCLES_TO_BUFFER * self.train_batches_per_epoch
    count = 0
    while self._training_queue.qsize() >= threshold and not self._stop_loop:
      time.sleep(0.01)
      count += 1
      if count >= 100 and np.log10(count) == np.round(np.log10(count)):
        tf.logging.info(
            "Waited {} times for training data to be consumed".format(count))

  def _construct_training_epoch(self):
    self._wait_to_construct_train_epoch()
    if self._stop_loop:
      return

    start_time = timeit.default_timer()
    map_args = [i for i in range(self.train_batches_per_epoch)]
    assert not self._current_epoch_order.shape[0]
    self._current_epoch_order = self._shuffle_producer.get()
    with multiprocessing.dummy.Pool(6) as pool:
      pool.map(self._get_training_batch, map_args)

    tf.logging.info("Epoch construction complete. Time: {:.1f} seconds".format(
      timeit.default_timer() - start_time))

  def _get_eval_batch(self, i):
    low_index = i * self._eval_users_per_batch
    high_index = (i + 1) * self._eval_users_per_batch

    users = np.repeat(self._eval_pos_users[low_index:high_index, np.newaxis],
                      1 + rconst.NUM_EVAL_NEGATIVES, axis=1)

    # Ordering:
    #   The positive items should be last so that they lose ties. However, they
    #   should not be masked out if the true eval positive happens to be
    #   selected as a negative. So instead, the positive is placed in the first
    #   position, and then switched with the last element after the duplicate
    #   mask has been computed.
    items = np.concatenate([
      self._eval_pos_items[low_index:high_index, np.newaxis],
      self.lookup_negative_items(negative_users=users[:, :-1].flatten())
        .reshape(-1, rconst.NUM_EVAL_NEGATIVES),
    ], axis=1)

    # We pad the users and items here so that the duplicate mask calculation
    # will include the padding. The metric function relies on every element
    # except the positive being marked as duplicate to mask out padded points.
    if users.shape[0] < self._eval_users_per_batch:
      pad_rows = self._eval_users_per_batch - users.shape[0]
      padding = np.zeros(shape=(pad_rows, users.shape[1]), dtype=np.int32)
      users = np.concatenate([users, padding.astype(users.dtype)], axis=0)
      items = np.concatenate([items, padding.astype(items.dtype)], axis=0)

    duplicate_mask = stat_utils.mask_duplicates(items, axis=1).astype(
        rconst.DUPE_MASK_DTYPE)

    items[:, (0, -1)] = items[:, (-1, 0)]
    duplicate_mask[:, (0, -1)] = duplicate_mask[:, (-1, 0)]

    assert users.shape == items.shape == duplicate_mask.shape

    return users.flatten(), items.flatten(), duplicate_mask.flatten()

  def _construct_eval_epoch(self):
    if self._stop_loop:
      return

    start_time = timeit.default_timer()
    map_args = [i for i in range(self.eval_batches_per_epoch)]
    with multiprocessing.dummy.Pool(6) as pool:
      eval_results = pool.map(self._get_eval_batch, map_args)

    self._eval_results = eval_results

    tf.logging.info("Eval construction complete. Time: {:.1f} seconds".format(
        timeit.default_timer() - start_time))

  def training_generator(self):
    for _ in range(self.train_batches_per_epoch):
      yield self._training_queue.get(timeout=300)

  def eval_generator(self):
    while self._eval_results is None:
      time.sleep(0.01)

    for i in self._eval_results:
      yield i


class DummyConstructor(threading.Thread):
  def run(self):
    pass

  def stop_loop(self):
    pass


class MaterializedDataConstructor(BaseDataConstructor):
  def __init__(self, *args, **kwargs):
    super(MaterializedDataConstructor, self).__init__(*args, **kwargs)
    self._negative_table = None
    self._per_user_neg_count = None

  def construct_lookup_variables(self):
    # Materialize negatives for fast lookup sampling.
    start_time = timeit.default_timer()
    inner_bounds = np.argwhere(self._train_pos_users[1:] -
                               self._train_pos_users[:-1])[:, 0] + 1
    index_bounds = [0] + inner_bounds.tolist() + [self._num_users]
    self._negative_table = np.zeros(shape=(self._num_users, self._num_items),
                                    dtype=rconst.ITEM_DTYPE)

    # Set the table to the max value to make sure the embedding lookup will fail
    # if we go out of bounds, rather than just overloading item zero.
    self._negative_table += np.iinfo(rconst.ITEM_DTYPE).max
    assert self._num_items < np.iinfo(rconst.ITEM_DTYPE).max

    # Reuse arange during generation. np.delete will make a copy.
    full_set = np.arange(self._num_items, dtype=rconst.ITEM_DTYPE)

    self._per_user_neg_count = np.zeros(
      shape=(self._num_users,), dtype=np.int32)

    # Threading does not improve this loop. For some reason, the np.delete
    # call does not parallelize well. Multiprocessing incurs too much
    # serialization overhead to be worthwhile.
    for i in range(self._num_users):
      positives = self._train_pos_items[index_bounds[i]:index_bounds[i+1]]
      negatives = np.delete(full_set, positives)
      self._per_user_neg_count[i] = self._num_items - positives.shape[0]
      self._negative_table[i, :self._per_user_neg_count[i]] = negatives

    tf.logging.info("Negative sample table built. Time: {:.1f} seconds".format(
      timeit.default_timer() - start_time))

  def lookup_negative_items(self, negative_users, **kwargs):
    negative_item_choice = stat_utils.very_slightly_biased_randint(
      self._per_user_neg_count[negative_users])
    return self._negative_table[negative_users, negative_item_choice]
