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
"""Statistics utility functions of NCF."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque
import multiprocessing
import sys
import threading
import time

import numpy as np


def random_int32():
  return np.random.randint(low=0, high=np.iinfo(np.int32).max, dtype=np.int32)


def _seeded_permutation(x, queue, count, count_lock, seed):
  seed = seed or struct.unpack("<L", os.urandom(4))[0]
  state = np.random.RandomState(seed=seed)
  output = np.arange(x, dtype=np.int32)
  state.shuffle(output)
  queue.put(output)
  with count_lock:
    count.set(count.get() - 1)


class AsyncPermuter(threading.Thread):
  def __init__(self, perm_size, num_workers=2, num_to_produce=None):
    super(AsyncPermuter, self).__init__()

    self._num_workers = num_workers

    self._num_to_produce = num_to_produce or np.inf
    self._started_count = 0
    self._max_queue_size = num_workers * 2

    self._pool = multiprocessing.Pool(num_workers)
    self._perm_size = perm_size
    self._manager = multiprocessing.Manager()
    self._result_queue = self._manager.Queue()
    self._active_count = self._manager.Value("i", 0)
    self._active_count_lock = self._manager.Lock()
    self._stop_loop = False

  def _loop_cond(self):
    with self._active_count_lock:
      return (self._started_count < self._num_to_produce or
              self._active_count.get()) and not self._stop_loop

  def run(self):
    while self._loop_cond():
      with self._active_count_lock:
        current_count = self._active_count.get()
        start_count = self._num_workers - current_count
        if self._result_queue.qsize() + current_count >= self._max_queue_size:
          start_count = 0

        self._active_count.set(self._active_count.get() + start_count)

      for _ in range(start_count):
        if self._started_count < self._num_to_produce:
          self._started_count += 1
          self._pool.apply_async(
              func=_seeded_permutation,
              args=(self._perm_size, self._result_queue, self._active_count,
                    self._active_count_lock, random_int32()))

      time.sleep(0.01)

    self._pool.close()
    self._pool.terminate()
    self.stop_loop()  # mark loop as closed.

  def get(self):
    if self._stop_loop and not self._result_queue.qsize():
      raise ValueError("No entries in result queue and permuter is no longer "
                       "producing entries.")
    return self._result_queue.get()

  def stop_loop(self):
    self._stop_loop = True


def very_slightly_biased_randint(max_val_vector):
  sample_dtype = np.uint64
  out_dtype = max_val_vector.dtype
  samples = np.random.randint(low=0, high=np.iinfo(sample_dtype).max,
                              size=max_val_vector.shape, dtype=sample_dtype)
  return np.mod(samples, max_val_vector.astype(sample_dtype)).astype(out_dtype)


def mask_duplicates(x, axis=1):  # type: (np.ndarray, int) -> np.ndarray
  """Identify duplicates from sampling with replacement.

  Args:
    x: A 2D NumPy array of samples
    axis: The axis along which to de-dupe.

  Returns:
    A NumPy array with the same shape as x with one if an element appeared
    previously along axis 1, else zero.
  """
  if axis != 1:
    raise NotImplementedError

  x_sort_ind = np.argsort(x, axis=1, kind="mergesort")
  sorted_x = x[np.arange(x.shape[0])[:, np.newaxis], x_sort_ind]

  # compute the indices needed to map values back to their original position.
  inv_x_sort_ind = np.argsort(x_sort_ind, axis=1, kind="mergesort")

  # Compute the difference of adjacent sorted elements.
  diffs = sorted_x[:, :-1] - sorted_x[:, 1:]

  # We are only interested in whether an element is zero. Therefore left padding
  # with ones to restore the original shape is sufficient.
  diffs = np.concatenate(
      [np.ones((diffs.shape[0], 1), dtype=diffs.dtype), diffs], axis=1)

  # Duplicate values will have a difference of zero. By definition the first
  # element is never a duplicate.
  return np.where(diffs[np.arange(x.shape[0])[:, np.newaxis],
                        inv_x_sort_ind], 0, 1)
