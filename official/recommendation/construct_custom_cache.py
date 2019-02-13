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
"""Coerce internally generated data into the format expected by NCF."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections
import os
import time

import pickle
import numpy as np

from official.recommendation import constants as rconst


def pkl_iterator(template, shards=16):
  for i in range(shards):
    print("shard {}".format(i))
    with open(template.format(i), "rb") as f:
      x = pickle.load(f, encoding='latin1')
      for j, data in enumerate(x):
        yield data

        if j == 400:
          break


def main(root):
  expansion = "4_16"
  train_template = os.path.join(root, expansion + "_correct_train.pkl_{}")
  test_template = os.path.join(root, expansion + "_correct_test.pkl_{}")
  # test_template = train_template
  num_shards = 1
  seed = 0

  item_counts = collections.defaultdict(int)
  user_id = -1
  num_train_pts = 0
  np.random.seed(seed)
  print("Starting precompute pass.")
  for train_items, test_items in zip(pkl_iterator(train_template, num_shards), pkl_iterator(test_template, num_shards)):
    user_id += 1

    # TODO(robieta): may as well use all the eval points?
    for i in train_items:
      item_counts[i] += 1
    item_counts[np.random.choice(test_items)] += 1
    num_train_pts += len(train_items)

  print("Computing dataset statistics.")
  num_positives = sum(item_counts.values())

  # Sort items by popularity to increase the efficiency of the bisection lookup
  item_map = sorted([(v, k) for k, v in item_counts.items()], reverse=True)
  item_map = {j: i for i, (_, j) in enumerate(item_map)}

  num_users = user_id + 1
  num_items = len(item_map)

  print("num_pts:  ", num_positives)
  print("num_users:", num_users)
  print("num_items:", num_items)

  assert num_users <= np.iinfo(rconst.USER_DTYPE).max
  assert num_items <= np.iinfo(rconst.ITEM_DTYPE).max

  # num_train_pts = num_positives - num_users
  train_users = np.zeros(shape=num_train_pts, dtype=rconst.USER_DTYPE) - 1
  train_items = np.zeros(shape=num_train_pts, dtype=rconst.ITEM_DTYPE) - 1
  eval_users = np.arange(num_users, dtype=rconst.USER_DTYPE)
  eval_items = np.zeros(shape=num_users, dtype=rconst.ITEM_DTYPE) - 1

  start_ind = 0
  np.random.seed(seed)
  print("Starting second pass.")
  for user_id, [user_train_items, user_test_items] in enumerate(
      zip(pkl_iterator(train_template, num_shards), pkl_iterator(test_template, num_shards))):
    user_train_items = [item_map[i] for i in user_train_items]
    eval_items[user_id] = item_map[np.random.choice(user_test_items)]

    train_users[start_ind:start_ind + len(user_train_items)] = user_id
    train_items[start_ind:start_ind + len(user_train_items)] = np.array(user_train_items, dtype=rconst.ITEM_DTYPE)
    start_ind += len(user_train_items)

  assert start_ind == num_train_pts
  assert not np.any(train_users == -1)
  assert not np.any(train_items == -1)
  assert not np.any(eval_items == -1)

  data = {
    rconst.TRAIN_USER_KEY: train_users,
    rconst.TRAIN_ITEM_KEY: train_items,
    rconst.EVAL_USER_KEY: eval_users,
    rconst.EVAL_ITEM_KEY: eval_items,
    rconst.USER_MAP: {i for i in range(num_users)},
    rconst.ITEM_MAP: item_map,
    "create_time": time.time() + int(1e10),  # never invalidate.
  }

  print("Writing record.")
  output_path = os.path.join(root, "{}_transformed_{}_shards_used_debug.pkl".format(expansion, num_shards))
  with open(output_path, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
  main("/tmp/ml_take_2")

