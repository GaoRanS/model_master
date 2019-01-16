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


import os
import pickle
import time

import numpy as np

from official.recommendation import constants as rconst


def main(raw_path):
  np.random.seed(0)

  user_data = []
  item_data = []
  user_eval_data = []
  item_eval_data = []

  with open(raw_path, "rb") as f:
    user_id = -1
    item_max = -1
    count = 0
    while True:
      user_id += 1
      try:
        items = pickle.load(f)
        count += len(items)
        item_max = max([item_max, max(items)])

        # items are sorted by index. For now just randomly choose one to be the
        # eval item.
        np.random.shuffle(items)

        user_eval_data.append(user_id)
        item_eval_data.append(items.pop)

        user_data.append([user_id] * len(items))
        item_data.append(items)

        if not (user_id + 1) % 10000:
          print(str(user_id + 1).ljust(10), "{:.2E}".format(count))

        # if user_id + 1 == 10000:
        #   break

      except EOFError:
        break

  data = {
    rconst.TRAIN_USER_KEY: np.concatenate(user_data).astype(rconst.USER_DTYPE),
    rconst.TRAIN_ITEM_KEY: np.concatenate(item_data).astype(rconst.ITEM_DTYPE),
    rconst.EVAL_USER_KEY: np.array(user_eval_data, dtype=rconst.USER_DTYPE),
    rconst.EVAL_ITEM_KEY: np.array(user_eval_data, dtype=rconst.ITEM_DTYPE),
    rconst.USER_MAP: {i for i in range(user_id + 1)},
    rconst.ITEM_MAP: {i for i in range(item_max + 1)},
    "create_time": time.time() + int(1e10),  # never invalidate.
  }

  output_path = os.path.join(os.path.split(raw_path)[0], "transformed.pkl")
  with open(output_path, "wb") as f:
    pickle.dump(data, f)


if __name__ == "__main__":
  main("/tmp/ml_extended/16_32_ext_user_item_sequences.pkl")

