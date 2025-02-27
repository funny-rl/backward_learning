# coding=utf-8
# Copyright 2019 Google LLC
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


"""Script allowing to replay a given trace file.
   Example usage:
   python replay.py --trace_file=/tmp/dumps/shutdown_20190521-165136974075.dump

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gfootball.env import script_helpers

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('fps', 80, 'How many frames per second to render')

def main(_):
  script_helpers.ScriptHelpers().replay("/user/hyunwoo/inverse_learning/results/replays/_11_vs_11_backward_learning/spectra_qmix_plus/SPECTra_QMIX+/1739961016/score_20250220-002055358712.dump", FLAGS.fps)

if __name__ == '__main__':
  app.run(main)