# Copyright 2021-2021 Jianwei Zhang All Right Reserved.
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
#
# =================================================================================

import time


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.acc_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.spc = 0.           # Seconds per call
        self.cps = 0.           # Calls per second

        self.total_time = 0.    # Not affected by self.reset()
        self.total_calls = 0    # Not affected by self.reset()

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.acc_time += self.diff
        self.total_time += self.diff
        self.calls += 1
        self.total_calls += 1
        self.spc = self.acc_time / self.calls
        self.cps = self.calls / self.acc_time
        return self.diff

    def reset(self):
        self.acc_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.spc = 0.
        self.cps = 0.

    def start(self):
        return self

    def __enter__(self):
        self.tic()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.toc()
