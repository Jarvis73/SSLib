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

import torch.nn.functional as F


def cross_entropy2d(inputs, target, weight=None, softmax_used=False, reduction='mean', ignore_index=250):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()

    if h != ht or w != wt:
        raise ValueError('sizes of input and label are not consistent')

    if softmax_used:
        loss = F.nll_loss(inputs, target, weight=weight, ignore_index=ignore_index, reduction=reduction)
    else:
        loss = F.cross_entropy(inputs, target, weight=weight, ignore_index=ignore_index, reduction=reduction)
    return loss
