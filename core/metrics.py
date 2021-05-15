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

import numpy as np


class Accumulator(object):
    def __init__(self, **kwargs):
        self.values = kwargs
        self.counter = {k: 0 for k, v in kwargs.items()}
        for k, v in self.values.items():
            if not isinstance(v, (float, int, list)):
                raise TypeError(f"The Accumulator does not support `{type(v)}`. Supported types: [float, int, list]")

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(self.values[k], list):
                self.values[k].append(v)
            else:
                self.values[k] = self.values[k] + v
            self.counter[k] += 1

    def mean(self, key, axis=None):
        if isinstance(key, str):
            if isinstance(self.values[key], list):
                return np.array(self.values[key]).mean(axis)
            else:
                return self.values[key] / self.counter[key]
        else:
            return [self.mean(k, axis) for k in key]

    def std(self, key, axis=None):
        if isinstance(key, str):
            if isinstance(self.values[key], list):
                return np.array(self.values[key]).std(axis)
            else:
                raise RuntimeError("`std` is not supported for (int, float). Use list instead.")
        elif isinstance(key, (list, tuple)):
            return [self.mean(k) for k in key]
        else:
            TypeError(f"`key` must be a str/list/tuple, got {type(key)}")


def _dice(ref, pred, num_classes=1, eps=1e-5):
    assert ref.shape == pred.shape, f"{ref.shape} vs {pred.shape}"

    dices = []
    for i in range(num_classes):
        n = 1 if num_classes == 1 else i
        rr, pp = ref[ref == n], pred[pred == n]
        a = np.count_nonzero(rr)
        b = np.count_nonzero(rr) + np.count_nonzero(pp)
        dices.append((2 * a + eps) / (b - a + eps))
    return dices


class DiceMetric(object):
    """ Dice Similarity Score """
    def __init__(self, num_classes=1, global_=False, eps=1e-5):
        self.num_classes = num_classes
        self.global_ = global_
        self.eps = eps

        # for global = True
        self.intersection = {i: 0. for i in range(self.num_classes)}
        self.union = {i: 0. for i in range(self.num_classes)}

        # for global = False
        self.dice = {i: 0. for i in range(self.num_classes)}

        self.count = 0

    def update(self, ref, pred):
        if self.global_:
            assert ref.shape == pred.shape, f"{ref.shape} vs {pred.shape}"

            for i in range(self.num_classes):
                n = 1 if self.num_classes == 1 else i
                rr, pp = ref[ref == n], pred[pred == n]
                a = np.count_nonzero(rr)
                b = np.count_nonzero(rr) + np.count_nonzero(pp)
                self.intersection[i] += a
                self.union[i] += b
        else:
            res = _dice(ref, pred, self.num_classes)
            for i in range(self.num_classes):
                self.dice[i] += res[i]

        self.count += 1

    def get_scores(self):
        if self.global_:
            res = [(2 * self.intersection[i] + self.eps) / (self.union[i] + self.eps)
                   for i in range(self.num_classes)]
            if len(res) == 1:
                return res[0]
        else:
            res = [self.dice[i] / (self.count + self.eps) for i in range(self.num_classes)]
            if len(res) == 1:
                return res[0]


class IoUMetric(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, ref, pred, n_class):
        mask = (ref >= 0) & (ref < n_class)
        hist = np.bincount(
            n_class * ref[mask].astype(int) + pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, ref, pred):
        for lt, lp in zip(ref, pred):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
