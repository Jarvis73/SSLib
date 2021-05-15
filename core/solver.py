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

import torch
from torch.optim.lr_scheduler import _LRScheduler


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, power=0.9, lr_end=0, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        self.lr_end = lr_end
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = (1 - self.last_epoch / self.max_iter) ** self.power
        factor = max(factor, 0)
        return [(base_lr - self.lr_end) * factor + self.lr_end for base_lr in self.base_lrs]


def get(model, opt, max_steps=100000):
    if isinstance(model, list):
        params_group = model
    elif isinstance(model, torch.nn.Module):
        params_group = model.parameters()
    else:
        raise TypeError(f"`model` must be an nn.Model or a list, got {type(model)}")

    # =================================================================================
    # Optimizer
    # =================================================================================
    if opt.opti == "sgd":
        optimizer_params = {"momentum": opt.sgd_momentum,
                            "weight_decay": opt.weight_decay,
                            "nesterov": opt.sgd_nesterov}
        optimizer = torch.optim.SGD(params_group, opt.lr, **optimizer_params)
    elif opt.opti == "adam":
        optimizer_params = {"betas": (opt.adam_beta1, opt.adam_beta2),
                            "eps": opt.adam_epsilon,
                            "weight_decay": opt.weight_decay}
        optimizer = torch.optim.Adam(params_group, opt.lr, **optimizer_params)
    else:
        raise ValueError("Not supported optimizer: " + opt.opti)

    # =================================================================================
    # Scheduler
    # =================================================================================
    if opt.lrp == "period_step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=opt.lr_step,
                                                    gamma=opt.lr_rate)
    elif opt.lrp == "custom_step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=opt.lr_boundaries,
                                                         gamma=opt.lr_rate)
    elif opt.lrp == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor=opt.lr_rate,
                                                               patience=opt.lr_patience,
                                                               threshold=opt.lr_min_delta,
                                                               cooldown=opt.cool_down,
                                                               min_lr=opt.lr_end)
    elif opt.lrp == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=max_steps,
                                                               eta_min=opt.lr_end)
    elif opt.lrp == "poly":
        scheduler = PolyLR(optimizer,
                           max_iter=max_steps,
                           power=opt.power,
                           lr_end=opt.lr_end)
    else:
        raise ValueError

    return optimizer, scheduler
