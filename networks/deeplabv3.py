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
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from networks.aspp import ASPP
from networks.resnet import resnet_builder, model_urls
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


def freeze_bn_func(m):
    if m.__class__.__name__.find('BatchNorm') != -1 or isinstance(m, SynchronizedBatchNorm2d) \
            or isinstance(m, nn.BatchNorm2d):
        m.weight.requires_grad = False
        m.bias.requires_grad = False


class DeepLabV3(nn.Module):
    """
    DeepLabV3 implementation.

    Parameters
    ----------
    backbone: str
        Backbone model. [resnet50|resnet101]
    output_stride: int
        Output stride size. [8|16]
    multi_grid: tuple
        Multi-grid dilation in the layers of the last block
    num_classes: int
        Number of segmentation classes
    pretrained: bool or str
        If False, no pretrained model will be loaded
        If None, imagenet pretrained model will be loaded (auto download)
        If a str, the specified pretrained model will be loaded
    freeze_bn: bool
        Freeze batch norm layer or not
    norm_layer:
        Normalization layer class. [None|nn.BatchNorm2d|SynchronizedBatchNorm2d]
        None means using default nn.BatchNorm2d
    """
    def __init__(self, backbone='resnet50',
                 output_stride=16,
                 multi_grid=(1, 2, 4),
                 num_classes=20,
                 pretrained=None,
                 freeze_bn=False,
                 norm_layer=None,
                 logger=None):
        super(DeepLabV3, self).__init__()
        self.backbone = backbone
        self.pretrained = pretrained
        self.freeze_bn = freeze_bn
        self.logger = logger

        self.encoder = resnet_builder(backbone, output_stride, multi_grid, norm_layer)
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ASPP(inc=256, midc=256, outc=256)
        )
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False)
        )

        self.initialize()

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        out = self.head(out)
        return out

    def initialize(self):
        if self.freeze_bn:
            self.apply(freeze_bn_func)

        if self.pretrained is False:
            pass
        else:
            if self.pretrained is None:
                pretrain_dict = model_zoo.load_url(model_urls[self.backbone])
                obj = self.encoder
                load_path = model_urls[self.backbone]
            else:
                pretrain_dict = torch.load(self.pretrained)[self.__class__.__name__]['state_dict']
                obj = self
                load_path = self.pretrained
            model_dict = {}
            state_dict = obj.state_dict()
            for k, v in pretrain_dict.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            obj.load_state_dict(state_dict)

            if self.logger:
                self.logger.info(f"Load checkpoint from {load_path}")
