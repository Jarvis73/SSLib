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

import torch.nn as nn

from models.aspp import ASPP
from models.resnet import resnet_builder


class DeepLabV3(nn.Module):
    def __init__(self, backbone='resnet50',
                 num_classes=20,
                 pretrained=None,
                 freeze_bn=False,
                 norm_layer=None):
        super(DeepLabV3, self).__init__()

        self.encoder = resnet_builder(backbone, pretrained, freeze_bn, norm_layer)

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

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        out = self.head(out)
        return out
