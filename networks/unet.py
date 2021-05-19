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
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, pool=False, up=False, up_in_channels=None):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )
        if pool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        if up:
            self.up = nn.ConvTranspose2d(up_in_channels, in_channels, kernel_size=2, stride=2)

    def forward(self, x):
        if hasattr(self, "up"):
            y, x = x
            x = torch.cat([y, self.up(x)])

        x = self.block(x)

        if hasattr(self, "pool"):
            y = self.pool(x)
            return x, y

        return x


class UNet2D(nn.Module):
    def __init__(self, init_channels=3, channels=64, num_classes=2, pretrained=None, norm_layer=None):
        super(UNet2D, self).__init__()
        norm_layer = norm_layer or nn.BatchNorm2d

        self.enc1 = ConvBlock(init_channels, channels, norm_layer=norm_layer, pool=True)
        self.enc2 = ConvBlock(channels * 1, channels * 2, norm_layer=norm_layer, pool=True)
        self.enc3 = ConvBlock(channels * 2, channels * 4, norm_layer=norm_layer, pool=True)
        self.enc4 = ConvBlock(channels * 4, channels * 8, norm_layer=norm_layer, pool=True)
        self.bridge = ConvBlock(channels * 8, channels * 16, norm_layer=norm_layer)
        self.dec4 = ConvBlock(channels * 16, channels * 8, norm_layer=norm_layer, up=True, up_in_channels=channels * 16)
        self.dec3 = ConvBlock(channels * 8, channels * 4, norm_layer=norm_layer, up=True, up_in_channels=channels * 8)
        self.dec2 = ConvBlock(channels * 4, channels * 2, norm_layer=norm_layer, up=True, up_in_channels=channels * 4)
        self.dec1 = ConvBlock(channels * 2, channels * 1, norm_layer=norm_layer, up=True, up_in_channels=channels * 2)
        self.final = nn.Conv2d(channels, num_classes, kernel_size=1)

        self.init_weights(pretrained)

    def forward(self, x):
        x1, x = self.enc1(x)
        x2, x = self.enc2(x)
        x3, x = self.enc3(x)
        x4, x = self.enc4(x)
        x = self.bridge(x)
        x = self.dec4((x4, x))
        x = self.dec3((x3, x))
        x = self.dec2((x2, x))
        x = self.dec1((x1, x))
        x = self.final(x)

        return x

    def init_weights(self, pretrained=None):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

        if pretrained:
            pretrain_dict = torch.load(pretrained)['state_dict']
            model_dict = {}
            state_dict = model.state_dict()
            for k, v in pretrain_dict.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            model.load_state_dict(state_dict)
