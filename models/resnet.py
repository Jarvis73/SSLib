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

from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

affine_par = True
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}
__all__ = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "wide_resnet50_2",
    "wide_resnet101_2",
    "resnet_builder"
]


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, base_width=64, dilation=1,
                 downsample=None, norm_layer=None):
        super(BottleNeck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))

        self.conv1 = conv1x1(inplanes, width, stride)
        self.bn1 = norm_layer(width, affine=affine_par)
        self.conv2 = conv3x3(width, width, dilation=dilation)
        self.bn2 = norm_layer(width, affine=affine_par)
        self.conv3 = conv1x1(width, width * self.expansion)
        self.bn3 = norm_layer(width * self.expansion, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class _ResNet(nn.Module):
    """ Resnet template used for semantic segmentation """
    def __init__(self, block, layers, width_per_group=64, norm_layer=None):
        super(_ResNet, self).__init__()
        self.inplanes = 64
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        self.init_weights()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        norm_layer = self._norm_layer
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion, affine=affine_par)
            )

        layers = [block(self.inplanes, planes, stride, self.base_width, dilation,
                        downsample=downsample, norm_layer=norm_layer)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.base_width,
                                dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.max_pool(out)

        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        return out4

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, (nn.BatchNorm2d, SynchronizedBatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def freeze_bn_func(m):
    if m.__class__.__name__.find('BatchNorm') != -1 or isinstance(m, SynchronizedBatchNorm2d) \
            or isinstance(m, nn.BatchNorm2d):
        m.weight.requires_grad = False
        m.bias.requires_grad = False


def initialize(name, model, freeze_bn=False, pretrained=None):
    if freeze_bn:
        model.apply(freeze_bn_func)

    if pretrained is False:
        pass
    else:
        if pretrained is None:
            pretrain_dict = model_zoo.load_url(model_urls[name])
        else:
            pretrain_dict = torch.load(pretrained)['state_dict']
        model_dict = {}
        state_dict = model.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=None, freeze_bn=False, norm_layer=None):
    model = _ResNet(BasicBlock, [2, 2, 2, 2], norm_layer=norm_layer)
    model = initialize('resnet18', model, freeze_bn, pretrained)
    return model


def resnet34(pretrained=None, freeze_bn=False, norm_layer=None):
    model = _ResNet(BasicBlock, [3, 4, 6, 3], norm_layer=norm_layer)
    model = initialize('resnet34', model, freeze_bn, pretrained)
    return model


def resnet50(pretrained=None, freeze_bn=False, norm_layer=None):
    model = _ResNet(BottleNeck, [3, 4, 6, 3], norm_layer=norm_layer)
    model = initialize('resnet50', model, freeze_bn, pretrained)
    return model


def resnet101(pretrained=None, freeze_bn=False, norm_layer=None):
    model = _ResNet(BottleNeck, [3, 4, 23, 3], norm_layer=norm_layer)
    model = initialize('resnet101', model, freeze_bn, pretrained)
    return model


def resnet152(pretrained=None, freeze_bn=False, norm_layer=None):
    model = _ResNet(BottleNeck, [3, 8, 36, 3], norm_layer=norm_layer)
    model = initialize('resnet152', model, freeze_bn, pretrained)
    return model


def wide_resnet50_2(pretrained=None, freeze_bn=False, norm_layer=None):
    model = _ResNet(BottleNeck, [3, 4, 6, 3], width_per_group=64 * 2, norm_layer=norm_layer)
    model = initialize('wide_resnet50_2', model, freeze_bn, pretrained)
    return model


def wide_resnet101_2(pretrained=None, freeze_bn=False, norm_layer=None):
    model = _ResNet(BottleNeck, [3, 4, 23, 3], width_per_group=64 * 2, norm_layer=norm_layer)
    model = initialize('wide_resnet101_2', model, freeze_bn, pretrained)
    return model


def resnet_builder(name, pretrained=None, freeze_bn=False, norm_layer=None):
    """
    ResNet builder for semantic segmentation tasks.

    Available models: resnet18, resnet34, resnet50, resnet101, resnet152, wide_resnet50_2, wide_resnet101_2

    Parameters
    ----------
    name: str
        model name
    pretrained: bool or str
        If False, no pretrained model will be loaded
        If None, imagenet pretrained model will be loaded (auto download)
        If a str, the specified pretrained model will be loaded
    freeze_bn: bool
        freeze batch norm layer or not
    norm_layer: class
        batch norm layer, either nn.BatchNorm2d or SynchronizedBatchNorm2d.
        nn.BatchNorm2d will be used by default.

    Returns
    -------
    A model instance
    """
    kwargs = {
        "pretrained": pretrained,
        "freeze_bn": freeze_bn,
        "norm_layer": norm_layer
    }

    if name == 'resnet18':
        model = resnet18(**kwargs)
    elif name == 'resnet34':
        model = resnet34(**kwargs)
    elif name == 'resnet50':
        model = resnet50(**kwargs)
    elif name == 'resnet101':
        model = resnet101(**kwargs)
    elif name == 'resnet152':
        model = resnet152(**kwargs)
    elif name == 'wide_resnet50_2':
        model = wide_resnet50_2(**kwargs)
    elif name == 'wide_resnet101_2':
        model = wide_resnet101_2(**kwargs)
    else:
        raise NotImplementedError(f"Model `{name}` is not implemented yet! "
                                  f"Select model from {list(model_urls.keys())}.")

    return model