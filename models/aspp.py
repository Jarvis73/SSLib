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


class ASPP(nn.Module):
    def __init__(self, inc, midc=256, outc=256, atrous=(1, 6, 12, 18)):
        super(ASPP, self).__init__()
        self.aspp_0 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inc, midc, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

        self.aspp_1 = nn.Sequential(
            nn.Conv2d(inc, midc,
                      kernel_size=3 if atrous[0] > 1 else 1,
                      stride=1,
                      padding=atrous[0] if atrous[0] > 1 else 0,
                      dilation=atrous[0], bias=False),
            nn.BatchNorm2d(midc),
            nn.ReLU())
        self.aspp_2 = nn.Sequential(
            nn.Conv2d(inc, midc, kernel_size=3, stride=1, padding=atrous[1], dilation=atrous[1], bias=False),
            nn.BatchNorm2d(midc),
            nn.ReLU())
        self.aspp_3 = nn.Sequential(
            nn.Conv2d(inc, midc, kernel_size=3, stride=1, padding=atrous[2], dilation=atrous[2], bias=False),
            nn.BatchNorm2d(midc),
            nn.ReLU())
        self.aspp_4 = nn.Sequential(
            nn.Conv2d(inc, midc, kernel_size=3, stride=1, padding=atrous[3], dilation=atrous[3], bias=False),
            nn.BatchNorm2d(midc),
            nn.ReLU())
        self.project = nn.Sequential(
            nn.Conv2d(5 * midc, outc, kernel_size=1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU())

    def forward(self, x):
        global_feat = self.aspp_0(x)
        global_feat = global_feat.expand(-1, -1, *x.shape[-2:])
        out = torch.cat((global_feat,
                         self.aspp_1(x),
                         self.aspp_2(x),
                         self.aspp_3(x),
                         self.aspp_4(x)), dim=1)  # [bs * query, 5c, h, w]
        out = self.project(out)
        return out
