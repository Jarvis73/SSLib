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
from sacred import Experiment
from config import MapConfig, settings
from utils.misc import find_snapshot
from pathlib import Path


ex = Experiment("SSLib", base_dir=Path(__file__).parent, save_git_info=False)
settings(ex)


@ex.command(unobserved=True)
def fix_ckpt(_run, _config):
    opt = MapConfig(_config)

    ckpt_path = find_snapshot(opt, interactive=False)
    ckpt = torch.load(ckpt_path)

    for k, v in ckpt.items():
        if k not in ["epoch", "epoch_iou", "best_iou"]:
            ckpt[k]["state_dict"] = ckpt[k].pop("model_state", None)
        else:
            print(ckpt[k])
    torch.save(ckpt, ckpt_path)


@ex.command(unobserved=True)
def inspect(_run, _config):
    opt = MapConfig(_config)

    ckpt_path = find_snapshot(opt, interactive=False)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    for k, v in ckpt.items():
        if k not in ["epoch", "epoch_iou", "best_iou"]:
            for x in ckpt[k]["state_dict"]:
                print(x)
        else:
            print(f"{k}: {ckpt[k]}")


if __name__ == "__main__":
    ex.run_commandline()
