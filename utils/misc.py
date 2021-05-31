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

from pathlib import Path


def search(exp_dir):
    ckpt_names = ["best.pth", "ckpt.pth"]
    for name in ckpt_names:
        ckpt_path = exp_dir / name
        if ckpt_path.exists():
            return ckpt_path
    return False


def try_possible_ckpt_names(base, exp_id, ckpt=None):
    if exp_id >= 0:
        # Search running results from `logdir` directory (with specific name)
        ckpt_path = search(base / str(exp_id))
        if ckpt_path:
            return ckpt_path

        # Search running results from `runs` directory (without specific name)
        for exp_dir in base.parent.glob("*/[0-9]*"):
            if exp_dir.is_dir() and int(exp_dir.name) == exp_id:
                ckpt_path = search(exp_dir)
                return ckpt_path

    # Use ckpt directly
    ckpt_file = Path(ckpt)
    if ckpt_file.exists():
        return ckpt_file

    return False


def find_snapshot(opt, interactive=True):
    ckpt_path = try_possible_ckpt_names(Path(opt.logdir), opt.ckpt_id, opt.ckpt)
    if ckpt_path:
        return ckpt_path
    else:
        print(f"Cannot find checkpoints from `opt.ckpt_id={opt.ckpt_id}` and `opt.ckpt={opt.ckpt}`.")

    if interactive:
        # Import readline module to improve the experience of input
        # noinspection PyUnresolvedReferences
        import readline

        while True:
            inputs = input("Please input a checkpoint file ([q] to skip):")
            if inputs in ["q", "Q"]:
                break
            ckpt_path = Path(inputs)
            if ckpt_path.exists():
                return ckpt_path
            else:
                print("Not found!")

    return None
