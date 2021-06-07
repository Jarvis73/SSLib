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

import unittest
from pathlib import Path

import torch
import torch.utils.data
from sacred import Experiment

from config import setup_runner
from data_kits.voc import VOC
from data_kits import get_composed_augmentations

PROJECT_DIR = Path(__file__).parent


class TestDataLoader(unittest.TestCase):
    def test_shape_dtype_pairs(self):
        ex = Experiment(name="Train", save_git_info=False, base_dir=Path(__file__).parents[1])

        @ex.config
        def config():
            name = "default"
            logdir = str(PROJECT_DIR / 'runs' / name)
            bs = 16

            dataset = 'voc'
            split = 'trainaug'
            no_droplast = False
            noshuffle = False
            noaug = False
            train_n = 0
            val_n = 0

            n_class = 21
            num_workers = 6

            rng = (0.5, 2.0)
            rsize = 0
            rcrop = [513, 513]

        @ex.command
        def run_(_run, _config):
            opt, logger = setup_runner(ex, _run, _config)

            # -------------------------------------------------------------------
            aug = get_composed_augmentations(opt)
            train_ds = VOC(opt, logger, 'trainaug', augmentations=aug)
            train_loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size=opt.bs,
                shuffle=True,
                num_workers=int(opt.num_workers),
                drop_last=True,
                pin_memory=True,
            )

            batch = next(iter(train_loader))
            print(batch["img"].shape)
            print(batch["lab"].shape)

            print()
            # -------------------------------------------------------------------
            aug = get_composed_augmentations(opt)
            val_ds = VOC(opt, logger, 'val')
            val_loader = torch.utils.data.DataLoader(
                val_ds,
                batch_size=1,
                shuffle=False,
                num_workers=int(opt.num_workers),
                drop_last=False,
                pin_memory=True,
            )

            batch = next(iter(val_loader))
            print(batch["img"].shape)
            print(batch["lab"].shape)
            print(batch["lab_full"].shape)

        ex.run("run_")
