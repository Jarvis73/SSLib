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

"""
References:

https://github.com/microsoft/ProDA/blob/main/data/__init__.py
"""

import importlib
import torch.utils.data
from data_kits.base_dataset import BaseDataset
from data_kits.augmentations import *


def find_dataset_by_name(name):
    """Import the module "data_kits/<name>.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data_kits." + name
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    for _name, cls in datasetlib.__dict__.items():
        if _name.lower() == name.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError(
            f"In {dataset_filename}.py, there should be a subclass of BaseDataset with class name "
            f"that matches {name} in lowercase.")

    return dataset


def get_composed_augmentations(opt):
    return Compose([RandomSized(opt.rng, opt.rsize),
                    RandomCrop(opt.rcrop),
                    RandomHorizontallyFlip(0.5)])


class CustomDatasetDataLoader(object):
    def __init__(self, opt, logger, splits=('train', 'val'), aug_cls=None):
        self.opt = opt
        self.logger = logger

        ds_cls = find_dataset_by_name(opt.dataset)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        if 'train' in splits or 'trainaug' in splits:
            split = 'train' if 'train' in splits else 'trainaug'
            data_aug = None if opt.noaug else (aug_cls or get_composed_augmentations(opt))
            self.train_dataset = ds_cls(opt, logger, augmentations=data_aug, split=split)
            self.logger.info(f"Dataset {self.train_dataset.__class__.__name__} for training was created")

            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=opt.bs,
                shuffle=not opt.noshuffle,
                num_workers=int(opt.num_workers),
                drop_last=not opt.no_droplast,
                pin_memory=True,
            )

        if 'val' in splits:
            self.val_dataset = ds_cls(opt, logger, augmentations=None, split='val')
            self.logger.info(f"Dataset {self.val_dataset.__class__.__name__} for validating was created")

            self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=opt.test_bs,
                shuffle=False,
                num_workers=int(opt.num_workers),
                drop_last=False,
                pin_memory=True,
            )

        if 'test' in splits:
            self.test_dataset = ds_cls(opt, logger, augmentations=None, split='test')
            self.logger.info(f"Dataset {self.test_dataset.__class__.__name__} for testing was created")

            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=opt.test_bs,
                shuffle=False,
                num_workers=int(opt.num_workers),
                drop_last=False,
                pin_memory=True,
            )
