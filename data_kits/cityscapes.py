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

import numpy as np
import torch
from PIL import Image

from data_kits.base_dataset import BaseDataset

DATAROOT = Path(__file__).parents[1] / "datasets" / "CityScape"
cached_data = {}


class CityScapes(BaseDataset):
    """ CityScapes Data Loader:
    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))
    mean_rgb = [0, 0, 0]

    def __init__(self, opt, logger, split='train', augmentations=None, cache=False):
        super(CityScapes, self).__init__(opt)

        self.opt = opt
        self.logger = logger
        self.cache = cache
        self.root = DATAROOT
        self.split = split
        self.augmentations = augmentations
        self.n_classes = opt.n_class
        self.img_size = (2048, 1024)
        self.files = {}
        self.paired_files = {}

        self.images_base = self.root / "leftImg8bit" / self.split
        self.annotations_base = self.root / "gtFine" / self.split

        self.files = sorted(self.images_base.glob("*/*.png"))

        # self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ["unlabelled",   "road",     "sidewalk",         "building",     "wall",
                            "fence",        "pole",     "traffic_light",    "traffic_sign", "vegetation",
                            "terrain",      "sky",      "person",           "rider",        "car",
                            "truck",        "bus",      "train",            "motorcycle",   "bicycle",]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))  # origin --> label

        if not self.files:
            raise Exception(f"No files for split=[{self.split}] found in {self.images_base}")

        self.logger.info("Found %d %s images" % (len(self.files), self.split))

    def __len__(self):
        actual = len(self.files)
        if self.split == 'train' and self.opt.train_n > 0:
            return min(actual, self.opt.train_n)
        if self.split == 'val' and self.opt.val_n > 0:
            return min(actual, self.opt.val_n)
        return len(self.files)

    def get_image(self, img_path):
        if self.cache and img_path in cached_data:
            return cached_data[img_path].copy()
        else:
            img = Image.open(img_path)  # 2048 x 1024
            img = np.array(img, dtype=np.uint8)
            if self.cache:
                cached_data[img_path] = img.copy()
            return img

    def get_label(self, lab_path):
        if self.cache and lab_path in cached_data:
            return cached_data[lab_path].copy()
        else:
            lab = Image.open(lab_path)  # 2048 x 1024
            lab = np.array(lab, dtype=np.uint8)
            lab = self.encode_segmap(lab)
            if self.cache:
                cached_data[lab_path] = lab.copy()
            return lab

    def __getitem__(self, index):
        img_path = self.files[index]
        lab_path = self.annotations_base / img_path.parts[-2] / (img_path.name[:-15] + "gtFine_labelIds.png")

        img = self.get_image(img_path)
        lab = self.get_label(lab_path)

        if self.augmentations is not None:
            img, lab = self.augmentations(img, lab)

        img, lab = self.transform(img, lab)

        return {
            "img": img,
            "lab": lab,
            "img_path": str(img_path)
        }

    def encode_segmap(self, mask):
        # Put all void classes to 250
        label_copy = 250 * np.ones(mask.shape, dtype=np.uint8)
        for k, v in list(self.class_map.items()):
            label_copy[mask == k] = v
        return label_copy

    def transform(self, img, lab, lp=None, check=True):

        img = (img.astype(np.float32) - self.mean_rgb) / 255.
        img = img.transpose(2, 0, 1)    # HWC -> CHW

        img = torch.from_numpy(img).float()
        lab = torch.from_numpy(lab).long()

        return img, lab
