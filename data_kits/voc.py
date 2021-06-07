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

DATAROOT = Path(__file__).parents[1] / "datasets" / "VOCdevkit/VOC2012"
cached_data = {}


class VOC(BaseDataset):
    """ Pascal VOC 2012 Data Loader:
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html

    Data is derived from Pascal VOC 2012, and can be downloaded from here:
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    """

    class_names = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "potted plant", "sheep", "sofa", "train", "tv/monitor"
    ]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, opt, logger, split='train', augmentations=None, cache=False):
        super(VOC, self).__init__(opt)

        self.opt = opt
        self.logger = logger
        self.cache = cache
        self.root = DATAROOT
        self.split = split
        assert self.split in ["train", "val", "trainaug"]
        self.augmentations = augmentations
        self.n_classes = opt.n_class
        self.files = {}

        self.images_base = self.root / "JPEGImages"
        self.split_file = self.root / "ImageSets/Segmentation" / f"{self.split}.txt"
        if 'aug' in self.split:
            self.annotations_base = self.root / "SegmentationClassAug"
        else:
            self.annotations_base = self.root / "SegmentationClass"

        self.files = [self.images_base / f"{img_id}.jpg" for img_id in self.split_file.read_text().splitlines()]
        self.ignore_index = 255

        self.logger.info("Found %d %s images" % (len(self.files), self.split))

    def __len__(self):
        actual = len(self.files)
        if self.split == 'train' and self.opt.train_n > 0:
            return min(actual, self.opt.train_n)
        if self.split == 'trainaug' and self.opt.train_n > 0:
            return min(actual, self.opt.train_n)
        if self.split == 'val' and self.opt.val_n > 0:
            return min(actual, self.opt.val_n)
        return actual

    def get_image(self, img_path):
        if self.cache and img_path in cached_data:
            return cached_data[img_path].copy()
        else:
            img = Image.open(img_path)
            img = np.array(img, dtype=np.uint8)
            if self.cache:
                cached_data[img_path] = img.copy()
            return img

    def get_label(self, lab_path):
        if self.cache and lab_path in cached_data:
            return cached_data[lab_path].copy()
        else:
            lab = Image.open(lab_path)
            lab = np.array(lab, dtype=np.uint8)
            if self.cache:
                cached_data[lab_path] = lab.copy()
            return lab

    def __getitem__(self, index):
        img_path = self.files[index]
        lab_path = self.annotations_base / f"{img_path.stem}.png"

        img = self.get_image(img_path)
        lab = self.get_label(lab_path)
        lab_full = None
        if self.split == 'val':
            lab_full = lab.copy()

        if self.augmentations is not None:
            img, lab = self.augmentations(img, lab)

        img, lab, lab_full = self.transform(img, lab, lab_full)

        batch = {
            "img": img,
            "lab": lab,
            "img_path": str(img_path)
        }
        if lab_full is not None:
            batch["lab_full"] = lab_full
        return batch

    def transform(self, img, lab, lab_full=None):

        img = (img.astype(np.float32) / 255 - self.mean) / self.std
        img = img.transpose(2, 0, 1)    # HWC -> CHW

        img = torch.from_numpy(img).float()
        lab = torch.from_numpy(lab).long()
        if lab_full is not None:
            lab_full = torch.from_numpy(lab_full)

        return img, lab, lab_full
