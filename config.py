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

import random
from pathlib import Path

import numpy as np
import torch
from sacred.config.custom_containers import ReadOnlyDict
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

from utils import loggers

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "datasets"


def add_observers(ex, config, fileStorage=False, MongoDB=True, db_name="DEFAULT"):
    if fileStorage:
        observer_file = FileStorageObserver(config["logdir"])
        ex.observers.append(observer_file)

    if MongoDB:
        observer_mongo = MongoObserver(url=f"localhost:{config['mongo_port']}", db_name=db_name)
        ex.observers.append(observer_mongo)


def settings(ex):
    # Track outputs
    ex.captured_out_filter = apply_backspaces_and_linefeeds

    @ex.config
    def configurations():
        root = str(PROJECT_DIR)         # root path
        model_name = "deeplabv3"        # select model, [deeplabv3|unet]
        backbone = "resnet50"           # for deeplabv3, [resnet50|resnet101]
        name = "default"                # experiment name
        logdir = str(PROJECT_DIR / 'runs' / name)
        mongo_port = 7000               # mongodb port
        seed = 1234                     # random seed

        # training
        lr = 0.00025
        bs = 4
        bn = "bn"                       # normalization layer [sync_bn|bn]
        epochs = 120                    # total training epochs
        weight_decay = 0.0005           # weight decay coefficient

        #model
        ckpt_id = -1                    # resume model by experiment id (only resume the best model)
        ckpt = ''                       # resume model by a specific path
        pretrained = "auto"             # automatic load pretrained weights from PyTorch, [auto|none]
        if model_name == "deeplabv3":
            freeze_bn = False           # freeze batch normalization layers
        elif model_name == "unet":
            init_c = 3                  # (unet) initial channels
            base_c = 64                 # (unet) base channels

        #data
        dataset = 'cityscapes'          # select dataset [voc|coco|cityscapes|gta5|synthia]
        no_droplast = False
        noshuffle = False               # do not use shuffle
        noaug = False                   # do not use data augmentation
        train_n = 0                     # If > 0, then #samples per epoch is set to <= train_n (for debug)
        val_n = 0                       # If > 0, then #samples is set to <= val_n (for debug)

        n_class = 19                    # number of classes
        num_workers = 6

        resize = 2200                   # resize long size
        rcrop = [1024, 512]             # rondom crop size

        # solver
        lrp = "poly"                    # Learning rate policy [custom_step|period_step|plateau|cosine|poly]
        if lrp == "custom_step":
            lr_boundaries = []          # (custom_step) Use the specified lr at the given boundaries
        if lrp == "period_step":
            lr_step = 999999999         # (period_step) Decay the base learning rate at a fixed step
        if lrp in ["custom_step", "period_step", "plateau"]:
            lr_rate = 0.1               # (period_step, plateau) Learning rate decay rate
        if lrp in ["plateau", "cosine", "poly"]:
            lr_end = 0.                 # (plateau, cosine, poly) The minimal end learning rate
        if lrp == "plateau":
            lr_patience = 30            # (plateau) Learning rate patience for decay
            lr_min_delta = 1e-4         # (plateau) Minimum delta to indicate improvement
            cool_down = 0               # (plateau)
            monitor = "val_loss"        # (plateau) Quantity to be monitored [val_loss|loss]
        if lrp == "poly":
            power = 0.9                 # (poly)

        opti = "sgd"                    # Optimizer for training [sgd|adam]
        if opti == "adam":
            adam_beta1 = 0.9            # (adam) Parameter
            adam_beta2 = 0.999          # (adam) Parameter
            adam_epsilon = 1e-8         # (adam) Parameter
        if opti == "sgd":
            sgd_momentum = 0.9          # (momentum) Parameter
            sgd_nesterov = False        # (momentum) Parameter


    @ex.config_hook
    def config_hook(config, command_name, logger):
        add_observers(ex, config, db_name=ex.path)
        ex.logger = loggers.get_global_logger(name=ex.path)
        return config


class MapConfig(ReadOnlyDict):
    """
    A wrapper for dict. This wrapper allow users to access dict value by `dot` operation.
    For example, you can access `cfg["split"]` by `cfg.split`, which makes the code more clear.
    Notice that the result object is a sacred.config.custom_containers.ReadOnlyDict, which is
    a read-only dict for preserving the configuration.
    Parameters
    ----------
    obj: ReadOnlyDict
        Configuration dict.
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, obj, **kwargs):
        new_dict = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    new_dict[k] = MapConfig(v)
                else:
                    new_dict[k] = v
        else:
            raise TypeError(f"`obj` must be a dict, got {type(obj)}")
        super(MapConfig, self).__init__(new_dict, **kwargs)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_rundir(opt, _run):
    if _run._id is not None:
        return str(Path(opt.logdir) / str(_run._id))
    else:
        return str(Path(opt.logdir).parent / 'None')
