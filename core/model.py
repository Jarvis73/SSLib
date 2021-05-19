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

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from networks.deeplabv3 import DeepLabV3
from networks.unet import UNet2D
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from networks.sync_batchnorm.replicate import DataParallelWithCallback, DataParallel
from utils.misc import find_snapshot
from utils.loggers import C as CC
from config import get_rundir
from core import solver
from core import losses as loss_kits


class Model(object):
    def __init__(self, opt, logger, run, datasets, isTrain=True):
        self.opt = opt
        self.logger = logger
        self.run = run
        self.datasets = datasets
        self.isTrain = isTrain

        self.n_class = opt.n_class
        self.best_iou = -1

        # Define logdir for saving checkpoints
        self.do_ckpt = False if self.run._id is None else True
        self.logdir = Path(get_rundir(opt, run))
        if self.isTrain:
            self.logdir.mkdir(parents=True, exist_ok=True)

        # Define model
        if opt.bn == 'sync_bn':
            BatchNorm = SynchronizedBatchNorm2d
        elif opt.bn == 'bn':
            BatchNorm = nn.BatchNorm2d
        else:
            raise NotImplementedError(f'batch norm choice {opt.bn} is not implemented')

        if opt.ckpt_id >= 0 or opt.ckpt:
            pretrained = find_snapshot(opt, interactive=False)
        elif opt.pretrained == "auto":
            pretrained = None
        else:
            pretrained = False
        if opt.model_name == "deeplabv3":
            self.model = DeepLabV3(opt.backbone, opt.output_stride, opt.multi_grid,
                                   self.n_class, pretrained, opt.freeze_bn, BatchNorm)
        elif opt.model_name == "unet":
            self.model = UNet2D(opt.init_c, opt.base_c, self.n_class, pretrained, BatchNorm)
        else:
            raise NotImplementedError(f"`{opt.model_name}` is not implemented. [deeplabv3|unet]")
        logger.info(f'The backbone is {self.model.__class__.__name__} ({opt.backbone})')
        self.model_DP = self.init_device(self.model, whether_DP=True)

        if self.isTrain:
            # Define optimizer and scheduler
            self.optimizers = []
            self.schedulers = []

            max_iters = opt.epochs * (len(datasets.train_dataset) // opt.bs)
            self.Opti, self.BaseSchedule = solver.get(self.model, opt, max_iters)
            self.optimizers.append(self.Opti)
            self.schedulers.append(self.BaseSchedule)

            self.celoss = loss_kits.cross_entropy2d
            self.do_step_lr = self.opt.lrp in ["cosine", "poly"]

    def step(self, x, y):
        prob = self.model(x)
        if tuple(prob.size()[-2:]) != tuple(x.size()[-2:]):
            prob = F.interpolate(prob, x.size()[-2:], mode='bilinear', align_corners=True)
        loss = self.celoss(inputs=prob, target=y)
        loss.backward()
        self.Opti.step()

        return loss.item()

    def init_device(self, net, whether_DP=False):
        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        net = net.to(device)
        if whether_DP:
            if self.opt.bn == "sync_bn":
                net = DataParallelWithCallback(net, device_ids=range(torch.cuda.device_count()))
            else:
                net = DataParallel(net, device_ids=range(torch.cuda.device_count()))
        return net

    def train(self):
        self.model.train()
        self.model_DP.train()

    def eval(self, logger=None):
        """Make specific models eval mode during test time"""
        self.model.eval()
        self.model_DP.eval()
        logger.info("Successfully set the model eval mode")

    def optimizer_zerograd(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step_lr(self, epoch_end=False):
        """
        Update learning rate by the specified learning rate policy.
        For 'cosine' and 'poly' policies, the learning rate is updated by steps.
        For other policies, the learning rate is updated by epochs.
        """
        if (not epoch_end and self.do_step_lr) \
                or (epoch_end and not self.do_step_lr):  # forward per step or per epoch
            for sche in self.schedulers:
                sche.step()

    def save(self, epoch, epoch_iou, save_path):
        state = {
            self.model.__class__.__name__: {
                "model_state": self.model.state_dict()
            },
            "epoch": epoch,
            "epoch_iou": epoch_iou,
            "best_iou": self.best_iou
        }

        torch.save(state, str(save_path))

    def snapshot(self, epoch=-1, epoch_iou=0., final=False, verbose=0):
        if final:
            # Save a checkpoint at the end of training.
            # noinspection PyProtectedMember
            if self.run._id is None:
                postfix = time.strftime("%y%m%d-%H%M%S", time.localtime())
                save_path = self.logdir / f"ckpt-{postfix}.pth"
                verbose = 1
            else:
                save_path = self.logdir / "ckpt.pth"
        elif self.do_ckpt:  # Save a checkpoint by the checkpoint interval
            save_path = self.logdir / "ckpt.pth"
        else:
            save_path = None

        if save_path:
            self.save(epoch, epoch_iou, save_path)
            if verbose:
                self.logger.info(CC.c(f" \\_/ Save checkpoint to {save_path}", CC.OKGREEN))
            return save_path

    def validation(self, datasets, device, epoch, running_metrics):
        self.eval(self.logger)
        torch.cuda.empty_cache()
        val_dataset = datasets.val_loader

        # Validate
        with torch.no_grad():
            tqdmm = tqdm(val_dataset, leave=False)
            for data_i in tqdmm:
                images = data_i["img"].to(device)
                labels = data_i["lab"].numpy()

                prob = self.model_DP(images)
                prob_up = F.interpolate(prob, size=images.size()[-2:], mode='bilinear', align_corners=True)
                pred = prob_up.argmax(1).cpu().numpy()
                running_metrics.update(labels, pred)

        # Record results
        score, class_iou = running_metrics.get_scores()
        for k, v in score.items():
            self.logger.info(f'{k}: {v}')
            self.run.log_scalar(k, float(v), epoch)
        for k, v in class_iou.items():
            self.logger.info(f'class{k}: {v:.4f}')
            self.run.log_scalar(f"class{k}", float(v), epoch)

        running_metrics.reset()
        torch.cuda.empty_cache()

        # Take snapshot
        self.snapshot(epoch, score["Mean IoU"])
        if score["Mean IoU"] > self.best_iou:
            self.best_iou = score["Mean IoU"]
            if self.do_ckpt:
                self.save(epoch, score["Mean IoU"], self.logdir / "best.pth")

        return score["Mean IoU"]
