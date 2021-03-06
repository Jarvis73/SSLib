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
from functools import partial

from networks.deeplabv3 import DeepLabV3
from networks.unet import UNet2D
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from networks.sync_batchnorm.replicate import DataParallelWithCallback, DataParallel
from utils.misc import find_snapshot
from utils.loggers import C as CC
from config import get_rundir
from core import solver
from core import losses as loss_kits
from core.metrics import Accumulator


class Model(object):
    def __init__(self, opt, logger, run, datasets=None, isTrain=True):
        self.opt = opt
        self.logger = logger
        self.run = run
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
            self.pretrained = find_snapshot(opt, interactive=False)
        elif opt.pretrained == "auto":
            self.pretrained = None
        else:
            self.pretrained = False
        if opt.model_name == "deeplabv3":
            self.model = DeepLabV3(opt.backbone, opt.output_stride, opt.multi_grid,
                                   self.n_class, self.pretrained, opt.freeze_bn, BatchNorm, self.logger)
        elif opt.model_name == "unet":
            self.model = UNet2D(opt.init_c, opt.base_c, self.n_class, self.pretrained, BatchNorm)
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

            self.celoss = partial(loss_kits.cross_entropy2d, ignore_index=opt.ignore_index)
            self.do_step_lr = self.opt.lrp in ["cosine", "poly"]

    def step(self, x, y):
        prob = self.model_DP(x)
        if tuple(prob.size()[-2:]) != tuple(x.size()[-2:]):
            prob = F.interpolate(prob, x.size()[-2:], mode='bilinear', align_corners=True)
        loss = self.celoss(inputs=prob, target=y)
        loss.backward()
        self.Opti.step()

        return loss.item()

    def test_step(self, x):
        _, _, h, w = x.shape
        if self.opt.tta:
            count = 0
            scales = self.opt.ms
            prob_sum = None
            count = 0
            for s in scales:
                xs = F.interpolate(x, (int(h * s), int(w * s)), mode='bilinear', align_corners=True)
                prob = self.model_DP(xs)
                if self.opt.flip:
                    borp = self.model_DP(torch.flip(xs, dims=[3]))  # prob -> borp
                    prob = (prob + torch.flip(borp, dims=[3])) / 2
                prob = F.interpolate(prob, (h, w), mode='bilinear', align_corners=True)
                if prob_sum is None:
                    prob_sum = prob
                else:
                    prob_sum = prob_sum + prob
                count += 1
            prob = prob_sum / count
        else:
            prob = self.model_DP(x)
            prob = F.interpolate(prob, (h, w), mode='bilinear', align_corners=True)
        return prob

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
                # Save model.state_dict(), not model_DP.state_dict()
                "state_dict": self.model.state_dict()
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
        val_dataset = datasets.val_loader
        acc = Accumulator(val_loss=0.)

        # Validate
        with torch.no_grad():
            tqdmm = tqdm(val_dataset, leave=False)
            for data_i in tqdmm:
                images = data_i["img"].to(device)
                labels = data_i["lab"].to(device)
                labels_np = data_i["lab"].numpy()

                prob = self.model_DP(images)
                if tuple(prob.size()[-2:]) != tuple(labels.size()[-2:]):
                    prob_up = F.interpolate(prob, labels.size()[-2:], mode='bilinear', align_corners=True)
                loss = self.celoss(inputs=prob_up, target=labels)
                acc.update(val_loss=loss)

                pred = prob_up.argmax(1).cpu().numpy()
                running_metrics.update(labels_np, pred)
        self.run.log_scalar('val_loss', float(acc.mean('val_loss')), epoch)

        # Record results
        score, class_iou = running_metrics.get_scores()
        for k, v in score.items():
            self.logger.info(f'{k}: {v}')
            self.run.log_scalar(k, float(v), epoch)
        for k, v in class_iou.items():
            self.logger.info(f'class{k}: {v:.4f}')
            self.run.log_scalar(f"class{k}", float(v), epoch)

        running_metrics.reset()

        # Take snapshot
        self.snapshot(epoch, score["Mean IoU"])
        if score["Mean IoU"] > self.best_iou:
            self.best_iou = score["Mean IoU"]
            if self.do_ckpt:
                self.save(epoch, score["Mean IoU"], self.logdir / "best.pth")

        return score["Mean IoU"]
