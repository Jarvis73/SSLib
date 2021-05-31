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

from tqdm import tqdm
import torch
import torch.nn.functional as F
from pathlib import Path
from sacred import Experiment

from config import settings, MapConfig, set_seed, get_rundir
from utils.loggers import get_global_logger
from utils.timer import Timer
from core.model import Model
from core.metrics import IoUMetric
from data_kits import CustomDatasetDataLoader


ex = Experiment("SSLib", base_dir=Path(__file__).parent, save_git_info=False)
settings(ex)


@ex.command
def train(_run, _config):
    # Setup
    opt = MapConfig(_config)
    set_seed(opt.seed)
    logger = get_global_logger(name=ex.path)
    logger.info(f"RUNDIR: {get_rundir(opt, _run)}")

    # Create dataset
    datasets = CustomDatasetDataLoader(opt, logger)
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    # Create model
    model = Model(opt, logger, _run, datasets, isTrain=True)
    timer = Timer()
    running_metrics = IoUMetric(opt.n_class)

    start_epoch = 0
    for epoch in range(start_epoch, opt.epochs):
        # 1. Training
        tqdmm = tqdm(datasets.train_loader, leave=False)
        for data_i in tqdmm:
            images = data_i['img'].to(device)
            labels = data_i['lab'].to(device)

            with timer.start():
                model.train()
                model.optimizer_zerograd()
                loss_ce = model.step(images, labels)

                tqdmm.set_description(f"[TRAIN] loss: {loss_ce:.4f} lr: {model.Opti.param_groups[0]['lr']:g}")
                model.step_lr()

        # 2. Validation
        miou = model.validation(datasets, device, epoch + 1, running_metrics)
        print_str = f"Epoch [{epoch + 1}/{opt.epochs}] LR: {model.Opti.param_groups[0]['lr']:g} " \
                    f"Mean IoU: {miou:.4f} ({model.best_iou:.4f}) Speed: {timer.cps:.2f}it/s"
        logger.info(print_str)

        model.step_lr(epoch_end=True)
        timer.reset()

    model.snapshot(opt.epochs, model.best_iou, final=True)
    return f"Mean IoU: {model.best_iou:.4f}"


@ex.command
def test(_run, _config):
    # Setup
    opt = MapConfig(_config)
    set_seed(opt.seed)
    logger = get_global_logger(name=ex.path)
    logger.info(f"RUNDIR: {get_rundir(opt, _run)}")

    # Create dataset
    # Annotations of split 'test' are not available. Test is only performed on valset.
    datasets = CustomDatasetDataLoader(opt, logger, splits=('val',))
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    # Create model
    model = Model(opt, logger, _run, isTrain=False)
    timer = Timer()
    running_metrics = IoUMetric(opt.n_class)
    model.eval(logger)

    # Validate
    with torch.no_grad():
        tqdmm = tqdm(datasets.val_loader, leave=False)
        for data_i in tqdmm:
            with timer.start():
                images = data_i["img"].to(device)
                labels = data_i["lab"].numpy()

                prob = model.model_DP(images)
                prob_up = F.interpolate(prob, size=images.size()[-2:], mode='bilinear', align_corners=True)
                pred = prob_up.argmax(1).cpu().numpy()
            running_metrics.update(labels, pred)
            score, class_iou = running_metrics.get_scores()

    # Record results
    score, class_iou = running_metrics.get_scores()
    for k, v in score.items():
        logger.info(f'{k}: {v}')
        _run.log_scalar(k, float(v), 0)
    for k, v in class_iou.items():
        logger.info(f'class{k}: {v:.4f}')
        _run.log_scalar(f"class{k}", float(v), 0)

    print_str = f"Mean IoU: {score['Mean IoU']:.4f} Speed: {timer.cps:.2f}it/s"
    logger.info(print_str)

    return f"Mean IoU: {score['Mean IoU']:.4f}"


if __name__ == "__main__":
    ex.run_commandline()
