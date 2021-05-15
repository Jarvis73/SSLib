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
from pathlib import Path
from sacred import Experiment

from config import settings, MapConfig, set_seed, get_rundir
from utils.loggers import get_global_logger
from utils.timer import Timer
from core.trainer import Trainer
from core.metrics import IoUMetric


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
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    # Create trainer
    trainer = Trainer(opt, logger, _run, isTrain=True)
    timer = Timer()
    running_metrics = IoUMetric(opt.n_class)

    trainer.iter = 0
    start_epoch = 0
    for epoch in range(start_epoch, opt.epochs):
        # 1. Training
        tqdmm = tqdm(datasets.train_loader, leave=False)
        for data_i in tqdmm:
            images = data_i['images'].to(device)
            labels = data_i['labels'].to(device)

            trainer.iter += 1
            i = trainer.iter

            with timer.start():
                trainer.train()
                trainer.optimizer_zerograd()
                loss_ce = trainer.step(images, labels)

                tqdmm.set_description(f"[TRAIN] loss: {loss_ce:.4f} lr: {trainer.Opti.param_groups[0]['lr']:g}")
                trainer.step_lr()

        # 2. Validation
        miou = trainer.validation(datasets, device, epoch, running_metrics)
        print_str = f"Epoch [{epoch + 1}/{opt.epochs}] LR: {trainer.Opti.param_groups[0]['lr']:g} " \
                    f"Mean IoU: {miou:.4f} ({trainer.best_iou:.4f}) Speed: {timer.cps:.2f}it/s"
        logger.info(print_str)

        trainer.step_lr(finish_epoch=True)
        timer.reset()

    trainer.snapshot(opt.epochs, trainer.best_iou, final=True)
    return f"Mean IoU: {trainer.best_iou:.4f}"


@ex.command
def test(_run, _config):
    # Setup
    opt = MapConfig(_config)
    set_seed(opt.seed)
    logger = get_global_logger(name=ex.path)
    logger.info(f"RUNDIR: {get_rundir(opt, _run)}")

    # Create dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    # Create trainer
    trainer = Trainer(opt, logger, _run, isTrain=False)
    timer = Timer()
    running_metrics = IoUMetric(opt.n_class)
    trainer.eval(logger)
    torch.cuda.empty_cache()

    # Validate
    with torch.no_grad():
        tqdmm = tqdm(datasets.val_loader)
        for data_i in tqdmm:
            with timer.start():
                images = data_i["images"].to(device)
                labels = data_i["labels"]

                prob = trainer.model_DP(images)
                prob_up = F.interpolate(prob, size=images.size()[-2:], mode='bilinear', align_corners=True)
                pred = prob_up.argmax(1).cpu().numpy()
            running_metrics.update(labels, pred)

    # Record results
    score, class_iou = running_metrics.get_scores()
    for k, v in score.items():
        logger.info(f'{k}: {v}')
        _run.log_scalar(k, float(v), iters)
    for k, v in class_iou.items():
        logger.info(f'class{k}: {v:.4f}')
        _run.log_scalar(f"class{k}", float(v), iters)

    print_str = f"Mean IoU: {miou:.4f} Speed: {timer.cps:.2f}it/s"
    logger.info(print_str)

    return f"Mean IoU: {trainer.best_iou:.4f}"


if __name__ == "__main__":
    ex.run_commandline()