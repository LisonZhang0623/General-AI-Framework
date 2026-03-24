import os
import math
import yaml
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from accelerate import Accelerator
from accelerate.utils import set_seed

from data.dataloader import get_loader, get_dataset
from models.model import EasyModel
from utils.misc import load_config, plot_losses, log_to_txt, save_checkpoint
from utils.loss import criterion
from utils.optimizer import build_optimizer, build_scheduler


def build_model(cfg):
    name = cfg["model"]["name"]
    model = EasyModel()
    # if name == "resnet18":
    #     model = models.resnet18(num_classes=num_classes)
    # elif name == "resnet50":
    #     model = models.resnet50(num_classes=num_classes)
    # else:
    #     raise ValueError(f"Unsupported model: {name}")
    return model


@torch.no_grad()
def validate(accelerator, model, val_loader, criterion):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    # 如果你的任务是分类，这里可以保留 accuracy
    # 如果是分割任务，argmax/accuracy 往往不合适，建议换成 dice / IoU
    total_correct = 0

    progress_bar = tqdm(
        val_loader,
        disable=not accelerator.is_local_main_process,
        desc="Validating",
        leave=False
    )

    for batch in progress_bar:
        # 兼容你的数据格式
        if len(batch) == 4:
            images, depth, gt, edge = batch
            labels = gt
        elif len(batch) == 2:
            images, labels = batch
        else:
            raise ValueError("Unexpected batch format in val_loader")

        outputs = model(images)
        loss = criterion(outputs, labels)

        # 如果是分类任务
        if outputs.dim() == 2:
            preds = outputs.argmax(dim=1)
            gathered_preds = accelerator.gather(preds)
            gathered_labels = accelerator.gather(labels)
            total_correct += (gathered_preds == gathered_labels).sum().item()
            total_samples += gathered_labels.numel()
        else:
            # 分割任务时，这里的 accuracy 仅占位，不建议作为核心指标
            gathered_labels = accelerator.gather(labels)
            total_samples += gathered_labels.numel()

        gathered_loss = accelerator.gather(loss.detach().unsqueeze(0))
        total_loss += gathered_loss.mean().item()

        progress_bar.set_postfix(val_loss=f"{total_loss / max(1, (progress_bar.n + 1)):.4f}")

    avg_loss = total_loss / len(val_loader)

    if total_correct > 0:
        acc = total_correct / total_samples
    else:
        acc = 0.0

    return avg_loss, acc


def main():
    cfg = load_config("config/train.yaml")
    print(cfg)

    accelerator = Accelerator(
        mixed_precision=cfg["train"]["mixed_precision"],
        gradient_accumulation_steps=cfg["train"]["grad_accum_steps"],
        log_with=None,
        project_dir=cfg["output_dir"],
    )

    log_file = os.path.join(cfg["output_dir"], "train_log.txt")

    # 只在主进程创建文件并写表头
    if accelerator.is_local_main_process:
        os.makedirs(cfg["output_dir"], exist_ok=True)
        with open(log_file, "w") as f:
            f.write("Epoch | Train Loss | Val Loss | Val Acc\n")

    set_seed(cfg["seed"])

    model = build_model(cfg)
    train_dataset, val_dataset = get_dataset(cfg)
    train_loader, val_loader = get_loader(cfg, train_dataset, val_dataset)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    train_losses = []
    val_losses = []

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch [{epoch}/{cfg['train']['epochs']}]",
            leave=True
        )

        for step, (images, depth, gt, edge) in enumerate(progress_bar, start=1):
            with accelerator.accumulate(model):
                outputs = model(images)
                loss = criterion(outputs, gt)

                accelerator.backward(loss)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            avg_train_loss = running_loss / step

            progress_bar.set_postfix(train_loss=f"{avg_train_loss:.4f}")

            # if step % cfg["train"]["log_interval"] == 0:
            #     accelerator.print(
            #         f"[Epoch {epoch}/{cfg['train']['epochs']}] "
            #         f"step={step}/{len(train_loader)} "
            #         f"loss={avg_train_loss:.4f}"
            #     )

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        val_loss, val_acc = validate(accelerator, model, val_loader, criterion)
        val_losses.append(val_loss)

        if accelerator.is_local_main_process:
            log_str = (
                f"Epoch {epoch} | "
                f"train_loss={epoch_train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f}"
            )
            log_to_txt(log_file, log_str)

        if epoch % cfg["save"]["save_every_epoch"] == 0:
            save_checkpoint(
                accelerator,
                model,
                optimizer,
                scheduler,
                epoch,
                cfg["output_dir"],
                cfg["save"].get("max_checkpoints", None),
            )

    if accelerator.is_local_main_process:
        plot_losses(train_losses, val_losses, cfg["output_dir"])


if __name__ == "__main__":
    main()