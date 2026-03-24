import os

import torch
import yaml
from matplotlib import pyplot as plt


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def log_to_txt(log_path, text):
    with open(log_path, "a") as f:
        f.write(text + "\n")

def plot_losses(train_losses, val_losses, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Loss curve saved to: {save_path}")

def save_checkpoint(accelerator, model, optimizer, scheduler, epoch, output_dir, max_checkpoints=None):
    os.makedirs(output_dir, exist_ok=True)

    unwrapped_model = accelerator.unwrap_model(model)
    save_path = os.path.join(output_dir, f"model_epoch_{epoch}.pth")

    if accelerator.is_local_main_process:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": unwrapped_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            save_path,
        )
        # print(f"Model saved to: {save_path}")

        # 超过最大保存数量时，删除最旧的 checkpoint
        if max_checkpoints is not None and max_checkpoints > 0:
            ckpt_files = []
            for fname in os.listdir(output_dir):
                if fname.startswith("model_epoch_") and fname.endswith(".pth"):
                    try:
                        ep = int(fname.replace("model_epoch_", "").replace(".pth", ""))
                        ckpt_files.append((ep, os.path.join(output_dir, fname)))
                    except ValueError:
                        continue

            ckpt_files.sort(key=lambda x: x[0])  # 按 epoch 从小到大排序

            while len(ckpt_files) > max_checkpoints:
                old_epoch, old_path = ckpt_files.pop(0)
                if os.path.exists(old_path):
                    os.remove(old_path)
                    # print(f"Removed old checkpoint: {old_path}")