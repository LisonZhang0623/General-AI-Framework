from torch.optim import lr_scheduler
import torch.optim as optim

def build_optimizer(cfg,model):
    optimizer_name = cfg['train']['optimizer']
    optimizer_func = 'Adam'

    if optimizer_name == 'SGD':
        optimizer_func = optim.SGD
    elif optimizer_name == 'AdamW':
        optimizer_func = optim.AdamW
    elif optimizer_name == 'Adam':
        optimizer_func = optim.Adam

    optimizer = optimizer_func(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"])
    )

    return optimizer

def build_scheduler(cfg, optimizer):
    scheduler_name = cfg['train']['lr_schedule']
    params = cfg['train'].get('scheduler_params', {})
    if scheduler_name == 'StepLR':
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params.get('step_size', 30),
            gamma=params.get('gamma', 0.1)
        )

    elif scheduler_name == 'CosineAnnealingLR':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=params.get('T_max', cfg['train']['epochs']),
            eta_min=params.get('eta_min', 0)
        )

    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return lr_scheduler
