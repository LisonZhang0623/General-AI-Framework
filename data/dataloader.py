from torch.utils.data import DataLoader
from data.datasets import CamObjDataset
from utils.misc import load_config


def get_dataset(cfg):
    train_path = cfg["data"]["train_path"]
    valid_path = cfg["data"]["valid_path"]
    imgsize = cfg["data"]["image_size"]
    train_dataset = CamObjDataset(
        image_root=f'{train_path}/images/',
        depth_root=f'{train_path}/depthsv2_vitl/rgb/',
        gt_root=f'{train_path}/masks/',
        edge_root=f'{train_path}/edge/',
        imgsize=imgsize
    )

    val_dataset = None
    if valid_path is not None:
        val_dataset = CamObjDataset(
            image_root=f'{valid_path}/images/',
            depth_root=f'{valid_path}/depthsv2_vitl/rgb/',
            gt_root=f'{valid_path}/masks/',
            edge_root=f'{valid_path}/edge/',
            imgsize=imgsize
        )

    return train_dataset, val_dataset

def get_loader(cfg, train_set, val_set=None):
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=1,
            shuffle=False,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=True,
            drop_last=False,
        )
    return train_loader, val_loader

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    cfg = load_config('../config/train.yaml')
    datas = get_dataset(cfg)
    loader,val_loader = get_loader(cfg, *datas)
    image, depth, gt, edge = next(iter(loader))

    img = image[0]  # 取第一张
    img = img.permute(1, 2, 0)  # CHW → HWC
    img = img.cpu().numpy()  # tensor → numpy

    plt.imshow(img)
    plt.axis('off')
    plt.show()

