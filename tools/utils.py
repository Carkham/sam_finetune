import random
import numpy as np
import torch
from torch.backends import cudnn
import os
import yaml
from .to_log import to_log


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False


def open_config(root):
    f = open(os.path.join(root, "config.yaml"))
    config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def load_ckpt(models, epoch, root):

    def _detect_latest():
        checkpoints = os.listdir(os.path.join(root, "logs"))
        checkpoints = [
            f for f in checkpoints
            if f.startswith("model_epoch_") and f.endswith(".pth")
        ]
        checkpoints = [
            int(f[len("model_epoch_"):-len(".pth")]) for f in checkpoints
        ]
        checkpoints = sorted(checkpoints)
        _epoch = checkpoints[-1] if len(checkpoints) > 0 else None
        return _epoch

    if epoch == -1:
        epoch = _detect_latest()
    if epoch is None:
        return -1
    for name, model in models.items():
        pth_path = os.path.join(root,
                                "logs/" + name + "_epoch_{}.pth".format(epoch))
        if not os.path.exists(pth_path):
            print("can't find pth file: {}".format(name))
            continue
        ckpt = torch.load(pth_path, map_location="cpu")
        # ckpt = {k: v for k, v in ckpt.items()}
        model.load_state_dict(ckpt)
        to_log("load model: {} from iter: {}".format(name, epoch))
    return epoch
