import sys

sys.path.append(".")
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from tools.utils import init_seeds, open_config, get_parameter_number, load_ckpt
from tools.to_log import to_log, set_file
import yaml
import numpy as np
from torch.utils.data import DataLoader
from torch.backends import cudnn
import random
from dataset import cityscapes, bdd100k
from nets import sam
import torch
import os
import argparse
import pylab

world_size = 1
local_rank = 0
device = None
log_file = None


def train(args, root):
    args_train = args['train']
    seed = args_train.get('seed', 2106346)
    # init_seeds(seed + local_rank)
    global log_file
    if local_rank == 0:
        if not os.path.exists(os.path.join(root, "logs/result/event")):
            os.makedirs(os.path.join(root, "logs/result/event"))
    args_eval = args["eval"]
    log_file = open(os.path.join(root, f"logs/{os.path.splitext(os.path.basename(args_eval['model']))[0]}_log.txt"), "w")
    set_file(log_file, rank=local_rank)
    to_log(args)
    args_data = args["data"]

    valid_loader = DataLoader(bdd100k.BDD100K(args_data["root"], split="val"), batch_size=1)

    sam_model = sam.get_model(args['train']).cuda()
    weight = torch.load(args_eval["model"], map_location="cpu")
    sam_model.load_state_dict(weight)
    to_log("load model from: {}".format(args_eval["model"]))
    for k, p in sam_model.named_parameters():
        p.requires_grad = False

    writer = SummaryWriter(os.path.join(root, "logs/result/event/"))
    with torch.no_grad():
        IoUs = 0.
        Rs = 0.
        losses = 0.
        pbar = tqdm(valid_loader)
        i = 0
        for img_org, img_gtf, points, labels, image_path in pbar:
            i += 1
            sam_model.eval()
            img_input = nn.functional.interpolate(img_org, [1024, 1024], mode="bilinear", align_corners=False, antialias=True).cuda()
            img_input = img_input.cuda()
            img_gtf, points, labels = img_gtf.cuda(), points.cuda(), labels.cuda()

            with torch.no_grad():
                image_embedding = sam_model.image_encoder(img_input)
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=(points, labels),
                    boxes=None,
                    masks=None,
                )
            low_res_masks, iou_predictions = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            upscaled_masks = sam_model.postprocess_masks(
                low_res_masks, [576, 1024], [720, 1280]).cuda()

            from torch.nn.functional import threshold, normalize

            binary_mask = normalize(
                threshold(upscaled_masks, 0.0, 0)).cuda()
            upscaled_masks = nn.Sigmoid()(upscaled_masks)

            losses += nn.BCELoss()(upscaled_masks, img_gtf.float())

            I = (binary_mask == 1) & (img_gtf == 1)
            U = (binary_mask == 1) | (img_gtf == 1)
            IoU = I.sum() / (U.sum() + 1e-8)
            R = I.sum() / ((img_gtf == 1).sum() + 1e-8)
            IoUs += IoU
            Rs += R
            pbar.set_description("mIoU: {}, mR: {}".format(IoUs / (i), Rs / (i)))
    IoU = IoUs / len(valid_loader)
    R = Rs / len(valid_loader)
    loss = losses / len(valid_loader)
    to_log("valid: \nIoU: {}\nRecall: {}".format(IoU, R))
    writer.add_scalar("valid/loss", loss, 0)
    writer.add_scalar("valid/IoU", IoU, 0)
    writer.add_scalar("valid/R", R, 0)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    torch.set_num_threads(3)
    args = parser.parse_args()
    train(open_config(args.root), args.root)
