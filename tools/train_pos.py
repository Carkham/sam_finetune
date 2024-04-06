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
from dataset import cityscapes
from torch import distributed as dist
from nets import sam
from nets.hrnet import seg_hrnet
from nets.hrnet.config import config, update_config
import torch
import os
import argparse
import pylab
from torch.cuda.amp.autocast_mode import autocast

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
    log_file = open(os.path.join(root, "logs/log.txt"), "w")
    set_file(log_file, rank=local_rank)
    to_log(args)
    args_data = args["data"]
    args_train = args["train"]

    train_loader = DataLoader(cityscapes.csdata(
        args_data["root"], split="train"), batch_size=args_data["bs"], num_workers=args_data["num_workers"])

    valid_loader = DataLoader(cityscapes.csdata(
        args_data["root"], split="valid"), 1, num_workers=1)

    sam_model = sam.get_model().cuda()
    update_config(config, path="nets/hrnet/config/w48.yaml")
    hrnet = seg_hrnet.get_seg_model(config)
    hrnet.init_weights("pretrain/hrnetv2_w48_imagenet_pretrained.pth")
    hrnet = hrnet.cuda()

    opt = optim.AdamW(
        sam_model.mask_decoder.parameters(), lr=args_train['lr'])
    # sch = optim.lr_scheduler.PolynomialLR(
    #     opt, args_train['total_iters'], power=args_train["power"])
    sch = optim.lr_scheduler.CosineAnnealingLR(
        opt, args_train['total_iters'], eta_min=args_train['lr'] / 10)
    opt_pos = optim.SGD(hrnet.parameters(), lr=args_train['lr_pos'])
    # sch_pos = optim.lr_scheduler.PolynomialLR(
    #     opt_pos, args_train['total_iters'], power=args_train["power"])
    sch_pos = optim.lr_scheduler.CosineAnnealingLR(
        opt_pos, args_train['total_iters'], eta_min=args_train['lr_pos'] / 10)

    writer = SummaryWriter(os.path.join(root, "logs/result/event/"))

    tot_iter = 0
    while True:
        if tot_iter == args_train["total_iters"]:
            break
        for img_org, img_gtf, points, labels, image_path in train_loader:
            if tot_iter == args_train["total_iters"]:
                break
            sam_model.train()
            hrnet.train()
            tot_iter += 1
            img_input = nn.functional.interpolate(
                img_org, [1024, 1024], mode="bilinear").cuda()
            img_gtf, points, labels = img_gtf.cuda(), points.cuda(), labels.cuda()

            with autocast():
                with torch.no_grad():
                    image_embedding = sam_model.image_encoder(img_input)
                    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                        points=(points, labels),
                        boxes=None,
                        masks=None,
                    )
                # print(sparse_embeddings.shape, dense_embeddings.shape)
                # print(image_embedding.shape, img_gtf.shape, points.shape, labels.shape)
                low_res_masks, iou_predictions = sam_model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                upscaled_masks = sam_model.postprocess_masks(
                    low_res_masks, [512, 1024], [1024, 2048]).cuda()

                from torch.nn.functional import threshold, normalize

                binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))
                # upscaled_masks = nn.Sigmoid()(upscaled_masks)
                loss = nn.BCEWithLogitsLoss()(upscaled_masks, img_gtf.float())

                pos_input = torch.cat(
                    [img_org[:, :, :1024, :].cuda(), binary_mask], 1)
                pos_input = nn.functional.interpolate(
                    pos_input, [512, 1024], mode="bilinear").cuda()
                pos_mask = hrnet(pos_input)
                pos_mask = nn.functional.interpolate(
                    pos_mask, [1024, 2048], mode="bilinear").cuda()
                
                loss_pos = nn.CrossEntropyLoss()(pos_mask, img_gtf.squeeze(1))

                tot_loss = loss + loss_pos

            opt.zero_grad()
            opt_pos.zero_grad()
            tot_loss.backward()
            opt.step()
            opt_pos.step()
            sch.step()
            sch_pos.step()

            # from torchvision.transforms import transforms
            # from PIL import ImageDraw
            # a = transforms.ToPILImage()(binary_mask.squeeze(0))
            # a.save("p.png")
            # b = transforms.ToPILImage()(img_gtf.squeeze(0).float())
            # b.save("gt.png")
            # bb = ImageDraw.Draw(b)
            # bb.point((int(points[0][0][0]), int(points[0][0][1])), fill="red")
            # b.save("gt_.png")
            # c = transforms.ToPILImage()(img_org.squeeze(0).float())
            # c.save("org.png")
            # print(points)
            # exit(0)


            if tot_iter % args_train['show_interval'] == 0 and local_rank == 0:
                to_log("iter: {}, epoch: {}, batch: {}/{}, loss: {}, loss_pos: {}".format(
                    tot_iter, tot_iter // len(
                        train_loader), tot_iter % len(train_loader),
                    len(train_loader), loss, loss_pos))
                writer.add_scalar("train/loss", loss, tot_iter)
                writer.add_scalar("train/loss_pos", loss_pos, tot_iter)

                writer.add_scalar(
                    "train/lr", opt.param_groups[0]['lr'], tot_iter)
            if tot_iter % args_train['snapshot_interval'] == 0:
                torch.save(sam_model.state_dict(), os.path.join(
                    root, "logs", "model-iter-{}.pth".format(tot_iter)))

            if tot_iter % args_train['valid_interval'] == 0 and local_rank == 0:
                with torch.no_grad():
                    IoUs = 0.
                    Rs = 0.
                    losses = 0.
                    loss_poses = 0.
                    IoUs_pos = 0.
                    Rs_pos = 0.
                    for img_org, img_gtf, points, labels, image_path in tqdm(valid_loader):
                        sam_model.eval()
                        hrnet.eval()
                        img_input = nn.functional.interpolate(
                            img_org, [1024, 1024], mode="bilinear").cuda()
                        img_gtf, points, labels = img_gtf.cuda(), points.cuda(), labels.cuda()

                        with torch.no_grad():
                            image_embedding = sam_model.image_encoder(
                                img_input)
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
                            low_res_masks, [512, 1024], [1024, 2048]).cuda()

                        from torch.nn.functional import threshold, normalize

                        binary_mask = normalize(
                            threshold(upscaled_masks, 0.0, 0))
                        upscaled_masks = nn.Sigmoid()(upscaled_masks)

                        losses += nn.BCELoss()(upscaled_masks, img_gtf.float())

                        pos_input = torch.cat(
                            [img_org[:, :, :1024, :].cuda(), binary_mask], 1)
                        pos_input = nn.functional.interpolate(
                            pos_input, [512, 1024], mode="bilinear").cuda()
                        pos_mask = hrnet(pos_input)
                        pos_mask = nn.functional.interpolate(
                            pos_mask, [1024, 2048], mode="bilinear").cuda()
                        loss_poses += nn.CrossEntropyLoss()(pos_mask, img_gtf.squeeze(1))
                        binary_mask_pos = pos_mask.argmax(1)

                        # tot_loss = loss + loss_pos

                        I = (binary_mask == 1) & (img_gtf == 1)
                        U = (binary_mask == 1) | (img_gtf == 1)
                        IoU = I.sum() / (U.sum() + 1e-8)
                        R = I.sum() / ((img_gtf == 1).sum() + 1e-8)
                        IoUs += IoU
                        Rs += R

                        I_pos = (binary_mask_pos == 1) & (img_gtf == 1)
                        U_pos = (binary_mask_pos == 1) | (img_gtf == 1)
                        IoU_pos = I_pos.sum() / (U_pos.sum() + 1e-8)
                        R_pos = I_pos.sum() / ((img_gtf == 1).sum() + 1e-8)
                        IoUs_pos += IoU_pos
                        Rs_pos += R_pos

                IoU = IoUs / len(valid_loader)
                R = Rs / len(valid_loader)
                IoU_pos = IoUs_pos / len(valid_loader)
                R_pos = Rs_pos / len(valid_loader)
                loss = losses / len(valid_loader)
                loss_pos = loss_poses / len(valid_loader)
                to_log("valid: \nIoU: {}\nRecall: {}".format(IoU, R))
                writer.add_scalar("valid/loss", loss, tot_iter)
                writer.add_scalar("valid/loss_pos", loss_pos, tot_iter)
                writer.add_scalar("valid/IoU", IoU, tot_iter)
                writer.add_scalar("valid/R", R, tot_iter)
                writer.add_scalar("valid/IoU_pos", IoU_pos, tot_iter)
                writer.add_scalar("valid/R_pos", R_pos, tot_iter)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    args = parser.parse_args()
    train(open_config(args.root), args.root)
