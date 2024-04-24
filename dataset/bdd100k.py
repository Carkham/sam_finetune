"""
BDD100K Dataset Loader
"""
import logging
import json
import os
import random

import numpy as np
from PIL import Image
# from skimage import color

from torch.utils import data
import torch
import torchvision.transforms as transforms
import dataset.cityscapes_labels as cityscapes_labels

trainid_to_name = cityscapes_labels.trainId2name
id_to_trainid = cityscapes_labels.label2trainid
trainid_to_trainid = cityscapes_labels.trainId2trainId
color_to_trainid = cityscapes_labels.color2trainId
num_classes = 19
ignore_label = 255
groot = '/home/r22user1/fedavgmodels/sam_finetune/dataset/bdd100k'
img_postfix = '.jpg'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
           153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    """
    Colorize a segmentation mask.
    """
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def add_items(items, aug_items, img_path, mask_path, mask_postfix, mode):
    """

    Add More items ot the list from the augmented dataset
    """

    if mode == "train":
        img_path = os.path.join(img_path, 'train')
        mask_path = os.path.join(mask_path, 'train')
    elif mode == "val":
        img_path = os.path.join(img_path, 'val')
        mask_path = os.path.join(mask_path, 'val')

    list_items = [name.split(img_postfix)[0] for name in
                  os.listdir(img_path)]
    for it in list_items:
        item = (os.path.join(img_path, it + img_postfix),
                os.path.join(mask_path, it + mask_postfix))
        items.append(item)


def make_cv_splits(img_dir_name):
    """
    Create splits of train/val data.
    A split is a lists of cities.
    split0 is aligned with the default Cityscapes train/val.
    """
    trn_path = os.path.join(groot, img_dir_name, 'train')
    val_path = os.path.join(groot, img_dir_name, 'val')

    trn_cities = ['train/' + c for c in os.listdir(trn_path)]
    val_cities = ['val/' + c for c in os.listdir(val_path)]

    # want reproducible randomly shuffled
    trn_cities = sorted(trn_cities)

    all_cities = val_cities + trn_cities
    num_val_cities = len(val_cities)
    num_cities = len(all_cities)

    cv_splits = []
    for split_idx in range(cfg.DATASET.CV_SPLITS):
        split = {}
        split['train'] = []
        split['val'] = []
        offset = split_idx * num_cities // cfg.DATASET.CV_SPLITS
        for j in range(num_cities):
            if j >= offset and j < (offset + num_val_cities):
                split['val'].append(all_cities[j])
            else:
                split['train'].append(all_cities[j])
        cv_splits.append(split)

    return cv_splits


def make_split_coarse(img_path):
    """
    Create a train/val split for coarse
    return: city split in train
    """
    all_cities = os.listdir(img_path)
    all_cities = sorted(all_cities)  # needs to always be the same
    val_cities = []  # Can manually set cities to not be included into train split

    split = {}
    split['val'] = val_cities
    split['train'] = [c for c in all_cities if c not in val_cities]
    return split


def make_test_split(img_dir_name):
    test_path = os.path.join(groot, img_dir_name, 'leftImg8bit', 'test')
    test_cities = ['test/' + c for c in os.listdir(test_path)]

    return test_cities


def make_dataset(mode):
    """
    Assemble list of images + mask files

    fine -   modes: train/val/test/trainval    cv:0,1,2
    coarse - modes: train/val                  cv:na

    path examples:
    leftImg8bit_trainextra/leftImg8bit/train_extra/augsburg
    gtCoarse/gtCoarse/train_extra/augsburg
    """
    items = []
    aug_items = []

    assert mode in ['train', 'val', 'test', 'trainval']
    img_dir_name = 'images'
    img_path = os.path.join(groot, img_dir_name)
    mask_path = os.path.join(groot, 'labels', 'masks')
    mask_postfix = '.png'
    # cv_splits = make_cv_splits(img_dir_name)
    if mode == 'trainval':
        modes = ['train', 'val']
    else:
        modes = [mode]
    for mode in modes:
        logging.info('{} fine cities: '.format(mode))
        add_items(items, aug_items, img_path, mask_path, mask_postfix, mode)

    # logging.info('Cityscapes-{}: {} images'.format(mode, len(items)))
    logging.info('BDD100K-{}: {} images'.format(mode, len(items) + len(aug_items)))
    return items, aug_items


class BDD100K(data.Dataset):

    def __init__(self, root='../data/cityscapes_video', **kwargs):
        self.root = root
        groot = root
        self.mode = kwargs.get('split')
        self.n_pos = 10
        self.n_neg = 10

        self.imgs, _ = make_dataset(self.mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.trans_img = transforms.Compose([
            transforms.Pad(padding=[0, 0, 0, 560]),
            transforms.ToTensor(),
            transforms.Normalize(*self.mean_std)
        ])

    def get_point(self, gtf, n_pos, n_neg):
        points = []
        labels = []
        for i in range(n_pos):
            for j in range(20):
                xh = random.randint(0, 575)
                yw = random.randint(0, 1023)
                if gtf[0, int(xh * 1.25 + 1), int(yw * 1.25 + 1)] == 1:
                    points.append([yw, xh])
                    labels.append(1)
                    break

        for i in range(n_neg):
            for j in range(20):
                xh = random.randint(0, 575)
                yw = random.randint(0, 1023)
                if gtf[0, int(xh * 1.25 + 1), int(yw * 1.25 + 1)] == 0:
                    points.append([yw, xh])
                    labels.append(-1)
                    break
        for i in range(20 - len(points)):
            points.append([0, 0])
            labels.append(-1)
        return points, labels

    def __getitem__(self, index):

        img_path, mask_path = self.imgs[index]

        img, mask = self.trans_img(Image.open(img_path).convert('RGB')), Image.open(mask_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        img_gtf = transforms.ToTensor()(mask)
        img_gtf = (img_gtf * 255).round().long()
        gtf = torch.zeros_like(img_gtf)
        gtf[img_gtf == 0] = 1
        img_gtf = gtf

        # mask = np.array(mask, dtype=np.uint8)
        # mask_copy = mask.copy()
        # for k, v in trainid_to_trainid.items():
        #     mask_copy[mask == k] = v

        # mask_copy = torch.from_numpy(mask_copy).long()
        # gtf = torch.zeros_like(mask_copy)
        # gtf[mask_copy == 0] = 1
        # img_gtf = gtf

        n_pos = random.randint(3, self.n_pos)
        n_neg = 0 if self.n_neg == 0 else self.n_neg
        points, labels = self.get_point(img_gtf, n_pos, n_neg)

        return img, img_gtf, torch.tensor(points), torch.tensor(labels), img_path.split('/')[-1]

    def __len__(self):
        return len(self.imgs)
