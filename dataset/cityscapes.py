import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import torch
import random
import math
import matplotlib as plt
from tqdm import tqdm
from cityscapesscripts.helpers import labels
from torchvision.transforms.functional import InterpolationMode

# seed = 2106346
# torch.manual_seed(seed)
# random.seed(seed)
# np.random.seed(seed)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class csdata(Dataset):

    def __init__(self, root='../data/cityscapes_video', **kwargs):
        super(csdata, self).__init__()
        self.split = kwargs.get('split')
        assert self.split is not None
        if self.split == "valid":
            self.split = "val"
        self.ann_path = os.path.join(root, "gtFine", self.split)
        self.img_path = os.path.join(root, "leftImg8bit", self.split)
        self.n_pos = kwargs.get("n_pos", 10)
        self.n_neg = kwargs.get("n_neg", 0)

        self.file_list = []
        for r, _, filenames in os.walk(self.ann_path):
            for filename in filenames:
                folder, a, b, _, tp = filename.split('.')[0].split('_')
                if tp != "labelTrainIds":
                    continue
                # print(folder, a, b)  # tp = "leftImg8bit" / "gtFine_labelTrainIds"
                self.file_list.append([folder, a, b])
        self.check_data()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.trans_img = transforms.Compose([transforms.Normalize(mean, std)])

        # random.seed(2333)

    def check_data(self):
        for (folder, a, b) in tqdm(self.file_list):
            org_path = os.path.join(
                self.img_path, folder,
                folder + "_" + a + "_" + b + "_leftImg8bit.png")
            # try:
            #     img_org = Image.open(org_path)
            # except:
            #     print(org_path)
            gtf_path = os.path.join(
                self.ann_path, folder,
                folder + "_" + a + "_" + b + "_gtFine_labelTrainIds.png")
            assert os.path.exists(org_path) and os.path.exists(gtf_path)
            # try:
            #     img_gtf = Image.open(gtf_path)
            # except:
            #     print(gtf_path)

    def __len__(self):
        return len(self.file_list)

    def get_point(self, gtf, n_pos, n_neg):
        points = []
        labels = []
        for i in range(n_pos):
            for j in range(20):
                xh = random.randint(0, 511)
                yw = random.randint(0, 1023)
                if gtf[0, xh * 2 + 1, yw * 2 + 1] == 1:
                    points.append([yw, xh])
                    labels.append(1)
                    break

        for i in range(n_neg):
            for j in range(20):
                xh = random.randint(0, 511)
                yw = random.randint(0, 1023)
                if gtf[0, xh * 2 + 1, yw * 2 + 1] == 0:
                    points.append([yw, xh])
                    labels.append(-1)
                    break
        for i in range(20 - len(points)):
            points.append([0, 0])
            labels.append(-1)
        return points, labels

    def __getitem__(self, id):

        transform = transforms.Compose([
            transforms.Pad(padding=[0, 0, 0, 1024]),
            transforms.ToTensor(),
        ])

        (folder, a, b) = self.file_list[id]
        org_path = os.path.join(
            self.img_path, folder,
            folder + "_" + a + "_" + b + "_leftImg8bit.png")
        img_org = Image.open(org_path)
        img_org = self.trans_img(transform(img_org))

        gtf_path = os.path.join(
            self.ann_path, folder,
            folder + "_" + a + "_" + b + "_gtFine_labelTrainIds.png")
        img_gtf = Image.open(gtf_path)
        img_gtf = transforms.ToTensor()(img_gtf)
        img_gtf = (img_gtf * 255).round().long()
        gtf = torch.zeros_like(img_gtf)
        gtf[img_gtf == 0] = 1
        img_gtf = gtf

        n_pos = random.randint(3, self.n_pos)
        n_neg = 0 if self.n_neg == 0 else self.n_neg
        points, labels = self.get_point(img_gtf, n_pos, n_neg)

        return img_org, img_gtf, torch.tensor(points), torch.tensor(labels), org_path.split('/')[-1]


if __name__ == "__main__":
    d = CSReconstruct(split="test", use_grad=False)
    # print(len(stat.folder), label_pool.count)
    print(len(d))
    print(d[0][0][0].shape)
