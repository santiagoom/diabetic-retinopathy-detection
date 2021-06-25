from __future__ import print_function, division
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import datasets, transforms
import os
import sys

from PIL import Image
import cv2


class DRDDataset(Dataset):

    def __init__(self, data_dir, csv_file, train=True, test_size=0.2, transform=None):
        csv_file = os.path.join(data_dir, csv_file)

        self.label_frame = pd.read_csv(csv_file)[:]

        self.images_array = []

        divide = int(round(len(self.label_frame) * (1 - test_size)))

        if train:
            self.label_frame = self.label_frame[:divide]
        else:
            self.label_frame = self.label_frame[divide:]

        print(self.label_frame)
        self.data_dir = data_dir
        self.transform = transform

        # for i in range(len(self.label_frame)):
        #     img_name = os.path.join(self.data_dir, "train", "{}{}".format(self.label_frame.iloc[i, 0], ".jpeg"))
        #     # print(i, img_name)
        #     image = Image.open(img_name)
        #     label = self.label_frame.iloc[i, 1]
        #     label = np.array(label)
        #     label = label
        #     sample = {'image_origin': image, 'label': label}
        #
        #     self.images_array.append(sample)

    def __len__(self):
        return len(self.label_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.data_dir, "train", "{}{}".format(self.label_frame.iloc[idx, 0], ".jpeg"))
        image = Image.open(img_name)
        # # image = np.asarray(image)
        label = self.label_frame.iloc[idx, 1]
        label = np.array(label)
        label = label

        if self.transform:
            # print(type(sample["image"]))
            # sample["image"] = self.transform(sample["image_origin"])
            image = self.transform(image)

        sample = {'image': image, 'label': label}

        return sample


def data_loder():
    data_dir = "/root/volume/data/images_eyepacs"
    csv_file = "trainLabels_part.csv"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize, ])

    drd_dataset = DRDDataset(data_dir, csv_file, train=False, transform=transform)

    for i in range(len(drd_dataset)):
        sample = drd_dataset[i]
        print(i, sample['image'].shape, sample['label'])

    train_loader = torch.utils.data.DataLoader(drd_dataset, )

    for batch, data in enumerate(train_loader):
        print(data)


def run():
    if (len(sys.argv) != 1):
        print("args error ...")
        sys.exit(0)

    data_loder()


if __name__ == "__main__":
    run()
