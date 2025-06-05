import os
import re
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from einops import rearrange
import SimpleITK as sitk
import albumentations as alb
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms, autoaugment

from config import get_args
from utils.tabular import Normalize, Normalize_test


class new_m12_dataset(Dataset):

    def __init__(self, image_path, patient_m2_idx, train):
        self.image_path = image_path
        self.patient_m2_idx = patient_m2_idx
        self.train = train
        self.img2name = pd.read_csv(r"data/newest_img2name_train.csv", encoding='utf-8', header=None)
        p_data = pd.read_csv(r"data/p_data.csv", encoding='unicode_escape', header=0)
        p_data = p_data.drop(columns=["dir","comments"])
        self.clinical_data = Normalize(p_data.fillna(p_data.mean(numeric_only=True)))

    def __len__(self):
        return len(self.image_path[0]['0']) + len(self.image_path[0]['1'])

    def __getitem__(self, idx):
        m1_image_path, m2_image_path = self.image_path

        if idx < len(m1_image_path['0']):
            m1_img_path = m1_image_path["0"][idx]

            # get patient
            patient_m1 = self.img2name.loc[self.img2name[0] == os.path.basename(m1_img_path)][1].item()

            # get clinical data
            t = self.clinical_data[self.clinical_data['Patient-AAA'] == patient_m1].values.tolist()[0][2:]
            ind = [1,4,8,9]
            tab_arr = torch.tensor([t[i] for i in ind])

            # get m1 image
            m1_image = Image.open(m1_img_path)
            if self.train:
                m1_image = process_train(m1_image)
            else:
                m1_image = process_eval(m1_image)

            # get m2 images
            weights = torch.tensor([1 / len(self.patient_m2_idx['0'][patient_m1]) for _ in range(len(self.patient_m2_idx['0'][patient_m1]))])

            resample_weights = torch.multinomial(weights, 1, replacement=True)  # sample 1 image

            m2_img_path = m2_image_path['0'][self.patient_m2_idx['0'][patient_m1][resample_weights]]
            m2_image = Image.open(m2_img_path)

            if self.train:
                m2_image = process_train(m2_image)
            else:
                m2_image = process_eval(m2_image)

            patient_m2 = self.img2name.loc[self.img2name[0] == os.path.basename(m2_img_path)][1].item()


            assert patient_m1==patient_m2  # make sure same patient
            patient_file_name = [patient_m1, patient_m1]
            images = [m1_image, m2_image]

            # get one-hot label
            label = torch.zeros(2)
            label[0] = 1.

        else:
            idx -= len(m1_image_path['0'])

            m1_img_path = m1_image_path["1"][idx]

            # get patient
            patient_m1 = self.img2name.loc[self.img2name[0] == os.path.basename(m1_img_path)][1].item()

            # get clinical data
            try:
                t = self.clinical_data[self.clinical_data['Patient-AAA'] == patient_m1].values.tolist()[0][2:]
                ind = [1,4,8,9]
                tab_arr = torch.tensor([t[i] for i in ind])
            except:
                import ipdb
                ipdb.set_trace()
                
            # get m1 image
            m1_image = Image.open(m1_img_path)
            if self.train:
                m1_image = process_train(m1_image)
            else:
                m1_image = process_eval(m1_image)

            # get m2 images
            weights = torch.tensor([1 / len(self.patient_m2_idx['1'][patient_m1]) for _ in range(len(self.patient_m2_idx['1'][patient_m1]))])
            resample_weights = torch.multinomial(weights, 1, replacement=True)  # sample 1 image
            m2_img_path = m2_image_path['1'][self.patient_m2_idx['1'][patient_m1][resample_weights]]
            m2_image = Image.open(m2_img_path)
            if self.train:
                m2_image = process_train(m2_image)
            else:
                m2_image = process_eval(m2_image)

            patient_m2 = self.img2name.loc[self.img2name[0] == os.path.basename(m2_img_path)][1].item()
            assert patient_m1==patient_m2  # make sure same patient
            patient_file_name = [patient_m1, patient_m2]

            images = [m1_image, m2_image]

            # get one-hot label
            label = torch.zeros(2)
            label[1] = 1.

        return images, label, patient_file_name, tab_arr


class new_m1_dataset(Dataset):

    def __init__(self, image_path, train):
        print("--using m1 dataset--")
        self.image_path = image_path
        self.train = train
        self.img2name = pd.read_csv(r"data/newest_img2name_train.csv", encoding='utf-8', header=None)
        p_data = pd.read_csv(r"data/p_data.csv", encoding='unicode_escape', header=0)
        p_data = p_data.drop(columns=["dir", "comments"])
        self.clinical_data = Normalize(p_data.fillna(p_data.mean(numeric_only=True)))

    def __len__(self):
        return len(self.image_path['0']) + len(self.image_path['1'])

    def __getitem__(self, idx):

        if idx < len(self.image_path['0']):
            img_path = self.image_path["0"][idx]

            # get patient
            patient = self.img2name.loc[self.img2name[0] == os.path.basename(img_path)][1].item()
            patient_file_name = [patient, patient]

            # get clinical data
            t = self.clinical_data[self.clinical_data['Patient-AAA'] == patient].values.tolist()[0][2:]
            ind = [1,4,8,9]
            tab_arr = torch.tensor([t[i] for i in ind])

            # get image
            if self.train:
                image = Image.open(img_path)
                image = process_train(image)
            else:
                image = Image.open(img_path)
                image = process_eval(image)

            # get one-hot label
            label = torch.zeros(2)
            label[0] = 1.

        else:
            idx -= len(self.image_path['0'])
            img_path = self.image_path["1"][idx]

            # get patient
            patient = self.img2name.loc[self.img2name[0] == os.path.basename(img_path)][1].item()
            patient_file_name = [patient, patient]

            # get tab
            t = self.clinical_data[self.clinical_data['Patient-AAA'] == patient].values.tolist()[0][2:]
            ind = [1,4,8,9]
            tab_arr = torch.tensor([t[i] for i in ind])

            # get image
            if self.train:
                image = Image.open(img_path)
                image = process_train(image)
            else:
                image = Image.open(img_path)
                image = process_eval(image)

            # get one-hot label
            label = torch.zeros(2)
            label[1] = 1.

        return image, label, patient_file_name, tab_arr


class new_m2_dataset(Dataset):

    def __init__(self, image_path, train):
        print("--using m2 dataset--")
        self.image_path = image_path
        self.train = train
        self.img2name = pd.read_csv(r"data/newest_img2name_train.csv", encoding='utf-8', header=None)
        p_data = pd.read_csv(r"data/p_data.csv", encoding='unicode_escape', header=0)
        p_data = p_data.drop(columns=["dir", "comments"])
        self.clinical_data = Normalize(p_data.fillna(p_data.mean(numeric_only=True)))

    def __len__(self):
        return len(self.image_path['0']) + len(self.image_path['1'])

    def __getitem__(self, idx):

        if idx < len(self.image_path['0']):

            img_path = self.image_path["0"][idx]

            # get patient
            patient = self.img2name.loc[self.img2name[0] == os.path.basename(img_path)][1].item()
            patient_file_name = [patient, patient]

            # get tab
            t = self.clinical_data[self.clinical_data['Patient-AAA'] == patient].values.tolist()[0][2:]
            ind = [1,4,8,9]
            tab_arr = torch.tensor([t[i] for i in ind])

            # get image
            if self.train:
                image = Image.open(img_path)
                image = process_train(image)
            else:
                image = Image.open(img_path)
                image = process_eval(image)

            # get one-hot label
            label = torch.zeros(2)
            label[0] = 1.

        else:
            idx -= len(self.image_path['0'])
            img_path = self.image_path["1"][idx]

            # get patient
            patient = self.img2name.loc[self.img2name[0] == os.path.basename(img_path)][1].item()
            patient_file_name = [patient, patient]

            # get tab
            t = self.clinical_data[self.clinical_data['Patient-AAA'] == patient].values.tolist()[0][2:]
            ind = [1,4,8,9]
            tab_arr = torch.tensor([t[i] for i in ind])

            # get image
            if self.train:
                image = Image.open(img_path)
                image = process_train(image)
            else:
                image = Image.open(img_path)
                image = process_eval(image)

            # get one-hot label
            label = torch.zeros(2)
            label[1] = 1.

        return image, label, patient_file_name, tab_arr


def get_resample_image(modal_path, num_sample, train):
    """
    Get N resampled image and concat by channel C, namely, n c h w -> (n c) h w
    args:
        modal_path: path of patient's modality dirctory that used for get images;
        num_samples: random resample N images;
        train: if train or not.
    """
    weights = torch.tensor([1 / len(os.listdir(modal_path)) for _ in os.listdir(modal_path)])
    resample_weights = torch.multinomial(weights, num_sample, replacement=True)  # sample function

    img_array = []
    for id in resample_weights:
        img_path = os.path.join(modal_path, os.listdir(modal_path)[id])
        image = Image.open(img_path)

        # transform
        if train:
            image = process_train(image)
        else:
            image = process_eval(image)

        img_array.append(image)

    img_array = [e.tolist() for e in img_array]
    img_array = torch.tensor(img_array)
    img_array = rearrange(img_array, "n c h w -> (n c) h w")

    return img_array

def process_train(image):
    opt = get_args()
    channel = np.array(image).transpose().shape[0]
    if channel != 3:
        convert = transforms.Grayscale(3)
        image = convert(image)

    image = np.array(image)
    transform = alb.Compose([
        alb.LongestMaxSize(max_size=1000, interpolation=0, p=1.0),
        alb.PadIfNeeded(min_height=1000, min_width=1000, border_mode=cv2.BORDER_REPLICATE, p=1),
        alb.RandomResizedCrop(opt.img_size, opt.img_size, scale=(0.8, 1.0)),
        alb.Normalize([0.359, 0.361, 0.379], [0.190, 0.190, 0.199]),
        ToTensorV2(),
    ])
    image = transform(image=image)["image"]

    return image


def process_eval(image):
    opt = get_args()
    channel = np.array(image).transpose().shape[0]
    if channel != 3:
        convert = transforms.Grayscale(3)
        image = convert(image)

    image = np.array(image)
    transform = alb.Compose([
        alb.LongestMaxSize(max_size=1000, interpolation=0, p=1.0),
        alb.PadIfNeeded(min_height=1000, min_width=1000, border_mode=cv2.BORDER_REPLICATE, p=1), # border_mode: BORDER_REPLICATE 边缘值填充 BORDER_CONSTANT 常数填充
        alb.Resize(opt.img_size, opt.img_size),
        alb.Normalize([0.359, 0.361, 0.379], [0.190, 0.190, 0.199]),
        ToTensorV2(),
    ])
    return transform(image=image)["image"]

def gray_to_rgb(img):
    convert = transforms.Grayscale(3)
    size = np.array(img).transpose().shape
    if size[0] != 3:
        img = convert(img)

    return img