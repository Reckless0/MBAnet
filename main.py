import os
import cv2
import json
import random
import logging
import timm
import numpy as np
from PIL import Image

import torch
from torch import nn
from torchsummary import summary

import matplotlib.pyplot as plt
from captum.attr import LRP
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from captum.attr import visualization as viz

from get_data import get_dataset
from train import train_model
from utils.visualize import draw_fold_ROC_PR, draw_total_ROC_PR, draw_test_ROC_PR
from config import get_args
from model import get_fused_model
from pretrain import *


if __name__ == '__main__':
    # basic settings
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(2)
    setup_seed(3407)

    y_real, y_score = [], []
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax_roc, ax_pr = ax[0], ax[1]
    score, lb = [], []
    torch.cuda.empty_cache()

    opt = get_args()

    tprs, aucs = [], []
    models = []
    test_models = []
    folds_optimal_threshold = {}

    if opt.modal == '12':
        modal=['1', '2']
    else: # single modal
        modal=opt.modal

    use_meta = opt.use_meta
    hospital_split = opt.hospital_split
    ext_video = opt.ext_video

    res_dir = opt.res_dir
    os.makedirs(res_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
            filename=res_dir+'/train.log',
            datefmt='%Y/%m/%d %H:%M:%S',
            format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
    
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger(__name__)

    for i in range(opt.k):
        print(f"第{i + 1}折：")
        train_data, test_data = get_dataset(data_dir = opt.dataset, k_th=i, modal_num=modal)
        print(len(train_data))
        print(len(test_data))

        if len(modal)==2:
            model1 = resnet101(opt, pretrained=True, n_classes=2)
            model2 = resnet101(opt, pretrained=True, n_classes=2)
            models = [model1, model2]
            model = get_fused_model(models, fuse_type='late', opt=opt).to(opt.device)
        
        elif len(modal)==1:
            model1 = resnet101(opt, pretrained=True, n_classes=2)
            models = [model1]
            model = get_fused_model(models, fuse_type='late', opt=opt).to(opt.device)

        print(list(model.children())[-1])

        import ipdb
        ipdb.set_trace()

        # training
        model_, score, lb, score_tensor, label_onehot = train_model(
            train_data, test_data, model, score, lb, i, opt,
        )
        test_models.append(model_)

        # draw ROC and PR curve for every fold
        y_real, y_score, aucs, fold_optimal_threshold = draw_fold_ROC_PR(
            score_tensor, label_onehot, y_real, y_score, aucs,
            k_th_fold=i, ax_roc=ax_roc, ax_pr=ax_pr, opt=opt, optimal_metrics='PR')
            
        # draw ROC and PR curve in total k folds
        ax_roc, ax_pr = draw_total_ROC_PR(y_real, y_score, aucs, ax_roc=ax_roc, ax_pr=ax_pr)
        plt.savefig('./roc_pr.png')