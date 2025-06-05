from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from torchvision.models import resnet50
import torch
import argparse
import cv2
import numpy as np

import os
import timm
import numpy as np
import cv2
from model import get_fused_model
from config import get_args


opt = get_args()

dir = r'D:\Person File\Research\BA_Project\BA_code_new\data\m1_masked+m2_full_total_5fold\fold_4'
new_dir = r'D:\Person File\Research\BA_Project\BA_code_new\grad_cam_result\fold_4'

# load model
model2 = timm.create_model('resnet101', in_chans=3, pretrained=True, num_classes=2).to(opt.device)
model = get_fused_model(model2, fuse_type='late').to(opt.device)
model_path = r'D:\Person File\Research\BA_Project\BA_code_new\roc_pr\总_run_on_m1_m2_full_only_both_dataset\m2_整张\try3\save_checkpoint\model_4.pth'
model_ = torch.load(model_path)

# crate grad_cam
target_layers = [model_.model_2[-2][-1]]
cam = GradCAM(model=model_, target_layers=target_layers, use_cuda=True)

for label in sorted(os.listdir(dir)):
    dir_l = os.path.join(dir, label)
    new_dir_l = os.path.join(new_dir, label)
    if not os.path.exists(new_dir_l):
        os.mkdir(new_dir_l)

    for patient in sorted(os.listdir(dir_l)):
        dir_p = os.path.join(dir_l, patient)
        new_dir_p = os.path.join(new_dir_l, patient)
        if not os.path.exists(new_dir_p):
            os.mkdir(new_dir_p)

        for modal in sorted(os.listdir(dir_p)):
            if modal == '2':
                dir_m = os.path.join(dir_p, modal)
                new_dir_m = os.path.join(new_dir_p, modal)
                if not os.path.exists(new_dir_m):
                    os.mkdir(new_dir_m)

                for img in sorted(os.listdir(dir_m)):
                    dir_i = os.path.join(dir_m, img)
                    new_dir_i = os.path.join(new_dir_m, img)

                    # read and process image
                    rgb_img = cv2.imread(dir_i, 1)[:, :, ::-1]
                    rgb_img = cv2.resize(rgb_img, (224, 224))
                    rgb_img = np.float32(rgb_img) / 255
                    input_tensor = preprocess_image(rgb_img, mean=[0.359, 0.361, 0.379], std=[0.190, 0.190, 0.199])

                    targets = None
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

                    grayscale_cam = grayscale_cam[0, :]
                    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
                    cv2.imwrite(new_dir_i, cam_image)

    #                 break
    #             break
    #
    #     break
    # break
