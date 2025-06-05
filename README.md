# Multimodal BA Diagnosis

## Content
- [Overview](#overview)
- [Requirements](#requirements)
- [Demo](#demo)
- [Visualization](#visualization)
- [Result](#result)

## Overview
Biliary atresia (BA) is a complex hepatobiliary disease in infants, which is still challenging to diagnose in current clinical practice. We aim to develop and validate a multimodal model by integrating conventional ultrasound, shear wave elastography, and clinical information to help diagnose BA.

## Requirements
运行本项目需要安装以下Python包：
```bash
pip install \
  torch \
  torchvision \
  numpy \
  scikit-learn \
  tensorboardX \
  Pillow \
  tqdm \
  SimpleITK \
  pandas \
  pretrainedmodels \
  matplotlib

## Demo
You can enter python run.sh in the terminal to run the code.

## Visualization
![实验结果描述](imgs/cams.png)

## Result
The model yields  an area under receiver operating characteristic curve (AUC) of 0.941 (95% confidence interval: 0.891-0.972) on the multi-center external validation dataset. Moreover, the diagnosis based on sonographic video sequences by the model yielded AUC 0.930 (95% confidence interval: 0.876-0.966).
