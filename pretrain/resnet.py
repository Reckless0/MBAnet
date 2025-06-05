import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.utils.model_zoo as model_zoo
from typing import Any, Optional, Tuple, Type


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, opt=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, input):
        residual = input
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, opt=None):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, input):
        residual = input
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.downsample:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    # 224*224
    def __init__(self, block, num_layer, opt, n_classes=1000, input_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 64, num_layer[0], 1, opt)
        self.layer2 = self._make_layer(block, 128, num_layer[1], 2, opt)
        self.layer3 = self._make_layer(block, 256, num_layer[2], 2, opt)
        self.layer4 = self._make_layer(block, 512, num_layer[3], 2, opt)
        # self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        # self.fc = nn.Linear(block.expansion * 512, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        if opt.use_meta:
            meta_input_dim = 4
            self.meta_proj1 = nn.Conv2d(meta_input_dim, 256, kernel_size=1, padding=0)
            self.meta_proj2 = nn.Conv2d(meta_input_dim, 512, kernel_size=1, padding=0)
            self.meta_proj3 = nn.Conv2d(meta_input_dim, 1024, kernel_size=1, padding=0)
            self.meta_proj4 = nn.Conv2d(meta_input_dim, 2048, kernel_size=1, padding=0)

    def _make_layer(self, block, out_channels, num_block, stride=1, opt=None):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, opt=opt))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels, opt=opt))
        return nn.Sequential(*layers)

    def forward(self, input, meta=None):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if meta != None:  
            meta = meta.view(-1, 4, 1, 1)
            attn1 = self.meta_proj1(meta)
            attn2 = self.meta_proj2(meta)
            attn3 = self.meta_proj3(meta)
            attn4 = self.meta_proj4(meta)

            x = self.layer1(x)
            x = x * attn1
            x = self.layer2(x)
            x = x * attn2
            x = self.layer3(x)
            x = x * attn3
            x = self.layer4(x)
            x = x * attn4 
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(opt, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], opt=opt, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(opt, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BottleNeck, [3, 4, 6, 3], opt=opt, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        # model_path = './initmodel/resnet50_v2.pth'
        # model.load_state_dict(torch.load(model_path), strict=False)
    return model


def resnet101(opt, pretrained=False,  **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BottleNeck, [3, 4, 23, 3], opt=opt, **kwargs)
    if pretrained:
        print("--using pretrained model--")
        try:
            state_dict = (model_zoo.load_url(model_urls['resnet101']))
        except:
            state_dict = torch.load("/data16/linrun/BA_code_new/models/Pretrained_ResNet/modelzoo_resnet101/resnet101-5d3b4d8f.pth", map_location='cpu')

        # model_path = "/data16/linrun/BA_code_new/models/Pretrained_ResNet/Gallbladder_res101_base_train_contain_empty_color_sgd_5"
        # print("pretrained path: {}".format(model_path))
        # ckpt = torch.load(model_path)
        # state_dict = ckpt['net']
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        model.load_state_dict(state_dict, strict=False) # strict=False
        # model.load_state_dict(torch.load(model_path), strict=False)
    return model


def resnet152(opt, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BottleNeck, [3, 8, 36, 3], opt=opt, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
        # model_path = './initmodel/resnet152_v2.pth'
        # model.load_state_dict(torch.load(model_path), strict=False)
    return model
