import torch
from torch import nn
import torch.nn.functional as F


class Late_Fuse_Model(torch.nn.Module):
    def __init__(self, models, opt):
        super(Late_Fuse_Model, self).__init__()
        if len(models)==2:
            model1, model2 = models
            self.model_1 = model1
            self.model_2 = model2

            if opt.use_meta:
                self.input_meta_dim=4

                self.proj = nn.Sequential(
                    nn.Conv2d(self.input_meta_dim, 512, kernel_size=1, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(512, 1024, kernel_size=1, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(1024, 2048, kernel_size=1, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(2048, 4096, kernel_size=1, padding=0),
                    )

        elif len(models)==1:
            model1 = models[0]
            self.model_1 = model1
            self.input_meta_dim=4
            if opt.use_meta:
                self.proj = nn.Sequential(
                    nn.Conv2d(self.input_meta_dim, 256, kernel_size=1, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(256, 512, kernel_size=1, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(512, 1024, kernel_size=1, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(1024, 2048, kernel_size=1, padding=0),
                    )
        else:
            raise ValueError('final meta_proj error.')
        
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AvgPool2d(kernel_size=7)

        if len(models)==1:
            self.fc = nn.Conv2d(2048, 2, kernel_size=1, padding=0)
        elif len(models)==2:
            self.fc = nn.Conv2d(4096, 2, kernel_size=1, padding=0)
        else:
            raise ValueError('fc error.')

    def forward(self, imgs, meta_data=None, modal_num=1):
        if meta_data !=None:
            if modal_num == 2:
                m1_img, m2_img = imgs
                x1 = self.model_1(m1_img, meta_data)
                x2 = self.model_2(m2_img, meta_data)
                x = torch.cat((x1, x2), dim=1)

            elif modal_num == 1:
                img = imgs
                x = self.model_1(img, meta_data)

            meta_data = meta_data.view(-1, self.input_meta_dim, 1, 1)
            meta_data = self.proj(meta_data)
            meta_data = self.sigmoid(meta_data)
            x = x * meta_data

        else:
            if modal_num == 2:
                m1_img, m2_img = imgs
                x1 = self.model_1(m1_img)
                x2 = self.model_2(m2_img)
                x = torch.cat((x1, x2), dim=1)

            elif modal_num == 1:
                img = imgs
                x = self.model_1(img)

            else: 
                raise ValueError('Error modal num!')

        x = self.avg_pool(x)
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        return x
    
def get_fused_model(model, fuse_type, opt):

    if fuse_type == "late":
        net = Late_Fuse_Model(model, opt)

        # # Contrastive learning
        # model = Late_Fuse_SupConResnet(models)
    elif fuse_type == "res_vit":
        net = resVit_Fuse_Model(model)

    elif fuse_type == "other":
        pass

    else:
        raise "model not found error!"

    return net