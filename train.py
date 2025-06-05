
import os
import cv2
import time
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from warmup_scheduler import GradualWarmupScheduler
from utils import perf_measure, EarlyStopping, SupConLoss
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import freeze_by_idxs, unfreeze_by_idxs


# basic settings
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class ComboIter(object):
    def __init__(self, my_loader):
        self.my_loader = my_loader
        self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]

    def __iter__(self):
        return self

    def __next__(self):
        batches = [loader_iter.next() for loader_iter in self.loader_iters]
        return self.my_loader.combine_batch(batches)

    def __len__(self):
        return len(self.my_loader)


class ComboLoader(object):
    """
    Mixup dataloaders 
    """
    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        return ComboIter(self)

    def __len__(self):
        return min([len(loader) for loader in self.loaders])

    # Customize the behavior of combining batches here.
    def combine_batch(self, batches):
        return batches

def modify_loader(train_data, loader):
    counts = [0, 0]
    for i in range(len(train_data)):
        counts[int(train_data.__getitem__(i)[1][1])] += 1
    print("counts:", counts)
    class_weights = [sum(counts) / c for c in counts]
    sample_weights = [class_weights[int(train_data.__getitem__(e)[1][1])] for e in range(len(train_data))]

    mod_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_data))
    mod_loader = DataLoader(loader.dataset, batch_size=loader.batch_size, sampler=mod_sampler,
                            num_workers=loader.num_workers, shuffle=False)
    return mod_loader

def get_combo_loader(train_data, loader, base_sampling='instance'):
    if base_sampling == 'instance':
        imbalanced_loader = loader
    else:
        imbalanced_loader = modify_loader(train_data, loader)

    balanced_loader = modify_loader(train_data, loader)
    combo_loader = ComboLoader([imbalanced_loader, balanced_loader])
    return combo_loader

# 用于处理one-hot格式的交叉熵损失
def cross_entropy_loss(input, target):
    return -(input.log_softmax(dim=-1) * target).sum(dim=-1).mean()



def train_model(train_data, test_data, net, total_score, total_label, kth, opt):
    lr = opt.learning_rate
    alpha = opt.alpha
    device = opt.device
    NUM_EPOCHS = opt.epochs
    batch_size = opt.batch_size
    num_workers = opt.num_workers
    patience = opt.patience
    checkpoint_dir = opt.res_dir

    torch.set_num_threads(2)
    setup_seed(3407)

    # log
    trlog = {}
    # train log
    trlog['args'] = vars(opt)
    trlog['train_loss'] = []
    trlog['train_acc'] = []
    # val log
    trlog['valid_loss(patient_level)'] = []
    trlog['valid_acc(patient_level)'] = []
    trlog['max_acc'] = 0.0
    trlog['max_acc_epoch'] = 0

    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    fc_param1 = list(net.children())[-1].parameters() # fc
    output_params = list(map(id, fc_param1))
    feature_params = filter(lambda p: id(p) not in output_params, net.parameters())
    
    optimizer = torch.optim.Adam([{'params': feature_params, 'lr': 1e-5},
                                  {'params': fc_param1, 'lr': 5e-3},
                                  ], lr=lr, weight_decay=5e-5)

    scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=100, total_epoch=10, after_scheduler=scheduler_cos)

    softmax = nn.Softmax(dim=1)
    combo_loader = get_combo_loader(train_data, train_dataloader, base_sampling="instance")
    early_stopping = EarlyStopping(patience, path=checkpoint_dir+'/model_' + str(kth) + '.pth', verbose=True)  # 关于 EarlyStopping 的

    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    iter_num = len(train_dataloader)
    total = len(train_dataloader) * batch_size

    for epoch in tqdm(range(NUM_EPOCHS)):
        net.train()
        running_train_loss, train_Correct, train_total = 0, 0, 0
        timestart = time.time()
        for batch in combo_loader:
            lam = np.random.beta(alpha, alpha)

            if len(opt.modal)==1:
                m1_imgs, labels, tab = batch[0][0], batch[0][1], batch[0][3]
                balanced_m1_imgs, balanced_labels, balanced_tab = batch[1][0], batch[1][1], batch[1][3]
                
                mixed_m1_imgs = (1 - lam) * m1_imgs + lam * balanced_m1_imgs
                mixed_labels = (1 - lam) * labels + lam * balanced_labels
                mixed_tab = (1 - lam) * tab + lam * balanced_tab

                mixed_m1_imgs = mixed_m1_imgs.to(device)
                mixed_labels = mixed_labels.to(device)
                mixed_tab = mixed_tab.to(device)

                if opt.use_meta:
                    preds = net((mixed_m1_imgs.to(torch.float32)),
                                mixed_tab.to(torch.float32), modal_num=1)
                else:
                    preds = net(mixed_m1_imgs.to(torch.float32), modal_num=1)

            elif len(opt.modal)==2:
                imgs, labels, patient, tab = batch[0][0], batch[0][1], batch[0][2], batch[0][3]
                balanced_imgs, balanced_labels, balanced_tab = batch[1][0], batch[1][1], batch[1][3]
                m1_imgs, m2_imgs = imgs
                balanced_m1_imgs, balanced_m2_imgs = balanced_imgs

                mixed_m1_imgs = (1 - lam) * m1_imgs + lam * balanced_m1_imgs
                mixed_m2_imgs = (1 - lam) * m2_imgs + lam * balanced_m2_imgs

                mixed_labels = (1 - lam) * labels + lam * balanced_labels
                mixed_tab = (1 - lam) * tab + lam * balanced_tab

                mixed_m1_imgs = mixed_m1_imgs.to(device)
                mixed_m2_imgs = mixed_m2_imgs.to(device)

                mixed_labels = mixed_labels.to(device)
                mixed_tab = mixed_tab.to(device)
                
                if opt.use_meta:
                    preds = net((mixed_m1_imgs.to(torch.float32), mixed_m2_imgs.to(torch.float32)),
                        mixed_tab.to(torch.float32), modal_num=2)
                else:
                    preds = net((mixed_m1_imgs.to(torch.float32), mixed_m2_imgs.to(torch.float32)), modal_num=2)

            train_loss = cross_entropy_loss(preds, mixed_labels)
            running_train_loss += train_loss

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_Correct += (torch.argmax(preds, dim=1) == torch.argmax(mixed_labels, dim=1)).float().sum()

        # evaluation
        net.eval()
        patient_level_preds = dict()
        patient_level_labels = dict()
        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))
        running_valid_loss = 0
        valid_total, valid_Correct = 0, 0
        val_score_list, val_label_list = [], []
        with torch.no_grad():

            for (imgs, labels, patients_file_name, tab) in test_dataloader:

                if len(opt.modal)==1:
                    m1_imgs = imgs.to(device)
                    labels = labels.to(device)
                    tab = tab.to(device)
                    if opt.use_meta:
                        preds = net((m1_imgs.to(torch.float32)),
                                tab.to(torch.float32), modal_num=1)     
                    else:
                        preds = net(m1_imgs.to(torch.float32), modal_num=1)
                     
                elif len(opt.modal)==2:
                    m1_imgs = imgs[0].to(device)
                    m2_imgs = imgs[1].to(device)
                    labels = labels.to(device)
                    tab = tab.to(device)
                    if opt.use_meta:
                        preds = net((m1_imgs.to(torch.float32), m2_imgs.to(torch.float32)),
                            tab.to(torch.float32), modal_num=2)
                    else:
                        preds = net((m1_imgs.to(torch.float32), m2_imgs.to(torch.float32)), modal_num=2)

                preds = softmax(preds)

                _, patients = patients_file_name

                for i in range(len(patients)):
                    if patients[i] not in patient_level_preds.keys():
                        patient_level_preds[patients[i]] = [preds[i]]
                    else:
                        patient_level_preds[patients[i]].append(preds[i])

                    if patients[i] not in patient_level_labels.keys():
                        patient_level_labels[patients[i]] = labels[i]

            # average over the same patient
            for p, p_pred in patient_level_preds.items():
                patient_preds = torch.zeros(1, 2).to(device)
                for pred in p_pred:
                    patient_preds += pred
            
                patient_preds /= len(p_pred)

                patient_label = patient_level_labels[p].reshape(1, 2)

                valid_loss = torch.nn.functional.cross_entropy(patient_preds, patient_label)
                running_valid_loss += valid_loss

                val_score_list.extend(np.array(patient_preds.cpu()))
                val_label_list.extend(np.array(patient_label.cpu()))

                prediction = torch.argmax(patient_preds, dim=1)
                true_labels = torch.argmax(patient_label, dim=1)

                class_total[true_labels] += 1
                if prediction == true_labels:
                    class_correct[prediction] += 1

                # valid_Correct += (prediction == true_labels).float().sum()

        # after evaluation
        avg_train_loss = running_train_loss / iter_num  # batch_num
        # train_acc = 100. * train_Correct / total  # batch_num * batch_size

        avg_valid_loss = running_valid_loss / len(patient_level_preds)
        # valid_acc = 100. * valid_Correct / len(patient_level_preds)
        val_AUC = roc_auc_score(np.array(val_label_list)[:, 1], np.array(val_score_list)[:, 1])

        trlog['train_loss'].append(avg_train_loss)
        trlog['valid_loss(patient_level)'].append(avg_valid_loss)

        # scheduler for every epoch
        scheduler.step()
        print("scheduler.last_epoch:", scheduler.last_epoch)

        print('epoch %d, time:%3f sec, conv_lr: %.7f, fc_lr: %.7f'
              % (epoch, time.time() - timestart, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))
        print('Train Loss: %.4f'
              % (running_train_loss))
        print('Valid AUC: %.3f, Valid Loss: %.3f, Sen: %.3f, Spe:%.3f, '
              % (val_AUC,
                 avg_valid_loss,
                 100 * class_correct[1] / (class_total[1] + 1e-6),
                 100 * class_correct[0] / (class_total[0] + 1e-6)
                 )
              )

        early_stopping(val_AUC, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    # Load early stopping params
    net = torch.load(checkpoint_dir + '/model_' + str(kth) + '.pth')

    # evaluation on best model
    net.eval()
    patient_level_preds = dict()
    patient_level_labels = dict()
    with torch.no_grad():
        score_list, label_list = [], []
        for (imgs, labels, patient_file_name, tab) in test_dataloader:
            if len(opt.modal)==1:
                    m1_imgs = imgs.to(device)
                    labels = labels.to(device)
                    tab = tab.to(device)
                    if opt.use_meta:
                        preds = net((m1_imgs.to(torch.float32)),
                                tab.to(torch.float32), modal_num=1)     
                    else:
                        preds = net(m1_imgs.to(torch.float32), modal_num=1)
                     
            elif len(opt.modal)==2:
                m1_imgs = imgs[0].to(device)
                m2_imgs = imgs[1].to(device)
                labels = labels.to(device)
                tab = tab.to(device)
                if opt.use_meta:
                    preds = net((m1_imgs.to(torch.float32), m2_imgs.to(torch.float32)),
                        tab.to(torch.float32), modal_num=2)
                else:
                    preds = net((m1_imgs.to(torch.float32), m2_imgs.to(torch.float32)), modal_num=2)

            preds = softmax(preds)

            _, patients = patient_file_name
            for i in range(len(patients)):
                # print(patients[i])
                if patients[i] not in patient_level_preds.keys():
                    patient_level_preds[patients[i]] = [preds[i]]
                else:
                    patient_level_preds[patients[i]].append(preds[i])

                if patients[i] not in patient_level_labels.keys():
                    patient_level_labels[patients[i]] = labels[i]

        # average over the same patient
        for p, p_pred in patient_level_preds.items():
            patient_preds = torch.zeros(1, 2).to(device)
            for pred in p_pred:
                patient_preds += pred
            patient_preds /= len(p_pred)

            patient_label = patient_level_labels[p].reshape(1, 2)

            score_list.extend(np.array(patient_preds.cpu()))
            label_list.extend(np.array(patient_label.cpu()))

        score_tensor = torch.tensor(np.array(score_list))
        label_onehot = torch.tensor(np.array(label_list))

        AUC = roc_auc_score(np.array(label_list)[:, 1], np.array(score_list)[:, 1])
        print("-----AUC:{}-----".format(AUC))

        total_score.extend(np.array(score_list)[:, 1])
        total_label.extend(np.array(label_list)[:, 1])

    return net, total_score, total_label, score_tensor, label_onehot