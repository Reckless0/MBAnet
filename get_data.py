import os
import torch
import random
import numpy as np
from collections import defaultdict

from data.dataset import new_m1_dataset, new_m2_dataset, new_m12_dataset


def get_dataset(data_dir, k_th, modal_num):
    # single modal
    if modal_num == '1' or modal_num == '2':
        train_path, test_path = dict(), dict()  # img path for train/test
        train_path['0'], train_path['1'], test_path['0'], test_path['1'] = [], [], [], []

        for n, fold in enumerate(sorted(os.listdir(data_dir))):
            fold_dir = os.path.join(data_dir, fold)

            for label in sorted(os.listdir(fold_dir)):
                label_dir = os.path.join(fold_dir, label)

                for patient in sorted(os.listdir(label_dir)):
                    patient_path = os.path.join(label_dir, patient)


                    for modal in sorted(os.listdir(patient_path)):
                        if modal == modal_num:
                            modal_path = os.path.join(patient_path, modal)

                            for img in sorted(os.listdir(modal_path)):
                                img_path = os.path.join(modal_path, img)

                                if n == k_th:
                                    test_path[label].append(img_path)
                                else:
                                    train_path[label].append(img_path)

    # multi-modal
    elif modal_num == ['1', '2']:
        m1_train_path, m1_test_path = dict(), dict()  # m1 img path for train/test
        m2_train_path, m2_test_path = dict(), dict()  # m2 img path for train/test
        m1_train_path['0'], m1_train_path['1'], m1_test_path['0'], m1_test_path['1'] = [], [], [], []
        m2_train_path['0'], m2_train_path['1'], m2_test_path['0'], m2_test_path['1'] = [], [], [], []
        train_patient_m2_idx, test_patient_m2_idx = dict(), dict()
        train_patient_m2_idx['0'], train_patient_m2_idx['1'], test_patient_m2_idx['0'], test_patient_m2_idx['1'] = dict(), dict(), dict(), dict()
    
        for n, fold in enumerate(sorted(os.listdir(data_dir))):
            fold_dir = os.path.join(data_dir, fold)

            for label in sorted(os.listdir(fold_dir)):
                label_dir = os.path.join(fold_dir, label)
        
                if n==k_th: # test
                    if test_patient_m2_idx[str(label)] == {}: 
                        m2_idx = 0
                    else:
                        last_pat = list(test_patient_m2_idx[str(label)])[-1]
                        m2_idx = test_patient_m2_idx[str(label)][last_pat][-1] +1 # last patient's last idx+1, as beginning m2_idx of next fold.
                else: # train
                    if train_patient_m2_idx[str(label)] == {}:
                        m2_idx = 0
                    else:
                        last_pat = list(train_patient_m2_idx[str(label)])[-1]
                        m2_idx = train_patient_m2_idx[str(label)][last_pat][-1] +1 # last patient's last idx+1, as beginning m2_idx of next fold.
        
                for patient in sorted(os.listdir(label_dir)):
        
                    patient_path = os.path.join(label_dir, patient)
        
                    if n == k_th:
                        test_patient_m2_idx[label][patient] = []
                    else:
                        train_patient_m2_idx[label][patient] = []
        
                    for modal in sorted(os.listdir(patient_path)):
                        assert len(modal_num) == 2
                        modal_path = os.path.join(patient_path, modal)
                        if modal == '1':
        
                            for img in sorted(os.listdir(modal_path)):
                                img_path = os.path.join(modal_path, img)
        
                                if n == k_th:
                                    m1_test_path[label].append(img_path)
                                else:
                                    m1_train_path[label].append(img_path)
        
                        elif modal == '2':
        
                            for img in sorted(os.listdir(modal_path)):
        
                                img_path = os.path.join(modal_path, img)
        
                                if n == k_th:
                                    m2_test_path[label].append(img_path)
                                    test_patient_m2_idx[label][patient].append(m2_idx)
                                else:
                                    m2_train_path[label].append(img_path)
                                    train_patient_m2_idx[label][patient].append(m2_idx)
        
                                m2_idx += 1

    if modal_num == '1':
        train_data = new_m1_dataset(image_path=train_path, train=True)
        test_data = new_m1_dataset(image_path=test_path, train=False)
    elif modal_num == '2':
        train_data = new_m2_dataset(image_path=train_path, train=True)
        test_data = new_m2_dataset(image_path=test_path, train=False)
    elif modal_num == ['1', '2']:
        train_data = new_m12_dataset(image_path=[m1_train_path, m2_train_path], patient_m2_idx=train_patient_m2_idx, train=True)
        test_data = new_m12_dataset(image_path=[m1_test_path, m2_test_path],patient_m2_idx=test_patient_m2_idx, train=False)
    else:
        raise ValueError('Modal num not found!')
    
    return train_data, test_data