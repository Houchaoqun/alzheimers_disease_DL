#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:52:27 2018

@author: reserch
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torchvision import models
import numpy as np

import matplotlib.pyplot as plt
import sys

data_transforms = {
    'test': transforms.Compose([
#            transforms.Grayscale(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
#            transforms.Normalize([0.5], [0.5])
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

### paramaters initial...
batch_size = 1
# print(len(sys.argv))
# target_resume_name = sys.argv[1]
target_resume_name = "test"
model_save_path = os.path.join('./pytorch_model', 'densenet201_96%_epoch50_single_subject_data_fold_02_train_val_test_entropy_keep_SliceNum_81.7t')
folder_name = 'single_subject_data_fold_02_train_val_test_entropy_keep_SliceNum_81'
# data_dir = os.path.join('/home/hcq/alzheimer_disease/ADNI_825/experiments_FineTunning', folder_name)
data_dir = os.path.join('/home/reserch/documents/deeplearning/alzheimers_disease/ADNI-825-Slice/experiments_FineTunning', folder_name)
test_datasets = datasets.ImageFolder(os.path.join(data_dir, target_resume_name), data_transforms['test'])
test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size,
                                          shuffle=False, num_workers=4)

## print model info
save_info_path = os.path.join("./result", target_resume_name + "_" + folder_name + ".txt")

import datetime
i = datetime.datetime.now()
date = str(i.year) + str("%02d"%i.month) + str("%02d"%i.day) + "-" + str("%02d"%i.hour) + str("%02d"%i.minute) + str("%02d"%i.second)

with open(save_info_path, "a+") as save_info_txt:
    print("===" + date + "===")
    print("folder_name = {}".format(folder_name))
    print("model_save_path = {}".format(model_save_path))
    save_info_txt.writelines("===" + date + "===" + "\n")
    save_info_txt.writelines("folder_name = {}".format(folder_name) + "\n")
    save_info_txt.writelines("model_save_path = {}".format(model_save_path) + "\n")

# print(" ={}".format())

print("*"*20)

# =============================================================================
# # 输出一个batch
# print('one batch tensor data: {}'.format(iter(test_loader).next()))
# # 输出batch数量
# batch_num = len(list(iter(test_loader)))
# print('len of batchtensor: {}'.format(batch_num))
# =============================================================================

class_names = test_datasets.classes
use_gpu = torch.cuda.is_available()
checkpoint = torch.load(model_save_path)

model = checkpoint['net']
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch'] + 1
best_epoch = checkpoint['epoch']
#print("best_acc = {}".format(best_acc))
#print("epoch = {}".format(checkpoint['epoch']))

if use_gpu:
    model.cuda()

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def visualize_model(model, num_images=4):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(test_loader):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return

def predict_test():
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if use_gpu:
            inputs = Variable(inputs.cuda())
            targets = Variable(targets.cuda())
        else:
            inputs, targets = Variable(inputs), Variable(targets)
            
        img_name = test_loader.dataset.imgs[batch_idx]
        img_name = img_name[0]
        img_name = img_name.split("/")[9]
        subject_id = img_name.split(".")[0]
        subject_id = subject_id.split("_")[1]
        print(subject_id)
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        pred_label = class_names[preds[0]]
        real_label = class_names[targets.data[0]]
        
        preds_softmax = F.softmax(outputs)
        p_AD = preds_softmax.data[0][0]
        p_NC = preds_softmax.data[0][1]
        # print('-'*10 + 'test' + '-'*10)
        # print("img_name = {}".format(img_name))
        # print("real label = {}, predicted label = {}".format(real_label, pred_label))
        # print("p_AD = {}, p_NC = {}.".format(p_AD, p_NC))
    #    print(preds_softmax)
    #    print(outputs)
    #    print(torch.max(outputs, 1)[1])
        if batch_idx == 10:
            break
# class subject_result():

### add by hcq 20180310
### test for specified subject

def predict_test2():
    model.eval()
    img_name_offset = 11

    ## final result
    num_correct = 0
    num_all_subject = 0
    
    # print(target_resume_name)
    subject_id_test_path = "./subject_id/"+ target_resume_name + "_single_subject_data_fold_02_train_val_test_entropy_keep_SliceNum_81.txt"
    # subject_id_test_path = "./subject_id/validation_single_subject_data_fold_02_train_val_test_entropy_keep_SliceNum_81.txt"
    
    with open(subject_id_test_path, "r") as subject_id_test_list:
        for item in subject_id_test_list:
            item = item.replace("\n", "")

            cur_subject_id = item.split(",")[0]
            label = item.split(",")[1]
            # print("--")
            # print("cur_subject_id = {}".format(cur_subject_id))

            num_AD = 0
            num_NC = 0
            num_total = 0

            for batch_idx, (inputs, targets) in enumerate(test_loader):
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    targets = Variable(targets.cuda())
                else:
                    inputs, targets = Variable(inputs), Variable(targets)
                
                img_name = test_loader.dataset.imgs[batch_idx]
                img_name = img_name[0]
                img_name = img_name.split("/")[img_name_offset]
                subject_name = img_name.split(".")[0]
                # print(subject_id)
                unique_id = subject_name.split("_")[0][0:4] ## To unique the subject id: id = 072 both in AD and NC
                subject_id = unique_id + "_" + subject_name.split("_")[1]

                if(cur_subject_id == subject_id):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    pred_label = class_names[preds[0]]
                    real_label = class_names[targets.data[0]]
                    # print("real label = {}, predicted label = {}".format(real_label, pred_label))
                    if(pred_label == "AD"):
                        num_AD += 1
                    elif(pred_label == "NC"):
                        num_NC += 1
                    num_total += 1
                    
                    # preds_softmax = F.softmax(outputs)
                    # p_AD = preds_softmax.data[0][0]
                    # p_NC = preds_softmax.data[0][1]
                    # print('-'*10 + 'test' + '-'*10)
                    # print("img_name = {}".format(img_name))
                    # print("real label = {}, predicted label = {}".format(real_label, pred_label))
                    # print("p_AD = {}, p_NC = {}.".format(p_AD, p_NC))

            ## calculate the final result
            num_all_subject += 1
            percentage = float(num_AD)/num_NC
            # print("percentage = {}".format(percentage))
            if((percentage>1 and label == "AD") or (percentage<1 and label == "NC")):
                dis_info = "subject_id = {}, label = {} ||| num_AD = {}, num_NC = {}, num_total = {}".format(cur_subject_id, label, num_AD, num_NC, num_total)
                num_correct += 1
            else:
                dis_info = "subject_id = {}, label = {} ||| num_AD = {}, num_NC = {}, num_total = {}  [error]".format(cur_subject_id, label, num_AD, num_NC, num_total)
            
            with open(save_info_path, "a+") as save_info_txt:
                save_info_txt.writelines(dis_info + "\n")
                print(dis_info)

        with open(save_info_path, "a+") as save_info_txt:
            print("***** final result information *****")
            info = "final acc = {} ({}/{})".format((float(num_correct)/num_all_subject), num_correct, num_all_subject)
            print(info)
            save_info_txt.writelines(info + "\n")

        

        ## python resume_model.py > result_test_single_subject_81.txt
        ## python resume_model.py > result_validation_single_subject_81.txt
        ## python resume_model.py > result_train_single_subject_81.txt


def predict_test3():
    model.eval()
    train_loss = 0
    correct =0
    total = 0
    criterion = nn.CrossEntropyLoss()
    from utils import progress_bar
    
    for batch_idx, (inputs,targets) in enumerate (test_loader):
        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            targets = Variable(targets.cuda())
        else:
            inputs, targets = Variable(inputs), Variable(targets)
            
        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, targets)
        
        train_loss += loss.data[0]
        total += batch_size
        correct += preds.eq(targets.data).cpu().sum()
        
        progress_bar(batch_idx, len(test_loader), 'test info: Acc: %.3f%% (%d/%d) | Loss: %.3f'
            % ( 100.*correct/total, correct, total, 
               train_loss/(batch_idx+1))) 
        
    
#visualize_model(model)
predict_test2()


