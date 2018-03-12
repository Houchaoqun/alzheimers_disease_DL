# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

import time
from model_densenet import densenet121, densenet169, densenet161
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from utils import progress_bar

## plot result
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

batch_size = 16
EPOCH = 10
use_gpu = torch.cuda.is_available()

model_name = "densenet161"
data_root = '/home/reserch/documents/deeplearning/alzheimers_disease/ADNI-825-Slice/experiments_FineTunning'
dataset_name = 'single_subject_data_fold_01_train_val_test_entropy_keep_SliceNum_81'
data_dir = os.path.join(data_root, dataset_name)
model_save_path = os.path.join('./pytorch_model', model_name + '_' + dataset_name + '.7t')

## save reslut into a txt file
save_info_path = os.path.join("./result_train", model_name + "_" + dataset_name + ".txt")
import datetime
i = datetime.datetime.now()
date = str(i.year) + str("%02d"%i.month) + str("%02d"%i.day) + "-" + str("%02d"%i.hour) + str("%02d"%i.minute) + str("%02d"%i.second)
with open(save_info_path, "a+") as save_info_txt:
    print("===" + date + "===")
    print("dataset_name = {}".format(dataset_name))
    print("model_name = {}".format(model_name))
    print("batch_size = {}".format(batch_size))
    print("model_save_path = {}".format(model_save_path))
    save_info_txt.writelines("===" + date + "===" + "\n")
    save_info_txt.writelines("dataset_name = {}".format(dataset_name) + "\n")
    save_info_txt.writelines("model_name = {}".format(model_name) + "\n")
    save_info_txt.writelines("batch_size = {}".format(batch_size) + "\n")
    save_info_txt.writelines("model_save_path = {}".format(model_save_path) + "\n")

## 
train_acc_list = []
train_loss_list = []
val_acc_list = []
val_loss_list = []


def display_net():
    net_from_web = alexnet()
    net_from_torchvision = models.alexnet()
    
    print(net_from_web)
    print(net_from_torchvision)

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

def data_prepare():
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  ## 3 channels -> RGB image 
        ]),
        'validation': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    
    image_datasets = {}
    dataloaders = {}
    
    ## train
    train_dir_name = 'train'
    image_datasets[train_dir_name] = datasets.ImageFolder(os.path.join(data_dir, train_dir_name), data_transforms[train_dir_name])
    dataloaders[train_dir_name] = torch.utils.data.DataLoader(image_datasets[train_dir_name], batch_size=batch_size, shuffle=True, num_workers=4)
    train_dataset_sizes = len(image_datasets[train_dir_name])
    print("train dataset_sizes = {}".format(train_dataset_sizes))
    
    ## val
    val_dir_name = 'validation'
    image_datasets[val_dir_name] = datasets.ImageFolder(os.path.join(data_dir, val_dir_name), data_transforms[val_dir_name])
    dataloaders[val_dir_name] = torch.utils.data.DataLoader(image_datasets[val_dir_name], batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataset_sizes = len(image_datasets[val_dir_name])
    print("validation dataset_sizes = {}".format(val_dataset_sizes))
    
    class_names = image_datasets['train'].classes
    print("class_names = {}".format(class_names))
    
    return dataloaders, class_names
    

def train(model, epoch, dataloaders_train, criterion, optimizer):
    
    train_acc = 0.0
    train_loss = 0.0
    total = 0
    print('-' * 10)
    print('Epoch {}/{}'.format(epoch, EPOCH - 1))
    
#    scheduler.step()
    for index, (inputs, targets) in enumerate(dataloaders_train):
        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            targets = Variable(targets.cuda())
        else:
            inputs, targets = Variable(inputs), Variable(targets)
        
        # zero the parameter gradients
        optimizer.zero_grad()
       
        ## forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, targets)
       
        ## backward
        loss.backward()
        optimizer.step()
       
        ## statistics
        train_loss += loss.data[0]
        total += batch_size
        train_acc += preds.eq(targets.data).cpu().sum()

        train_avg_acc = 100.*train_acc/total
        train_avg_loss = train_loss/(index+1)

        progress_bar(index, len(dataloaders['train']), 'train info: Acc: %.3f%% (%d/%d) | Loss: %.3f'
                        % ( train_avg_acc, train_acc, total, train_avg_loss))
    
    ## save result into txt file
    train_acc_list.append(round(train_avg_acc, 2))
    train_loss_list.append(round(train_avg_loss, 2))
    with open(save_info_path, "a+") as save_info_txt:
        save_info_txt.writelines("train_avg_acc = {}, train_avg_loss = {}".format(train_avg_acc, train_avg_loss) + "\n")

    
best_acc = 0.0
def validation(model, epoch, dataloaders_val, criterion, optimizer):
    
    global best_acc
    global best_epoch
    model.eval()
    val_loss = 0
    correct =0
    total = 0
    # best_acc = 0.0
    for batch_idx, (inputs,targets) in enumerate (dataloaders_val):
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
        
        val_loss += loss.data[0]
        total += batch_size
        correct += preds.eq(targets.data).cpu().sum()

        val_avg_acc = 100.*correct/total
        val_avg_loss = val_loss/(batch_idx+1)
        progress_bar(batch_idx, len(dataloaders_val), 'val info: Acc: %.3f%% (%d/%d) | Loss: %.3f' % ( val_avg_acc, correct, total, val_avg_loss))
    
    ## save result into txt file
    val_acc_list.append(round(val_avg_acc, 2))
    val_loss_list.append(round(val_avg_loss, 2))
    with open(save_info_path, "a+") as save_info_txt:
        save_info_txt.writelines("val_avg_acc = {}, val_avg_loss = {}".format(val_avg_acc, val_avg_loss) + "\n")
    
    if val_avg_acc > best_acc:
        print('Saving..')
        state = {
            'net':model,
            'acc': val_avg_acc,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
        }
        
        torch.save(state, model_save_path)
        best_acc = val_avg_acc
        best_epoch = epoch

def _plot_it():

    ## acc
    # train_acc = [52.2, 63.2, 65.5, 67.8, 69.5, 72.9, 77.2, 83.2, 87.2, 92.1, 93.1, 95.2]
    # val_acc = [50.2, 60.2, 62.5, 65.8, 66.5, 70.9, 75.2, 80.2, 85.2, 90.1, 90.1, 90.2]    
    plt.figure(figsize=(10, 6))       # 确定图片大小
    plt.subplot(211)                # 确定第1个图的位置

    # 绘图
    plt.plot(train_acc_list,lw=1,label="train_acc",color='blue')
    plt.plot(train_acc_list,'ro', color='blue')

    plt.plot(val_acc_list,lw=1,label="val_acc",color='red')
    plt.plot(val_acc_list,'r*', color='red')
    # 添加细节
    plt.title("train-val acc info", size=20)
    plt.xlabel('epoch',size=10)
    plt.ylabel('trends',size=15)
    # plt.axis('tight')
    plt.xlim(0, EPOCH)
    plt.ylim(0,100)
    plt.legend(loc=0)

    ### loss
    plt.subplot(212)    # 确定第2个图的位置
    plt.plot(train_loss_list,lw=1,label="train_loss",color='blue')
    plt.plot(train_loss_list,'ro', color='blue')

    plt.plot(val_loss_list,lw=1,label="val_loss",color='red')
    plt.plot(val_loss_list,'r*', color='red')
    # 添加细节
    plt.title("train-val loss info", size=20)
    plt.xlabel('epoch',size=10)
    plt.ylabel('trends',size=15)
    # plt.axis('tight')
    plt.xlim(0, EPOCH)
    plt.ylim(np.min(val_loss_list)-1, np.max(val_loss_list)+1)
    plt.legend(loc=0)
    
    plt.savefig('result_train/' + model_name + '.png', format='png')
    plt.show()
    
if __name__ == '__main__':
    
    ### main
    ### train and validation
    since = time.time()
    
    dataloaders, class_names = data_prepare()
    dataloaders_train = dataloaders['train']
    dataloaders_val = dataloaders['validation']
    num_classes = 2

    model = densenet161(pretrained=True)
    num_features = 2208
    model.classifier = nn.Linear(num_features, num_classes)
    
    if use_gpu:
        model = model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    params = [{'params': model.features.parameters(), 'lr':1e-5},
              {'params': model.classifier.parameters(), 'lr': 1e-3}
             ]
    optimizer = optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), weight_decay = 5e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    with open(save_info_path, "a+") as save_info_txt:
        save_info_txt.writelines("features = 1e-5 \n")
        save_info_txt.writelines("classifier = 1e-3 \n")
        save_info_txt.writelines("optimizer = optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), weight_decay = 5e-4) \n")
        save_info_txt.writelines("exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) \n")
    
    for epoch in range(EPOCH):
        exp_lr_scheduler.step()
        with open(save_info_path, "a+") as save_info_txt:
            save_info_txt.writelines("Epoch = {}/{}".format(epoch, EPOCH) + "\n")
        train(model, epoch, dataloaders_train, criterion, optimizer)
        validation(model, epoch, dataloaders_val, criterion, optimizer)
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    with open(save_info_path, "a+") as save_info_txt:
        save_info_txt.writelines('[Time used] train complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60) + "\n")


