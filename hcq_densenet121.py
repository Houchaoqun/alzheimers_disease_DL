# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import time
from model_densenet import densenet121, densenet169
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
from utils import progress_bar

batch_size = 32
EPOCH = 120
use_gpu = torch.cuda.is_available()

data_root = '/home/hcq/alzheimer_disease/ADNI_825/experiments_FineTunning'
dataset_name = 'single_subject_data_fold_01_train_val_test_entropy_keep_SliceNum_81'
data_dir = os.path.join(data_root, dataset_name)
model_save_path = os.path.join('./pytorch_model', dataset_name + '_densenet.7t')

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
    #            transforms.Grayscale(),  ## change image to gray image
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
    #            transforms.Normalize([0.5], [0.5]),  ## 1 channel -> gray image 
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  ## 3 channels -> RGB image 
        ]),
        'validation': transforms.Compose([
    #            transforms.Grayscale(),  ## change image to gray image
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
    #            transforms.Normalize([0.5], [0.5]),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    
    ## 
#==============================================================================
#     image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                           data_transforms[x])
#                   for x in ['train', 'val']}
#     dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
#                                                  shuffle=True, num_workers=4)
#                   for x in ['train', 'val']}
#==============================================================================
    image_datasets = {}
    dataloaders = {}
#    dataset_sizes = {}
    
    ## train
    train_dir_name = 'train'
    image_datasets[train_dir_name] = datasets.ImageFolder(os.path.join(data_dir, train_dir_name), data_transforms[train_dir_name])
    dataloaders[train_dir_name] = torch.utils.data.DataLoader(image_datasets[train_dir_name], batch_size=batch_size, shuffle=True, num_workers=4)
    train_dataset_sizes = len(image_datasets[train_dir_name])
    print("train dataset_sizes = {}".format(train_dataset_sizes))
    
    ## val
    val_dir_name = 'validation'
    image_datasets[val_dir_name] = datasets.ImageFolder(os.path.join(data_dir, val_dir_name), data_transforms[val_dir_name])
    dataloaders[val_dir_name] = torch.utils.data.DataLoader(image_datasets[train_dir_name], batch_size=batch_size, shuffle=True, num_workers=4)
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
        # train_acc += torch.sum(preds == targets.data)
        train_acc += preds.eq(targets.data).cpu().sum()
#==============================================================================
#         train_loss += loss.data[0]
#         total += batch_size
#         correct += preds.eq(targets.data).cpu().sum()
#==============================================================================
        progress_bar(index, len(dataloaders['train']), 'train info: Acc: %.3f%% (%d/%d) | Loss: %.3f'
                        % ( 100.*train_acc/total, train_acc, total, 
                           train_loss/(index+1)))
        
    # len_train_data = len(dataloaders_train)
    # epoch_train_loss = train_loss / len_train_data
    # epoch_train_acc = train_acc / len_train_data
    
    # print("epoch_train_loss = {}, epoch_train_acc = {}".format(epoch_train_loss, epoch_train_acc))
    

def validation(model, epoch, dataloaders_val, criterion, optimizer):
    
    global best_acc
    global best_epoch
#    print('-' * 5 + 'validation' + '-' * 5)
    model.eval()
    val_loss = 0
    correct =0
    total = 0
    best_acc = 0.0
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
        
        progress_bar(batch_idx, len(dataloaders_val), 'val info: Acc: %.3f%% (%d/%d) | Loss: %.3f'
            % ( 100.*correct/total, correct, total, val_loss/(batch_idx+1)))
        
    acc = 100.*correct/total  
    if acc > best_acc:
        print('Saving..')
        state = {
            'net':model,
            'acc': acc,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
        }
        
        torch.save(state, model_save_path)
        best_acc = acc
        best_epoch = epoch
    
    
    
if __name__ == '__main__':
    
    ### main
    ### train and validation
    since = time.time()
    
    dataloaders, class_names = data_prepare()
    dataloaders_train = dataloaders['train']
    dataloaders_val = dataloaders['validation']
    num_classes = 2

    # model = alexnet(pretrained=True)
    # model.classifier = nn.Sequential(
    #     nn.Dropout(),
    #     nn.Linear(256 * 6 * 6, 4096),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(),
        
    #     nn.Linear(4096, 4096),
    #     nn.ReLU(inplace=True),
    #     nn.Linear(4096, num_classes),
    # )

    model = densenet121(pretrained=True)
    num_features = 1024
    model.classifier = nn.Linear(num_features, num_classes)
    

    if use_gpu:
        model = model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    params = [{'params': model.features.parameters(), 'lr':1e-5},
              {'params': model.classifier.parameters(), 'lr': 1e-3}
             ]
    optimizer = optim.Adam(params, lr=1e-3, betas=(0.9, 0.999))
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    for epoch in range(EPOCH):
        exp_lr_scheduler.step()
        train(model, epoch, dataloaders_train, criterion, optimizer)
        validation(model, epoch, dataloaders_val, criterion, optimizer)
        
    
    
#==============================================================================
#     dataloaders, class_names = data_prepare()
#     stop = 0
#     for index, (inputs,targets) in enumerate (dataloaders['train']):
#         print('='*10)
#         if use_gpu:
#             inputs = Variable(inputs.cuda())
#             targets = Variable(targets.cuda())
#         else:
#             inputs, targets = Variable(inputs), Variable(targets)
#             
#         img_name = dataloaders['train'].dataset.imgs[index]
#         print("index = {}, targets = {}".format(index, targets))
#         print("img_name = {}".format(img_name[0]))
# #        
# #        out = torchvision.utils.make_grid(inputs)
# #        imshow(out, title=class_names[targets])
#         
#         stop = stop+1
#         if (stop > 10):
#             break
#==============================================================================
            
        
#==============================================================================
#     inputs, classes = next(iter(dataloaders['train']))
#     for index, x in enumerate(classes):
#         print("index = {}, x = {}".format(index, x))
# #        img_name = inputs.dataset.imgs[index]
# #        print("img_name = {}".format(img_name))
#         
#     # Make a grid from batch
#     out = torchvision.utils.make_grid(inputs)
#     imshow(out, title=[class_names[x] for x in classes])
#==============================================================================
    
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))










