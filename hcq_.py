# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torchvision import models
from model_vgg import vgg16_bn,vgghcq_bn
from utils import progress_bar
import time 
import copy


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

batch_size = 32
num_epochs = 200

model_save_path = os.path.join('./pytorch_model','best_single_subject_data_fold_01_train_val_test_entropy_keep_SliceNum_81.t7')
data_dir = '/home/reserch/documents/deeplearning/alzheimers_disease/ADNI-825-Slice/experiments_FineTunning/single_subject_data_fold_01_train_val_test_entropy_keep_SliceNum_81'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'validation']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True if x=='train' else False, num_workers=4)
              for x in ['train', 'validation']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
print("train dataset_sizes = {}".format(dataset_sizes['train']))
print("validation dataset_sizes = {}".format(dataset_sizes['validation']))
class_names = image_datasets['train'].classes
print("class_names = {}".format(class_names))

use_gpu = torch.cuda.is_available()

model = models.vgg16_bn(pretrained=True)
#model = vgghcq_bn()

# =============================================================================
# for param in model.parameters():
#     param.requires_grad = False
# =============================================================================

# =============================================================================
# self.classifier = nn.Sequential(
#     nn.Linear(512 * 7 * 7, 4096),
#     nn.ReLU(True),
#     nn.Dropout(),
#     nn.Linear(4096, 4096),
#     nn.ReLU(True),
#     nn.Dropout(),
#     nn.Linear(4096, num_classes),
# )
# =============================================================================

model.classifier = nn.Sequential(
        ### fc 1
        nn.Linear(512 * 7 * 7, 1024),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        
        nn.Linear(1024, 512),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        
        ### predictions
        nn.Linear(512, 2))

if use_gpu:
    model = model.cuda()

## 
# =============================================================================
# Examples::
# 
# >>> loss = nn.CrossEntropyLoss()
# >>> input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
# >>> target = autograd.Variable(torch.LongTensor(3).random_(5))
# >>> output = loss(input, target)
# >>> output.backward()
# =============================================================================
criterion = nn.CrossEntropyLoss()

#optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)  ## optimize specified paramaters
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  ## optimize all paramaters

## optimize features and classifier with different learning rate.
params = [{'params': model.features.parameters(), 'lr': 1e-5},
          {'params': model.classifier.parameters(), 'lr': 1e-3}
         ]

#optimizer = optim.SGD(params, lr=0.001, momentum=0.9)  ## 
optimizer = optim.Adam(params, lr=1e-3, betas=(0.9, 0.999))

# Decay LR by a factor of 0.1 every step_size epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


best_acc = 0
best_epoch = 0
best_model_wts = copy.deepcopy(model.state_dict())

def train(epoch):
    
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('*' * 20)
    
#    print('-' * 5 + 'train' + '-' * 5)
    
    
    model.train()  ## set the model as a state of train.
    train_loss = 0
    correct =0
    total = 0
    
    for batch_idx, (inputs,targets) in enumerate (dataloaders['train']):
        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            targets = Variable(targets.cuda())
        else:
            inputs, targets = Variable(inputs), Variable(targets)
            
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, targets)
    
        # backward
        loss.backward()
        optimizer.step()
        
        # statistics
#        running_loss += loss.data[0] * inputs.size(0)
#        running_corrects += torch.sum(preds == targets.data)
        train_loss += loss.data[0]
        total += batch_size
        correct += preds.eq(targets.data).cpu().sum()
        
        progress_bar(batch_idx, len(dataloaders['train']), 'train info: Acc: %.3f%% (%d/%d) | Loss: %.3f'
            % ( 100.*correct/total, correct, total, 
               train_loss/(batch_idx+1))) 
        
#    epoch_loss = running_loss / len(dataloaders['train'])
#    epoch_acc = running_corrects / len(dataloaders['train'])
    
#    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                'train', epoch_loss, epoch_acc))
        
   
    
def validation(epoch):
    global best_acc
    global best_epoch
#    print('-' * 5 + 'validation' + '-' * 5)
    model.eval()
    train_loss = 0
    correct =0
    total = 0
    best_acc = 0.0
    for batch_idx, (inputs,targets) in enumerate (dataloaders['validation']):
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
        
        progress_bar(batch_idx, len(dataloaders['validation']), 'val info: Acc: %.3f%% (%d/%d) | Loss: %.3f'
            % ( 100.*correct/total, correct, total, 
               train_loss/(batch_idx+1))) 
        
    acc = 100.*correct/total  
    if acc > best_acc:
        print('Saving..')
        
        best_acc = acc
        best_epoch = epoch
        best_model_wts = copy.deepcopy(model.state_dict())
        
        state = {
            'net':model,
            'acc': best_acc,
            'epoch': best_epoch,
            'optimizer': optimizer.state_dict(),
        }
        
        torch.save(state, model_save_path)
        

# =============================================================================
# def test():
#     print('-'*5 + 'test' + '-'*5)
# #    model.eval()
# =============================================================================
    
    
### main
### train and validation
since = time.time()

for epoch in range(num_epochs):
    exp_lr_scheduler.step()
    train(epoch)
    validation(epoch)

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
