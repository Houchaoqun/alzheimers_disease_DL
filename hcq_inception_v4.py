# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

## refercen:
## https://github.com/pytorch/vision/pull/43/commits/59197ef1663560f52efb02f36a0eeb6474a30499

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torchvision import models
from model_inception_v4 import inceptionv4
from utils import progress_bar
import time
import copy

data_transforms = {
    'train': transforms.Compose([
            transforms.Resize(340),
            transforms.RandomCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  ## 3 channels -> RGB image 
    ]),
    'validation': transforms.Compose([
            transforms.Resize(340),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

batch_size = 20
num_epochs = 200
num_class = 2

dataset_fold_name = 'single_subject_data_fold_01_train_val_test_entropy_keep_SliceNum_81'
model_save_path = os.path.join('./pytorch_model', dataset_fold_name + '.t7')
data_dir = os.path.join('/home/reserch/documents/deeplearning/alzheimers_disease/ADNI-825-Slice/experiments_FineTunning', dataset_fold_name)
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'validation']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True if x=='train' else False, num_workers=4)
              for x in ['train', 'validation']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}

class_names = image_datasets['train'].classes
print("train dataset_sizes = {}".format(dataset_sizes['train']))
print("validation dataset_sizes = {}".format(dataset_sizes['validation']))
print("class_names = {}".format(class_names))

use_gpu = torch.cuda.is_available()
model = inceptionv4(pretrained=True)
print("[DONE] inceptionv4 pretrained imagenet.")

## transfer learning --> keras, inception-v4
# net_ft = AveragePooling2D((8,8), border_mode='valid')(net)
# net_ft = Dropout(dropout_keep_prob)(net_ft)
# net_ft = Flatten()(net_ft)
# predictions_ft = Dense(output_dim=num_classes, activation='softmax')(net_ft)

model.classif = nn.Sequential(
        # ### fc 1
        # nn.Linear(1536 * 1 * 1, 512),
        # nn.ReLU(True),
        # nn.Dropout(p=0.2),
        
        ### predictions
        nn.Linear(1536 * 1 * 1, num_class))

if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()

## optimize features and classifier with different learning rate.
params = [{'params': model.features.parameters(), 'lr': 1e-5},
          {'params': model.classif.parameters(), 'lr': 1e-3}
         ]

optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay = 5e-4)  ## 
# optimizer = optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), weight_decay = 3e-4)

# Decay LR by a factor of 0.1 every step_size epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


### train 
###

best_acc = 0
best_epoch = 0
best_model_wts = copy.deepcopy(model.state_dict())

def train(epoch):
    
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('*' * 20)
    
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
        train_loss += loss.data[0]
        total += batch_size
        correct += preds.eq(targets.data).cpu().sum()
        
        progress_bar(batch_idx, len(dataloaders['train']), 'train info: Acc: %.3f%% (%d/%d) | Loss: %.3f'
            % ( 100.*correct/total, correct, total, 
               train_loss/(batch_idx+1))) 

### validation
###

def validation(epoch):
    global best_acc
    global best_epoch

    model.eval()

    val_loss = 0
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
        
        val_loss += loss.data[0]
        total += batch_size
        correct += preds.eq(targets.data).cpu().sum()
        
        progress_bar(batch_idx, len(dataloaders['validation']), 'val info: Acc: %.3f%% (%d/%d) | Loss: %.3f'
            % ( 100.*correct/total, correct, total, val_loss/(batch_idx+1))) 
        
    acc = 100.*correct/total
    # print("acc = 100*{}/{}".format(correct, total))
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
        


    
### main
### train and validation
since = time.time()

for epoch in range(num_epochs):
    exp_lr_scheduler.step()
    train(epoch)
    validation(epoch)

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))