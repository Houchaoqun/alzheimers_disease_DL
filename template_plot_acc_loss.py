# -*- coding: utf-8 -*-
### reference: https://www.cnblogs.com/hanbb/p/7846452.html
### author: houchaoqun
### data: 20180312

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

def _plot_it():

    ## acc
    train_acc_list = [52.2, 63.2, 65.5, 67.8, 69.5, 72.9, 77.2, 83.2, 87.2, 92.1, 93.1, 95.2]
    val_acc_list = [50.2, 60.2, 62.5, 65.8, 66.5, 70.9, 75.2, 80.2, 85.2, 90.1, 90.1, 90.2]    
    plt.figure(figsize=(10, 6))       # 确定图片大小
    plt.subplot(211)                  # 确定第1个图的位置

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
    train_loss_list = [52.2, 63.2, 65.5, 67.8, 69.5, 72.9, 77.2, 83.2, 87.2, 92.1, 93.1, 95.2]
    val_loss_list = [50.2, 60.2, 62.5, 65.8, 66.5, 70.9, 75.2, 80.2, 85.2, 90.1, 90.1, 90.2]
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
    _plot_it()