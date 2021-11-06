import argparse
import torch.optim as optim
import numpy as np
import json
from time import time
from opacus import PrivacyEngine
import torch.nn as nn
import torchvision
from torchvision import models, datasets, transforms
from torch.optim.lr_scheduler import MultiStepLR
from utils import ImageNetData, load_data, acc, verify_check_points_folder, training_monitor
from Optim import SGD_AGC
from models.models import model_dict
from imagenet import *
from train_utils import TrainOneEpoch
import warnings
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import sys


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.stdout.write('train model on:'+str(device)+'\n')
sys.stdout.flush()

def main(model_name="scatternet_cnn",
         data_path='~/projects/rrg-xihe/dataset/imagenet12',batch_size=256, mini_batch_size=2,
         epochs=10, optim="SGD", momentum=0.9,weight_decay=0,
         lr=1, lr_decay_epoch=[], lr_decay_factor=1,
         noise_multiplier=1, max_grad_norm=1, SGD_AGC_clip = np.inf,
         val_batch_size=64,
         CheckPointPATH='./checkpoint/', checkpoint=None):

    # load the data
    training_params = {
        "shuffle": True,
        "num_workers": 3,
        "batch_size": mini_batch_size
    }

    val_params = {
        "shuffle": True,
        "num_workers": 3,
        "batch_size": val_batch_size
    }
    if batch_size%mini_batch_size != 0:
        warnings.warn('batch_size should be an integral multiple of mini_batch_size')
        return None
    sys.stdout.write('start loading\n')
    sys.stdout.flush()
    train_data, test_data, val_data = load_data(data_path)
    # create dataloaders
    train_data = torch.utils.data.DataLoader(train_data, **training_params)
    val_data = torch.utils.data.DataLoader(val_data, **val_params)
    # test_data = torch.utils.data.DataLoader(test_data, **params)
    sys.stdout.write('Done\n')
    sys.stdout.flush()
    model = model_dict[model_name]
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # initialize optimizer
    if optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay)
    elif optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    else:
        optimizer = SGD_AGC(
                # The optimizer needs all parameter names
                # to filter them by hand later
                named_params=model.named_parameters(),
                lr=lr,
                momentum=momentum,
                clipping=SGD_AGC_clip,
                weight_decay=weight_decay,
                #nesterov=config['nesterov']
            )
    scheduler = MultiStepLR(optimizer, milestones=lr_decay_epoch, gamma=lr_decay_factor)
    privacy_engine = PrivacyEngine(
        model,
        sample_rate=batch_size / len(train_data.dataset),
        alphas=[10, 100],
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    ).to(device)

    # TODO: fix the privacy engine parameters
    privacy_engine.attach(optimizer)

    monitor = training_monitor(set_size=len(train_data.dataset))
    start_epoch=0
    # load check point
    if checkpoint:
        model, optimizer, start_epoch, train_loss = load_checkpoint(checkpoint=CheckPointPATH+checkpoint,
                                                              model=model, optimizer=optimizer,
                                                              epoch=0, train_loss=0)
    # main training loop
    for epoch in range(start_epoch+1, epochs):
        sys.stdout.write('EPOCH: '+str(epoch)+'\n')
        sys.stdout.flush()
        monitor.reset()
        model.train()
        start = time()
        TrainOneEpoch(model=model,
                      criterion=criterion,
                      train_data=train_data,
                      optimizer=optimizer,
                      device=device,
                      monitor=monitor,
                      steps=batch_size//mini_batch_size,
                      )
        end = time()
        scheduler.step()
        train_acc_top1, train_acc_top5, train_loss = monitor.get_acc_loss()
        val_top1_acc, val_top5_acc, val_loss = acc(model, val_data)
        sys.stdout.write('saving\n')
        sys.stdout.flush()
        CheckPoint = {'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'train_loss':train_loss,
                      'train_acc_top1':train_acc_top1,
                      'train_acc_top5':train_acc_top5,
                      'val_loss': val_loss,
                      'val_acc_top1': val_top1_acc,
                      'val_acc_top5': val_top5_acc
                      }
        print('| epoch: ',epoch, '| time: ', end-start,'| train_loss: ',train_loss,'| train_acc_top1: ',train_acc_top1,'| train_acc_top5: ',train_acc_top5,'| val_loss: ',val_loss,'| val_top1_acc: ',val_top1_acc,'| val_top5_acc: ',val_top5_acc,'|')
        sys.stdout.flush()
        torch.save(CheckPoint, CheckPointPATH + model_name + 'epoch' + str(epoch) + '.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="scatter_resnet", choices=["alexnet", "scatter_resnet", "resnet18"])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--mini_batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--optim', type=str, default="SGD", choices=["SGD", "Adam", 'SGD_AGC'])
    parser.add_argument('--SGD_AGC_clip', type=float, default=999999)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_decay_epoch', nargs='+', help='<Required> Set flag', required=False, default=[])
    parser.add_argument('--lr_decay_factor', type=float, default=0.1)

    parser.add_argument('--noise_multiplier', type=float, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=1)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--data_path',type=str,default='~/salon/large-scale-dpsgd/data')
    parser.add_argument('--CheckPointPATH', type=str, default=' ')
    parser.add_argument('--checkpoint', type=str, default=None)

    #parser.add_argument('--data_path',type=str,default='~/projects/rrg-xihe/dataset/imagenet12')
    args = parser.parse_args()
    verify_check_points_folder()
    main(**vars(args))
