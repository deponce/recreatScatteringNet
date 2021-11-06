'''
Dataloader for imagenet
'''

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from imagenet import *
import torch.nn as nn
import os
import warnings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class training_monitor():
    def __init__(self, set_size=100):
        self._set_size = set_size
        self._top1cnt = 0
        self._top5cnt = 0
        self._training_loss = 0
    def reset(self):
        self._top1cnt = 0
        self._top5cnt = 0
        self._training_loss = 0
    def update(self,model_output, target, loss):
        self._training_loss += loss.item()
        top_5 = torch.topk(model_output,dim=1,k=5).indices
        top_5_diff = (top_5-target[:,None])<=1
        self._top5cnt += torch.sum(top_5_diff)
        top_1_diff = (top_5[:,0]-target)<=1
        self._top1cnt += torch.sum(top_1_diff)
    def avg_loss(self):
        return self._training_loss/self._set_size
    def top1accuracy(self):
        return self._top1cnt/self._set_size
    def top5accuracy(self):
        return self._top5cnt/self._set_size
    def get_acc_loss(self):
        return [self.top1accuracy(),self.top5accuracy(),self.avg_loss()]

class ImageNetData(Dataset):
    def __init__(self, samples=None, targets=None, transform=None, augment=False, height=256, width=256, C_in=3):
        self.samples = samples
        self.targets = targets
        self.transform = transform
        self.augment = augment
        self.height = height
        self.width = width
        self.C_in = C_in
    def __len__(self):
        return self.samples.shape[0]
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.transform is not None:
            # should be a single sample at a time here
            sample = sample.reshape(self.height, self.width, self.C_in)
            sample = self.transform(sample)
        # recasting
        sample = np.array(sample).astype(np.double)
        sample = torch.from_numpy(sample)
        # if we're augmenting the data, then the PIL images
        # will return the domain of the image to [0,255]
        # so we need to renormalize each sample
        if self.augment:
            return (sample / 255.) - 0.5, self.targets[idx]
        # on the other hand, if we are not augmenting, there is no need for this additional normalization
        return sample, self.targets[idx]


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    

def load_data(filepath= './data'):
    # return the different datasets
    #filepath = './data'
    transform = transforms.Compose(
        [transforms.Resize(256),transforms.CenterCrop(224),
         transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    val_dataset = ImageNet(root=filepath, split='val',transform=transform)
    train_dataset = ImageNet(root=filepath, split='train',transform=transform)
    return train_dataset, None, val_dataset


def acc(model, dataloader, test=False):
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    top1cnt = 0
    top5cnt = 0
    loss_cnt = 0
    for idx,(batch, target) in enumerate(dataloader):
        batch, target = batch.to(device), target.to(device)
        outputs = model(batch)
        top_5 = torch.topk(outputs, dim=1, k=5).indices
        top_5_diff = (top_5 - target[:, None])<=1
        top5cnt += torch.sum(top_5_diff)
        top_1_diff = (top_5[:, 0] - target)<=1
        top1cnt += torch.sum(top_1_diff)
        loss = criterion(outputs, target).reshape(1)
        loss_cnt += loss.item()
    avg_loss = loss_cnt/len(dataloader.dataset)
    top1_accuracy = top1cnt/len(dataloader.dataset)
    top5_accuracy = top5cnt/len(dataloader.dataset)

    return top1_accuracy,top5_accuracy,avg_loss

def verify_check_points_folder(root='./', path='checkpoint/'):
    if os.path.isdir(root+path):
        print('check points folder is exist')
        if os.listdir(root+path):
            warnings.warn('check points folder is not empty')
    else:
        print('creat a check points folder:', root+path)
        os.mkdir(root+path)
    return None

def load_checkpoint(checkpoint, model, optimizer, epoch, train_loss):
    try:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_acc_top1']
    except:
        warnings.warn('the checkpoint file did not exist')
    return model, optimizer, epoch, train_loss
