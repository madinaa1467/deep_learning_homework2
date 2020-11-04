
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from io import open
import glob
import os

import pandas as pd

from os import listdir
from PIL import Image as PImage

import matplotlib.pyplot as plt 
import matplotlib.image as img


# In[37]:


def findFiles(path): return glob.glob(path)

def createLabels(path, outputFile, isTrain):
    classes = ['airplane', 'bird', 'dog', 'frog', 'horse']
    df = pd.DataFrame(columns=['id', 'label'])
    
#     print(findFiles(path + '*.png'))
    # for i, class in enumerate(classes):
    
    for filename in findFiles(path + '/*.png'):
        
        file = os.path.basename(filename)
        img_class = os.path.basename(filename).split('_')[1].split('.')[0]
        
        if isTrain:
            file = os.path.basename(os.path.dirname(filename)) + '/' + file
        
        df = df.append({'id': file, 'label': classes.index(img_class)}, ignore_index=True)

    df.to_csv(outputFile)
#     return df


createLabels(r'data/data1/train/*/', r'data/data1/train.csv', True)
createLabels(r'data/data1/test/', r'data/data1/test.csv', False)


# In[38]:


# print(train_labels)


# In[39]:


class MyDataset(Dataset):
    def __init__(self, data, path , transform = None):
#         super().__init__()
        self.data = data.values
        self.path = path
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        img_name,label = self.data[index]
        img_path = os.path.join(self.path, img_name)
#         image = img.imread(img_path)
        image = PImage.open(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# In[40]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
#         declare layers here  [64, 3, 32, 32]
#         self.conv1 = nn.Conv2d(3, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.fc1 = nn.Linear(5*5*64, 128)
#         self.fc2 = nn.Linear(128, 10)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(720, 1024)
        self.fc2 = nn.Linear(1024, 5)

    def forward(self, x):
        # define forward propagation here
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2)
#         x = x.view(-1, 5*5*64)
#         x = self.fc1(x)
#         x = self.fc2(x)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# In[67]:


def validate(model, device, validation_loader, criterion):
    model.eval()
    valid_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
#             valid_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            valid_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    valid_loss /= len(validation_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, len(validation_loader.dataset),
        100. * correct / len(validation_loader.dataset)))
    
    return valid_loss
        


# In[69]:


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
#         loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return loss.item()
            
def loadDatabase():
    
    train_path = r'data/data1/train/'
    test_path = r'data/data1/test/'

    train_labels = pd.read_csv(r'data/data1/train.csv', index_col=[0])
    test_labels = pd.read_csv(r'data/data1/test.csv', index_col=[0])

    train_data, validation_data = train_test_split(train_labels, stratify=train_labels.label, test_size=0.1)
    
    train_data = MyDataset(train_data, train_path, transforms.ToTensor() )
    validation_data = MyDataset(validation_data, train_path, transforms.ToTensor() )
    test_data = MyDataset(test_labels, test_path, transforms.ToTensor() )


    train_loader = DataLoader(train_data, batch_size=64, num_workers=0, shuffle=True)
    validation_loader = DataLoader(train_data, batch_size=64, num_workers=0, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000, num_workers=0, shuffle=True)
    
    return train_loader, validation_loader, test_loader
    
def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, validation_loader, test_loader = loadDatabase()

    model = Net().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
    criterion = nn.CrossEntropyLoss()

    counter, train_losses, valid_losses = [], [], []

    for epoch in range(1, 35 + 1):
        
        train_losses.append( train(model, device, train_loader, optimizer, criterion, epoch) )
        valid_losses.append( validate(model, device, validation_loader, criterion) )
        counter.append(epoch)
    
    plt.figure(figsize=(9, 6))
    plt.ylabel("Loss")
    plt.xlabel("Number of Epochs")
    plt.plot(counter, train_losses, "r", label = "Train loss")
    plt.plot(counter, valid_losses, "b", label = "Validation loss")
    plt.title("Loss")
    plt.show()

if __name__ == '__main__':
    main()

