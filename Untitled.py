
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


# In[2]:


# test5
# 909
# 888


# In[3]:


def findFiles(path): return glob.glob(path)

def loadDatabase(path, outputFile, isTrain):
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
    return df

# test = loadDatabase(r'data/test/', 'test')

train_labels = loadDatabase(r'data/data1/train/*/', r'data/data1/train.csv', True)
test_labels = loadDatabase(r'data/data1/test/', r'data/data1/test.csv', False)

train_path = r'data/data1/train/'
test_path = r'data/data1/test/'


# In[4]:


# train_labels = pd.read_csv(r'data/data1/train.csv')
# test_labels = pd.read_csv(r'data/data1/test.csv')


# In[5]:


# train_labels['label'].value_counts()

# classes = ['airplane', 'bird', 'dog', 'frog', 'horse']

# plt.figure(figsize = (8,8))
# plt.pie(train_labels.groupby('label').size(), labels = classes, autopct='%1.1f%%', shadow=True, startangle=90)
# plt.show()


# In[46]:


fig,ax = plt.subplots(1,5,figsize = (15,3))

for i,idx in enumerate(train_labels[train_labels['label'] == 0]['id'][-5:]):
    path = os.path.join(train_path,idx)
    print(path)
    ax[i].imshow(img.imread(path))
#     plt.imshow(PImage.open(path))
#     plt.show()


# In[45]:


# fig,ax = plt.subplots(1,5,figsize = (15,3))

# for i,idx in enumerate(train_labels[train_labels['label'] == 0]['id'][-5:]):
#     path = os.path.join(train_path,idx)
#     print(path)
#     ax[i].imshow(img.imread(path))
    
# fig,ax = plt.subplots(1,5,figsize = (15,3))
# for i,idx in enumerate(train_labels[train_labels['label'] == 1]['id'][:5]):
#     path = os.path.join(train_path,idx)
#     ax[i].imshow(img.imread(path))

def imshow(image, ax=None, title=None, normalize=True):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax
    
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
        image = img.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# In[26]:


means = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Normalization
train_transform = transforms.Compose([transforms.ToPILImage(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(means,std)])

test_transform = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(means,std)])

valid_transform = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(means,std)])

# Splitting the Dataset
train_data, valid_data = train_test_split(train_labels, stratify=train_labels.label, test_size=0.1)

train_data = MyDataset(train_data, train_path, train_transform )
valid_data = MyDataset(valid_data, train_path, valid_transform )
test_data = MyDataset(test_labels, test_path, test_transform )


# In[27]:


# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.ToTensor()
# ])

# batch_size = 25 

# # dataloader = torch.utils.data.DataLoader(*torch_dataset*, batch_size=4096, shuffle=False, num_workers=4)
# train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle=True, num_workers=0)
# valid_loader = DataLoader(dataset = valid_data, batch_size = batch_size, shuffle=False, num_workers=0)
# test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle=False, num_workers=0)



# pop_mean = []
# pop_std0 = []
# pop_std1 = []
# for i, data in enumerate(train_loader, 0):
#     # shape (batch_size, 3, height, width)
#     numpy_image = data['image'].numpy()
    
#     # shape (3,)
#     batch_mean = np.mean(numpy_image, axis=(0,2,3))
#     batch_std0 = np.std(numpy_image, axis=(0,2,3))
#     batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)
    
#     pop_mean.append(batch_mean)
#     pop_std0.append(batch_std0)
#     pop_std1.append(batch_std1)

# # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
# pop_mean = np.array(pop_mean).mean(axis=0)
# pop_std0 = np.array(pop_std0).mean(axis=0)
# pop_std1 = np.array(pop_std1).mean(axis=0)


# # Normalization
# train_transform = transforms.Compose([transforms.ToPILImage(),
#                                       transforms.ToTensor(),
#                                       transforms.Normalize(pop_mean,pop_std0)])

# # test_transform = transforms.Compose([transforms.ToPILImage(),
# #                                      transforms.ToTensor(),
# #                                      transforms.Normalize(means,std)])

# # valid_transform = transforms.Compose([transforms.ToPILImage(),
# #                                      transforms.ToTensor(),
# #               
# # Splitting the Dataset
# train_data, valid_data = train_test_split(train_labels, stratify=train_labels.label, test_size=0.1)

# train_data = MyDataset(train_data, train_path, train_transform )
# valid_data = MyDataset(valid_data, train_path, valid_transform )
# test_data = MyDataset(test_labels, test_path, test_transform )



# In[64]:


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


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    
    train_loss = 0.0
    valid_loss = 0.0
#     for data, target in train_loader:
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        
        
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
def loadDatabase():
    train_dataset = datasets.ImageFolder(
        root=train_path,
        transform=transforms.ToTensor()
    )
    test_dataset = datasets.ImageFolder(
        root=test_path,
        transform=transforms.ToTensor()
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    test_loader = DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return train_loader, test_loader

# for batch_idx, (data, target) in enumerate(load_dataset()):
#     print(batch_idx)
    
def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = 25 
    
    train_loader, test_loader = loadDatabase()

    model = Net().to(device)
#     optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(1, 10+ 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

if __name__ == '__main__':
    main()


# In[48]:


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# In[58]:


# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")

# # traindir = os.path.join(args.data, 'train')
# # valdir = os.path.join(args.data, 'val')
# # testdir = os.path.join(args.data, 'test')
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# train_loader = DataLoader(
#     datasets.ImageFolder(train_path,
#                          transforms.Compose([
#                              transforms.RandomSizedCrop(224),
#                              transforms.RandomHorizontalFlip(),
#                              transforms.ToTensor(),
#                              normalize,
#                          ])),
#     batch_size=10,
#     shuffle=True,
#     num_workers=0,
#     pin_memory=True)

# # val_loader = data.DataLoader(
# #     datasets.ImageFolder(valdir,
# #                          transforms.Compose([
# #                              transforms.Scale(256),
# #                              transforms.CenterCrop(224),
# #                              transforms.ToTensor(),
# #                              normalize,
# #                          ])),
# #     batch_size=args.batch_size,
# #     shuffle=True,
# #     num_workers=args.workers,
# #     pin_memory=True)

# # test_loader = DataLoader(
# #     TestImageFolder(test_path,
# #                     transforms.Compose([
# #                         transforms.Scale(256),
# #                         transforms.CenterCrop(224),
# #                         transforms.ToTensor(),
# #                         normalize,
# #                     ])),
# #     batch_size=1,
# #     shuffle=False,
# #     num_workers=0,
# #     pin_memory=False)

# batch_size = 25 # tran = 64 , test = 1000
# #     shuffle=True # test = True
    
# # train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle=True, num_workers=0)
# # valid_loader = DataLoader(dataset = valid_data, batch_size = batch_size, shuffle=False, num_workers=0)
# # test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle=False, num_workers=0)

# model = Net().to(device)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# for epoch in range(1, 2):
#     train(model, device, train_loader, optimizer, epoch)
# #     test(model, device, test_loader)


# In[61]:


def load_dataset():
    train_dataset = datasets.ImageFolder(
        root=train_path,
        transform=transforms.ToTensor()
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return train_loader

# for batch_idx, (data, target) in enumerate(load_dataset()):
#     print(batch_idx)
    
def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = 25 
    
    train_loader = load_dataset()

    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(1, 2):
        train(model, device, train_loader, optimizer, epoch)
#         test(model, device, test_loader)

if __name__ == '__main__':
    main()


# In[40]:


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    
#     train_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('../data', train=True, download=True,
#                        transform=transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))
#                        ])),
#         batch_size=64, shuffle=True)
    
#     test_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('../data', train=False, transform=transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))
#                        ])),
#         batch_size=1000, shuffle=True)

#     num_epochs = 35
#     num_classes = 5
#     learning_rate = 0.001
    
    batch_size = 25 # tran = 64 , test = 1000
#     shuffle=True # test = True
    
    train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset = valid_data, batch_size = batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle=False, num_workers=0)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(1, 2):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

if __name__ == '__main__':
    main()


# In[ ]:


list = [2, 3, 4, 5]
a, b, c = list

