# -*- coding: utf-8 -*-
"""373Final.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1EnFwJpvUX69lt3MgNyo_jlO0JUMsV7cS
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
import sklearn.model_selection
import torch
import cv2

from google.colab import drive
drive.mount('/content/gdrive')

#training data
data_labeled = np.load('/content/gdrive/MyDrive/cellsCountDataZip/counting-cells-in-microscopy-images-2024/train_data.npz')

#checks how well we did... use later
test_images = np.load('/content/gdrive/MyDrive/cellsCountDataZip/counting-cells-in-microscopy-images-2024/test_images.npz')

X_labeled = data_labeled['X'] #then split into x train and xval  #numpy array
y_labeled = data_labeled['y'] #then split into ytrain and yval
#use xtrain to predict ytrain.. use xlabeled to predict ylabeled

#splitting the data into training and validation sets
X_train, X_val = sk.model_selection.train_test_split(X_labeled, train_size=.8)
y_train, y_val = sk.model_selection.train_test_split(y_labeled, train_size=.8)

#what is test images?

i = np.random.randint(len(X_train))
x = X_train[i]
y = y_train[i]
print(x.shape)
plt.figure(figsize=(3,3))
plt.imshow(y, cmap='gray')

class CellsDataset():
  def __init__(self, X, y):
    # X and y are numpy arrays
    # X contains a bunch of images (Maybe 1600 images, for example)
    # y contains corresponding target images
    self.X = X
    self.y = y

  def __len__(self):
    return len(self.X)

  def __getitem__(self, i):

    x = self.X[i]/255.0  # x is the ith image in our dataset
    y = self.y[i]
    x = x.reshape((1, 128, 128))
    y = y.reshape((1, 128, 128))

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return x, y

class CellsDatasetTest():
  def __init__(self, X):
    # X and y are numpy arrays
    # X contains a bunch of images (Maybe 1600 images, for example)
    # no labels in test data
    self.X = X

  def __len__(self):
    return len(self.X)

  def __getitem__(self, i):

    x = self.X[i]/255.0
    x = x.reshape((1, 128, 128))

    x = torch.tensor(x, dtype=torch.float32)

    return x

#class, dataloader objects
dataset_train = CellsDataset(X_train, y_train)
dataset_val = CellsDataset(X_val, y_val)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True) #why only x and not y? #grabs a batch of 16 x_train obs
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=16, shuffle=False)

x_batch, y_batch = next(iter(dataloader_train))

class UnetCNN(torch.nn.Module): #change this to Unet
  def __init__(self):
    super().__init__()
                            # depth, num filters, 3x3 filter, 0 padding as to maintain size
    self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, padding='same') #does the input have a depth of 1 or 16?
    self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding='same')
    self.conv2_1 = torch.nn.Conv2d(64, 64, kernel_size=3, padding='same')
    self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding='same') #128 filters with depth of 64
    self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3, padding='same')
    self.conv4_1 = torch.nn.Conv2d(128, 128, kernel_size=3, padding='same')
    self.conv5 = torch.nn.Conv2d(128, 256, kernel_size=3, padding='same')
    self.conv6 = torch.nn.Conv2d(256, 256, kernel_size=3, padding='same')
    self.conv6_1 = torch.nn.Conv2d(256, 256, kernel_size=3, padding='same')
    self.conv7 = torch.nn.Conv2d(256, 512, kernel_size=3, padding='same')
    self.conv8 = torch.nn.Conv2d(512, 512, kernel_size=3, padding='same')
    self.conv8_1 = torch.nn.Conv2d(512, 512, kernel_size=3, padding='same')
    self.conv9 = torch.nn.Conv2d(512, 1024, kernel_size=3, padding='same')
    self.conv10 = torch.nn.Conv2d(1024, 1024, kernel_size=3, padding='same')
    self.conv15 = torch.nn.Conv2d(64, 1, kernel_size=3, padding='same')
#upconv:
    self.conv11 = torch.nn.Conv2d(1024, 512, kernel_size=3, padding='same')
    self.conv11_1 = torch.nn.Conv2d(1024, 512, kernel_size=3, padding='same')
    self.conv12 = torch.nn.Conv2d(512, 256, kernel_size=3, padding='same')
    self.conv12_1 = torch.nn.Conv2d(512, 256, kernel_size=3, padding='same')
    self.conv13 = torch.nn.Conv2d(256, 128, kernel_size=3, padding='same')
    self.conv13_1 = torch.nn.Conv2d(256, 128, kernel_size=3, padding='same')
    self.conv14 = torch.nn.Conv2d(128, 64, kernel_size=3, padding='same')
    self.conv14_1 = torch.nn.Conv2d(128, 64, kernel_size=3, padding='same')


    #self.dense1 = torch.nn.Linear(64*14*14, 10)

    self.relu = torch.nn.ReLU()
    self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    self.upsample = torch.nn.Upsample(scale_factor=2) #upsampling
    #directly use torch.cat , doesnt need to be in the constrictor ie.) x = torch.cat((x, x2), 0)

    #unet needds forwards which contains all opertaions we perform
    #upconv, copy and crop
      #up conv has 2 setps: upsampling (doubles like reverse max pool)0and apply apply conv layer
    #dims: input has 1x128x128
    #conv 1, 2 uses 64

##every upconv need to be concatenated!

  def forward(self, x): #specified the exact calulation that our nn performs
    # import pdb
    # pdb.set_trace()

    x = self.conv1(x)
    x = self.relu(x) #sets any negative enteries to 0 #64,128,128

    x = self.conv2(x)
    x1 = self.relu(x) #64,128,128

    x = self.maxpool(x1) #after this step we have size 64x64x64
    ####
    x = self.conv3(x)
    x = self.relu(x) #128,64,64

    x = self.conv4(x)
    x2 = self.relu(x) #128,64,64

    x = self.maxpool(x2) #128x32x32
    ###
    x = self.conv5(x) #256,32,32
    x = self.relu(x)

    x = self.conv6(x)
    x3 = self.relu(x) #256,32,32

    x = self.maxpool(x3) #256x16x16
    ###
    x = self.conv7(x)#512,16,16
    x = self.relu(x)

    x = self.conv8(x)
    x4 = self.relu(x)#512,16,16

    x = self.maxpool(x4) #512x8x8
    ###
    x = self.conv9(x) #1024,8,8
    x = self.relu(x)

    x = self.conv10(x)
    x = self.relu(x)#1024,8,8

    x = self.upsample(x) #1024x16x16 #torch.nn.Upsample()
    x = self.conv11(x)


    xA = self.relu(x) #512,16,16
    #relu???????? when upconv? ^?
    #concat depthwise: xA + x4

    ##concat!! #1024,16,16

    x = torch.cat((xA,x4), dim = 1)#

    #first step after the first concat:
    x = self.conv11_1(x)
    x = self.relu(x)#512,16,16


    x = self.conv8_1(x)
    x = self.relu(x) #512,16,16
    ##upsample
    x = self.upsample(x) #512x32x32
    #upconv:
    x = self.conv12(x)
    xB = self.relu(x)#256,32, 32
    #relu?^

    #concat xb and x3 #512x32x32
    x = torch.cat((xB,x3), dim = 1)

    ###
    x = self.conv12_1(x)
    x = self.relu(x) #256,32,32

    x = self.conv6_1(x)
    x = self.relu(x) #256,32,32

    #upsample:
    x = self.upsample(x) #256x64x64
    #upconv; star c
    x = self.conv13(x)
    xC = self.relu(x) #128,64,64

    #concat xc and x2; should be 256x64x64
    x = torch.cat((xC,x2), dim = 1)

    # x = torch.cat((xc,x2), dim = 1)

    x = self.conv13_1(x) #128x64x64
    x = self.relu(x)

    x = self.conv4_1(x)
    x = self.relu(x)#128x64x64

    #upsample:
    x = self.upsample(x) #128x128x128
    #upconv
    x = self.conv14(x)
    xD = self.relu(x) #64,128,128

    #concat: xd and x1 #128,128,128
    x = torch.cat((xD,x1), dim = 1)


    x = self.conv14_1(x)
    x = self.relu(x) #64x128x128

    x = self.conv2_1(x)
    x = self.relu(x)#64x128x128

    x = self.conv15(x) #1, 128, 128
    return x


#sql practice problems
#python and pandas // be able to do a kagglwe contest ... like housing prices, read in csv, handle missing data. feature engineer, rando, forest ,

#main???

# softmax = torch.nn.Softmax() # should be sigmoid
sigmoid = torch.nn.Sigmoid()
#stochasic gradient descent
model = UnetCNN() #change to uNet
device = torch.device('cuda') #gpu noises
#device = torch.device('cpu')
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=.003)
#loss_fun = torch.nn.CrossEntropyLoss() #is this what i want?
loss_fun = torch.nn.BCEWithLogitsLoss()


#training loop!
num_epochs = 6
ace_vals_train = []
ace_vals_val = []

for ep in range(num_epochs):
  print(f'ep is: {ep}')
  for x_batch, y_batch in dataloader_train:
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    outputs = model(x_batch) #is outputs a 16x1 col vector describing 16 images? i need to consider one at a time and round?
    loss = loss_fun(outputs, y_batch) #bug here :Target 1 is out of bounds.
    model.zero_grad()
    loss.backward()
    optimizer.step()

  #with an
  with torch.no_grad():
    ace = 0
    #4 send ola pync
    for x_batch, y_batch in dataloader_train:
      x_batch = x_batch.to(device)
      y_batch = y_batch.to(device)
      outputs = model(x_batch)
      loss = loss_fun(outputs, y_batch)

      ace = ace + loss * len(y_batch)

    #2acc, 3ace

    ace = ace / len(dataset_train)
    ace = ace.item()
    ace_vals_train.append(ace)
    #print ace and acc_train
    print(f'Average cross entropy (training): {ace}')


    #validation loop
    ace = 0
    #4send ola pync
    for x_batch, y_batch in dataloader_val:
      x_batch = x_batch.to(device)
      y_batch = y_batch.to(device)
      outputs = model(x_batch)
      loss = loss_fun(outputs, y_batch)
      ace = ace + loss * len(y_batch)

    #2acc 3ace pp

    ace = ace / len(dataset_val)
    ace = ace.item()
    ace_vals_val.append(ace)

    print(f'Average cross entropy (validation): {ace}')



#bro why is she so bad? why is she getting worse?
#monday ml club oh:6:30

plt.figure()
plt.plot(ace_vals_train)
plt.figure()
plt.plot(ace_vals_val)

sigmoid = torch.nn.Sigmoid()

with torch.no_grad():
  x_batch, y_batch = next(iter(dataloader_val))
  x_batch = x_batch.to(device)
  y_pred = sigmoid(model(x_batch))
  y_pred = y_pred.cpu().numpy()
  y_batch = y_batch.numpy()
  i = np.random.randint(len(x_batch))
  img = x_batch[i].cpu()[0]
  plt.figure(figsize=(2,2))
  plt.imshow(img, cmap='gray')

  plt.figure(figsize=(2,2))
  plt.imshow(y_batch[i][0])
  plt.title('ground truth labels')

  plt.figure(figsize=(2,2))
  plt.imshow(y_pred[i][0], cmap='gray')
  plt.title('predicted probabilities')

#after training, when considering test data, we round the output of our trained model!

#what is the name of our trained model? // how to access?.. outputs!


dataset_test = CellsDatasetTest(test_images['X'])

dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=16, shuffle=False)

x_batch = next(iter(dataloader_test))#KeyError: '0 is not a file in the archive'

cell_counts = []
with torch.no_grad():
  for x_batch in dataloader_test:
        x_batch = x_batch.to(device)
        #y_batch = y_batch.to(device)
        outputs = model(x_batch)
        probabilities = sigmoid(outputs)
        labels = torch.round(probabilities) #rounded output

        for i in range(len(x_batch)):
          labels_i = labels[i].cpu().numpy().astype('uint8')[0]

          info = cv2.connectedComponents(labels_i)
          count = info[0]
          cell_counts.append(count)

cell_counts