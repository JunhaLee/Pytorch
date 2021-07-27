#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.init
from torchsummary import summary as summary_
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
from matplotlib import pyplot as plt
import sys
import warnings
warnings.filterwarnings(action='ignore')


# In[2]:


img_path = sys.argv[1]


# In[4]:


#if len(sys.argv) != 2:
#    print("Insufficient arguments")
#    sys.exit()


# In[5]:


print("File path : " + img_path)


# In[6]:


class VGG11(nn.Module):
    def __init__(self,init_weights: bool = True):
        super(VGG11, self).__init__()
        self.convnet = nn.Sequential(
            # Input Channel (RGB: 3)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2), 
        )

        self.fclayer = nn.Sequential(
            nn.Linear(512 * 8 * 8, 4096),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 5),
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.convnet(x)
        x = torch.flatten(x, 1)
        x = self.fclayer(x)
        return x


# In[7]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG11().to(device)
"""
from torchsummary import summary as summary_
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_IDS = [0,1]
model = VGG11().to(device)
summary_(model,(3,256,256),batch_size = 128)
VGG11 = torch.nn.parallel.DataParallel(VGG11, device_ids=DEVICE_IDS)
"""


# In[8]:


model.load_state_dict(torch.load('model20.pt'))
model.eval()


# In[19]:


from torchvision.transforms import ToTensor
X_test = torch.empty(0,3,256,256)
img = Image.open(img_path)
data = ToTensor()(img).unsqueeze(0)
X_test =torch.cat([X_test,data],dim=0)


# In[16]:



with torch.no_grad(): 
    X_test = X_test.to(device)
    result = model(X_test)


# In[17]:


result = torch.argmax(result, 1)


# In[18]:


if result[0] == 0:
    plt.title('Prediction : Hat')
elif result[0] == 1:
    plt.title('Prediction : Outer')
elif result[0] == 2:
    plt.title('Prediction : Top')
elif result[0] == 3:
    plt.title('Prediction : Bottom')
elif result[0] == 3:
    plt.title('Prediction : Shoes')


# In[ ]:


ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.imshow(img)
plt.show() 


# In[ ]:




