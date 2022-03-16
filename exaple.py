# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import torch as T
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 
import numpy as np

train_transform = train_transforms = transforms.Compose([transforms.Resize(48),
                                       transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(32),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5],
                                                            [0.5, 0.5, 0.5])])
train_data = datasets.ImageFolder('./train', transform=train_transform)
print(train_data)