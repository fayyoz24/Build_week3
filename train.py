from torchvision.transforms import RandomHorizontalFlip, RandomCrop, ToTensor, Grayscale
from torch.utils.data import WeightedRandomSampler, DataLoader, random_split
from sklearn.metrics import classification_report
import config as cfg
from utils import Early_stopping, LRScheduler
from torchvision import transforms, datasets
import model
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import pandas as pd
import torch.nn as nn
import torch
import math
from torch.optim import SGD


# configure the device to use for training model, gpu or cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'\n\n[INFO] Current training device: {device}')

# initialize a list of preprocessing steps to apply on each image during
# training/validating and testing

train_tarnsform = transforms.Compose([
            Grayscale(num_output_channels=1),
            RandomCrop((48,48)),
            RandomHorizontalFlip(),
            ToTensor()
])

test_transform = transforms.Compose([
            Grayscale(num_output_channels=1),
            ToTensor()
])

# loading all the images within the specified folder and apply different augmentation
train_data = datasets.ImageFolder(cfg.train_directory, transform=train_tarnsform)
test_data = datasets.ImageFolder(cfg.test_directory, transform=test_transform)

# extract the class labels and total number of classes
classes = train_data.classes
num_of_classes = len(classes)
print(f'[INFO] Class labels: {classes}'
      f'\nNumber of {num_of_classes}')

# use train samples to generate train/validation set
num_train_samples = len(train_data)
train_size = math.floor(num_train_samples * cfg.train_size)
val_size = math.ceil(num_train_samples * cfg.val_size)
print(f'[INFO] Train samples: {train_size} ...\t Validation samples: {val_size}...')

# randomly splti the training dataset intp train and validation set
train_data, val_data = random_split(train_data, [train_size, val_size])

# modify the data transform applied towards the validation set
val_data.dataset.transforms = test_transform

# get the labels in train data
train_classes = [label for _, label in train_data]
# print(train_classes)
# count each labels within each classes
class_count = Counter(train_classes)
print(f'[INFO] Total samples: {class_count}')

# Compute and determine the weights to be applied on each category
# depending on the number of samples available
class_weight = torch.Tensor([len(train_classes) / c for c in pd.Series(class_count).sort_index().values])

"""
Initialize a placeholder for each target image and iterate via the train dataset,
get the weights for each class and modify the default sample weight to its
corresponding class weight already computed
"""
sample_weight = [0] * len(train_data)
for idx, (image, label) in enumerate(train_data):
      weight = class_weight[label]
      sample_weight[idx] = weight

#  define a sampler which randomly sample labels from the train dataset
sampler = WeightedRandomSampler(weights=sample_weight, 
                              num_samples=len(train_data),
                              replacement=True)

# load our own dataset and store each sample with their corresponding labels
train_dataloader = DataLoader(train_data, batch_size=cfg.batch_size, sampler=sampler)
val_dataloader = DataLoader(val_data, batch_size=cfg.batch_size)
test_dataloader = DataLoader(test_data, batch_size=cfg.batch_size)

# Initialize the model and send it ot device
model = model.Emotion(num_of_channels=1, num_of_classes=num_of_classes)
model = model.to(device)

# Initilaize our optimizer and loss function
optimzer = SGD(model.parameters(), cfg.lr)
criterion = nn.CrossEntropyLoss()

# Initialize the learning rate scheduler and early stopping mechanism
lr_scheduler = LRScheduler(optimzer)
early_stopping = Early_stopping()

# calculate the steps per epoch for training and validation set
train_steps = len(train_dataloader.dataset) // cfg.batch_size
val_steps = len(val_dataloader) // cfg.batch_size

# Initialize a dictionary to save the training history
history = {
      "train accuracy" : [],
      "train LOSS" : [],
      "val accuracy" : [],
      "val LOSS" : [],
}

# Iterate through the epochs
print(f'[INFO] Training the model...')
start_time = datetime.now()

for epoch in range(0, cfg.num_of_epochs):
      print(f'[INFO] epoch: {(epoch + 1)} / {cfg.num_of_epochs}')
      """
      Training the model
      """
      # setthe model to training mode
      model.train()
      """
      Initilaize the total training and validation loss and
      the total number of prediztions in both steps
      """
      total_train_loss = 0
      total_val_loss = 0
      train_correct = 0
      val_correct = 0
      # iterate through the training set
      for i, (data, target) in enumerate(iter(train_dataloader)):
            # move the data into the device used for training 
            data, target = data.to(device), target.to(device)

            # perform a forward pass and calculate the training loss
            predictions = model(data)
            loss = criterion(predictions, target)
            """
            Zero the gradients accumulated from the previous operation,
            perform a backward pass, then update the model parameters
            """
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()

            # add the training loss and keep track of the number of correct predictions
            total_train_loss += loss
            train_correct += (predictions.argmax(1) == target).type(torch.float).sum().item()
            if i % 30 == 0:
                  print(f'iterations: {i}'
                        f'Train_loss: {total_train_loss/(i+1)}')

      """
      Validating the model
      """
      model.eval() # disable dropout and dropout layers
      """
      prevents pytorch from calculating the gradients, reducing
      memory usage and speeding up the computation time (no back prop)
      """
      with torch.set_grad_enabled(False):
            # iterate through the validation set
            for (data, target) in val_dataloader:
                  # move the data into the device used for testing
                  data, target = data.to(device), target.to(device)

                  # perform a forward pass and calculate the training loss
                  predictions = model(data)
                  loss = criterion(predictions, target)

                  # add the training loss and keep track of the number of correct predictions
                  total_val_loss += loss
                  val_correct += (predictions.argmax(1) == target).type(torch.float).sum().item()

      # calculate the average training and validation loss
      avg_train_loss = total_train_loss / train_steps
      avg_val_loss = total_val_loss / val_steps

      # calculate the train and validation accuracy
      train_correct = train_correct / len(train_dataloader.dataset)
      val_correct = val_correct / len(val_dataloader.dataset)

      print(f'trian loss: {avg_train_loss: .3f}  .. train accuracy : {train_correct: .2f}')
      print(f'val loss: {avg_val_loss: .3f}  .. val accuracy : {val_correct: .2f}')

      # update the training and validation results
      history['train LOSS'].append(avg_train_loss.cpu().detach().numpy())
      history['train accuracy'].append(train_correct)
      history['val LOSS'].append(avg_val_loss.cpu().detach().numpy())
      history['val accuracy'].append(val_correct)

      # execute the learning rate scheduler and early stopping
      validation_loss = avg_val_loss.cpu().detach().numpy()
      lr_scheduler(validation_loss)
      early_stopping(validation_loss)

      # stop the training procedure due to no improvement while validating the model
      if early_stopping.early_stop_enabled:
            break

print(f'[INFO] Total training time: {datetime.now() - start_time} ...')

# move model back to cpu and save the trained model to disk
if device == 'cuda':
      model = model.to('cpu')
torch.save(model.state_dict(), 'model.pth')

#plotting the trainnig loss and accuracy overtime
plt.style.use('ggplot')
plt.figure()
plt.plot(history['train accuracy'], label='train accuracy')
plt.plot(history['val accuracy'], label='val accuracy')
plt.plot(history['train LOSS'], label='train LOSS')
plt.plot(history['val LOSS'], label='val LOSS')
plt.ylabel('loss/Accuracy')
plt.xlabel('No of epochs')
plt.title('Training Loss and Accuracy')
plt.legend(loc='upper right')
plt.savefig('plots')

# evaluate the model based on test set
model = model.to(device)
with torch.set_grad_enabled(False):
      # set the evaluation model
      model.eval()

      predictions = []

      # iterate throught the test set
      for (data, _) in test_dataloader:
            # move the data into the device used for testing
            data = data.to(device)

            # perform a forward pass and calculate the training loss
            output = model(data)
            output = output.argmax(axis=1).cpu().numpy()
            predictions.extend(output)

# evaluate the network
print('[INFO] evaluating network...')
actual = [label for _, label in test_data]
print(classification_report(actual, predictions, target_names=test_data.classes))
