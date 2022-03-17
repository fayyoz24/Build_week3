from ast import arg
from torchvision.transforms import ToPILImage, Grayscale, ToTensor, Resize
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import cv2
from model import Emotion
import torch 
import argparse


# cascade = 'https://github.com/akmadan/Emotion_Detection_CNN/blob/main/haarcascade_frontalface_default.xml'
# print(cascade)
parser = argparse.ArgumentParser()
model = Emotion(num_of_channels=1, num_of_classes=5)
model.load_state_dict(torch.load('./model.pth'))
# model.eval()