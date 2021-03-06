from ast import arg
from statistics import mode
from torchvision.transforms import ToPILImage, Grayscale, ToTensor, Resize
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import cv2
from model import Emotion
import torch 
import argparse


parser = argparse.ArgumentParser()
model = Emotion(num_of_channels=1, num_of_classes=5)
model.load_state_dict(torch.load('./model.pth'))
model.eval()









cascade = './haarcascade_frontalface_default.xml'
# print(cascade)
face_classifier = cv2.CascadeClassifier(cascade)



emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



# transform = transforms.ToTensor()

# # Convert the image to PyTorch tensor
# tensor = transform(image)

# # print the converted image tensor
# print(tensor)



        if np.sum([roi_gray])!=0:
            roi = roi_gray
            roi = roi_gray.astype('float')/255.0
            transform = transforms.ToTensor()
            roi = transform(roi)
            # roi = ToTensor(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = model.forward(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()