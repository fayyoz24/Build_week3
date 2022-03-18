# import cv2
from __future__ import print_function

import pandas as pd
import torch
from face_dection import frame_prep
import torch.nn.functional as F
# import streamlit as st
import cv2
# st.write('ncoasmvsd')
modul, face_cascade, data_transform, emotion_dict = frame_prep()

vs = cv2.VideoCapture(0,cv2.CAP_DSHOW)
# frame_window = st.image([])
while True:
    (grabbed, img) = vs.read()
   
    # Read the frame
 
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    # cavas = np.zeros((300, 300,3), dtype='uint8')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        
        frame = img.copy()
        frame = data_transform(frame)
        frame = frame.unsqueeze(0)
        predictions = modul(frame)
        prob = F.softmax(predictions, dim=1)
        top_p, top_class = prob.topk(1, dim=1)
        top_p, top_class = top_p.item(), top_class.item()
        emotion_prob = [p.item() for p in prob[0]]
        emotion_value = emotion_dict.values()

        face_emotion = emotion_dict[top_class]
        face_text = f'{face_emotion}:{top_p*100:.2f}%'
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, face_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
        1.05, (0,255,0),2)
        cv2.putText(img, "FAYYOZJON ", (x-50, y-50), cv2.FONT_HERSHEY_SIMPLEX,
        1.05, (0,0,255),2)
    # frame_window.image(img)
    cv2.imshow('IMAGE', img)
    
    # Stop if escape key is pressed
    k = cv2.waitKey(1) & 0xff
    if k==ord("q"):
        break
