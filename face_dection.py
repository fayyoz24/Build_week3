
from torchvision.transforms import ToPILImage, Grayscale, ToTensor, Resize
from torchvision import transforms

import cv2
import model 
import torch 
import torch.nn.functional as F

def frame_prep():
    haar_file = './haarcascade_frontalface_default.xml'

    face_cascade = cv2.CascadeClassifier(haar_file)

    emotion_dict = {
        0: "Happy",
        1: "Disgust",
        2: "Neutral",
        3: "Fear",
        4: "Angry"
    }

    modul = model.Emotion(num_of_channels=1,num_of_classes= 5)
    modul.load_state_dict(torch.load('./model.pth'))
    modul.eval()

    data_transform = transforms.Compose([
        ToPILImage(),
        Grayscale(num_output_channels=1),
        Resize((48,48)),
        ToTensor()
    ])
    return modul, face_cascade, data_transform, emotion_dict
# # img = 
# image = cv2.imread('./netral1.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # Detect the faces
# # cavas = np.zeros((300, 300,3), dtype='uint8')
# faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# # Draw the rectangle around each face
# for (x, y, w, h) in faces:
#     img = image.copy()
#     img = data_transform(img)
#     img = img.unsqueeze(0)
#     predictions = modul(img)
#     prob = F.softmax(predictions, dim=1)
#     top_p, top_class = prob.topk(1, dim=1)
#     top_p, top_class = top_p.item(), top_class.item()
#     emotion_prob = [p.item() for p in prob[0]]
#     emotion_value = emotion_dict.values()

#     face_emotion = emotion_dict[top_class]
#     face_text = f'{face_emotion}:{top_p*100:.2f}%'
#     cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
#     cv2.putText(image, face_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
#     1.05, (0,255,0),2)
# # cv2.waitkey(1)
# cv2.imshow('img',image)
# cv2.waitKey()
# cv2.destroyAllWindows()






        # for (i, (emotion, prob)) in enumerate(zip(emotion_value, emotion_prob)):
        #     prob_next = f'{emotion}: {prob * 100:.2f}%'
        #     width = int(prob*300)
        #     cv2.rectangle(cavas, (5,(i*50)+5), (width, (i*50)+50), (0,0,255), -1)
        #     cv2.putText(cavas, prob_next, (5, (i*50)+30),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)