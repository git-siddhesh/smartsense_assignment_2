# open the camera using cv2 
# detect the face and create ractangle around it
# save the image in the folder with the name of the class as test 
# load the model
# predict the class of the image

import cv2
import numpy as np
import os
import time
import argparse


parser = argparse.ArgumentParser(description='Predicting a class')
parser.add_argument('--model', type=str, default='vgg', help='model name')
parser.add_argument('--blocks', type=int, default=16, help='number of blocks 1,3,5,..')
# # add num of clicks
# parser.add_argument('--clicks', type=int, default=5, help='number of clicks')
args = parser.parse_args()

# In[]: __________________________ Open the camera __________________________
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
os.makedirs('test', exist_ok=True)
while True:
    rval, frame = vc.read()
    cv2.imshow("preview", frame)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    elif key == 32: # space to capture
        cv2.imwrite(f"test/test_1.jpg", frame)
        break

cv2.destroyWindow("preview")
vc.release()

# Display the image with the predicted class name


imagePath = f"test/test_1.jpg"
img = cv2.imread(imagePath)
cv2.imshow("Image", img)
cv2.waitKey(0)

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.imshow("Image2", img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()