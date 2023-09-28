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
parser.add_argument('--file', type=str, default=None, help='file name')

# # add num of clicks
# parser.add_argument('--clicks', type=int, default=5, help='number of clicks')
args = parser.parse_args()
assert  args.model in ['vgg'], "Current model is not supported please try vgg"
assert args.blocks in [1,3,16], "Current number of blocks is not supported please try 1,3,16"
assert os.path.exists(args.file), "File does not exist"

# In[]: __________________________ Open the camera __________________________

if  args.file == None:
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
imagePath = None
if args.file == None:
    imagePath = f"test/test_1.jpg"
else:
    imagePath = args.file
img = cv2.imread(imagePath)
cv2.imshow("Image", img)

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.imshow("Image2", img)

# load the model.h5 file from the model folder
# predict the class of the image

# In[]: __________________________ Load the model __________________________
# load the model

import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model(f"models/{args.model}_{args.blocks}.h5")

model.summary()

# In[]: __________________________ Predict the class of the image __________________________
# predict the class of the image
img = cv2.imread(imagePath)

img = cv2.resize(img, (128, 128))
img = np.expand_dims(img, axis=0)

class_name_int_map = dict()
for i, class_name in enumerate(os.listdir("dataset/train")):
    class_name_int_map[i] = class_name

print(class_name_int_map)

prediction = model.predict(img)
print("Prediction: ", prediction)

class_num = np.argmax(prediction)
print("Class: ", class_num, ": ", class_name_int_map[class_num])

img = cv2.imread(imagePath)
# write the class name on the image
img - cv2.putText(img, class_name_int_map[class_num], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 0, 0), 1, cv2.LINE_AA)
   
cv2.imshow("Prediction", img)
cv2.waitKey(0)

cv2.destroyAllWindows()