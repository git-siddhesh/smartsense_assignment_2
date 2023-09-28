import cv2
import numpy as np
import os
import time

import argparse
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--model', type=str, default='vgg', help='model name')
parser.add_argument('--blocks', type=int, default=16, help='number of blocks 1,3,5,..')
parser.add_argument('--username', type=str, default='new_user', help='username')
# add num of clicks
parser.add_argument('--clicks', type=int, default=5, help='number of clicks')
args = parser.parse_args()



# In[]: __________________________ Cli application __________________________
# 2. open the camera
# 4. click some photos
# 5. create a folder with the name of the class as username and save the photos in that folder 



# In[]: __________________________ Open the camera __________________________
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
# os.makedirs('my_data', exist_ok=True)
if os.path.exists(f'data/{args.username}'):
    print(f'User {args.username} already exists')
os.makedirs(f'data/{args.username}', exist_ok=True)
i = 0
while True:
    rval, frame = vc.read()
    cv2.imshow("preview", frame)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    elif key == 32: # space to capture
        cv2.imwrite(f"data/{args.username}/{args.username}_{i+1}.jpg", frame)
        i += 1
        if i == args.clicks:
            break

cv2.destroyWindow("preview")
vc.release()


