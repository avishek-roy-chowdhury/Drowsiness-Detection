import cv2
from google.colab.patches import cv2_imshow
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model

vid=cv2.VideoCapture('/content/drive/MyDrive/drowsy/dowsiness/32.mp4')
new_model=keras.models.load_model('/content/drive/MyDrive/drowsy/model')

def main(x):
  x=np.array(x)
  y_predict=new_model.predict(x)
  y_class=[np.argmax(i) for i in y_predict]
  return y_class


x=[]
count=0
add=0
while(True):
  success,frame=vid.read()
  if success==False:
    break
  image=cv2.imread(str(frame))
  resize_img=cv2.resize(frame,(227,227),3)
  x.append(resize_img)

  p=main(x)
  if p[0]==0:
    add+=1
    x.pop()
  else:
    x.pop()
  # cv2_imshow(resize_img)
  count+=1
  # if count==60:
  #   cv2_imshow(resize_img)
  # if count==1:
  #   break