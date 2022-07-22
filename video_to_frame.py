import cv2
from google.colab.patches import cv2_imshow

count=0
for i in range(1,16):
  path="/content/drive/MyDrive/drowsy/dowsiness/"+str(i)+".mp4"
  vid=cv2.VideoCapture(path)
  while(True):
    success,frame=vid.read()
    if success==False:
      break
    cv2.imwrite("/content/drive/MyDrive/drowsy/practicefolder/drowsiness/"+str(count)+'.jpg',frame)
    count+=1

count=0
for i in range(1,16):
  path="/content/drive/MyDrive/drowsy/normal/"+str(i)+".mp4"
  vid=cv2.VideoCapture(path)
  while(True):
    success,frame=vid.read()
    if success==False:
      break
    cv2.imwrite("/content/drive/MyDrive/drowsy/practicefolder/normal/"+str(count)+'.jpg',frame)
    count+=1