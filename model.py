from glob import glob
import cv2
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input,Lambda,Dense,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential

drowsy_image={'Drowsiness':glob('/content/drive/MyDrive/drowsy/practicefolder/drowsiness/*'),
 'Normal':glob('/content/drive/MyDrive/drowsy/practicefolder/normal/*')
 }

drowsy_labels={
    'Drowsiness':0,
    'Normal':1,
}
for drowsy_name, images in drowsy_image.items():
  print(drowsy_name)
  print(len(images))

x, y=[], []
for drowsy_name, images in drowsy_image.items():
  for image in images:
    img=cv2.imread(str(image))
    resize_img=cv2.resize(img,(224,224),3)
    x.append(resize_img)
    y.append(drowsy_labels[drowsy_name])

x=np.array(x)
y=np.array(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

model =Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(224,224,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(100,  activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2,  activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history=model.fit(x_train,y_train,batch_size=16,epochs=48)