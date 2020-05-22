from __future__ import absolute_import, division, print_function, unicode_literals
import glob
import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.preprocessing import image
path = glob.glob(r"C:\Users\ashwi\Desktop\FinalYearProject\FYP\database\Training Data\Apple___Apple_scab/*.jpg")#Apple_Apple_scab
cv_img = []
for img in path:
    n = cv2.imread(img)
    n=np.double(n)/255
    b=cv2.resize(n,(32,32))
    cv_img.append(b)


path = glob.glob(r"C:\Users\ashwi\Desktop\FinalYearProject\FYP\database\Training Data\Apple___Black_rot/*.jpg") #Apple_Black_rot
cv_img1 = []
for img in path:
    n = cv2.imread(img)
    n=np.double(n)/255
    b=cv2.resize(n,(32,32))
    cv_img1.append(b)


path = glob.glob(r"C:\Users\ashwi\Desktop\FinalYearProject\FYP\database\Training Data\Apple___Cedar_apple_rust/*.jpg")#Apple_cedar_apple_rust
cv_img2 = []
for img in path:
    n = cv2.imread(img)
    n=np.double(n)/255
    b=cv2.resize(n,(32,32))
    cv_img2.append(b)


path = glob.glob(r"C:\Users\ashwi\Desktop\FinalYearProject\FYP\database\Training Data\Apple___healthy/*.jpg")#Apple_healthy
cv_img3 = []
for img in path:
    n = cv2.imread(img)
    n=np.double(n)/255
    b=cv2.resize(n,(32,32))
    cv_img3.append(b)
    
train_data= np.concatenate(((cv_img),(cv_img1),(cv_img2),(cv_img3)),axis=0)
a= [0]*1816
b= [1]*1787
c= [2]*1560
d= [3]*1809
train_label= a+b+c+d
print(np.size(train_data,0),np.size(train_data,1),np.size(train_data,2))
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(train_data, train_label, epochs=20)

###########testing###########
path = glob.glob(r"C:\Users\ashwi\Desktop\FinalYearProject\FYP\database\Test Data\Apple__Apple_scab/*.jpg")#Apple_Apple_scab
cv_img = []
for img in path:
    n = cv2.imread(img)
    n=np.double(n)/255
    b=cv2.resize(n,(32,32))
    cv_img.append(b)


path = glob.glob(r"C:\Users\ashwi\Desktop\FinalYearProject\FYP\database\Test Data\Apple__Black_rot/*.jpg") #Apple_Black_rot
cv_img1 = []
for img in path:
    n = cv2.imread(img)
    n=np.double(n)/255
    b=cv2.resize(n,(32,32))
    cv_img1.append(b)


path = glob.glob(r"C:\Users\ashwi\Desktop\FinalYearProject\FYP\database\Test Data\Apple__Cedar_apple_rust/*.jpg")#Apple_cedar_apple_rust
cv_img2 = []
for img in path:
    n = cv2.imread(img)
    n=np.double(n)/255
    b=cv2.resize(n,(32,32))
    cv_img2.append(b)


path = glob.glob(r"C:\Users\ashwi\Desktop\FinalYearProject\FYP\database\Test Data\Apple_healthy/*.jpg")#Apple_healthy
cv_img3 = []
for img in path:
    n = cv2.imread(img)
    n=np.double(n)/255
    b=cv2.resize(n,(32,32))
    cv_img3.append(b)
    
test_data= np.concatenate(((cv_img),(cv_img1),(cv_img2),(cv_img3)),axis=0)
a=[0]*200 
b=[1]*200 
c=[2]*200 
d=[3]*199 
test_label= a+b+c+d
test_loss, test_acc = model.evaluate(test_data,  test_label, verbose=2)

print('\nTest accuracy:', test_acc)
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()
tst=cv2.imread(file_path)
n=np.double(tst)/255
b=cv2.resize(n,(32,32))
b=image.img_to_array(b)
b=np.expand_dims(b,axis=0)
predictions = model.predict(b)
print((predictions[0][0]),(predictions[0][1]))

