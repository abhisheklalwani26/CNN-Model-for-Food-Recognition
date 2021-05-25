import cv2
from numpy import asarray
from tensorflow import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import StandardScaler
import numpy as np


model = keras.models.load_model('128')
scaler = StandardScaler()
image2 = load_img('i9.jpg', target_size=(64, 64))
image2 = img_to_array(image2)
roi = image2.reshape(-1, 1)
scaler.fit(roi)
roi = scaler.transform(roi)
roi = roi.reshape(64, 64, 3)
roi = np.expand_dims(roi, axis=0)
ypred = model.predict(roi)
max = 0
ind = 0
labels = ["Bhel Puri", "Burger", "Butter Chicken", "Chicken Lollipop",
          "Chocolate Bar", "Chocolate Cake", "Gulab Jamun",  "Palak Paneer", "Pizza"]
for i in range(9):
    if ypred[0][i] > max:
        max = ypred[0][i]
        ind = i

print(labels[ind])
