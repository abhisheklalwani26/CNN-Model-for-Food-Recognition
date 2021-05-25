import cv2
from sklearn import preprocessing
from tensorflow import keras


model = keras.models.load_model('Food/train_set')

im = cv2.imread('i9.jpg')
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
roi = cv2.resize(im_gray, (64, 64), interpolation=cv2.INTER_AREA)
roi.reshape(-1)
roi = preprocessing.scale(roi)
roi = roi.reshape(1, 64, 64, 1)
ypred = model.predict(roi)
max = 0
ind = 0
for i in range(9):
    if ypred[0][i] > max:
        max = ypred[0][i]
        ind = i
labels = ["Bhel Puri", "Burger", "Butter Chicken", "Chicken Lollipop",
          "Chocolate Bar", "Chocolate Cake", "Gulab Jamun",  "Palak Paneer", "Pizza"]
print(labels[ind])
