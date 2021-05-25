from tkinter import *
from tkinter import filedialog
import os
import tkinter as tk
from PIL import Image, ImageTk
import cv2
from numpy import asarray
from tensorflow import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import StandardScaler
import numpy as np


def showImage():
    l.config(text="")
    fln = filedialog.askopenfilename(initialdir=os.getcwd(
    ), title="Select Image", filetypes=(("JPG", "*.jpg"), ("JPEG", "*.jpeg")))
    img = Image.open(fln)
    im1 = img
    im1 = im1.save("pred.jpg")
    img.thumbnail((350, 350))
    img = ImageTk.PhotoImage(img)
    lbl.configure(image=img)
    lbl.image = img


def predict():
    model = keras.models.load_model('128')
    scaler = StandardScaler()
    image2 = load_img('pred.jpg', target_size=(64, 64))
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

    prediction = labels[ind]
    l.config(text=prediction)

    return True


root = Tk()

frm = Frame(root)
frm.pack(side=BOTTOM, padx=15, pady=15)
lbl = Label(root)
lbl.pack()

btn = Button(frm, text="Select", command=showImage)
btn.pack(side=tk.LEFT)
btn1 = Button(frm, text="Predict", command=predict)
btn1.pack(side=tk.LEFT, padx=10)

l = Label(root)
l.config(font=("Courier", 14))
l.place(relx=0.5, rely=0.5, anchor='sw')
l.pack(side=BOTTOM)


root.title("CNN Model")
root.geometry("350x450")
root.mainloop()
