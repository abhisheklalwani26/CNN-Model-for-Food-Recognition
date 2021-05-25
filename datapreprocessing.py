import cv2
import glob
from numpy import savez_compressed
from numpy import asarray
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import StandardScaler

image_data = list()
image_label = list()
scaler = StandardScaler()


def datacsv(label):
    # run this script for all labels
    dirList = glob.glob("Food/train_set/"+label+"/*.jpg")

    for img_path in dirList:
        file_name = img_path.split("/")[2]
        print(file_name)
        image2 = load_img(img_path, target_size=(64, 64))
        image2 = img_to_array(image2)
        roi = image2.reshape(-1, 1)
        scaler.fit(roi)
        roi = scaler.transform(roi)
        roi = roi.reshape(64, 64, 3)
        image_data.append(roi)
        image_label.append(labels.index(label))
        print(labels.index(label))


labels = ["Bhel Puri", "Burger", "Butter Chicken", "Chicken Lollipop",
          "Chocolate Bar", "Chocolate Cake", "Gulab Jamun",  "Palak Paneer", "Pizza"]

emotions = {0: "Bhel Puri", 1: "Burger", 2: "Butter Chicken",
            3: "Chicken Lollipop", 4: "Chocolate Bar", 5: "Chocolate Cake", 6: "Gulab Jamun", 7: "Palak Paneer", 8: "Pizza"}

for j in range(9):
    datacsv(labels[j])
image_data = asarray(image_data)
savez_compressed('color.npz', image_data, image_label)
