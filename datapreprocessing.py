import cv2
import glob
from numpy import savez_compressed
from sklearn import preprocessing

image_data = list()
image_label = list()


def datacsv(label):
    # run this script for all labels
    dirList = glob.glob("train_set/"+label+"/*.jpg")

    for img_path in dirList:
        file_name = img_path.split("/")[1]
        print(file_name)
        im = cv2.imread(img_path)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(im_gray, (64, 64), interpolation=cv2.INTER_AREA)
        roi.reshape(-1)
        roi = preprocessing.scale(roi)
        roi.reshape(64, 64)
        image_data.append(roi)
        image_label.append(labels.index(label))
        print(labels.index(label))


labels = ["Bhel Puri", "Burger", "Butter Chicken", "Chicken Lollipop",
          "Chocolate Bar", "Chocolate Cake", "Gulab Jamun",  "Palak Paneer", "Pizza"]

emotions = {0: "Bhel Puri", 1: "Burger", 2: "Butter Chicken",
            3: "Chicken Lollipop", 4: "Chocolate Bar", 5: "Chocolate Cake", 6: "Gulab Jamun", 7: "Palak Paneer", 8: "Pizza"}

for j in range(9):
    datacsv(labels[j])

savez_compressed('1.npz', image_data, image_label)
