from keras import models
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from tensorflow.keras import layers
from numpy import load
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

data = load('1.npz')
X, y = data['arr_0'], data['arr_1']

trainX, testX, trainY, testY = train_test_split(
    X, y, test_size=0.3, random_state=1)

print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)
trainY = np_utils.to_categorical(trainY, num_classes=9)
testY = np_utils.to_categorical(testY, num_classes=9)
trainX = trainX.reshape(len(trainX), 64, 64, 1)
testX = testX.reshape(len(testX), 64, 64, 1)

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), padding='same',
                        activation='relu', input_shape=(64, 64, 1)))
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))


model.add(layers.Flatten())

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(9, activation='softmax'))

model.compile(loss=categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model.fit(trainX, trainY,
          batch_size=32,
          epochs=80,
          verbose=1,
          validation_data=(testX, testY),
          # shuffle=True
          )

model.save('train_set')
