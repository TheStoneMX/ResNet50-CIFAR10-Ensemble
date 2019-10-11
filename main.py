from keras.applications.resnet50 import ResNet50, preprocess_input
from keras import models
from keras import layers
from keras import optimizers

from keras.utils import np_utils
from keras.models import load_model
from keras.datasets import cifar10
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

# load the training and testing data, then scale it into the
# range [0, 1]
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(y_train)
testY = lb.transform(y_test)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(200, 200, 3))


# loop over the number of models to train
for i in np.arange(0,5):
    # initialize the optimizer and model
    print("[INFO] training model {}/{}".format(i + 1, 5))

    model = models.Sequential()
    model.add(layers.UpSampling2D((2,2)))
    model.add(layers.UpSampling2D((2,2)))
    model.add(layers.UpSampling2D((2,2)))
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])

    history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

    # save the model to disk
    p = ["./models/model_{}.model".format(i)]
    model.save(os.path.sep.join(p))

    # evaluate the network
    predictions = model.predict(x_test, batch_size=64)
    report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames)

    # save the classification report to file
    p = ["./output/model_{}.txt".format(i)]
    f = open(os.path.sep.join(p), "w")
    f.write(report)
    f.close()
