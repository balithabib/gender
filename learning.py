# import necessary packages
import matplotlib

from model import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import random
import cv2
import os
import glob

matplotlib.use("Agg")

# handle command line arguments
param = argparse.ArgumentParser()
param.add_argument("-d", "--dataset", required=True,
                   help="path to input dataset (i.e., directory of images)")
param.add_argument("-m", "--model", type=str, default="detection_entrained.model",
                   help="path to output model")

args = param.parse_args()

# initial parameters
epochs = 100
batch_size = 64
dimension_image = (96, 96, 3)

data = []
labels = []

# load image files from dataset

image_files = [f for f in glob.glob(args.dataset + "/*/*", recursive=True) if not os.path.isdir(f)]
random.seed(42)
random.shuffle(image_files)

# create groud-truth label from the image path
for img in image_files:

    image = cv2.imread(img)

    image = cv2.resize(image, (dimension_image[0], dimension_image[1]))
    image = img_to_array(image)
    data.append(image)

    label = img.split(os.path.sep)[-2]
    if label == "woman":
        label = 1
    else:
        label = 0

    labels.append([label])

# pre-processing
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# split dataset for training and validation
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2,
                                                  random_state=42)
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# augmenting datset 
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# build model
model = Model.build(width=dimension_image[0], height=dimension_image[1], depth=dimension_image[2], classes=2)

# compile the model
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])

# train the model
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),
                        validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // batch_size,
                        epochs=epochs, verbose=1)

# save the model to disk
model.save(args.model)
