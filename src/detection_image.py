import argparse
import os
import random

import cv2
import cvlib as cv
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# command line arguments
arguments = argparse.ArgumentParser()
arguments.add_argument("-i", "--image", required=True,
                       help="path to input image")

args = arguments.parse_args()

# read image
image = cv2.imread(args.image)

if image is None:
    print("Could not read image")
    exit()

# load pre-trained model
model_path = "./model/detection.model"
model = load_model(model_path)

# detect faces in the image
face, confidence = cv.detect_face(image)

classes = ['man', 'woman']

# loop through detected faces
for index, current_face in enumerate(face):
    # rand color
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # get corner points of face rectangle
    (start_x, start_y) = current_face[0], current_face[1]
    (end_x, end_y) = current_face[2], current_face[3]

    # draw rectangle over face
    cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color, 2)

    # crop the detected face region
    face_crop = np.copy(image[start_y:end_y, start_x:end_x])

    # pre-processing for gender detection model
    face_crop = cv2.resize(face_crop, (96, 96))
    face_crop = face_crop.astype("float") / 255.0
    face_crop = img_to_array(face_crop)
    face_crop = np.expand_dims(face_crop, axis=0)

    # apply gender detection on face
    conf = model.predict(face_crop)[0]
    print(conf)
    print(classes)

    # get label with max accuracy
    index = np.argmax(conf)
    label = classes[index]

    label = "{}: {}%".format(label, int(conf[index] * 100))

    y = start_y - 10 if start_y - 10 > 10 else start_y + 10

    # write label and confidence above face rectangle
    cv2.putText(image, label, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# display output
cv2.imshow("Result", image)

# press any key to close window
cv2.waitKey()

# save output
name = os.path.basename(args.image).split('.')[0]
cv2.imwrite("./output/{}-output.jpg".format(name), image)

# release resources
cv2.destroyAllWindows()
