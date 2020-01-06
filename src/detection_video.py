# import necessary packages
import argparse

import cv2
import cvlib as cv
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# command line arguments
arguments = argparse.ArgumentParser()
arguments.add_argument("-v", "--video", required=True,
                       help="path to input video")
args = arguments.parse_args()

model_path = "./model/detection.model"

# load model
model = load_model(model_path)

# open video
video = cv2.VideoCapture(args.video)

if not video.isOpened():
    print("Could not open video")
    exit()

classes = ['man', 'woman']

# loop through frames
while video.isOpened():

    # read frame from web_cam
    status, frame = video.read()

    if not status:
        print("Could not read frame")
        exit()

    # apply face detection from  open_cv
    face, confidence = cv.detect_face(frame)

    print(face)
    print(confidence)

    # loop through detected faces
    for index, current_face in enumerate(face):

        # color
        color = (0, 255, 0)

        # get corner points of face rectangle
        (start_x, start_y) = current_face[0], current_face[1]
        (end_x, end_y) = current_face[2], current_face[3]

        # draw rectangle over face
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

        # crop the detected face region
        face_crop = np.copy(frame[start_y:end_y, start_x:end_x])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # pre_processing for gender detection model
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

        Y = start_y - 10 if start_y - 10 > 10 else start_y + 10

        # write label and confidence above face rectangle
        cv2.putText(frame, label, (start_x, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # display output
    cv2.imshow("Result", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
video.release()
cv2.destroyAllWindows()
