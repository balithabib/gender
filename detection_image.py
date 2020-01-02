
import argparse
import cv2
import cvlib as cv
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# command line arguments
param = argparse.ArgumentParser()
param.add_argument("-i", "--image", required=True,
                   help="path to input image")

args = param.parse_args()


# read input image
img = cv2.imread(args.image)

if img is None:
    print("Could not read input image")
    exit()

# load pre-trained model
model_path = "gender_detection.model"
model = load_model(model_path)

# detect faces in the image
face, confidence = cv.detect_face(img)

classes = ['man', 'woman']

# loop through detected faces
for index, currentFace in enumerate(face):
    # get corner points of face rectangle
    (startX, startY) = currentFace[0], currentFace[1]
    (endX, endY) = currentFace[2], currentFace[3]

    # draw rectangle over face
    cv2.rectangle(img, (startX, startY), (endX, endY), (255, 0, 0), 2)

    # crop the detected face region
    face_crop = np.copy(img[startY:endY, startX:endX])

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

    label = "{}: {:.2f}%".format(label, conf[index] * 100)

    Y = startY - 10 if startY - 10 > 10 else startY + 10

    # write label and confidence above face rectangle
    cv2.putText(img, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 0, 0), 2)

# display output
cv2.imshow("results", img)

# press any key to close window           
cv2.waitKey()

# save output
cv2.imwrite("gender_detection.jpg", img)

# release resources
cv2.destroyAllWindows()
