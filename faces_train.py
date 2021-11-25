import os
import cv2 as cv
import numpy as np

people = ['rock', 'vin', 'gal']
dir = r'C:\Users\srini\Desktop\Charan\Pesu\School Project\Facerecogtest\Faces\train'
haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []


def create_train():
    for person in people:
        path = os.path.join(dir, person)
        label = people.index(person)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y + h, x:x + w]
                features.append(faces_roi)
                labels.append(label)


create_train()
print('training completed..............')
features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
# train the recognizer on features and labels lists
face_recognizer.train(features, labels)
np.save('features.npy', features)
np.save('labels.npy', labels)

# use recognizer
dir1 = r'C:\Users\srini\Desktop\Charan\Pesu\School Project\Facerecogtest\Faces\val\rock'
for img_name in os.listdir(dir1): 
    img_path = os.path.join(dir1, img_name)
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow('person', gray)

    # detect face

    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y + h, x:x + w]
        label, confidence = face_recognizer.predict(faces_roi)
        label = people[label]
        if confidence > 100:
            print('No records')
            label = 'Unknown'

        print('label=', label, ' confidence=', confidence)
        cv.putText(img, str(label), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    cv.imshow('detected face', img)
    cv.waitKey(0)
