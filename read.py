import time

import cv2 as cv

"""
#reading images:
img = cv.imread('Photos/cat.jpg')  # syntax to read an image in opencv
cv.imshow('Cat', img)  # syntax to show an image in opencv,arg1=title,arg2=img to be shown
cv.waitKey(0)  # ask the new window to n=be shown until a key is pressed
"""
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# reading videos
capture = cv.VideoCapture(0)  # use arg=0 if u want to use webcam,else path of video
while True:
    isTrue, frame = capture.read()  # isTrue gives a boolean if the frame was read or not,read each frame individually
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # convert img to grey img
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    for (x, y, w, h) in faces_rect:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    cv.imshow('rock face', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):  # stop video when 'd' key is pressed
        break
capture.release()  # release capture pointer
cv.destroyAllWindows()  # destroy the window
