import cv2 as cv

img = cv.imread('Faces/train/rock/download.jfif')  # syntax to read an image in opencv
cv.imshow('rock', img)  # syntax to show an image in opencv,arg1=title,arg2=img to be shown

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert img to grey img
cv.imshow('GRAY rock', gray)  # syntax to show an image in opencv,arg1=title,arg2=img to be shown

haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
print('faces found=', len(faces_rect))

for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
cv.imshow('rock face',img)
cv.waitKey(0)  # ask the new window to n=be shown until a key is pressed
