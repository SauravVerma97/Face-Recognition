#Capstone, Face Recognition

import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    faces = face_cascade.detectMultiScale(frame, scaleFactor = 1.05, minNeighbors = 5)
    for (x, y, w, h) in faces:
    	frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('w'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()