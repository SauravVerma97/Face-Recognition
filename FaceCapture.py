#Capstone, Face Recognition
import numpy as np
import dlib
import cv2
import sys

cap = cv2.VideoCapture(0)

def boundingBox(parameter = 0):

	while(True):
	    # Capture frame-by-frame
	    ret, frame = cap.read()
	    frame = cv2.flip(frame, 1)

	    if parameter == 0:
		    #Detect the faces - Dlib
		    faceRects = dnnFaceDetector(frame, 1)

		    #Using Dlib - Blue Colour	
		    for faceRect in faceRects:
		    	x1 = faceRect.rect.left()
		    	y1 = faceRect.rect.top()
		    	x2 = faceRect.rect.right()
		    	y2 = faceRect.rect.bottom()
		    	frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

	    if parameter == 1:
		    #Detect the faces - OpenCV
		    faces = face_cascade.detectMultiScale(frame, scaleFactor = 1.1, minNeighbors = 7)

		    #Using OpenCV - Green Colour
		    for (x, y, w, h) in faces:
		    	frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

	    # Display the resulting frame
	    cv2.imshow('frame', frame)
	    if cv2.waitKey(1) & 0xFF == ord('w'):
	        break

	# When everything is done, release the capture
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":

	#Load files for OpenCV and Dlib
	face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
	dnnFaceDetector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")

	value = int(sys.argv[1])
	if value == 1:
		print("Using Haar Cascade from OpenCV - Faster")
	else:
		print("Using CNN Face Detection from Dlib - More Accurate")

	boundingBox(value)	
