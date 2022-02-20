# import openCV module
import cv2
import numpy as np

# create face_cascade variable from classifier file
face_cascade = cv2.CascadeClassifier(r"face detector.xml")

# read image file to be used
img = cv2.imread("images\IMG_4.jpg", 1)

# convert image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces in image file
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)

# draw rectangle over the detected faces
for x,y,w,h in faces:
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)

print('Original Dimensions : ', img.shape)

scale_percent = 20  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

print('Resized Dimensions : ', resized.shape)


# show image
cv2.imshow("Resized image", resized)

# print the number of faces
print("Face count is:", len(faces))

cv2.waitKey(0)

cv2.destroyAllWindows()