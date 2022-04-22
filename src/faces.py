import numpy as np
import cv2
import pickle

# read face classifier file
faceCascade = cv2.CascadeClassifier(r"cascades\data\haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.yml')

labels = {}
with open('labels.pickle', 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}

video_capture = cv2.VideoCapture(0)

while True:

    # capture frame by frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        #print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        if conf>= 45 and conf <= 85:
            print(id_)
            print (labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 0, 0)
            stroke = 1
            cv2.putText(frame, name, (x,y), font, 0.5, color, stroke, cv2.LINE_AA)

        img_item = 'test_image.png'
        img_item2 = 'test_image2.jpeg'
        cv2.imwrite(img_item, roi_gray)
        cv2.imwrite(img_item2, roi_color)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0))



    # sDisplay resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
print("Face count is: ", len(faces))
video_capture.release()
cv2.destroyAllWindows()