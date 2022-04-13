import os
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

def res_480():
    cap.set(3, 640)
    cap.set(4, 480)

res_480()
while True:
    ret, frame = cap.read()
    cv2.imshow('feame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cap.destroyAllWindows()