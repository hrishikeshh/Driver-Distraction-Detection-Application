import cv2
import numpy as np

cap = cv2.VideoCapture('people-walking.mp4')
#cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv2.imshow('original',fgmask)
    cv2.imshow('fg', fgmask)
    k = cv2.waitKey(30) & 0xff
    
    if k == 27:
        break

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
