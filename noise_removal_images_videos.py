import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    #HSV hue sat value
    lower_color = np.array([10,10,10])
    upper_color = np.array([100,255,255])

    mask = cv2.inRange(hsv, lower_color, upper_color)
    res = cv2.bitwise_and(frame, frame, mask = mask)
    
    kernel = np.ones((15,15),np.float32)/225
    
    smoothed = cv2.filter2D(res, -1, kernel)
    blur = cv2.GaussianBlur(res, (15,15), 0)
    medianBlur = cv2.medianBlur(res,15)
    bilateralBlur = cv2.bilateralFilter(res, 15, 75, 75)
    
    
    cv2.imshow('frame', frame)
    #cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    cv2.imshow('blur', blur)
    cv2.imshow('smoothed', smoothed)
    cv2.imshow('medianBlur', medianBlur)
    cv2.imshow('bIlteralblur', bilateralBlur)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()

