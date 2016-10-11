import cv2
import numpy as np

cap = cv2.VideoCapture(0)

#for storing the video input
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter('outputVideo.avi', fourcc , 20.0, (640,480))

while True:
    #Analysis
    
    #this keep checking for the next frame 
    #if found then true 
    ret, frame = cap.read()
    #to get a modified frame BGR to Gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    out.write(frame)
    
    cv2.imshow('frame',frame)
    cv2.imshow('gray', gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#To release the camera 
cap.release()
#To release the video recorder 
out.release()

cv2.destroyAllWindows()


