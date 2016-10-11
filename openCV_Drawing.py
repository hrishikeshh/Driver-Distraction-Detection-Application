import numpy as np 
import cv2

img = cv2.imread('img2.jpg',cv2.IMREAD_COLOR)
#BGR
#line - name - start co - ending co - color - pixel
#cv2.line(img, (0,0) ,(150,150),(255,0,0), 5)

#rectangle
#cv2.rectangle(img, (15,25), (200,150), (0,0,0), 15)

#circle
#cv2.rectangle(img, (100,63), 50 , (200,12,34), -1)

pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
#pts = pts.reshape((-1,1,2))

cv2.polylines(img, [pts], True, (0,220,200),5 )

font = cv2.FONT_HERSHEY_SIMPLEX

#text - (image , text, starting Co, font, Size, Color, spacing, Antialising )
cv2.putText(img, 'OpenCV Tuts!', (0,130),font, 1 , (255,23,2), 1, cv2.LINE_AA)


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


