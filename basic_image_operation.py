import numpy as np
import cv2

img = cv2.imread('image3.jpg', cv2.IMREAD_COLOR)

#location of the pixel
px = img[55,55]

#pixel can be also filled with any color 
#img[55,55] = [255,255,255]
#print(px)

#Region of Image 
roi = img[100:150, 100:150]

#ROI can also be filled with a specific color 
img[100:150, 100:150] = [255,255,255]

watch_face = img[37:111, 107:194]

img[0:74,0:87] = watch_face


print(roi)

cv2.imshow('Modified Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


