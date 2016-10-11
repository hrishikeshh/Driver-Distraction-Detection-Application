import cv2
import numpy as np
import matplotlib.pyplot as plt

#opening the image with filter
#by default It removes alpha channel if no filter is mentioned 
img = cv2.imread('img1.jpg',cv2.IMREAD_GRAYSCALE)
#Other options 
#IMREAD_COLOR - 1
#IMREAD_UNCHANGED = -1
#can also use numerical values of the filter direclty 

#In CV its BGR instead of RBG


#Using cv2
cv2.imshow('Title of the window', img)
cv2.waitKey(3000)
cv2.destroyAllWindows()

'''
#using Matplotlib

plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.plot([50,100],[80,100], 'c',linewidth = 5)
plt.show()

'''

'''
cv2.imwrite('greyScale_img1.png', img)
cv2.imshow('Grey' ,img)
'''


