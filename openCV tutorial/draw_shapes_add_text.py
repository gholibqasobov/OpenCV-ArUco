import cv2 as cv
import numpy as np

img = cv.imread('Photos/cat_in_stop.jpg')
# cv.imshow('Cat', img)



"""
2 ways of writing text on image:
    on the original one 
    creating blank image
"""

# blank image
blank = np.zeros((500, 500, 3), dtype='uint8')
# cv.imshow('Blank', blank)

# Paint the image a certain colour
# r, g, b
blank[:] = 0, 255, 0

# cv.imshow('Green', blank)

# Paint a certain range
img[0:100, 0:100] = 0, 0, 175
# cv.imshow('Red', img)


# Draw a Rectangle
cv.rectangle(blank, (0, 0), (150, 150), (0, 0, 0), thickness=-1)
cv.imshow('Rectangle', blank)


# Draw a Circle
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 150, (30, 100, 200), thickness=2)
cv.imshow('Circle', blank)


# Draw a Line

cv.line(blank, (10, 10), (100, 200), (255, 255, 255), thickness=3)
cv.imshow('line', blank)


# Write text

cv.putText(img, 'Hello', (255, 255), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, (255, 0, 0), thickness=2)
cv.imshow('text', img)
cv.waitKey(0)
